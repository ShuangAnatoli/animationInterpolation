import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from datetime import timedelta

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = 'save_checkpoints/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    if tenFlow.dim() == 3:
        tenFlow = tenFlow.unsqueeze(1)
    if tenFlow.size(1) != 2:
        raise ValueError(f"tenFlow must have 2 channels. Got {tenFlow.size(1)} channels.")

    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU instead of ReLU
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU instead of ReLU
    )

#IFBLOCK 6 conv layers total 
class IFBlock(nn.Module):
    def __init__(self, in_planes, c=48):  # Increased channel count
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c, 3, 2, 1),
            conv(c, 2*c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(2*c, 2*c),
            conv(2*c, 2*c),
            conv(2*c, 2*c),
            conv(2*c, 2*c),
        )
        self.lastconv = nn.ConvTranspose2d(2*c, 5, 4, 2, 1)

    def forward(self, x, flow=None, scale=1):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1./scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1./scale, mode="bilinear", align_corners=False) * 1./scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale*2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale*2
        mask = tmp[:, 4:5]
        return flow, mask

#8, 16, 32, 48, 64 (no changes after 64)
c = 48

# Context Layers
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = conv(3, c)
        self.conv2 = conv(c, 2*c)
        self.conv3 = conv(2*c, 4*c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = warp(x, flow)
        
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f2 = warp(x, flow)
        
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f3 = warp(x, flow)
        
        return [f1, f2, f3]

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # Input has 17 channels 2×RGB images (6) + 2×warped images (6) + mask (1) + flow (4)
        self.down0 = conv(17, 2*c)
        self.down1 = conv(4*c, 4*c)
        self.down2 = conv(8*c, 8*c)
        #3c 3d
        self.up0 = deconv(8*c, 4*c)
        self.up1 = deconv(4*c, 2*c)
        self.up2 = deconv(2*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        
        c0_0 = F.interpolate(c0[0], size=(s0.size(2), s0.size(3)), mode="bilinear", align_corners=False)
        c1_0 = F.interpolate(c1[0], size=(s0.size(2), s0.size(3)), mode="bilinear", align_corners=False)
        s1 = self.down1(torch.cat((s0, c0_0, c1_0), 1))
        
        c1_1 = F.interpolate(c1[1], size=(s1.size(2), s1.size(3)), mode="bilinear", align_corners=False)
        s2 = self.down2(torch.cat((s1, c0_1, c1_1), 1))
        
        x = self.up0(s2)
        x = self.up1(x)
        x = self.up2(x)
        x = self.conv(x)
        return x

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=64)
        self.block1 = IFBlock(13+4, c=64)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] if x.shape[1] > 6 else None
        
        # First stage flow estimation
        flow, mask = self.block0(torch.cat((img0, img1), 1), None, scale=scale[0])
        
        # Second stage flow refinement
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        
        flow_d, mask_d = self.block1(torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[1])
        flow = flow + flow_d
        mask = mask + mask_d
        
        # Final warping
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        
        # Apply mask
        mask_final = torch.sigmoid(mask)
        merged = warped_img0 * mask_final + warped_img1 * (1 - mask_final)
        
        # Apply contextual enhancement
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        
        # Apply Unet refinement
        refined = self.unet(img0, img1, warped_img0, warped_img1, mask_final, flow, c0, c1)
        refined = refined[:, :3] * 2 - 1
        
        
        if refined.size(2) != merged.size(2) or refined.size(3) != merged.size(3):
            refined = F.interpolate(refined, size=(merged.size(2), merged.size(3)), mode='bilinear', align_corners=False)
        
       
        final_output = torch.clamp(merged + refined, 0, 1)
        
        return flow, mask_final, final_output

class FrameInterpolationDataset(Dataset):
    def __init__(self, data_dir, transform=None, resize=None, cache_size=100, augment=True):
        self.data_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.frame_pairs = self._load_frame_pairs()
        self.cache = {}
        self.cache_size = cache_size
        self.augment = augment
        
    def _load_frame_pairs(self):
        frame_pairs = []
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Dataset directory does not exist: {self.data_dir}")
            
        for seq in os.listdir(self.data_dir):
            seq_dir = os.path.join(self.data_dir, seq)
            
            if not os.path.isdir(seq_dir):
                continue
                
            frames = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            if len(frames) < 3:
                continue
                
            for i in range(len(frames) - 2):
                frame_pairs.append((
                    os.path.join(seq_dir, frames[i]), 
                    os.path.join(seq_dir, frames[i+2]), 
                    os.path.join(seq_dir, frames[i+1])
                ))
                
        if not frame_pairs:
            raise ValueError(f"No valid frame pairs found in {self.data_dir}")
            
        return frame_pairs

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        img0_path, img1_path, gt_path = self.frame_pairs[idx]
        
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        gt = cv2.imread(gt_path)
        
        if img0 is None or img1 is None or gt is None:
            raise ValueError(f"Could not read one of the images: {self.frame_pairs[idx]}")
        
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        
        if self.resize:
            img0 = cv2.resize(img0, self.resize, interpolation=cv2.INTER_AREA)
            img1 = cv2.resize(img1, self.resize, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, self.resize, interpolation=cv2.INTER_AREA)
        
        if self.augment:
            if np.random.random() > 0.5:
                img0 = np.flip(img0, axis=1).copy()
                img1 = np.flip(img1, axis=1).copy()
                gt = np.flip(gt, axis=1).copy()
            
            if np.random.random() > 0.5:
                img0 = np.flip(img0, axis=0).copy()
                img1 = np.flip(img1, axis=0).copy()
                gt = np.flip(gt, axis=0).copy()
            
            if np.random.random() > 0.5:
                brightness = 0.9 + np.random.random() * 0.2
                img0 = np.clip(img0 * brightness, 0, 255).astype(np.uint8)
                img1 = np.clip(img1 * brightness, 0, 255).astype(np.uint8)
                gt = np.clip(gt * brightness, 0, 255).astype(np.uint8)
        
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            gt = self.transform(gt)
        
        result = torch.cat((img0, img1, gt), 0)
    
        if len(self.cache) < self.cache_size:
            self.cache[idx] = result
            
        return result
    
def train_with_amp(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, 
                   num_epochs=10, patience=5, start_epoch=0, best_val_loss=float('inf'), best_val_psnr=0.0):
    scaler = GradScaler()
    patience_counter = 0
    
    # Runtime tracking
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for i, data in enumerate(train_dataloader):
            data = data.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                flow, mask, final_output = model(data)
                loss = criterion(final_output, data[:, 6:9])
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        val_loss, val_psnr = validate_with_amp(model, val_dataloader, criterion)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_remaining = num_epochs - (epoch + 1)
        est_time_remaining = avg_epoch_time * epochs_remaining
        
        epoch_time_str = str(timedelta(seconds=int(epoch_time)))
        est_remaining_str = str(timedelta(seconds=int(est_time_remaining)))
        total_elapsed_str = str(timedelta(seconds=int(time.time() - total_start_time)))
        
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Train Loss: {avg_train_loss:.6f}, "
              f"Validation Loss: {val_loss:.6f}, Validation PSNR: {val_psnr:.4f} dB")
        print(f"Time: {epoch_time_str} | Total: {total_elapsed_str} | Remaining: {est_remaining_str}")
        
        checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
            'psnr': val_psnr,
        }, checkpoint_path)

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_psnr_model.pth")
            print(f"Model saved with improved validation PSNR: {best_val_psnr:.4f} dB")
            patience_counter = 0
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_loss_model.pth")
            print(f"Model saved with improved validation loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    total_training_time = time.time() - total_start_time
    total_time_str = str(timedelta(seconds=int(total_training_time)))
    avg_epoch_time = total_training_time / min(num_epochs, epoch+1)
    avg_epoch_time_str = str(timedelta(seconds=int(avg_epoch_time)))
    
    print(f"Training completed in {total_time_str} ({avg_epoch_time_str} per epoch)")
    print(f"Best validation PSNR: {best_val_psnr:.4f} dB")
    
    with open(f"{checkpoint_dir}/training_time_stats.txt", "w") as f:
        f.write(f"Total training time: {total_time_str}\n")
        f.write(f"Average epoch time: {avg_epoch_time_str}\n")
        f.write(f"Total epochs: {epoch+1}\n")
        f.write(f"Best validation PSNR: {best_val_psnr:.4f} dB\n")
        f.write(f"Final learning rate: {scheduler.get_last_lr()[0]:.8f}\n")
        
        f.write("\nEpoch times:\n")
        for i, e_time in enumerate(epoch_times):
            e_time_str = str(timedelta(seconds=int(e_time)))
            f.write(f"Epoch {i+1}: {e_time_str}\n")
    
    return best_val_psnr, total_time_str

def validate_with_amp(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):
                flow, mask, final_output = model(data)
                gt = data[:, 6:9]
                loss = criterion(final_output, gt)
            
            total_loss += loss.item()
            
            mse = F.mse_loss(final_output, gt).item()
            if mse > 0:
                psnr = 10 * np.log10(1.0 / mse)
            else:
                psnr = float('inf')
            total_psnr += psnr
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr

def load_checkpoint_and_resume(checkpoint_path, model, optimizer, scheduler):
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        best_val_loss = checkpoint.get('loss', float('inf'))
        best_val_psnr = checkpoint.get('psnr', 0.0)
        
        print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Best validation loss: {best_val_loss:.6f}, Best PSNR: {best_val_psnr:.4f} dB")
        
        return model, optimizer, scheduler, start_epoch, best_val_loss, best_val_psnr
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e
    
if __name__ == "__main__":
    data_dir_train = "datasets/train_10k"
    data_dir_val = "datasets/test_2k"
    batch_size = 16
    resize = (256, 256)
    world_size = torch.cuda.device_count()
    
    load_from_checkpoint = True
    checkpoint_path = f"{checkpoint_dir}/model_epoch_49.pth"
    
    # For single GPU
    if world_size <= 1:
        model = IFNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        
        start_epoch = 0
        best_val_loss = float('inf')
        best_val_psnr = 0.0

        if load_from_checkpoint:
            try:
                model, optimizer, scheduler, start_epoch, best_val_loss, best_val_psnr = load_checkpoint_and_resume(
                    checkpoint_path, model, optimizer, scheduler
                )
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch instead.")

                model = IFNet().to(device)
                optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
                scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
                start_epoch = 0
                best_val_loss = float('inf')
                best_val_psnr = 0.0
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        train_dataset = FrameInterpolationDataset(
            data_dir=data_dir_train, 
            transform=transform, 
            resize=resize,
            augment=True
        )
        
        val_dataset = FrameInterpolationDataset(
            data_dir=data_dir_val, 
            transform=transform, 
            resize=resize,
            augment=False
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        criterion = nn.L1Loss()

        best_psnr, total_time = train_with_amp(
            model, 
            train_dataloader, 
            val_dataloader, 
            optimizer, 
            scheduler, 
            criterion, 
            num_epochs=50, 
            patience=10,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            best_val_psnr=best_val_psnr
        )
        print(f"Training completed in {total_time} with best PSNR of {best_psnr:.4f} dB")
    else:

        print("Distributed training not implemented in this example")