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
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = 'save_checkpoints/'
# print(os.listdir(checkpoint_dir))
# Optimized warping module with caching
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

# Simplified convolution and deconvolution helpers
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # ReLU is faster than PReLU
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ReLU(inplace=True)  # ReLU is faster than PReLU
    )

# Simplified IFBlock with fewer layers
class IFBlock(nn.Module):
    def __init__(self, in_planes, c=32):  # Reduced channel count
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
        )  # Reduced from 8 to 4 layers
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

# Simplified Contextnet
c = 16  # Reduced channel count
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

# Streamlined Unet with fewer parameters
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = conv(17, 2*c)
        self.down1 = conv(4*c, 4*c)
        self.down2 = conv(8*c, 8*c)
        self.up0 = deconv(16*c, 4*c)
        self.up1 = deconv(8*c, 2*c)
        self.up2 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        # Downsample with fewer feature maps
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        
        c0_0 = F.interpolate(c0[0], size=(s0.size(2), s0.size(3)), mode="bilinear", align_corners=False)
        c1_0 = F.interpolate(c1[0], size=(s0.size(2), s0.size(3)), mode="bilinear", align_corners=False)
        s1 = self.down1(torch.cat((s0, c0_0, c1_0), 1))
        
        c0_1 = F.interpolate(c0[1], size=(s1.size(2), s1.size(3)), mode="bilinear", align_corners=False)
        c1_1 = F.interpolate(c1[1], size=(s1.size(2), s1.size(3)), mode="bilinear", align_corners=False)
        s2 = self.down2(torch.cat((s1, c0_1, c1_1), 1))
        
        # Upsample
        c0_2 = F.interpolate(c0[2], size=(s2.size(2), s2.size(3)), mode="bilinear", align_corners=False)
        c1_2 = F.interpolate(c1[2], size=(s2.size(2), s2.size(3)), mode="bilinear", align_corners=False)
        x = self.up0(torch.cat((s2, c0_2, c1_2), 1))
        
        x = self.up1(torch.cat((x, F.interpolate(s1, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)), 1))
        x = self.up2(torch.cat((x, F.interpolate(s0, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)), 1))
        
        x = self.conv(x)
        return x

# Simplified IFNet with fewer stages
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=32)  # Reduced channel count
        self.block1 = IFBlock(13+4, c=32)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:] if x.shape[1] > 6 else None
        
        # First stage
        flow, mask = self.block0(torch.cat((img0, img1), 1), None, scale=scale[0])
        
        # Second stage
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
        
        # Ensure refined has the same dimensions as merged
        refined = F.interpolate(refined, size=(merged.size(2), merged.size(3)), mode='bilinear', align_corners=False)
        
        # Final result
        final_output = torch.clamp(merged + refined, 0, 1)
        
        return flow, mask_final, final_output

# Optimized Dataset with caching and resizing
class FrameInterpolationDataset(Dataset):
    def __init__(self, data_dir, transform=None, resize=None, cache_size=100):
        self.data_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.frame_pairs = self._load_frame_pairs()
        self.cache = {}
        self.cache_size = cache_size
        
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
        
        # Read images
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        gt = cv2.imread(gt_path)
        
        # Check for errors
        if img0 is None or img1 is None or gt is None:
            raise ValueError(f"Could not read one of the images: {self.frame_pairs[idx]}")
        
        # Convert BGR to RGB
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        
        # Resize to reduce memory footprint
        if self.resize:
            img0 = cv2.resize(img0, self.resize, interpolation=cv2.INTER_AREA)
            img1 = cv2.resize(img1, self.resize, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, self.resize, interpolation=cv2.INTER_AREA)
        
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            gt = self.transform(gt)
        
        result = torch.cat((img0, img1, gt), 0)
        
        # Cache the result
        if len(self.cache) < self.cache_size:
            self.cache[idx] = result
            
        return result

# Optimized training function with mixed precision
def train_with_amp(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=10, patience=5):
    scaler = GradScaler()  # For mixed precision training
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for i, data in enumerate(train_dataloader):
            data = data.to(device, non_blocking=True)  # non_blocking=True can help with async data transfers
            
            # Mixed precision training
            with autocast(device_type='cuda'):  # <-- Fix here
                flow, mask, final_output = model(data)
                loss = criterion(final_output, data[:, 6:9])
            
            # Scale gradients and optimize
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        val_loss, val_psnr = validate_with_amp(model, val_dataloader, criterion)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Train Loss: {avg_train_loss:.6f}, "
              f"Validation Loss: {val_loss:.6f}, Validation PSNR: {val_psnr:.4f} dB")
        
        checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'psnr': val_psnr,
        }, checkpoint_path)

        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model saved with improved validation loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return best_val_loss

# Optimized validation function with mixed precision
def validate_with_amp(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device, non_blocking=True)
            
            with autocast(device_type='cuda'):  # Specify device_type='cuda'
                flow, mask, final_output = model(data)
                gt = data[:, 6:9]
                loss = criterion(final_output, gt)
            
            total_loss += loss.item()
            
            # Calculate PSNR
            mse = F.mse_loss(final_output, gt).item()
            if mse > 0:
                psnr = 10 * np.log10(1.0 / mse)
            else:
                psnr = float('inf')
            total_psnr += psnr
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr

# Optimized validation function with mixed precision
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
            
            # Calculate PSNR
            mse = F.mse_loss(final_output, gt).item()
            if mse > 0:
                psnr = 10 * np.log10(1.0 / mse)
            else:
                psnr = float('inf')
            total_psnr += psnr
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr

# Setup distributed training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Main training function for distributed training
def main_worker(rank, world_size, data_dir_train, data_dir_val, batch_size, resize):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = IFNet().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Define the image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets with resizing
    train_dataset = FrameInterpolationDataset(data_dir=data_dir_train, transform=transform, resize=resize)
    val_dataset = FrameInterpolationDataset(data_dir=data_dir_val, transform=transform, resize=resize)
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler
    )
    
    # Define optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.L1Loss()
    
    # Train with mixed precision
    train_with_amp(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=4, patience=3)
    
    # Clean up
    cleanup()

# Main program entry
if __name__ == "__main__":
    # Configuration
    # print(os.listdir(checkpoint_dir))
    data_dir_train = "datasets/train_10k"
    data_dir_val = "datasets/test_2k"
    batch_size = 16  # Increased from 2
    resize = (256, 256)  # Resize images to reduce memory usage
    world_size = torch.cuda.device_count()
    
    # For single GPU, use this:
    if world_size <= 1:
        # Create model
        model = IFNet().to(device)
        
        # Define the image transformation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Create datasets with resizing
        train_dataset = FrameInterpolationDataset(data_dir=data_dir_train, transform=transform, resize=resize)
        val_dataset = FrameInterpolationDataset(data_dir=data_dir_val, transform=transform, resize=resize)
        
        # Create data loaders
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
        
        # Define optimizer and criterion
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.L1Loss()
        
        # Train with mixed precision
        train_with_amp(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs=100, patience=3)
    else:
        # Use distributed training for multiple GPUs
        mp.spawn(
            main_worker,
            args=(world_size, data_dir_train, data_dir_val, batch_size, resize),
            nprocs=world_size,
            join=True
        )



