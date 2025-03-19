import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Warping module
backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    # Ensure tenFlow has the correct shape (batch_size, 2, height, width)
    if tenFlow.dim() == 3:
        tenFlow = tenFlow.unsqueeze(1)  # Add channel dimension if missing
    if tenFlow.size(1) != 2:
        raise ValueError(f"tenFlow must have 2 channels (horizontal and vertical flow). Got {tenFlow.size(1)} channels.")

    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

# Convolution and deconvolution helpers
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )

# IFBlock module
class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow=None, scale=1):  # Ensure `scale` is a default argument
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask

# Contextnet module
c = 16
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = conv(3, c)
        self.conv2 = conv(c, 2*c)
        self.conv3 = conv(2*c, 4*c)
        self.conv4 = conv(4*c, 8*c)

    def forward(self, x, flow):
        x = self.conv1(x)  # Output: (batch, c, H/2, W/2)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = warp(x, flow)  # Output: (batch, c, H/2, W/2)
        
        x = self.conv2(x)  # Output: (batch, 2*c, H/4, W/4)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f2 = warp(x, flow)  # Output: (batch, 2*c, H/4, W/4)
        
        x = self.conv3(x)  # Output: (batch, 4*c, H/8, W/8)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f3 = warp(x, flow)  # Output: (batch, 4*c, H/8, W/8)
        
        x = self.conv4(x)  # Output: (batch, 8*c, H/16, W/16)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f4 = warp(x, flow)  # Output: (batch, 8*c, H/16, W/16)
        
        return [f1, f2, f3, f4]

# Unet module
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = conv(17, 2*c)  # Input: (batch, 17, H, W) -> Output: (batch, 2*c, H/2, W/2)
        self.down1 = conv(4*c, 4*c)  # Input: (batch, 4*c, H/2, W/2) -> Output: (batch, 4*c, H/4, W/4)
        self.down2 = conv(8*c, 8*c)  # Input: (batch, 8*c, H/4, W/4) -> Output: (batch, 8*c, H/8, W/8)
        self.down3 = conv(16*c, 16*c)  # Input: (batch, 16*c, H/8, W/8) -> Output: (batch, 16*c, H/16, W/16)
        self.up0 = deconv(32*c, 8*c)  # Input: (batch, 32*c, H/16, W/16) -> Output: (batch, 8*c, H/8, W/8)
        self.up1 = deconv(16*c, 4*c)  # Input: (batch, 16*c, H/8, W/8) -> Output: (batch, 4*c, H/4, W/4)
        self.up2 = deconv(8*c, 2*c)  # Input: (batch, 8*c, H/4, W/4) -> Output: (batch, 2*c, H/2, W/2)
        self.up3 = deconv(4*c, c)  # Input: (batch, 4*c, H/2, W/2) -> Output: (batch, c, H, W)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)  # Input: (batch, c, H, W) -> Output: (batch, 3, H, W)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        # Downsample
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))  # Output: (batch, 2*c, H/2, W/2)
        
        # Ensure c0[0] and c1[0] match the spatial dimensions of s0
        c0_0_resized = F.interpolate(c0[0], size=(s0.size(2), s0.size(3)), mode="bilinear", align_corners=False)
        c1_0_resized = F.interpolate(c1[0], size=(s0.size(2), s0.size(3)), mode="bilinear", align_corners=False)
        s1 = self.down1(torch.cat((s0, c0_0_resized, c1_0_resized), 1))  # Output: (batch, 4*c, H/4, W/4)
        
        # Ensure c0[1] and c1[1] match the spatial dimensions of s1
        c0_1_resized = F.interpolate(c0[1], size=(s1.size(2), s1.size(3)), mode="bilinear", align_corners=False)
        c1_1_resized = F.interpolate(c1[1], size=(s1.size(2), s1.size(3)), mode="bilinear", align_corners=False)
        s2 = self.down2(torch.cat((s1, c0_1_resized, c1_1_resized), 1))  # Output: (batch, 8*c, H/8, W/8)
        
        # Ensure c0[2] and c1[2] match the spatial dimensions of s2
        c0_2_resized = F.interpolate(c0[2], size=(s2.size(2), s2.size(3)), mode="bilinear", align_corners=False)
        c1_2_resized = F.interpolate(c1[2], size=(s2.size(2), s2.size(3)), mode="bilinear", align_corners=False)
        s3 = self.down3(torch.cat((s2, c0_2_resized, c1_2_resized), 1))  # Output: (batch, 16*c, H/16, W/16)
        
        # Upsample
        # Ensure c0[3] and c1[3] match the spatial dimensions of s3
        c0_3_resized = F.interpolate(c0[3], size=(s3.size(2), s3.size(3)), mode="bilinear", align_corners=False)
        c1_3_resized = F.interpolate(c1[3], size=(s3.size(2), s3.size(3)), mode="bilinear", align_corners=False)
        x = self.up0(torch.cat((s3, c0_3_resized, c1_3_resized), 1))  # Output: (batch, 8*c, H/8, W/8)
        
        # Upsample and concatenate
        # Ensure s2 matches the spatial dimensions of x
        s2_resized = F.interpolate(s2, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        x = self.up1(torch.cat((x, s2_resized), 1))  # Output: (batch, 4*c, H/4, W/4)
        
        # Ensure s1 matches the spatial dimensions of x
        s1_resized = F.interpolate(s1, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        x = self.up2(torch.cat((x, s1_resized), 1))  # Output: (batch, 2*c, H/2, W/2)
        
      
        s0_resized = F.interpolate(s0, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        x = self.up3(torch.cat((x, s0_resized), 1))  # Output: (batch, c, H, W)
        
       
        x = self.conv(x)  

# StrokeLevelModel module
class StrokeLevelModel(nn.Module):
    def __init__(self):
        super(StrokeLevelModel, self).__init__()
        c = 24

        self.fuse_block = nn.Sequential(
            nn.Conv2d(7, 2 * c, 3, 1, 1),  # Expects 7 input channels
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.fuse_block1 = nn.Sequential(
            nn.Conv2d(6, 2 * c, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.fuse_block2 = nn.Sequential(
            nn.Conv2d(6, 2 * c, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Add a channel reduction layer
        self.channel_reduction = nn.Conv2d(144, 9, 1, 1, 0)  # Reduce 144 channels to 9

        self.final_fuse_block = nn.Sequential(
            nn.Conv2d(9, 2 * c, 3, 1, 1),  # Expects 9 input channels
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * c, 3, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.points_fuse = nn.Sequential(
            nn.Conv2d(1, 2 * c, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2 * c, 1, 3, 1, 1),  # Output 1 channel
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, img0, img1, points, region_flow):
        B, _, H, W = img0.size()

        # Ensure region_flow has 2 channels (horizontal and vertical flow)
        if region_flow.size(1) == 4:
            region_flow = region_flow[:, :2]  # Use only the first 2 channels
        elif region_flow.size(1) != 2:
            raise ValueError(f"region_flow must have 2 channels (horizontal and vertical flow). Got {region_flow.size(1)} channels.")

        # Warp the images using the flow
        warped_img0 = warp(img0, region_flow[:, :2])  # Use the first 2 channels for flow
        warped_img1 = warp(img1, region_flow[:, :2])  # Use the first 2 channels for flow

        # Process points
        points = self.points_fuse(points)

        # Fuse features
        fused_img0 = self.fuse_block1(torch.cat([warped_img0, warped_img1], dim=1))
        fused_img1 = self.fuse_block2(torch.cat([warped_img0, warped_img1], dim=1))

        # Final fusion
        x = self.fuse_block(torch.cat([warped_img0, warped_img1, points], dim=1))

        # Concatenate fused_img0, fused_img1, and x
        concat_features = torch.cat([fused_img0, fused_img1, x], dim=1)

        # Reduce the number of channels to 9
        concat_features = self.channel_reduction(concat_features)

        # Generate the final output
        pred = self.final_fuse_block(concat_features)
        pred = torch.sigmoid(pred)

        return pred

# IFNet module with StrokeLevelModel
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, c=90)
        self.block1 = IFBlock(13+4, c=90)
        self.block2 = IFBlock(13+4, c=90)
        self.block_tea = IFBlock(16+4, c=90)
        self.contextnet = Contextnet()
        self.unet = Unet()
        self.stroke_level_model = StrokeLevelModel()  # Add StrokeLevelModel

    def forward(self, x, scale=[4, 2, 1], timestep=0.5):  # Ensure `scale` is a default argument
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]  # In inference time, gt is None
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        
        # Resize `tmp` to match the spatial dimensions of `merged[2]`
        res = F.interpolate(tmp, size=(merged[2].size(2), merged[2].size(3)), mode="bilinear", align_corners=False)
        res = res[:, :3] * 2 - 1
        
        # Add `res` to `merged[2]`
        merged[2] = torch.clamp(merged[2] + res, 0, 1)

        # Pass the output through the StrokeLevelModel
        points = torch.zeros_like(img0[:, :1])  # Dummy points tensor (replace with actual points if available)
        stroke_output = self.stroke_level_model(img0, img1, points, flow)
        
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill, stroke_output

# Dataset class
class FrameInterpolationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.frame_pairs = self._load_frame_pairs()

    def _load_frame_pairs(self):
        frame_pairs = []
        for seq in os.listdir(self.data_dir):
            seq_dir = os.path.join(self.data_dir, seq)
            frames = sorted(os.listdir(seq_dir))
            for i in range(len(frames) - 2):
                frame_pairs.append((os.path.join(seq_dir, frames[i]), os.path.join(seq_dir, frames[i+2]), os.path.join(seq_dir, frames[i+1])))
        return frame_pairs

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        img0_path, img1_path, gt_path = self.frame_pairs[idx]
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        gt = cv2.imread(gt_path)
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            gt = self.transform(gt)
        return torch.cat((img0, img1, gt), 0)

# Training loop
def train(model, dataloader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill, stroke_output = model(data)
            loss = criterion(merged[2], data[:, 6:9]) + criterion(stroke_output, data[:, 6:9])  # Add stroke_output loss
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item()}")

# Main script
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 4

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = FrameInterpolationDataset(data_dir="datasets/train_10k", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, and loss function
    model = IFNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    # Train the model
    train(model, dataloader, optimizer, criterion, num_epochs=num_epochs)


    #change loss function to the original papers
    #early stopping
    #save the best model
    #hyperparameter tuning
    #generate images
    #draw the model
    #make an animation sw (optional)


