import os
import torch
import cv2
import numpy as np
import shutil
from combined import IFNet, warp
import torchvision.transforms as transforms
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create eval_data directory
eval_dir = "eval_data"
if os.path.exists(eval_dir):
    # Clear the directory if it already exists
    shutil.rmtree(eval_dir)
os.makedirs(eval_dir)
print(f"Created directory: {eval_dir}")

# Initialize model
model = IFNet().to(device)

# Load checkpoint if available
checkpoint_path = "save_checkpoints/model_epoch_50.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with PSNR: {checkpoint.get('psnr', 'N/A')} dB")
else:
    print("No checkpoint found, using uninitialized model")

model.eval()

# Define the preprocessing transforms as in your test script
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to preprocess images (same as in your test.py)
def preprocess_images(img0_path, img1_path, gt_path=None):
    # Read images
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
    # Check if images were loaded successfully
    if img0 is None or img1 is None:
        raise ValueError(f"Could not read images: {img0_path}, {img1_path}")
    
    gt = None
    if gt_path and os.path.exists(gt_path):
        gt = cv2.imread(gt_path)
        if gt is None:
            print(f"Warning: Could not read ground truth image: {gt_path}")
            gt = None
        else:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    # Save original dimensions for later
    original_size = (img0.shape[0], img0.shape[1])
    
    # Store original images for display
    orig_img0 = img0.copy()
    orig_img1 = img1.copy()
    
    # Resize to model's expected input size
    img0_resized = cv2.resize(img0, (256, 256))
    img1_resized = cv2.resize(img1, (256, 256))
    
    # Apply transformations
    img0_tensor = transform(img0_resized)
    img1_tensor = transform(img1_resized)
    
    # Stack tensors - make sure everything is on the same device
    input_tensor = torch.cat((img0_tensor, img1_tensor), 0).unsqueeze(0).to(device)
    
    return input_tensor, original_size, orig_img0, orig_img1, gt

# Function to denormalize and convert tensor to image (same as in your test.py)
def tensor_to_image(tensor):
    # Make sure tensor is on CPU for numpy conversion
    tensor = tensor.cpu()
    
    # Denormalize
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy array
    img = tensor.numpy().transpose(1, 2, 0) * 255
    return img.astype(np.uint8)

# Counter for output filename indexing
counter = 1

# Process all subdirectories in the dataset
dataset_dir = "datasets/test_2k"
print(f"Looking for frames in: {dataset_dir}")

# Get all immediate subdirectories in the test_2k folder
subdirs = []
try:
    # Get all items in the dataset directory
    items = os.listdir(dataset_dir)
    # Filter to get only directories
    subdirs = [item for item in items if os.path.isdir(os.path.join(dataset_dir, item))]
    print(f"Found {len(subdirs)} subdirectories in {dataset_dir}")
except Exception as e:
    print(f"Error listing subdirectories: {e}")

# Process each subdirectory
for subdir in subdirs:
    subdir_path = os.path.join(dataset_dir, subdir)
    
    # Check if this directory contains the required frames
    frame1_path = os.path.join(subdir_path, "frame1.png")
    frame2_path = os.path.join(subdir_path, "frame2.png")
    frame3_path = os.path.join(subdir_path, "frame3.png")
    
    # Check for different possible extensions if .png doesn't exist
    if not os.path.exists(frame1_path):
        for ext in ['.jpg', '.jpeg', '']:
            test_path = os.path.join(subdir_path, f"frame1{ext}")
            if os.path.exists(test_path):
                frame1_path = test_path
                break
    
    if not os.path.exists(frame2_path):
        for ext in ['.jpg', '.jpeg', '']:
            test_path = os.path.join(subdir_path, f"frame2{ext}")
            if os.path.exists(test_path):
                frame2_path = test_path
                break
    
    if not os.path.exists(frame3_path):
        for ext in ['.jpg', '.jpeg', '']:
            test_path = os.path.join(subdir_path, f"frame3{ext}")
            if os.path.exists(test_path):
                frame3_path = test_path
                break
    
    if not (os.path.exists(frame1_path) and os.path.exists(frame2_path) and os.path.exists(frame3_path)):
        print(f"Skipping {subdir_path} - missing required frames")
        continue
    
    print(f"Processing {subdir_path}")
    
    try:
        # Preprocess images
        input_tensor, original_size, orig_img0, orig_img1, gt = preprocess_images(
            frame1_path, frame3_path, frame2_path
        )
        
        # Generate interpolation
        start_time = time.time()
        with torch.no_grad():
            # Call the model to generate the interpolated frame
            flow, mask, interpolated = model(input_tensor)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        
        # Convert output tensor to image
        interpolated_img = tensor_to_image(interpolated[0])
        
        # Resize back to original dimensions if needed
        if interpolated_img.shape[:2] != original_size:
            interpolated_img = cv2.resize(interpolated_img, (original_size[1], original_size[0]))
        
        # Save the interpolated frame and ground truth
        output_io = os.path.join(eval_dir, f"io{counter}.png")
        output_gt = os.path.join(eval_dir, f"gt{counter}.png")
        
        # Save images (convert RGB to BGR for OpenCV)
        cv2.imwrite(output_io, cv2.cvtColor(interpolated_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(output_gt, cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
        
        print(f"Saved pair {counter}: {output_io} and {output_gt}")
        
        # Increment the counter for the next pair
        counter += 1
        
    except Exception as e:
        print(f"Error processing {subdir_path}: {e}")
        import traceback
        traceback.print_exc()

print(f"Processing complete. Generated {counter-1} pairs of images in {eval_dir}")