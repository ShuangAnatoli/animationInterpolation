import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab


from combined import IFNet, warp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = IFNet().to(device)


checkpoint_path = "save_checkpoints/model_epoch_50.pth"
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']} with PSNR: {checkpoint.get('psnr', 'N/A')} dB")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_images(img0_path, img1_path, gt_path=None):
    # Read images
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)
    
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
    
    original_size = (img0.shape[0], img0.shape[1])

    orig_img0 = img0.copy()
    orig_img1 = img1.copy()
    
    img0_resized = cv2.resize(img0, (256, 256))
    img1_resized = cv2.resize(img1, (256, 256))

    img0_tensor = transform(img0_resized)
    img1_tensor = transform(img1_resized)
    

    input_tensor = torch.cat((img0_tensor, img1_tensor), 0).unsqueeze(0).to(device)
    
    return input_tensor, original_size, orig_img0, orig_img1, gt

def tensor_to_image(tensor):
    tensor = tensor.cpu()
    
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    
    img = tensor.numpy().transpose(1, 2, 0) * 255
    return img.astype(np.uint8)


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)

def calculate_ssim(img1, img2):
    
    if img1.ndim == 3 and img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return ssim(gray1, gray2)
    return ssim(img1, img2)


def calculate_cd(img1, img2):
    lab1 = rgb2lab(img1 / 255.0)
    lab2 = rgb2lab(img2 / 255.0)
    
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=2))
    return np.mean(delta_e)

def calculate_ie(interpolated, gt):
    return np.mean(np.abs(interpolated.astype(np.float32) - gt.astype(np.float32)))

def interpolate_frames(img0_path, img1_path, output_path, gt_path=None):
    input_tensor, original_size, img0, img1, gt = preprocess_images(img0_path, img1_path, gt_path)
    
    start_time = time.time()
    with torch.no_grad():
        flow, mask, interpolated = model(input_tensor)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    
    interpolated_img = tensor_to_image(interpolated[0])
    
    interpolated_img = cv2.resize(interpolated_img, (original_size[1], original_size[0]))
    

    interpolated_img_bgr = cv2.cvtColor(interpolated_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, interpolated_img_bgr)
    
    metrics = {}
    if gt is not None:
        metrics['psnr'] = calculate_psnr(interpolated_img, gt)
        metrics['ssim'] = calculate_ssim(interpolated_img, gt)
        metrics['cd'] = calculate_cd(interpolated_img, gt)
        metrics['ie'] = calculate_ie(interpolated_img, gt)
        
        print(f"Metrics (compared to ground truth):")
        print(f"  PSNR: {metrics['psnr']:.4f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  Color Difference (CD): {metrics['cd']:.4f}")
        print(f"  Interpolation Error (IE): {metrics['ie']:.4f}")
    
  
    return img0, img1, interpolated_img, gt, metrics


def display_results(img0, img1, interpolated, gt, metrics, output_path):
    
    has_gt = gt is not None
    
    
    plt.figure(figsize=(15, 5 if not has_gt else 10))
    
    
    plt.subplot(2 if has_gt else 1, 3, 1)
    plt.imshow(img0)
    plt.title('Frame 1')
    plt.axis('off')
    
    plt.subplot(2 if has_gt else 1, 3, 2)
    plt.imshow(interpolated)
    plt.title('Interpolated Frame')
    plt.axis('off')
    
    plt.subplot(2 if has_gt else 1, 3, 3)
    plt.imshow(img1)
    plt.title('Frame 2')
    plt.axis('off')
    
    
    if has_gt:
        plt.subplot(2, 3, 4)
        plt.imshow(gt)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
 
        diff = np.abs(interpolated.astype(np.float32) - gt.astype(np.float32))
        plt.imshow(diff.astype(np.uint8))
        plt.title('Difference')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.axis('off')
        metrics_text = "\n".join([
            f"PSNR: {metrics['psnr']:.2f} dB",
            f"SSIM: {metrics['ssim']:.4f}",
            f"CD: {metrics['cd']:.2f}",
            f"IE: {metrics['ie']:.2f}"
        ])
        plt.text(0.1, 0.5, metrics_text, fontsize=12)
        plt.title('Metrics')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_comparison.png'))
    plt.show()

#CHANGE FILE PATH
test_pairs = [
    
    ("test_frames/frame1.png", "test_frames/frame3.png", "results/scene1_interpolated.png", "test_frames/frame2.png"),
   
]


os.makedirs("results", exist_ok=True)

for test_item in test_pairs:
    img0_path, img1_path, output_path = test_item[0], test_item[1], test_item[2]
    gt_path = test_item[3] if len(test_item) > 3 else None
    
    print(f"Processing: {img0_path} and {img1_path}")
    try:
        img0, img1, interpolated, gt, metrics = interpolate_frames(img0_path, img1_path, output_path, gt_path)
        display_results(img0, img1, interpolated, gt, metrics, output_path)
    except Exception as e:
        print(f"Error processing frames: {e}")
        import traceback
        traceback.print_exc()