import os
import math
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import lpips
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from eval.distance_transform_v0 import ChamferDistance2dMetric
import torch
from pytorch_msssim import ssim_matlab as ssim
import glob
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def calc_psnr(pred, gt):
    return -10 * math.log10(((pred - gt) * (pred - gt)).mean())

# Initialize metrics and lists
loss_fn_alex = lpips.LPIPS(net='alex')
transform = transforms.Compose([
    transforms.PILToTensor()
])
cd = ChamferDistance2dMetric()
torch.set_printoptions(precision=4)

# Initialize metric lists
psnrlisto = []
ssimlisto = []
ielisto = []
cdlisto = []

# Define the directory with the eval data
eval_dir = "eval_data"

# Check if directory exists
if not os.path.exists(eval_dir):
    print(f"Error: Directory '{eval_dir}' not found")
else:
    # Find all io (interpolated) files
    io_files = glob.glob(os.path.join(eval_dir, "io*.png"))
    print(f"Found {len(io_files)} interpolated images in {eval_dir}")
    
    # Sort files to ensure proper matching
    io_files.sort(key=lambda x: int(re.findall(r'io(\d+)\.png', x)[0]))
    
    # Process each pair of images
    for io_path in io_files:
        # Extract the index number
        index_match = re.search(r'io(\d+)\.png', io_path)
        if not index_match:
            print(f"Skipping {io_path} - unable to extract index")
            continue
            
        index = index_match.group(1)
        gt_path = os.path.join(eval_dir, f"gt{index}.png")
        
        # Check if the corresponding ground truth exists
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth image '{gt_path}' not found for '{io_path}'")
            continue
        
        print(f"Processing pair {index}: {io_path} and {gt_path}")
        
        try:
            # Load predicted image
            pio = Image.open(io_path).convert("RGB").resize((384, 192))
            pred1o = transform(pio).unsqueeze(0).float()/255.
            predo = np.asarray(pio)
            
            # Load ground truth image
            gi = Image.open(gt_path).convert("RGB").resize((384, 192))
            gt1 = transform(gi).unsqueeze(0).float()/255.
            gt = np.asarray(gi)
            
            # Calculate metrics
            try:
                psnr_val = psnr(predo, gt)
                psnrlisto.append(psnr_val)
                print(f"PSNR: {psnr_val:.4f}")
            except Exception as e:
                print(f"Error calculating PSNR: {e}")
                
            try:
                ie_val = math.sqrt(mse(predo, gt))
                ielisto.append(ie_val)
                print(f"IE: {ie_val:.4f}")
            except Exception as e:
                print(f"Error calculating IE: {e}")
                
            try:
                cd_val = cd.calc(pred1o, gt1)
                cdlisto.append(cd_val)
                print(f"CD: {cd_val:.4f}")
            except Exception as e:
                print(f"Error calculating Chamfer Distance: {e}")
                
            try:
                ssim_val = ssim(pred1o, gt1)
                ssimlisto.append(ssim_val)
                print(f"SSIM: {ssim_val:.4f}")
            except Exception as e:
                print(f"Error calculating SSIM: {e}")
                
        except Exception as e:
            print(f"Error processing images: {e}")
            import traceback
            traceback.print_exc()

# Calculate averages with safety checks
psnr_avgo = np.average(psnrlisto) if psnrlisto else 0
ie_avgo = np.average(ielisto) if ielisto else 0
cd_avgo = np.average(cdlisto) if cdlisto else 0
ssim_avgo = np.average(ssimlisto) if ssimlisto else 0

# Print individual metrics for all processed pairs
print("\nMetrics for each processed pair:")
for i, (psnr_val, ssim_val, cd_val, ie_val) in enumerate(zip(psnrlisto, ssimlisto, cdlisto, ielisto)):
    print(f"Pair {i+1}:")
    print(f"  PSNR: {psnr_val:.4f}")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  CD: {cd_val:.4f}")
    print(f"  IE: {ie_val:.4f}")

# Print average results
print("\nFinal Results (Averages):")
print(f"Number of image pairs processed: {len(psnrlisto)}")
print("cdval: {:.4f}".format(cd_avgo))
print("ssim: {:.4f}".format(ssim_avgo))
print("psnr: {:.4f}".format(psnr_avgo))
print("ie: {:.4f}".format(ie_avgo))

# Optionally save results to a file
results_file = "eval_results.txt"
with open(results_file, "w") as f:
    f.write("Evaluation Results\n")
    f.write(f"Number of image pairs processed: {len(psnrlisto)}\n")
    f.write("cdval: {:.4f}\n".format(cd_avgo))
    f.write("ssim: {:.4f}\n".format(ssim_avgo))
    f.write("psnr: {:.4f}\n".format(psnr_avgo))
    f.write("ie: {:.4f}\n".format(ie_avgo))

print(f"\nResults saved to {results_file}")