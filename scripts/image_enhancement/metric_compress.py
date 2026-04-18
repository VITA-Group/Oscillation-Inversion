import os
from PIL import Image
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score

# Define paths
degtype = 'compress'
output_dir = f"/home/zy3724/git_repos/flatten/metric_dataset/celebmetric/celeb_subset_{degtype}_output"
input_dir = f"/home/zy3724/git_repos/flatten/metric_dataset/celebmetric/celeb_subset_512"  # Directory containing input images
piscart_dir = os.path.join(output_dir, "piscart")
output_selected_dir = os.path.join(output_dir, "output_selected_dir")  # Your results directory

# Ensure that the necessary directories exist
os.makedirs(piscart_dir, exist_ok=True)
os.makedirs(output_selected_dir, exist_ok=True)

# LPIPS model
lpips_alex = lpips.LPIPS(net='alex')  # Use AlexNet for LPIPS calculation

def calculate_psnr(image1, image2):
    """Calculate PSNR between two images."""
    return psnr(np.array(image1), np.array(image2))

def calculate_lpips(image1, image2):
    """Calculate LPIPS between two images."""
    # Convert images to tensors
    image1_tensor = lpips.im2tensor(np.array(image1))
    image2_tensor = lpips.im2tensor(np.array(image2))
    # Compute LPIPS score
    with torch.no_grad():
        return lpips_alex(image1_tensor, image2_tensor).item()

def check_and_resize(image_path, output_path):
    """Resize image to 512x512 if needed and save to output_path."""
    if os.path.exists(output_path):
        pass  # Image already resized
    else:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((512, 512))
        img_resized.save(output_path)

def process_images_between_dirs(dir1, dir2, result_name):
    """Process images between two directories and compute metrics."""
    files1 = {f for f in os.listdir(dir1) if f.endswith(".png")}
    files2 = {f for f in os.listdir(dir2) if f.endswith(".png")}

    common_files = files1 & files2  # Process only files present in both directories

    if not common_files:
        print(f"No common images found between {dir1} and {dir2} for {result_name}.")
        return

    lpips_scores = []
    psnr_scores = []

    # Ensure images in dir2 are resized if necessary
    resized_dir2 = os.path.join(output_dir, f"{result_name}_resized")
    os.makedirs(resized_dir2, exist_ok=True)

    for file_name in common_files:
        image1_path = os.path.join(dir1, file_name)
        image2_path = os.path.join(dir2, file_name)
        resized_image2_path = os.path.join(resized_dir2, file_name)

        # Resize images in dir2 if not already resized
        check_and_resize(image2_path, resized_image2_path)

        # Load both images
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(resized_image2_path).convert('RGB')

        # Calculate LPIPS
        lpips_score = calculate_lpips(image1, image2)
        lpips_scores.append(lpips_score)

        # Calculate PSNR
        psnr_score = calculate_psnr(image1, image2)
        psnr_scores.append(psnr_score)

        print(f"{result_name} - Processed {file_name}: LPIPS={lpips_score}, PSNR={psnr_score}")

    # Calculate the FID score between dir1 and resized_dir2
    fid_value = fid_score.calculate_fid_given_paths(
        [dir1, resized_dir2],
        batch_size=50,
        device='cuda',
        dims=2048
    )

    # Output the results
    average_lpips = np.mean(lpips_scores)
    average_psnr = np.mean(psnr_scores)

    print(f"\nHere is the {result_name} average LPIPS number: {average_lpips}")
    print(f"Here is the {result_name} average PSNR number: {average_psnr}")
    print(f"Here is the {result_name} FID score: {fid_value}")

def main():
    # Process piscart images
    process_images_between_dirs(input_dir, piscart_dir, "piscart")

    # Process our results
    process_images_between_dirs(input_dir, output_selected_dir, "our_result")

if __name__ == "__main__":
    main()
