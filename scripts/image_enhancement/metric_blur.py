import os
import shutil
from PIL import Image
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score

# Define paths
output_dir = "/home/zy3724/git_repos/flatten/metric_dataset/celebmetric/celeb_subset_blur_output"
input_dir = os.path.join(output_dir, "input_dir")
enhanced_dir = "/home/zy3724/git_repos/flatten/metric_dataset/celebmetric/celeb_subset_blur_enhanced"
piscart_dir = os.path.join(output_dir, "piscart")
output_selected_dir = os.path.join(output_dir, "output_selected_dir")  # Your results directory
center_select = 4  # The center index for selecting images

# Create necessary directories
os.makedirs(input_dir, exist_ok=True)
os.makedirs(piscart_dir, exist_ok=True)
os.makedirs(output_selected_dir, exist_ok=True)  # Ensure the directory exists

# LPIPS model
lpips_alex = lpips.LPIPS(net='alex')  # Use AlexNet for LPIPS calculation

def calculate_psnr(image1, image2):
    """Calculate PSNR between two images."""
    return psnr(np.array(image1), np.array(image2))

def calculate_lpips(image1, image2):
    """Calculate LPIPS between two images."""
    # Convert PIL Images to NumPy arrays
    image1_np = np.array(image1)
    image2_np = np.array(image2)
    # Convert NumPy arrays to tensors using lpips.im2tensor
    image1_tensor = lpips.im2tensor(image1_np)
    image2_tensor = lpips.im2tensor(image2_np)
    # Compute LPIPS score
    with torch.no_grad():
        return lpips_alex(image1_tensor, image2_tensor).item()

def check_and_resize(image_path, output_path):
    """Check if the image already exists in output folder and resize if not."""
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping resizing...")
    else:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((512, 512))
        img_resized.save(output_path)
        print(f"Copied and resized {image_path} -> {output_path}")

def clear_directory(directory):
    """Remove all files in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Cleared all files in {directory}")

def update_output_selected_dir(center_select):
    """Remove existing images and move new images into output_selected_dir based on center_select."""
    # Clear the output_selected_dir
    clear_directory(output_selected_dir)
    
    # Define the source directory where images are located
    source_dir = output_dir  # Images are located directly in output_dir
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    
    # Move and rename images matching center_select to output_selected_dir
    for filename in os.listdir(source_dir):
        # Check if the filename matches the pattern *_blur_inversion_center{center_select}.png
        if filename.endswith(f"_blur_inversion_center{center_select}.png"):
            source_path = os.path.join(source_dir, filename)
            
            # Remove '_blur_inversion_center{center_select}.png' to get the new filename
            image_index_512 = filename[:-len(f"_blur_inversion_center{center_select}.png")]
            new_filename = image_index_512 + '.png'
            dest_path = os.path.join(output_selected_dir, new_filename)
            
            shutil.copy(source_path, dest_path)
            print(f"Copied and renamed {source_path} to {dest_path}")

def process_images_between_dirs(dir1, dir2, result_name):
    """Process images between two directories and compute metrics."""
    files1 = [f for f in os.listdir(dir1) if f.endswith(".png")]
    files2 = [f for f in os.listdir(dir2) if f.endswith(".png")]

    lpips_scores = []
    psnr_scores = []

    # Ensure images in dir2 are resized if necessary
    resized_dir2 = os.path.join(output_dir, f"{result_name}_resized")
    os.makedirs(resized_dir2, exist_ok=True)

    for file_name in files2:
        image_index = file_name.split('_')[0]  # Extract image index
        image2_path = os.path.join(dir2, file_name)
        resized_image2_path = os.path.join(resized_dir2, f"{image_index}_512.png")

        # Resize images in dir2 if not already resized
        check_and_resize(image2_path, resized_image2_path)

        # Find corresponding image in dir1
        image1_name = f"{image_index}_512.png"
        image1_path = os.path.join(dir1, image1_name)

        if os.path.exists(image1_path):
            # Load both images
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(resized_image2_path).convert('RGB')

            # Calculate LPIPS
            lpips_score = calculate_lpips(image1, image2)
            lpips_scores.append(lpips_score)

            # Calculate PSNR
            psnr_score = calculate_psnr(image1, image2)
            psnr_scores.append(psnr_score)

            print(f"{result_name} - Processed {image1_name}: LPIPS={lpips_score}, PSNR={psnr_score}")
        else:
            print(f"Image {image1_name} not found in {dir1}, skipping...")

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
    # Update output_selected_dir based on center_select
    update_output_selected_dir(center_select)

    # Process piscart images
    process_images_between_dirs(input_dir, piscart_dir, "piscart")

    # Process our results
    process_images_between_dirs(input_dir, output_selected_dir, "our_result")

if __name__ == "__main__":
    main()
