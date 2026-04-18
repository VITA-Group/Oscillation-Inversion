
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["WANDB_MODE"] = "disabled"



import time
import tqdm
import numpy as np
import itertools

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer
from diffusers.image_processor import VaeImageProcessor

#from FastSAM.fastsam import FastSAM, FastSAMPrompt

import torch
import torch.nn.functional as F



import math

#from cam_utils import orbit_camera, OrbitCamera





#from grid_put import mipmap_linear_grid_put_2d

import copy
import glob
import wandb
from PIL import Image, ImageDraw

import torchvision.transforms.functional as TF

import datetime
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.utils.checkpoint as checkpoint


import PIL
from PIL import Image

import math
import numpy as np
import torch

import PIL
from PIL import Image
def image_pt2pil(img): # input image tensor shoud be of shape 3,512,512
    input_img_torch_resized = img.permute(1, 2, 0)
    input_img_np = input_img_torch_resized.detach().cpu().numpy()
    input_img_np = (input_img_np * 255).astype(np.uint8)
    return Image.fromarray((input_img_np).astype(np.uint8)) 

def resize_image(image_path, new_width=None, new_height=None):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print("Failed to open image:", e)
        return None

    # Get original dimensions
    image = image.convert("RGB")
    orig_width, orig_height = image.size

    # Calculate new dimensions
    if new_width is not None:
        # Calculate the new height maintaining the aspect ratio
        aspect_ratio = orig_height / orig_width
        new_height = int(new_width * aspect_ratio)
    elif new_height is not None:
        # Calculate the new width maintaining the aspect ratio
        aspect_ratio = orig_width / orig_height
        new_width = int(new_height * aspect_ratio)
    else:
        print("Either new_width or new_height must be provided.")
        return None


     # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image, new_height, new_width


PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}


import torchvision
from torchvision import transforms


import matplotlib.pyplot as plt
import matplotlib.cm as cm


from src.flux_utils_multi import flux_optim

# Get list of input and output files
root_dir = os.path.join(os.path.dirname(__file__), '..')
name = 'women2old'
input_dir = os.path.join(root_dir, f'demo/{name}/source')
output_dir = os.path.join(root_dir, f'demo/{name}/target')
save_dir = os.path.join(root_dir, 'outputs/cache_visual')

name2 = 'women2black'
output_dir_2 = os.path.join(root_dir, f'demo/{name2}/target')



files_source = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
files_target = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
files_target_2 = sorted([f for f in os.listdir(output_dir_2) if f.endswith('.png')])

pils_source = []
pils_target = []
pils_target_2 = []


for f in files_source:
    source_img = Image.open(os.path.join(input_dir, f)).convert("RGB").resize((512,512))
    print(source_img.size)
    pils_source.append(source_img)

for f in files_target:
    target_img = Image.open(os.path.join(output_dir, f)).convert("RGB").resize((512,512))
    print(target_img.size)
    pils_target.append(target_img)

for f in files_target_2:
    target_img = Image.open(os.path.join(output_dir_2, f)).convert("RGB").resize((512,512))
    print(target_img.size)
    pils_target_2.append(target_img)

# print(len(pils_source)) 1
# print(pils_target) 1

optim_iters = 6
reoptim_iters = 30
edited_sequence = [] 
going_sequence = []
for r in range(reoptim_iters):
    print(r,r,r,r,r,r,r,r,r,r)
    print(pils_source)
    print(pils_target)
    ipviews_edited_images = flux_optim( pils_source , pils_target, pils_target_2, text_prompt = 'black skin old women, no letter, no number', optim_iters = optim_iters, pils_sde = None )
    edited_sequence = edited_sequence + ipviews_edited_images[0] 
    pils_source = [ ipviews_edited_images[0][-1] ]



# print(len(ipviews_edited_images)) 1
# print(len(ipviews_edited_images[0])) 4
def make_image_grid(images, save_path=None, row_width = optim_iters, name = 'None'):
    """
    Creates a grid of PIL images with 10 images per row
    Args:
        images: List of PIL images to arrange in a grid
        save_path: Path to save the grid image
    Returns:
        PIL Image containing the grid
    """
    # Get dimensions of input images (assuming all same size)
    w, h = images[0].size
    
    # Calculate number of rows needed
    num_images = len(images)
    images_per_row = row_width
    num_rows = (num_images + images_per_row - 1) // images_per_row  # Ceiling division
    
    # Create blank image for grid
    grid_w = w * images_per_row
    grid_h = h * num_rows
    grid_img = Image.new('RGB', (grid_w, grid_h), color='black')  # Black background
    
    # Paste each image into the grid
    for idx, img in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        grid_img.paste(img, (col * w, row * h))

        
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        grid_img.save(os.path.join(save_path, f'edit_sequence_grid_{name}.png'))


    return grid_img


savename = f'oldblackmix_optim_{optim_iters}_reoptim_{reoptim_iters}_P003001'
# Create and save grid for each set of edited images
#save_dir = '/home/zy3724/git_repos/dualdreamer/harmonydreamer/demo_difftrans/cache_visual'
grid = make_image_grid(edited_sequence, save_path=save_dir, row_width = optim_iters, name = f'{savename}')

# Save individual images from edited_sequence with index
for i, img in enumerate(edited_sequence):
    save_path = os.path.join(root_dir, f'outputs/edited_sequence_{i:03d}.png')
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

# Create and save gif from the sequence
gif_path = os.path.join(save_dir, f'edit_sequence_grid_{savename}.gif')
# Create directory if it doesn't exist
os.makedirs(os.path.dirname(gif_path), exist_ok=True)
# Save as GIF with 8 fps (125ms delay between frames)
edited_sequence[0].save(
    gif_path,
    save_all=True,
    append_images=edited_sequence[1:],
    duration=125,  # 125ms delay for 8fps
    loop=0
)


# # Save individual images from edited_sequence with index
# for i, img in enumerate(going_sequence):
#     save_path = f'/home/zy3724/git_repos/osci_interp/osci_interp_visual_optim3/edited_sequence_{i:03d}_going.png'
#     # Create directory if it doesn't exist
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     img.save(save_path)
