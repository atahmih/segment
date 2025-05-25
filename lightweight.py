import os
import numpy as np
import torch
from torchvision import transforms
import requests
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import AutoPipelineForInpainting
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import argparse

from utils import setup_sam, find_wall_segments
# there's a part of SAM that outputs a float64 tensor, causing an error on the mps device. 
# Explicitly casting it to float32
import segment_anything.utils.transforms as sam_transforms
original_apply_coords = sam_transforms.ResizeLongestSide.apply_coords

def patched_apply_coords(self, coords, original_size):
    # Force float32 instead of float64
    coords = coords.astype(np.float32)
    result = original_apply_coords(self, coords, original_size)
    return result.astype(np.float32)  # Ensure output is also float32
# Apply the monkey patch
sam_transforms.ResizeLongestSide.apply_coords = patched_apply_coords

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def modify_walls(image_path, prompt, negative_prompt, output_path='modified_room.png'):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set up SAM
    sam = setup_sam()
    
    # Find wall segments
    print('Finding wall segments...')
    wall_masks = find_wall_segments(image_rgb, sam)
    
    if not wall_masks:
        print('No wall segments found!')
        return
    
    print(f'Found {len(wall_masks)} potential wall segments')
    
    # Combine wall masks into a single mask
    combined_mask = np.zeros_like(wall_masks[0]['segmentation'], dtype=bool)
    for mask in wall_masks:
        combined_mask = combined_mask | mask['segmentation']
    
    # Convert combined mask to image format for inpainting
    mask_image = Image.fromarray((combined_mask).astype(np.uint8) * 255)
    mask_image.save('test_mask.png', format='PNG')
    # Convert original image to PIL format
    original_image = Image.fromarray(image_rgb)
    
    # Load Stable Diffusion inpainting pipeline
    print('Loading Stable Diffusion inpainting model...')
    pipe = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint",  # Lighter alternative
        torch_dtype=torch.float16
    )

    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
       
    pipe = pipe.to(device)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_vae_slicing()
    # pipe.enable_xformers_memory_efficient_attention()
    # Modify walls based on prompt
    print(f"Applying prompt: '{prompt}'")
    images = pipe(
        prompt=prompt,
        negative_prompt = negative_prompt,
        image=original_image,
        mask_image=mask_image,
        guidance_scale=7.5,
        num_inference_steps=25
    ).images
    
    # Save the result
    images[0].save(output_path)
    print(f'Modified image saved to {output_path}')
    
    # Also show a visualization of the wall mask
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.imshow(combined_mask, alpha=0.5)
    plt.title('Detected Walls')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(images[0])
    plt.title('Modified Walls')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('wall_modification_process.png')
    plt.close()
    
    return images[0]

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Modify room walls based on a prompt')
    # parser.add_argument('--image', required=True, help='Path to the input room image')
    # parser.add_argument('--prompt', required=True, help='Text prompt describing the desired wall modification')
    # parser.add_argument('--output', default='modified_room.png', help='Path to save the output image')
    
    # args = parser.parse_args()
    image = 'input_images/image2.png'
    prompt = 'Rustic walls, interior design, photorealistic, same lighting as original'
    negative_prompt = 'furniture, bed, ceiling, floor, lamps, windows, distortion, blurry, low quality, artifacts'
    output = 'output_images/output2-1.png'
    modify_walls(image, prompt, negative_prompt, output)
    print('Done!')