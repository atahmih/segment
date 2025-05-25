import os
import requests
import base64
from PIL import Image, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import io
from dotenv import load_dotenv
import json
import time
import re

import tempfile

import torch
from openai import OpenAI
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from utils import setup_sam, find_wall_segments

# Load environment variables from .env file
load_dotenv()

client = OpenAI()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def encode_image(image_array):
    # Resizes and converts to BufferedReader file
    if isinstance(image_array, np.ndarray):
        # Convert numpy array to a PIL Image
        image_resized = cv2.resize(image_array, (720, 720), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    else:
        # If it's already a PIL Image
        image = image_array.resize((720, 720))
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_file.name, format="PNG")
    
    # Open the temporary file as a BufferedReader
    return open(temp_file.name, "rb")

def decode_and_save_image(base64_string, output_path):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)
    return image

def segment_walls_with_gpt4v(image_path, api_key, prompt):
    print('Analyzing image to identify walls...')
    base64_image = encode_image(image_path)
    response = client.responses.create(
        model = 'gpt-4o-mini',
        input = [{
            'role': 'user',
            'content': [
                {
                    'type': 'input_text',
                    'text': 'Please analyze this image of a room. I need you to identify the walls in the image. Return ONLY a JSON object with the coordinates of polygons that outline the walls. The format should be: {\"walls\": [{\"points\": [[x1, y1], [x2, y2], ...]}]}. Include only visible wall surfaces, not furniture, floors, or ceilings.'
                },
                {
                    'type': 'input_image',
                    'image_url': f'data:image/jpeg;base64,{base64_image}'
                },
            ]
        }],
    )
    
    
    # Parse the response
    try:
        response_data = response.output_text
        print('Converting response to json...')
        # Extract the json content from the output
        match = re.search(r'\{.*\}', response_data, re.DOTALL)
        data = ''
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print("Failed to parse JSON:", e)
        
        # wall_data = json.loads(json_str)
        return data
        # return 'ok'
    except Exception as e:
        print(f"Error parsing response: {e}")
        # print(content)
        return None

def create_mask_from_polygons(image_path, wall_data):
    # Load the original image to get dimensions
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill in white where the walls are
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    # Draw the polygons onto the mask
    for wall in wall_data.get("walls"):
        # points = np.array(wall["points"], dtype=np.int32)
        # cv2.fillPoly(mask, [points], 255)
        polygon = patches.Polygon(wall["points"], closed=True, facecolor='white', fill=False, linewidth=2)
        ax.add_patch(polygon)
    
    plt.savefig('gpt_mask.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    return mask

def modify_walls_with_dalle(image, mask, prompt, api_key, output_path):
    print(f"Modifying walls with prompt: '{prompt}'")
    encoded_image = encode_image(image)
    encoded_mask = encode_image(mask)
    response = client.images.edit(
        model = 'dall-e-2',
        image = encoded_image,
        mask = encoded_mask,
        # image = open(f'{image_path}', 'rb'),
        # mask = open(f'{mask_path}', 'rb'),
        prompt = prompt,
        # prompt =  f"Modify ONLY the walls in this room to have: {prompt}. Keep all furniture, fixtures, lighting, and other room elements EXACTLY the same. Focus only on changing the wall appearance.",
        n = 1,
        size='1024x1024',
    )

    print('Downloading image')
    # print(response.data[0].url)
    image_url = response.data[0].url
    # get the image
    image_response = requests.get(image_url)
    with open(output_path, 'wb') as file:
        file.write(image_response.content)
    modified_image = Image.open(io.BytesIO(image_response.content))
    return modified_image
   
def segment_with_sam(image_rgb, image_name, mask_path):
    sam = setup_sam()
    sam.to(device)
    print('Finding wall segments...')
    wall_masks = find_wall_segments(image_rgb, sam)
    
    # Combine wall masks into a single mask.
    combined_mask = np.zeros_like(wall_masks[0]['segmentation'], dtype=bool)
    for mask in wall_masks:
        combined_mask = combined_mask | mask['segmentation']
    # plt.imshow(combined_mask, alpha=0.5)
    saved_mask = Image.fromarray(combined_mask).astype(np.uint8) * 255
    saved_mask.save(mask_path, format='PNG')
    # we flip the bool values using ~ because the default segments are the other objects in the image (not the walls)
    # inverted_mask = ~combined_mask

    # print('Creating transparent image')
    # # Convert mask into transparent PNG 
    # # the 'a' in rgba is alpha, with 0 for transparent and 255 for opaque
    # rgba = np.zeros((inverted_mask.shape[0], inverted_mask.shape[1], 4), dtype=np.uint8)
    # rgba[..., :3] = 255  # Set the first 3 channels to 255 ie white
    # # select 4th channel (index 3), convert from bool to 1 or 0 and multiply by 255
    # rgba[..., 3] = inverted_mask.astype(np.uint8) * 255 

    # # Save the transparent mask
    # transparent_mask = Image.fromarray(rgba)
    # transparent_mask.save(mask_path, format='PNG')

    # mask_io.seek(0)
    print(f'Saved transparent image mask at {mask_path}')

 
def visualize_results(original_image_path, mask_path, output_image_path, image_name):
    # Load images
    original = cv2.imread(original_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    mask = np.array(Image.open(mask_path).resize((original.shape[1], original.shape[0])))  # Load with alpha channel if present
    

    # mask_image = Image.fromarray((mask_array).astype(np.uint8) * 255)
    
    # Create an RGB image where:
    # - False (0) pixels are black
    # - True (255) pixels are red
    # colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    # colored_mask[mask_array == 0] = [0, 0, 0]  # Black for background
    # colored_mask[mask_array > 0] = [255, 230, 0]  # Red for segments
    
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # # Ensure mask is properly sized for overlay
    # if mask.shape[:2] != original.shape[:2]:
    #     mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Create overlay
    # overlay = original.copy()
    # condition = mask > 0  # Use the mask directly
    # overlay[condition] = original[condition] * 0.5 + np.array([255, 0, 0]) * 0.5  # Highlight mask in red
    
    modified = cv2.imread(output_image_path)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)
    modified = cv2.resize(modified, (original.shape[1], original.shape[0]))
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(original)
    plt.imshow(mask, cmap='viridis')
    plt.title('Detected Walls')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(modified)
    plt.title('Modified Walls')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{image_name}_combined.png')
    plt.close()

def process_image(image_name, prompt, output_path):
    image_path = image_name + '.png'
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Step 1: Segment walls using GPT-4 Vision
    # wall_data = segment_walls_with_gpt4v(image_path, api_key, prompt)
    # if not wall_data:
    #     return
    # print(f'Segment polygons: {wall_data}')

    # # Step 2: Create a mask image
    # mask = create_mask_from_polygons(image_path, wall_data)
    # mask_path = "wall_mask.png"
    # cv2.imwrite(mask_path, mask)
    # print(f"Wall mask saved to {mask_path}")

    # image = Image.open(image_path)
    # image = image.resize((1024, 1024))

    # Segmenting with SAM
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask_path = f'{image_name}_transparent_mask.png'
    mask_path = f'{image_name}_mask.png'
    # Comment out if segment mask has already been generated
    # Can add an if condition to check if the segment mask exists
    segment_with_sam(image_rgb, image_name, mask_path)

    # Convert combined mask to image format for inpainting - NOT NEEDED
    # mask_image = Image.fromarray((inverted_mask).astype(np.uint8) * 255)
    # plt.savefig(f'{image_name}_mask.png', dpi=100, bbox_inches='tight', pad_inches=0)
    # plt.close()
    transparent_mask = Image.open(mask_path)
    # # Step 3: Modify walls using DALL-E
    modified_image = modify_walls_with_dalle(image_rgb, transparent_mask, prompt, api_key, output_path)
    if modified_image:
        # Step 4: Visualize the results
        visualize_results(image_path, mask_path, output_path, image_name)

if __name__ == '__main__':
    image = 'image1'
    image_path = f'input_images/{image}'
    for i in range(1):
        prompt = 'Paint ONLY the walls with blue with white stripes. Maintain exact original appearance of all furniture, decorations, ceiling, floor, windows, and lighting. Do not alter any furniture, lighting, floor, or decorations.'
        # prompt = 'In the masked area only, create a weathered red brick wall effect. Ensure the texture stops strictly at the segment edges.'
        output = f'output_images/{image}-output-trial_{i}.jpg'
        process_image(image_path, prompt, output)
        print('Done!')