import os
import requests
import torch
import cv2
import numpy as np
import segment_anything.utils.transforms as sam_transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def download_file(url, save_path):
    '''Download a file from a URL if it doesn't exist locally.'''
    if not os.path.exists(save_path):
        print(f'Downloading {url} to {save_path}...')
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Download complete!')
    else:
        print(f'File already exists at {save_path}')

def setup_sam(checkpoint_path='sam_vit_b_01ec64.pth', model_type='vit_b'):
    '''Download and set up the Segment Anything Model.'''
    # Download SAM checkpoint if it doesn't exist
    if not os.path.exists(checkpoint_path):
        sam_url = f'https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_path}'
        download_file(sam_url, checkpoint_path)
    
    # Set up SAM model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    for param in sam.parameters():
        param.data = param.data.float()
    sam.to(device)
    return sam

def find_wall_segments(image, sam):
    '''Use SAM to generate segments and filter for likely wall segments.'''
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        # points_per_side=16,
        # pred_iou_thresh=0.86,
        # stability_score_thresh=0.92
    )

    # Ensure image isn't too large, else buffer error
    # Also, images must stay on CPU
    h, w, _ = image.shape
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    masks = mask_generator.generate(image)
    
    # Sort masks by area (descending)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    # print(masks)
    # Filter for likely wall segments (typically large, rectangular segments)
    wall_masks = []
    for mask in masks:
        # Skip very small segments
        if mask['area'] < (image.shape[0] * image.shape[1]) * 0.05:
            continue
            
        # Calculate aspect ratio of the bounding box
        bbox = mask['bbox']  # [x, y, w, h]
        aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
        
        # Skip segments that are too square (furniture) or too narrow (objects)
        if 0.2 < aspect_ratio < 5.0:
            # Check if the segment touches the edge of the image (walls often do)
            binary_mask = mask['segmentation']
            edge_contact = (
                np.any(binary_mask[0, :]) or  # Top edge
                np.any(binary_mask[-1, :]) or  # Bottom edge
                np.any(binary_mask[:, 0]) or  # Left edge
                np.any(binary_mask[:, -1])  # Right edge
            )
            
            if edge_contact:
                wall_masks.append(mask)
    print('Found segments')
    return wall_masks
