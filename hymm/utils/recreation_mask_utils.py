import numpy as np
import torch
import json
from sklearn.cluster import DBSCAN

def calculate_box_distance(box1, box2):
    """Calculate normalized distance between two bounding boxes"""
    # Calculate centers
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    
    # Calculate center distance
    center_dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # Consider box sizes for normalization
    size1 = max(box1[2] - box1[0], box1[3] - box1[1])
    size2 = max(box2[2] - box2[0], box2[3] - box2[1])
    avg_size = (size1 + size2) / 2
    
    # Normalize distance by average size
    normalized_dist = center_dist / max(avg_size, 1)
    
    return normalized_dist

def cluster_recreation_boxes(boxes, eps=2.0, min_samples=1):
    """Cluster recreation boxes using DBSCAN algorithm"""
    if len(boxes) <= 1:
        return {0: list(range(len(boxes)))} if boxes else {}
    
    # Calculate pairwise distances
    n_boxes = len(boxes)
    distances = np.zeros((n_boxes, n_boxes))
    
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            dist = calculate_box_distance(boxes[i], boxes[j])
            distances[i, j] = distances[j, i] = dist
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = clustering.fit_predict(distances)
    
    # Group boxes by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label == -1:  # Noise points get individual clusters
            label = f"noise_{idx}"
        
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    return clusters

def create_recreation_cluster_mask(box_indices, boxes, shape_h, shape_w, margin_ratio=0.05, 
                       margin_ratio_w=None, margin_ratio_h=None, min_margin_pixels=0):
    """
    Create recreation mask for a single cluster of boxes with separate width/height margin control
    
    Args:
        box_indices: Indices of boxes in the cluster
        boxes: List of bounding boxes
        shape_h, shape_w: Shape of the target mask
        margin_ratio: Default margin ratio for both dimensions (if margin_ratio_w/h not specified)
        margin_ratio_w: Specific margin ratio for width dimension
        margin_ratio_h: Specific margin ratio for height dimension
        min_margin_pixels: Minimum margin in pixels to ensure visibility in latent space
    """
    mask = np.zeros((shape_h, shape_w), dtype=np.uint8)
    
    if not box_indices:
        return mask
    
    # Find bounding box of the entire cluster
    cluster_boxes = [boxes[i] for i in box_indices]
    
    min_x = min(box[0] for box in cluster_boxes)
    min_y = min(box[1] for box in cluster_boxes)
    max_x = max(box[2] for box in cluster_boxes)
    max_y = max(box[3] for box in cluster_boxes)
    
    # Add margin to cluster bounding box
    cluster_w = max_x - min_x
    cluster_h = max_y - min_y
    
    # Use specific ratios if provided, otherwise fall back to default
    margin_ratio_w = margin_ratio_w if margin_ratio_w is not None else margin_ratio
    margin_ratio_h = margin_ratio_h if margin_ratio_h is not None else margin_ratio
    
    # Calculate margins
    margin_w = cluster_w * margin_ratio_w
    margin_h = cluster_h * margin_ratio_h
    
    # Apply minimum margin constraint
    if min_margin_pixels > 0:
        margin_w = max(margin_w, min_margin_pixels)
        margin_h = max(margin_h, min_margin_pixels)
    
    x1 = max(0, int(min_x - margin_w))
    y1 = max(0, int(min_y - margin_h))
    x2 = min(shape_w, int(max_x + margin_w))
    y2 = min(shape_h, int(max_y + margin_h))
    
    mask[y1:y2, x1:x2] = 1
    
    return mask

def create_clustered_masks(boxes, shape_h, shape_w, eps=2.0, margin_ratio=0.05,
                          margin_ratio_w=None, margin_ratio_h=None, min_margin_pixels=0):
    """
    Create separate masks for each cluster of OCR boxes with separate margin control
    
    Args:
        boxes: List of OCR bounding boxes
        shape_h, shape_w: Shape of the target mask
        eps: DBSCAN clustering threshold
        margin_ratio: Default margin ratio for both dimensions
        margin_ratio_w: Specific margin ratio for width dimension
        margin_ratio_h: Specific margin ratio for height dimension
        min_margin_pixels: Minimum margin in pixels to ensure visibility in latent space
    """
    if not boxes:
        empty_mask = np.zeros((shape_h, shape_w), dtype=np.uint8)
        return empty_mask
    
    # Cluster the boxes
    clusters = cluster_recreation_boxes(boxes, eps=eps)
    
    # Create masks for each cluster
    combined_mask = np.zeros((shape_h, shape_w), dtype=np.uint8)
    
    for cluster_id, box_indices in clusters.items():
        cluster_mask = create_recreation_cluster_mask(
            box_indices, boxes, shape_h, shape_w, 
            margin_ratio, margin_ratio_w, margin_ratio_h, min_margin_pixels
        )
        combined_mask = np.logical_or(combined_mask, cluster_mask).astype(np.uint8)

    return combined_mask
    
def parse_recreation_result(recreation_result_str):
    """Parse recreation result string to extract structured data"""
    try:
        if isinstance(recreation_result_str, str):
            return json.loads(recreation_result_str)
        else:
            return recreation_result_str
    except Exception as e:
        raise ValueError(f"Failed to parse recreation result: {e}")

def map_recreation_boxes_to_processed_space(boxes, original_height, original_width, 
                                   target_height, target_width):
    """
    Map recreation boxes from original video space to processed video space
    following the exact same resize and center crop logic used in resize_and_center_crop.
    
    This function replicates the two-step process:
    1. Resize: Scale image so the shorter side fits the target (maintaining aspect ratio)
    2. Center crop: Crop from the center to get exact target dimensions
    
    Args:
        boxes: List of recreation boxes in original space [x1, y1, x2, y2]
        original_height, original_width: Original video dimensions
        target_height, target_width: Target dimensions after processing
    
    Returns:
        List of mapped boxes in processed space
    """
    if not boxes:
        return []
    
    # Step 1: Calculate scale factor (same as in resize_and_center_crop)
    # This ensures the shorter side of the original fits exactly into the target
    scale_factor = max(target_width / original_width, target_height / original_height)
    
    # Calculate resized dimensions after scaling
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    # Step 2: Calculate center crop offsets
    # These are the offsets used to crop from the resized image to get target size
    crop_left = (resized_width - target_width) / 2
    crop_top = (resized_height - target_height) / 2
    
    mapped_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Step 1: Apply resize scaling to the bounding box coordinates
        # Scale all coordinates by the same factor used for image resizing
        scaled_x1 = x1 * scale_factor
        scaled_y1 = y1 * scale_factor
        scaled_x2 = x2 * scale_factor
        scaled_y2 = y2 * scale_factor
        
        # Step 2: Apply center crop offset
        # Subtract the crop offsets to get coordinates in the final cropped space
        cropped_x1 = scaled_x1 - crop_left
        cropped_y1 = scaled_y1 - crop_top
        cropped_x2 = scaled_x2 - crop_left
        cropped_y2 = scaled_y2 - crop_top
        
        # Step 3: Clamp coordinates to target bounds
        # Ensure all coordinates are within the target image bounds
        final_x1 = max(0, min(target_width, cropped_x1))
        final_y1 = max(0, min(target_height, cropped_y1))
        final_x2 = max(0, min(target_width, cropped_x2))
        final_y2 = max(0, min(target_height, cropped_y2))
        
        # Only add boxes that have valid area after mapping
        if final_x2 > final_x1 and final_y2 > final_y1:
            mapped_boxes.append([final_x1, final_y1, final_x2, final_y2])
    return mapped_boxes


def convert_recreation_mask_to_latent_space(processed_mask, target_height, target_width, 
                                latent_t, latent_h, latent_w, vae_spatial_stride=16):
    """Convert processed recreation mask to latent space mask"""
    # Calculate expected latent dimensions from target dimensions
    expected_latent_h = target_height // vae_spatial_stride
    expected_latent_w = target_width // vae_spatial_stride
    
    # Always resize mask to match actual latent dimensions (latent_h, latent_w)
    # This ensures the mask matches the latent tensor shape exactly
    mask_tensor = torch.from_numpy(processed_mask).float().unsqueeze(0)  # Add batch dimension
    resized_mask = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0),  # Add channel dimension
        size=(latent_h, latent_w),  # Use actual latent dimensions
        mode='nearest'
    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    
    # Convert to boolean
    latent_mask = resized_mask > 0.5
    
    # Expand to all frames
    latent_mask = latent_mask.unsqueeze(0).expand(latent_t, -1, -1)
    
    return latent_mask