"""
Evaluation metrics for heatmap-based net detection.

Implements:
- Percentage of Correct Keypoints (PCK) at multiple thresholds
- Angular error for direction prediction
- Hitpoint distance error
- Stratified analysis by distance and orientation bins
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

sys.path.append(str(Path(__file__).parent.parent))
import config


def extract_peak_from_heatmap(heatmap, threshold=0.1):
    """
    Extract peak location from heatmap.
    
    Args:
        heatmap: (B, 1, H, W) predicted heatmap
        threshold: Minimum confidence threshold
        
    Returns:
        peaks: (B, 2) peak locations in pixels [y_px, z_px] or None if no peak
        confidences: (B,) peak confidences
    """
    B, _, H, W = heatmap.shape
    
    # Flatten spatial dimensions
    heatmap_flat = heatmap.view(B, H * W)
    
    # Find max value and location
    confidences, max_indices = torch.max(heatmap_flat, dim=1)
    
    # Convert flat indices to 2D coordinates
    y_px = max_indices // W
    z_px = max_indices % W
    
    peaks = torch.stack([y_px, z_px], dim=1).float()  # (B, 2)
    
    # Mask out low confidence predictions
    valid = confidences > threshold
    peaks[~valid] = -1  # Invalid marker
    
    return peaks, confidences, valid


def pixels_to_meters(pixel_coords, image_size=(512, 512), 
                     y_range=(0, 20), z_range=(-18.2, 18.2)):
    """
    Convert pixel coordinates to meters.
    
    Args:
        pixel_coords: (B, 2) or (N, 2) coordinates [y_px, z_px]
        image_size: (height, width) in pixels
        y_range: (min, max) range in meters (forward)
        z_range: (min, max) range in meters (lateral)
        
    Returns:
        meter_coords: (B, 2) or (N, 2) coordinates [y_m, z_m]
    """
    H, W = image_size
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    # Pixel to meters scaling
    y_px = pixel_coords[..., 0]
    z_px = pixel_coords[..., 1]
    
    y_m = y_min + (y_px / H) * (y_max - y_min)
    z_m = z_min + (z_px / W) * (z_max - z_min)
    
    return torch.stack([y_m, z_m], dim=-1)


def compute_pck(pred_points, target_points, thresholds=[0.5, 1.0, 2.0]):
    """
    Compute Percentage of Correct Keypoints at multiple thresholds.
    
    Args:
        pred_points: (B, 2) predicted points in meters [y, z]
        target_points: (B, 2) target points in meters [y, z]
        thresholds: List of distance thresholds in meters
        
    Returns:
        pck_dict: Dictionary with PCK@threshold values
    """
    # Compute Euclidean distance
    distances = torch.norm(pred_points - target_points, dim=1)  # (B,)
    
    pck_dict = {}
    for thresh in thresholds:
        pck = (distances < thresh).float().mean().item() * 100
        # Format key consistently (integer thresholds without decimal, float with)
        if thresh == int(thresh):
            key = f'PCK@{int(thresh)}m'
        else:
            key = f'PCK@{thresh}m'
        pck_dict[key] = pck
    
    # Also return mean distance
    pck_dict['mean_distance_error'] = distances.mean().item()
    pck_dict['median_distance_error'] = distances.median().item()
    
    return pck_dict


def compute_angular_error(pred_direction, target_direction, valid_mask=None):
    """
    Compute angular error between predicted and target directions.
    
    Args:
        pred_direction: (B, 2) predicted direction vectors [ty, tz]
        target_direction: (B, 2) target direction vectors [ty, tz]
        valid_mask: (B,) boolean mask for valid predictions
        
    Returns:
        angular_errors: (B,) angular errors in degrees
        mean_error: Mean angular error
    """
    # Normalize vectors
    pred_norm = pred_direction / (torch.norm(pred_direction, dim=1, keepdim=True) + 1e-6)
    target_norm = target_direction / (torch.norm(target_direction, dim=1, keepdim=True) + 1e-6)
    
    # Compute cosine similarity (clamped to [-1, 1])
    cos_sim = torch.sum(pred_norm * target_norm, dim=1).clamp(-1, 1)
    
    # Convert to angular error in degrees
    angular_errors = torch.rad2deg(torch.acos(cos_sim))
    
    # Apply valid mask if provided
    if valid_mask is not None:
        angular_errors = angular_errors[valid_mask]
    
    if len(angular_errors) > 0:
        mean_error = angular_errors.mean().item()
        median_error = angular_errors.median().item()
    else:
        mean_error = 0.0
        median_error = 0.0
    
    return angular_errors, mean_error, median_error


def compute_metrics(predictions, targets):
    """
    Compute comprehensive evaluation metrics from predictions and targets.
    
    Args:
        predictions: List of dicts with 'heatmap', 'direction', 'visibility', 'hitpoint'
        targets: List of dicts with 'heatmap', 'direction', 'visibility', 'hitpoint'
        
    Returns:
        metrics: Dictionary of all computed metrics
    """
    all_pred_points = []
    all_target_points = []
    all_pred_directions = []
    all_target_directions = []
    all_confidences = []
    all_valid = []
    
    # Process all batches
    for pred_batch, target_batch in zip(predictions, targets):
        # Extract peak from heatmap
        peaks_px, confidences, valid = extract_peak_from_heatmap(
            pred_batch['heatmap'],
            threshold=0.1
        )
        
        # Convert to meters
        pred_points_m = pixels_to_meters(peaks_px)
        target_points_m = target_batch['hitpoint']  # Already in meters [py, pz]
        
        # Store
        all_pred_points.append(pred_points_m)
        all_target_points.append(target_points_m)
        all_pred_directions.append(pred_batch['direction'])
        all_target_directions.append(target_batch['direction'])
        all_confidences.append(confidences)
        all_valid.append(valid)
    
    # Concatenate all batches
    pred_points = torch.cat(all_pred_points, dim=0)
    target_points = torch.cat(all_target_points, dim=0)
    pred_directions = torch.cat(all_pred_directions, dim=0)
    target_directions = torch.cat(all_target_directions, dim=0)
    confidences = torch.cat(all_confidences, dim=0)
    valid = torch.cat(all_valid, dim=0)
    
    # Only evaluate on valid detections
    pred_points = pred_points[valid]
    target_points = target_points[valid]
    pred_directions = pred_directions[valid]
    target_directions = target_directions[valid]
    
    metrics = {}
    
    # Detection rate
    metrics['detection_rate'] = (valid.float().mean().item() * 100)
    
    if len(pred_points) > 0:
        # PCK metrics
        pck_metrics = compute_pck(pred_points, target_points, 
                                  thresholds=[0.5, 1.0, 2.0, 5.0])
        metrics.update(pck_metrics)
        
        # Angular error
        _, mean_angular, median_angular = compute_angular_error(
            pred_directions, target_directions
        )
        metrics['mean_angular_error'] = mean_angular
        metrics['median_angular_error'] = median_angular
        
        # Confidence stats
        metrics['mean_confidence'] = confidences[valid].mean().item()
        
        # Stratified analysis by distance
        distances = target_points[:, 0]  # Y-coordinate (forward distance)
        distance_bins = [(0, 5), (5, 10), (10, 15), (15, 20)]
        
        for d_min, d_max in distance_bins:
            mask = (distances >= d_min) & (distances < d_max)
            if mask.sum() > 0:
                bin_pck = compute_pck(
                    pred_points[mask],
                    target_points[mask],
                    thresholds=[1.0]
                )
                metrics[f'PCK@1m_dist_{d_min}-{d_max}m'] = bin_pck['PCK@1m']
    else:
        # No valid detections
        metrics.update({
            'PCK@0.5m': 0.0,
            'PCK@1m': 0.0,
            'PCK@2m': 0.0,
            'PCK@5m': 0.0,
            'mean_distance_error': 999.0,
            'median_distance_error': 999.0,
            'mean_angular_error': 180.0,
            'median_angular_error': 180.0,
            'mean_confidence': 0.0
        })
    
    return metrics


def visualize_predictions(predictions, targets, save_path, num_samples=8):
    """
    Visualize predictions vs targets.
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    # Select random samples
    total_samples = sum(pred['heatmap'].shape[0] for pred in predictions)
    sample_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    sample_idx = 0
    for pred_batch, target_batch in zip(predictions, targets):
        batch_size = pred_batch['heatmap'].shape[0]
        
        for i in range(batch_size):
            if sample_idx not in sample_indices:
                sample_idx += 1
                continue
            
            row_idx = list(sample_indices).index(sample_idx)
            
            # Extract peak
            peak_px, conf, valid = extract_peak_from_heatmap(
                pred_batch['heatmap'][i:i+1]
            )
            
            # Target hitpoint in meters
            target_point = target_batch['hitpoint'][i].numpy()
            
            # Convert peak to meters
            if valid[0]:
                pred_point = pixels_to_meters(peak_px[0:1])[0].numpy()
            else:
                pred_point = np.array([-999, -999])
            
            # Plot input (intensity channel)
            ax = axes[row_idx, 0]
            # Assuming input is loaded somewhere - skip for now
            ax.set_title("Input (Intensity)")
            ax.axis('off')
            
            # Plot predicted heatmap
            ax = axes[row_idx, 1]
            heatmap_vis = pred_batch['heatmap'][i, 0].cpu().numpy()
            im = ax.imshow(heatmap_vis, cmap='hot', aspect='auto')
            ax.plot(peak_px[0, 1].item(), peak_px[0, 0].item(), 'g*', 
                   markersize=15, label=f'Pred (conf={conf[0]:.2f})')
            ax.set_title(f"Predicted Heatmap")
            ax.legend()
            plt.colorbar(im, ax=ax)
            
            # Plot target heatmap
            ax = axes[row_idx, 2]
            target_heatmap = target_batch['heatmap'][i, 0].cpu().numpy()
            im = ax.imshow(target_heatmap, cmap='hot', aspect='auto')
            ax.set_title("Target Heatmap")
            plt.colorbar(im, ax=ax)
            
            # Add error text
            if valid[0]:
                error = np.linalg.norm(pred_point - target_point)
                fig.text(0.02, 1 - (row_idx + 0.5) / num_samples, 
                        f"Error: {error:.2f}m", fontsize=10)
            
            sample_idx += 1
            
            if sample_idx >= len(sample_indices):
                break
        
        if sample_idx >= len(sample_indices):
            break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")
