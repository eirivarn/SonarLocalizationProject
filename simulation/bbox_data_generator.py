"""
Oriented bounding box data generator for net detection.

Generates synthetic sonar images with oriented bounding box annotations.
Each sample contains:
- Sonar image (intensity)
- Oriented bounding box: [center_y, center_z, width, height, angle]
"""
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import VoxelGrid, VoxelSonar, Material, EMPTY, NET, ROPE, FISH, BIOMASS
from config import (
    SONAR_CONFIG, SCENE_CONFIG, DATA_GEN_CONFIG, DATASET_DIR
)

# Import scene generation from original data_generator
from data_generator import (
    create_random_scene,
    generate_random_sonar_position, 
    find_net_intersection,
    polar_to_cartesian,
    _trajectory_state
)


def compute_oriented_bbox_from_net_segments(position, direction, net_segments, 
                                            range_m, fov_deg):
    """
    Compute oriented bounding box from visible net segments in sonar frame.
    
    Args:
        position: Sonar position [x, y] in world
        direction: Sonar direction [dx, dy] (normalized)
        net_segments: List of (x1, y1, x2, y2) net segments in world
        range_m: Sonar max range
        fov_deg: Sonar field of view
        
    Returns:
        bbox: dict with center_y, center_z, width, height, angle or None if no visible net
    """
    position = np.array(position)
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    
    # Sonar frame: forward = direction, right = perpendicular
    forward = direction
    right = np.array([-direction[1], direction[0]])
    
    fov_rad = np.deg2rad(fov_deg)
    
    # Collect all net points that are visible to sonar
    visible_points_y = []
    visible_points_z = []
    
    for seg in net_segments:
        x1, y1, x2, y2 = seg
        
        # Sample points along segment (including endpoints)
        num_samples = 20
        for t in np.linspace(0, 1, num_samples):
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            point_world = np.array([px, py])
            
            # Transform to sonar frame
            point_local = point_world - position
            point_y = np.dot(point_local, forward)
            point_z = np.dot(point_local, right)
            
            # Check if point is within sonar range and FOV
            distance = np.sqrt(point_y**2 + point_z**2)
            if distance > range_m or distance < 0.1:
                continue
            
            # Check if within FOV
            angle = np.arctan2(point_z, point_y)
            if abs(angle) > fov_rad / 2:
                continue
            
            # Point is visible
            visible_points_y.append(point_y)
            visible_points_z.append(point_z)
    
    if len(visible_points_y) < 5:
        # Not enough visible points
        return None
    
    # Convert to numpy arrays
    points_y = np.array(visible_points_y)
    points_z = np.array(visible_points_z)
    points = np.column_stack([points_z, points_y])  # Stack as (z, y) for rotation
    
    # Fit oriented bounding box using PCA
    # Compute centroid
    centroid = points.mean(axis=0)
    
    # Compute covariance matrix
    centered = points - centroid
    cov = np.cov(centered.T)
    
    # Eigendecomposition to get principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (largest first = main direction)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Main axis (net direction)
    main_axis = eigenvectors[:, 0]
    
    # Project points onto principal axes to get extent
    projected = centered @ eigenvectors
    
    # Compute width and height (with small padding)
    width = (projected[:, 0].max() - projected[:, 0].min()) * 1.1
    height = (projected[:, 1].max() - projected[:, 1].min()) * 1.1
    
    # Ensure minimum dimensions
    width = max(width, 0.5)
    height = max(height, 0.3)
    
    # Compute angle from main axis
    # main_axis is [dz, dy]
    angle_rad = np.arctan2(main_axis[0], main_axis[1])
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to [-90, 90]
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180
    
    bbox = {
        'center_y': float(centroid[1]),  # centroid is [z, y]
        'center_z': float(centroid[0]),
        'width': float(width),
        'height': float(height),
        'angle': float(angle_deg),
    }
    
    return bbox


def generate_sample(scene_id, sample_id):
    """
    Generate a single training sample with oriented bounding box.
    
    Args:
        scene_id: Scene identifier for trajectory continuity
        sample_id: Sample identifier
        
    Returns:
        dict with image, valid_mask, net_visible, bbox
    """
    global _trajectory_state
    
    # Create scene (reuse if same scene_id)
    state_key = f"scene_{scene_id}"
    
    if state_key not in _trajectory_state:
        # New scene - create environment and reset trajectory
        grid, fish_data, cage_center, cage_radius, net_segments = create_random_scene(seed=scene_id)
        _trajectory_state[state_key] = {
            'grid': grid,
            'fish_data': fish_data,
            'cage_center': cage_center,
            'cage_radius': cage_radius,
            'net_segments': net_segments,
            'prev_position': None,
            'prev_direction': None,
        }
    
    # Get scene data
    scene_data = _trajectory_state[state_key]
    grid = scene_data['grid']
    cage_center = scene_data['cage_center']
    cage_radius = scene_data['cage_radius']
    net_segments = scene_data['net_segments']
    
    # Generate sonar position using trajectory
    position, direction, _ = generate_random_sonar_position(
        cage_center, cage_radius,
        DATA_GEN_CONFIG['distance_range'],
        DATA_GEN_CONFIG['angle_range'],
        prev_position=scene_data['prev_position'],
        prev_direction=scene_data['prev_direction']
    )
    
    # Update trajectory state
    scene_data['prev_position'] = position
    scene_data['prev_direction'] = direction
    
    # Create and scan sonar
    sonar = VoxelSonar(
        position=position,
        direction=direction,
        range_m=SONAR_CONFIG['range_m'],
        fov_deg=SONAR_CONFIG['fov_deg'],
        num_beams=SONAR_CONFIG['num_beams']
    )
    
    polar_image = sonar.scan(grid)
    
    # Convert to Cartesian
    sonar_image = polar_to_cartesian(polar_image)
    
    # Compute oriented bounding box from visible net segments
    bbox = compute_oriented_bbox_from_net_segments(
        position, direction, net_segments,
        SONAR_CONFIG['range_m'], SONAR_CONFIG['fov_deg']
    )
    
    # Net is visible if we got a valid bounding box
    net_visible = bbox is not None
    
    # Prepare output dictionary
    output = {
        'image': sonar_image[0].astype(np.float32),      # Intensity channel
        'valid_mask': sonar_image[1].astype(np.float32), # Valid mask
        'net_visible': net_visible,
        'bbox': bbox,  # None if not visible, dict otherwise
        # Metadata
        'scene_id': scene_id,
        'sample_id': sample_id,
    }
    
    return output


def visualize_sample(sample_data, output_path):
    """
    Visualize a sonar sample with oriented bounding box overlay.
    
    Args:
        sample_data: Dictionary with image, bbox, etc.
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Handle both dict and npz formats
    if isinstance(sample_data, dict):
        intensity = sample_data.get('image')
        bbox = sample_data.get('bbox')
        net_visible = sample_data.get('net_visible', False)
        sample_id = sample_data.get('sample_id', 0)
    else:
        # Assume npz file
        intensity = sample_data['image']
        bbox = sample_data['bbox'].item() if sample_data['net_visible'] else None
        net_visible = sample_data['net_visible']
        sample_id = sample_data.get('sample_id', 0)
    
    # Set dark grey background
    ax.set_facecolor('#2a2a2a')
    fig.patch.set_facecolor('#2a2a2a')
    
    # Calculate extent
    range_m = SONAR_CONFIG['range_m']
    fov_deg = SONAR_CONFIG['fov_deg']
    z_max = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    extent = [-z_max, z_max, 0, range_m]
    
    # Auto-scale intensity for better visualization
    # Use percentile-based normalization to handle outliers
    vmin = np.percentile(intensity, 1)
    vmax = np.percentile(intensity, 99.5)
    
    # Avoid division by zero
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    
    print(f"  Intensity range: [{intensity.min():.6f}, {intensity.max():.6f}]")
    print(f"  Display range: [{vmin:.6f}, {vmax:.6f}]")
    
    # Plot image with auto-scaling
    ax.imshow(intensity, cmap='hot', vmin=vmin, vmax=vmax, 
              extent=extent, origin='lower', aspect='auto')
    
    # Draw oriented bounding box if net is visible
    if net_visible and bbox is not None:
        cy = bbox['center_y']
        cz = bbox['center_z']
        w = bbox['width']
        h = bbox['height']
        angle = bbox['angle']
        
        # Create oriented rectangle
        # matplotlib Rectangle uses bottom-left corner, so compute that
        # Also need to convert from center-based to corner-based
        t = patches.Rectangle(
            (-w/2, -h/2), w, h,  # Start from center, will transform
            angle=angle,
            linewidth=2.5,
            edgecolor='red',
            facecolor='none',
            alpha=0.9,
            label='Oriented BBox'
        )
        
        # Transform: rotate then translate to center
        from matplotlib.transforms import Affine2D
        trans = Affine2D().rotate_deg(angle).translate(cz, cy) + ax.transData
        t.set_transform(trans)
        ax.add_patch(t)
        
        # Draw center point
        ax.plot(cz, cy, 'r+', markersize=15, markeredgewidth=2, alpha=0.9, 
                label='BBox Center')
        
        title = f'Sample {sample_id} | BBox: {w:.1f}m × {h:.1f}m @ {angle:+.0f}°'
    else:
        title = f'Sample {sample_id} | NO NET'
    
    ax.set_xlim(-z_max, z_max)
    ax.set_ylim(0, range_m)
    ax.set_aspect('equal')
    ax.set_xlabel('Z (m)', color='white', fontsize=10)
    ax.set_ylabel('Y (m)', color='white', fontsize=10)
    ax.set_title(title, fontsize=12, pad=15, color='white')
    ax.grid(True, alpha=0.2, linewidth=0.5, color='white')
    ax.tick_params(colors='white')
    if net_visible:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#2a2a2a')
    plt.close()


def generate_dataset(split='train', num_samples=None):
    """
    Generate a dataset split with oriented bounding boxes.
    
    Args:
        split: 'train', 'val', or 'test'
        num_samples: Number of samples (uses config if None)
    """
    if num_samples is None:
        num_samples = DATA_GEN_CONFIG[f'num_{split}']
    
    output_dir = DATASET_DIR / f"{split}_bbox"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images subdirectory for visualizations
    viz_interval = DATA_GEN_CONFIG.get('viz_interval', 10)
    
    if viz_interval > 0:
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        print(f"\nGenerating {num_samples} {split} samples (bbox)...")
        print(f"Output directory: {output_dir}")
        print(f"Saving visualizations every {viz_interval} samples to: {images_dir}")
    else:
        images_dir = None
        print(f"\nGenerating {num_samples} {split} samples (bbox)...")
        print(f"Output directory: {output_dir}")
        print(f"Visualizations disabled")
    
    for i in tqdm(range(num_samples)):
        # Generate sample
        sample_data = generate_sample(scene_id=i, sample_id=i)
        
        # Save as compressed .npz
        sample_path = output_dir / f"sample_{i:06d}.npz"
        np.savez_compressed(sample_path, **sample_data)
        
        # Save visualization if enabled and on interval
        if viz_interval > 0 and i % viz_interval == 0:
            viz_path = images_dir / f"sample_{i:06d}.png"
            visualize_sample(sample_data, viz_path)
    
    print(f"✓ Generated {num_samples} samples")
    if viz_interval > 0:
        print(f"✓ Saved {num_samples // viz_interval} visualization images")
    
    # Save dataset metadata
    save_dataset_metadata(output_dir)


def save_dataset_metadata(output_dir):
    """Save dataset metadata."""
    range_m = SONAR_CONFIG['range_m']
    fov_deg = SONAR_CONFIG['fov_deg']
    image_size = DATA_GEN_CONFIG['image_size']
    
    z_extent = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    height, width = image_size
    
    metadata = {
        'image_size': image_size,
        'range_m': range_m,
        'fov_deg': fov_deg,
        'z_extent': z_extent,
        'pixel_to_meters': {
            'y': range_m / height,
            'z': 2 * z_extent / width,
        },
        'origin_pixel': {
            'u': width / 2,  # z=0 is at center horizontally
            'v': 0,          # y=0 is at bottom
        },
        'bbox_info': 'Bounding boxes computed from visible net segments using PCA',
        'sonar_config': SONAR_CONFIG,
        'scene_config': SCENE_CONFIG,
        'data_gen_config': DATA_GEN_CONFIG,
    }
    
    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved dataset metadata to {metadata_path}")


def main():
    """Generate all dataset splits with oriented bounding boxes."""
    print("=" * 80)
    print("SONAR NET DETECTION - ORIENTED BBOX DATASET GENERATION")
    print("=" * 80)
    
    print("\nBounding boxes computed from visible net segments (PCA-based)")
    
    # Generate datasets
    generate_dataset('train')
    generate_dataset('val')
    generate_dataset('test')
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
