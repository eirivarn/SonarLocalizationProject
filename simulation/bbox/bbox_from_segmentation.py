"""
Bounding box data generator using semantic segmentation.

Uses ground truth material IDs to accurately place oriented bounding boxes around net regions.
"""
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import FISH, NET, ROPE, BIOMASS
from core import SONAR_CONFIG, SCENE_CONFIG, DATA_GEN_CONFIG, DATASET_DIR
from core import create_random_scene, generate_random_sonar_position, polar_to_cartesian, _trajectory_state
from core import (
    VoxelGridWithMaterials, VoxelSonarWithSegmentation, 
    MATERIAL_IDS, MATERIAL_COLORS, create_semantic_visualization
)


def polar_to_cartesian_segmentation(polar_segmentation):
    """
    Convert polar segmentation to Cartesian (preserves integer labels).
    
    Args:
        polar_segmentation: (range_bins, num_beams) array of uint8 material IDs
        
    Returns:
        cart_segmentation: (height, width) array of uint8 material IDs
    """
    range_m = SONAR_CONFIG['range_m']
    fov_deg = SONAR_CONFIG['fov_deg']
    output_size = DATA_GEN_CONFIG['image_size']
    
    num_range_bins, num_beams = polar_segmentation.shape
    height, width = output_size
    
    # Calculate extent
    x_extent = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    y_extent = range_m
    
    # Create coordinate grids
    j_grid, i_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixels to meters in sonar frame
    z_m = (j_grid - width/2) * (2 * x_extent / width)
    y_m = i_grid * (y_extent / height)
    
    # Convert to polar coordinates
    r_m = np.sqrt(z_m**2 + y_m**2)
    theta_rad = np.arctan2(z_m, y_m)
    
    # Create mask for points within sonar cone
    valid_mask = (r_m <= range_m) & (np.abs(theta_rad) <= np.deg2rad(fov_deg/2))
    
    # Find corresponding indices in polar image
    r_idx = ((r_m / range_m) * (num_range_bins - 1)).astype(np.int32)
    theta_idx = ((theta_rad + np.deg2rad(fov_deg/2)) / np.deg2rad(fov_deg) * (num_beams - 1)).astype(np.int32)
    
    # Clamp indices
    r_idx = np.clip(r_idx, 0, num_range_bins - 1)
    theta_idx = np.clip(theta_idx, 0, num_beams - 1)
    
    # Create output (default to 0 = empty)
    cart_seg = np.zeros((height, width), dtype=np.uint8)
    cart_seg[valid_mask] = polar_segmentation[r_idx[valid_mask], theta_idx[valid_mask]]
    
    return cart_seg


def compute_bbox_from_segmentation(segmentation_map, material_id=MATERIAL_IDS['net']):
    """
    Compute oriented bounding box from semantic segmentation map.
    
    Args:
        segmentation_map: (H, W) array of material IDs in sonar image space
        material_id: Which material to create bbox for (default: net)
        
    Returns:
        bbox: dict with center_y, center_z, width, height, angle in meters
              or None if material not found
    """
    # Find all pixels of the target material
    mask = segmentation_map == material_id
    
    if np.sum(mask) < 10:  # Need at least 10 pixels
        return None
    
    # Get pixel coordinates
    coords = np.argwhere(mask)  # Returns (row, col) = (v, u)
    
    if len(coords) < 10:
        return None
    
    # Convert pixel coordinates to meters
    # Sonar frame: Y is forward (range), Z is lateral (width)
    range_m = SONAR_CONFIG['range_m']
    fov_deg = SONAR_CONFIG['fov_deg']
    image_size = DATA_GEN_CONFIG['image_size']
    
    height, width = image_size
    z_max = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    
    # Convert to meters
    # v (row) = 0 at y=0 (bottom), v=height at y=range_m (top)
    # u (col) = 0 at z=-z_max (left), u=width at z=+z_max (right)
    points_v = coords[:, 0]  # Rows
    points_u = coords[:, 1]  # Cols
    
    points_y = (points_v / height) * range_m
    points_z = (points_u / width) * (2 * z_max) - z_max
    
    # Stack as (z, y) for PCA
    points = np.column_stack([points_z, points_y])
    
    # Fit oriented bounding box using PCA
    centroid = points.mean(axis=0)
    
    # Compute covariance
    centered = points - centroid
    cov = np.cov(centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Main axis
    main_axis = eigenvectors[:, 0]
    
    # Project points onto principal axes
    projected = centered @ eigenvectors
    
    # Compute width and height with padding
    width_bbox = (projected[:, 0].max() - projected[:, 0].min()) * 1.05
    height_bbox = (projected[:, 1].max() - projected[:, 1].min()) * 1.05
    
    # Ensure minimum dimensions
    width_bbox = max(width_bbox, 0.3)
    height_bbox = max(height_bbox, 0.2)
    
    # Compute angle from main axis
    # main_axis is [dz, dy] - we want angle from horizontal (z-axis)
    # For matplotlib, 0° = horizontal right, positive = counter-clockwise
    angle_rad = np.arctan2(main_axis[1], main_axis[0])  # arctan2(dy, dz)
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to [-90, 90]
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180
    
    bbox = {
        'center_y': float(centroid[1]),  # centroid is [z, y]
        'center_z': float(centroid[0]),
        'width': float(width_bbox),
        'height': float(height_bbox),
        'angle': float(angle_deg),
    }
    
    return bbox


def generate_sample_with_bbox(scene_id, sample_id):
    """
    Generate a sample with intensity, segmentation, and bounding box.
    
    Returns:
        dict with image, segmentation, net_visible, bbox
    """
    global _trajectory_state
    
    # Create scene (reuse if same scene_id)
    state_key = f"scene_{scene_id}"
    
    if state_key not in _trajectory_state:
        # Create scene - but we need to use VoxelGridWithMaterials
        # Let's create a custom scene creation function
        grid, fish_data, cage_center, cage_radius, net_segments = create_scene_with_materials(seed=scene_id)
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
    
    # Generate sonar position
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
    
    # Scan with segmentation
    sonar = VoxelSonarWithSegmentation(
        position=position,
        direction=direction,
        range_m=SONAR_CONFIG['range_m'],
        fov_deg=SONAR_CONFIG['fov_deg'],
        num_beams=SONAR_CONFIG['num_beams']
    )
    
    polar_intensity, polar_segmentation = sonar.scan_with_segmentation(grid)
    
    # Convert to Cartesian
    cart_intensity = polar_to_cartesian(polar_intensity)
    
    # Convert segmentation separately (no interpolation, preserve integer labels)
    cart_segmentation = polar_to_cartesian_segmentation(polar_segmentation)
    
    # Compute bounding box from segmentation
    bbox = compute_bbox_from_segmentation(cart_segmentation, material_id=MATERIAL_IDS['net'])
    
    net_visible = bbox is not None
    
    # Prepare output
    output = {
        'image': cart_intensity[0].astype(np.float32),
        'valid_mask': cart_intensity[1].astype(np.float32),
        'segmentation': cart_segmentation.astype(np.uint8),  # Already (512, 512)
        'net_visible': net_visible,
        'bbox': bbox,
        'scene_id': scene_id,
        'sample_id': sample_id,
    }
    
    return output


def create_scene_with_materials(seed=None):
    """
    Create a random scene using VoxelGridWithMaterials.
    
    This is a modified version of create_random_scene that uses material tracking.
    """
    if seed is not None:
        np.random.seed(seed)
    
    world_size = SCENE_CONFIG['world_size_m']
    voxel_size = SONAR_CONFIG['voxel_size']
    
    # Create grid with material tracking
    grid_size = int(world_size / voxel_size)
    grid = VoxelGridWithMaterials(grid_size, grid_size, voxel_size)
    
    # Cage parameters
    cage_center = np.array([world_size / 2, world_size / 2])
    cage_radius = SCENE_CONFIG['cage_radius']
    
    # Create circular net
    num_segments = 64
    net_segments = []
    
    for i in range(num_segments):
        angle1 = 2 * np.pi * i / num_segments
        angle2 = 2 * np.pi * (i + 1) / num_segments
        
        x1 = cage_center[0] + cage_radius * np.cos(angle1)
        y1 = cage_center[1] + cage_radius * np.sin(angle1)
        x2 = cage_center[0] + cage_radius * np.cos(angle2)
        y2 = cage_center[1] + cage_radius * np.sin(angle2)
        
        net_segments.append((x1, y1, x2, y2))
        
        # Draw net segment
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        
        num_points = 15
        for t in np.linspace(0, 1, num_points):
            pos = p1 + t * (p2 - p1)
            grid.set_circle(pos, 0.15, NET)
    
    # Add fish
    fish_data = []
    num_fish = np.random.randint(
        SCENE_CONFIG.get('fish_count_range', [5, 15])[0],
        SCENE_CONFIG.get('fish_count_range', [5, 15])[1] + 1
    )
    
    for _ in range(num_fish):
        # Fish inside cage
        angle = np.random.rand() * 2 * np.pi
        radius = np.random.rand() * (cage_radius - 1.0)
        
        fish_pos = cage_center + radius * np.array([np.cos(angle), np.sin(angle)])
        fish_size = np.random.uniform(0.2, 0.5)
        
        grid.set_circle(fish_pos, fish_size, FISH)
        fish_data.append({'position': fish_pos, 'size': fish_size})
    
    # Add biomass
    if np.random.rand() < SCENE_CONFIG.get('biomass_probability', 0.3):
        num_biomass = np.random.randint(1, 4)
        for _ in range(num_biomass):
            angle = np.random.rand() * 2 * np.pi
            radius = np.random.rand() * (cage_radius - 0.5)
            
            bio_pos = cage_center + radius * np.array([np.cos(angle), np.sin(angle)])
            bio_size = np.random.uniform(0.3, 0.8)
            
            grid.set_circle(bio_pos, bio_size, BIOMASS)
    
    return grid, fish_data, cage_center, cage_radius, net_segments


def visualize_bbox_sample(sample_data, output_path):
    """
    Visualize sample with intensity, segmentation, and bounding box.
    
    Args:
        sample_data: Dictionary with image, segmentation, bbox
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Set dark background
    for ax in axes:
        ax.set_facecolor('#2a2a2a')
    fig.patch.set_facecolor('#2a2a2a')
    
    # Get data
    if isinstance(sample_data, dict):
        intensity = sample_data['image']
        segmentation = sample_data['segmentation']
        bbox = sample_data.get('bbox')
        net_visible = sample_data.get('net_visible', False)
        sample_id = sample_data.get('sample_id', 0)
    else:
        # NPZ file
        intensity = sample_data['image']
        segmentation = sample_data['segmentation']
        bbox = sample_data['bbox'].item() if sample_data['net_visible'] else None
        net_visible = sample_data['net_visible']
        sample_id = sample_data.get('sample_id', 0)
    
    # Calculate extent
    range_m = SONAR_CONFIG['range_m']
    fov_deg = SONAR_CONFIG['fov_deg']
    z_max = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    extent = [-z_max, z_max, 0, range_m]
    
    # 1. Intensity with bbox overlay
    vmin = np.percentile(intensity, 1)
    vmax = np.percentile(intensity, 99.5)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    
    axes[0].imshow(intensity, cmap='hot', vmin=vmin, vmax=vmax,
                   extent=extent, origin='lower', aspect='auto')
    
    if net_visible and bbox is not None:
        cy = bbox['center_y']
        cz = bbox['center_z']
        w = bbox['width']
        h = bbox['height']
        angle = bbox['angle']
        
        # Draw oriented rectangle
        rect = patches.Rectangle(
            (-w/2, -h/2), w, h,
            angle=0,  # Don't apply angle here, use transform instead
            linewidth=3,
            edgecolor='cyan',
            facecolor='none',
            alpha=0.9,
            label='BBox from Segmentation'
        )
        
        from matplotlib.transforms import Affine2D
        trans = Affine2D().rotate_deg(angle).translate(cz, cy) + axes[0].transData
        rect.set_transform(trans)
        axes[0].add_patch(rect)
        
        # Draw center
        axes[0].plot(cz, cy, 'c+', markersize=15, markeredgewidth=3, alpha=0.9)
        
        title0 = f'Intensity + BBox | {w:.1f}m × {h:.1f}m @ {angle:+.0f}°'
        axes[0].legend(loc='upper right', fontsize=10, framealpha=0.8)
    else:
        title0 = 'Intensity (NO NET)'
    
    axes[0].set_title(title0, fontsize=13, color='white', pad=10)
    axes[0].set_xlabel('Z (m)', color='white')
    axes[0].set_ylabel('Y (m)', color='white')
    axes[0].grid(True, alpha=0.2, color='white')
    axes[0].tick_params(colors='white')
    
    # 2. Semantic segmentation with bbox
    h_img, w_img = segmentation.shape
    rgb_seg = np.zeros((h_img, w_img, 3), dtype=np.float32)
    
    for material_id, color in MATERIAL_COLORS.items():
        mask = segmentation == material_id
        rgb_seg[mask] = color
    
    axes[1].imshow(rgb_seg, extent=extent, origin='lower', aspect='auto')
    
    if net_visible and bbox is not None:
        # Draw same bbox on segmentation
        rect2 = patches.Rectangle(
            (-w/2, -h/2), w, h,
            angle=0,  # Don't apply angle here, use transform instead
            linewidth=3,
            edgecolor='white',
            facecolor='none',
            alpha=0.9,
            linestyle='--',
            label='BBox'
        )
        
        trans2 = Affine2D().rotate_deg(angle).translate(cz, cy) + axes[1].transData
        rect2.set_transform(trans2)
        axes[1].add_patch(rect2)
        
        axes[1].plot(cz, cy, 'w+', markersize=15, markeredgewidth=3, alpha=0.9)
    
    axes[1].set_title('Segmentation (Ground Truth)', fontsize=13, color='white', pad=10)
    axes[1].set_xlabel('Z (m)', color='white')
    axes[1].set_ylabel('Y (m)', color='white')
    axes[1].grid(True, alpha=0.2, color='white')
    axes[1].tick_params(colors='white')
    
    # Add legend for materials
    from matplotlib.patches import Patch
    legend_elements = []
    for material_id in sorted(MATERIAL_COLORS.keys()):
        if np.any(segmentation == material_id):
            material_name = [k for k, v in MATERIAL_IDS.items() if v == material_id][0]
            color = MATERIAL_COLORS[material_id]
            legend_elements.append(Patch(facecolor=color, edgecolor='white', label=material_name))
    
    if legend_elements:
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#2a2a2a')
    plt.close()


def test_bbox_from_segmentation():
    """Test bounding box generation from segmentation."""
    print("=" * 80)
    print("BOUNDING BOX FROM SEGMENTATION TEST")
    print("=" * 80)
    
    output_dir = Path("test_outputs/bbox_from_segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = 10
    
    for i in range(num_samples):
        print(f"\n--- Sample {i} ---")
        
        sample_data = generate_sample_with_bbox(scene_id=i, sample_id=i)
        
        if sample_data['net_visible']:
            bbox = sample_data['bbox']
            print(f"  Net VISIBLE")
            print(f"  BBox: center=({bbox['center_y']:.2f}, {bbox['center_z']:.2f})")
            print(f"  Size: {bbox['width']:.2f} × {bbox['height']:.2f} m")
            print(f"  Angle: {bbox['angle']:+.1f}°")
        else:
            print(f"  Net NOT visible")
        
        # Visualize
        viz_path = output_dir / f"sample_{i:03d}.png"
        visualize_bbox_sample(sample_data, viz_path)
        print(f"  Saved: {viz_path}")
    
    print(f"\n✓ Generated {num_samples} samples with semantic bboxes")
    print(f"✓ Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    test_bbox_from_segmentation()
