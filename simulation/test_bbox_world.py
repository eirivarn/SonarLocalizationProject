"""
Test script for bounding box generation from simulation world.

Tests the complete pipeline:
1. Use real simulation scenes from data_generator (cage, fish, biomass)
2. Generate sonar scans with semantic segmentation
3. Compute oriented bounding boxes from net pixels
4. Visualize results

Integration with data_generator.py:
- Uses create_scene_with_materials() instead of create_random_scene()
  (material-tracking version for semantic segmentation)
- Uses generate_random_sonar_position() for realistic trajectories
- Applies same image processing (log scale, normalization)
- Uses same polar_to_cartesian() for multi-channel output
- Adds semantic segmentation + bbox on top of standard pipeline
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import real simulation world
from data_generator import generate_random_sonar_position, polar_to_cartesian
from config import SONAR_CONFIG, DATA_GEN_CONFIG, SCENE_CONFIG
from semantic_segmentation import VoxelSonarWithSegmentation, VoxelGridWithMaterials
from bbox_from_segmentation import (
    compute_bbox_from_segmentation, polar_to_cartesian_segmentation,
    visualize_bbox_sample, MATERIAL_IDS
)
from simulation import NET, ROPE, FISH, BIOMASS, DEBRIS_LIGHT, DEBRIS_MEDIUM, DEBRIS_HEAVY


def create_realistic_scene_with_materials(seed=None):
    """
    Create realistic scene using VoxelGridWithMaterials.
    
    This is EXACTLY create_random_scene from data_generator.py,
    but using VoxelGridWithMaterials instead of VoxelGrid.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create world with material tracking
    world_size = int(SCENE_CONFIG['world_size_m'] / SONAR_CONFIG['voxel_size'])
    grid = VoxelGridWithMaterials(world_size, world_size, voxel_size=SONAR_CONFIG['voxel_size'])
    
    cage_center = np.array(SCENE_CONFIG['cage_center'])
    cage_radius = SCENE_CONFIG['cage_radius']
    num_sides = SCENE_CONFIG['num_sides']
    
    # Randomize current if enabled
    if DATA_GEN_CONFIG['vary_current']:
        current_strength = np.random.uniform(*DATA_GEN_CONFIG['current_range'])
    else:
        current_strength = SCENE_CONFIG['current_strength']
    
    current_direction = np.array([0.0, 1.0])
    net_sag = SCENE_CONFIG['net_sag']
    
    # Store net segments for ground truth calculation
    net_segments = []
    all_net_points = []
    
    # Create cage net (EXACT same logic as data_generator.py)
    for i in range(num_sides):
        angle1 = (i / num_sides) * 2 * np.pi
        angle2 = ((i + 1) / num_sides) * 2 * np.pi
        
        # Panel corners (base positions)
        x1_base = cage_center[0] + cage_radius * np.cos(angle1)
        y1_base = cage_center[1] + cage_radius * np.sin(angle1)
        x2_base = cage_center[0] + cage_radius * np.cos(angle2)
        y2_base = cage_center[1] + cage_radius * np.sin(angle2)
        
        # Apply current deflection
        deflection1 = current_strength * max(0, (y1_base - cage_center[1]) / cage_radius) * current_direction
        deflection2 = current_strength * max(0, (y2_base - cage_center[1]) / cage_radius) * current_direction
        
        lateral_factor1 = np.sin(angle1) * 0.4
        lateral_factor2 = np.sin(angle2) * 0.4
        deflection1 += np.array([lateral_factor1 * current_strength * 0.3, 0])
        deflection2 += np.array([lateral_factor2 * current_strength * 0.3, 0])
        
        x1 = x1_base + deflection1[0]
        y1 = y1_base + deflection1[1]
        x2 = x2_base + deflection2[0]
        y2 = y2_base + deflection2[1]
        
        # Create net segments and store points
        num_segments_per_panel = 20
        start_idx = 1 if i > 0 else 0
        
        for j, t in enumerate(np.linspace(0, 1, num_segments_per_panel + 1)[start_idx:]):
            x_linear = x1 + t * (x2 - x1)
            y_linear = y1 + t * (y2 - y1)
            
            # Add sag
            sag = net_sag * (1 - (2*t - 1)**2)
            
            panel_dx = x2 - x1
            panel_dy = y2 - y1
            panel_length = np.sqrt(panel_dx**2 + panel_dy**2)
            if panel_length > 0:
                perp_x = -panel_dy / panel_length
                perp_y = panel_dx / panel_length
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                to_center_x = cage_center[0] - mid_x
                to_center_y = cage_center[1] - mid_y
                if perp_x * to_center_x + perp_y * to_center_y < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y
                
                x = x_linear + sag * perp_x
                y = y_linear + sag * perp_y
            else:
                x = x_linear
                y = y_linear
            
            x = np.clip(x, 0.2, SCENE_CONFIG['world_size_m'] - 0.2)
            y = np.clip(y, 0.2, SCENE_CONFIG['world_size_m'] - 0.2)
            
            all_net_points.append((x, y))
            
            # Set voxels with material tracking
            grid.set_box(
                np.array([x - 0.08, y - 0.08]),
                np.array([x + 0.08, y + 0.08]),
                NET
            )
            
            if j % 3 == 0:
                grid.set_box(
                    np.array([x - 0.12, y - 0.12]),
                    np.array([x + 0.12, y + 0.12]),
                    ROPE
                )
    
    # Helper to check overlap (same as data_generator)
    def is_overlapping_net(grid_check, position, radius):
        voxel_size = grid_check.voxel_size
        check_radius_voxels = int(radius / voxel_size) + 2
        cx, cy = grid_check.world_to_voxel(position)
        
        for dx in range(-check_radius_voxels, check_radius_voxels + 1):
            for dy in range(-check_radius_voxels, check_radius_voxels + 1):
                vx = cx + dx
                vy = cy + dy
                
                if grid_check.is_inside(vx, vy):
                    if grid_check.density[vx, vy] > 0.25 and grid_check.density[vx, vy] < 0.4:
                        if grid_check.reflectivity[vx, vy] > 0.15 and grid_check.reflectivity[vx, vy] < 0.3:
                            world_x = (vx + 0.5) * voxel_size
                            world_y = (vy + 0.5) * voxel_size
                            dist = np.sqrt((world_x - position[0])**2 + (world_y - position[1])**2)
                            if dist < radius:
                                return True
        return False
    
    # Add biomass (EXACT same logic)
    if DATA_GEN_CONFIG['randomize_biomass']:
        num_biomass = SCENE_CONFIG['num_biomass_patches']
        attempts = 0
        max_attempts = num_biomass * 10
        biomass_count = 0
        
        while biomass_count < num_biomass and attempts < max_attempts:
            attempts += 1
            angle = np.random.rand() * 2 * np.pi
            r = cage_radius + np.random.randn() * 0.5
            x = cage_center[0] + r * np.cos(angle)
            y = cage_center[1] + r * np.sin(angle)
            patch_size = np.random.uniform(*SCENE_CONFIG['biomass_size_range'])
            
            if not is_overlapping_net(grid, np.array([x, y]), patch_size):
                grid.set_circle(np.array([x, y]), patch_size, BIOMASS)
                biomass_count += 1
    
    # Add fish (EXACT same logic)
    fish_data = []
    if DATA_GEN_CONFIG['randomize_fish']:
        num_fish = SCENE_CONFIG['num_fish']
        attempts = 0
        max_attempts = num_fish * 10
        fish_count = 0
        
        while fish_count < num_fish and attempts < max_attempts:
            attempts += 1
            angle = np.random.rand() * 2 * np.pi
            r_fraction = 0.6 + 0.35 * np.random.rand()
            r = cage_radius * r_fraction
            
            x = cage_center[0] + r * np.cos(angle)
            y = cage_center[1] + r * np.sin(angle)
            
            swim_angle = np.random.rand() * 2 * np.pi
            fish_length = np.random.uniform(*SCENE_CONFIG['fish_length_range'])
            fish_width = fish_length * SCENE_CONFIG['fish_width_ratio']
            
            if not is_overlapping_net(grid, np.array([x, y]), fish_length):
                fish_data.append({
                    'position': np.array([x, y]),
                    'radii': np.array([fish_length, fish_width]),
                    'orientation': swim_angle,
                })
                
                grid.set_ellipse(
                    np.array([x, y]),
                    np.array([fish_length, fish_width]),
                    swim_angle,
                    FISH
                )
                fish_count += 1
    
    # Create segments from continuous loop
    for i in range(len(all_net_points)):
        x1, y1 = all_net_points[i]
        x2, y2 = all_net_points[(i + 1) % len(all_net_points)]
        net_segments.append((x1, y1, x2, y2))
    
    return grid, fish_data, cage_center, cage_radius, net_segments


def generate_real_sample_with_bbox(scene_id, sample_id):
    """
    Generate sample using REAL simulation world from data_generator.
    
    This version uses the same scene generation and trajectory logic as the
    real data_generator, but replaces VoxelGrid with VoxelGridWithMaterials
    to enable semantic segmentation.
    
    Returns:
        Dictionary with image, segmentation, bbox, and metadata
    """
    # Create or reuse scene
    if not hasattr(generate_real_sample_with_bbox, '_scene_cache'):
        generate_real_sample_with_bbox._scene_cache = {}
    
    scene_key = f"scene_{scene_id}"
    if scene_key not in generate_real_sample_with_bbox._scene_cache:
        # Create scene using REALISTIC version with material tracking
        grid, fish_data, cage_center, cage_radius, net_segments = create_realistic_scene_with_materials(seed=scene_id)
        
        generate_real_sample_with_bbox._scene_cache[scene_key] = {
            'grid': grid,
            'fish_data': fish_data,
            'cage_center': cage_center,
            'cage_radius': cage_radius,
            'net_segments': net_segments,
            'prev_position': None,
            'prev_direction': None,
        }
    
    scene_data = generate_real_sample_with_bbox._scene_cache[scene_key]
    grid = scene_data['grid']
    
    # Generate sonar position using SAME trajectory logic as real data_generator
    position, direction, _ = generate_random_sonar_position(
        scene_data['cage_center'],
        scene_data['cage_radius'],
        DATA_GEN_CONFIG['distance_range'],
        DATA_GEN_CONFIG['angle_range'],
        prev_position=scene_data['prev_position'],
        prev_direction=scene_data['prev_direction']
    )
    
    # Update trajectory state (maintains continuity like real generator)
    scene_data['prev_position'] = position
    scene_data['prev_direction'] = direction
    
    # Create sonar with segmentation capability
    sonar = VoxelSonarWithSegmentation(
        position=position,
        direction=direction,
        range_m=SONAR_CONFIG['range_m'],
        fov_deg=SONAR_CONFIG['fov_deg'],
        num_beams=SONAR_CONFIG['num_beams']
    )
    
    # Scan scene - returns both intensity and material segmentation
    intensity, segmentation = sonar.scan_with_segmentation(grid)
    
    # Apply SAME image processing as real data_generator
    if DATA_GEN_CONFIG.get('log_scale', True):
        intensity = 10 * np.log10(np.maximum(intensity, 1e-10))
    
    if DATA_GEN_CONFIG.get('normalize', True):
        intensity = np.clip((intensity + 60) / 60, 0, 1)
    
    # Convert to Cartesian using SAME function as real data_generator
    multi_channel = polar_to_cartesian(
        intensity,
        range_m=SONAR_CONFIG['range_m'],
        fov_deg=SONAR_CONFIG['fov_deg'],
        output_size=DATA_GEN_CONFIG['image_size']
    )
    
    # Extract channels (same as real data_generator output format)
    image = multi_channel[0]           # Intensity
    valid_mask = multi_channel[1]      # Valid mask
    y_map = multi_channel[2]           # Y coordinates
    z_map = multi_channel[3]           # Z coordinates
    
    # Convert segmentation to Cartesian
    cart_segmentation = polar_to_cartesian_segmentation(segmentation)
    
    # Compute bbox from segmentation (our new feature!)
    bbox = compute_bbox_from_segmentation(cart_segmentation)
    net_visible = bbox is not None
    
    return {
        'image': image,
        'valid_mask': valid_mask,
        'y_map': y_map,
        'z_map': z_map,
        'segmentation': cart_segmentation,
        'net_visible': net_visible,
        'bbox': bbox,
        'sonar_position': position,
        'sonar_direction': direction,
        'scene_id': scene_id,
        'sample_id': sample_id,
    }


def test_bbox_generation():
    """Test bounding box generation from simulation world."""
    print("=" * 80)
    print("BOUNDING BOX GENERATION TEST - REAL SIMULATION WORLD")
    print("=" * 80)
    
    output_dir = Path("test_outputs/bbox_world_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = 20  # Total samples to generate
    scene_id = 0      # Use same scene for all samples (like real data collection)
    
    stats = {
        'total': 0,
        'net_visible': 0,
        'net_not_visible': 0,
        'bbox_widths': [],
        'bbox_heights': [],
        'bbox_angles': [],
    }
    
    print(f"\nGenerating {num_samples} samples from simulator...")
    print(f"Using real simulation world with cage, fish, and biomass")
    print(f"Trajectory continues across samples (realistic inspection path)")
    
    for sample_id in tqdm(range(num_samples), desc="Generating"):
        # Generate sample with real world
        # All samples use same scene_id (continuous trajectory in same environment)
        sample_data = generate_real_sample_with_bbox(scene_id, sample_id)
        
        stats['total'] += 1
        
        if sample_data['net_visible']:
            stats['net_visible'] += 1
            bbox = sample_data['bbox']
            stats['bbox_widths'].append(bbox['width'])
            stats['bbox_heights'].append(bbox['height'])
            stats['bbox_angles'].append(bbox['angle'])
        else:
            stats['net_not_visible'] += 1
        
        # Save visualization (like real simulator output)
        viz_path = output_dir / f"sample_{sample_id:04d}.png"
        visualize_bbox_sample(sample_data, viz_path)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total samples: {stats['total']}")
    print(f"Net visible: {stats['net_visible']} ({100*stats['net_visible']/stats['total']:.1f}%)")
    print(f"Net not visible: {stats['net_not_visible']} ({100*stats['net_not_visible']/stats['total']:.1f}%)")
    
    if stats['bbox_widths']:
        print(f"\nBounding box statistics:")
        print(f"  Width:  {np.mean(stats['bbox_widths']):.2f} ± {np.std(stats['bbox_widths']):.2f} m")
        print(f"          Range: [{np.min(stats['bbox_widths']):.2f}, {np.max(stats['bbox_widths']):.2f}] m")
        print(f"  Height: {np.mean(stats['bbox_heights']):.2f} ± {np.std(stats['bbox_heights']):.2f} m")
        print(f"          Range: [{np.min(stats['bbox_heights']):.2f}, {np.max(stats['bbox_heights']):.2f}] m")
        print(f"  Angle:  {np.mean(stats['bbox_angles']):.1f} ± {np.std(stats['bbox_angles']):.1f}°")
        print(f"          Range: [{np.min(stats['bbox_angles']):.1f}, {np.max(stats['bbox_angles']):.1f}]°")
    
    print(f"\n✓ Generated {num_samples} samples from simulator")
    print(f"✓ Saved visualizations to: {output_dir}")


def test_material_distribution():
    """Test material distribution in segmentation maps."""
    print("\n" + "=" * 80)
    print("MATERIAL DISTRIBUTION TEST - REAL SIMULATION")
    print("=" * 80)
    
    num_samples = 10
    scene_id = 100  # Different scene for this test
    material_counts = {name: 0 for name in MATERIAL_IDS.keys()}
    
    print(f"\nAnalyzing material distribution from {num_samples} samples...")
    
    for sample_id in tqdm(range(num_samples), desc="Analyzing"):
        sample_data = generate_real_sample_with_bbox(scene_id, sample_id)
        segmentation = sample_data['segmentation']
        
        # Count pixels per material
        unique_materials, counts = np.unique(segmentation, return_counts=True)
        
        for mat_id, count in zip(unique_materials, counts):
            mat_name = [k for k, v in MATERIAL_IDS.items() if v == mat_id][0]
            material_counts[mat_name] += count
    
    # Print distribution
    total_pixels = sum(material_counts.values())
    print(f"\nMaterial distribution across {num_samples} samples:")
    for mat_name, count in sorted(material_counts.items(), key=lambda x: -x[1]):
        percentage = 100 * count / total_pixels
        print(f"  {mat_name:15s}: {count:8d} pixels ({percentage:5.2f}%)")


def test_bbox_accuracy():
    """Test bbox placement accuracy by checking overlap with net pixels."""
    print("\n" + "=" * 80)
    print("BOUNDING BOX ACCURACY TEST - REAL SIMULATION")
    print("=" * 80)
    
    num_scenes = 5
    samplamples = 10
    scene_id = 200  # Different scene for this test
    
    print(f"\nTesting bbox accuracy on {num_samples} samples...")
    
    accuracies = []
    
    for sample_id in range(num_samples):
        sample_data = generate_real_sample_with_bbox(scene_id, sample_id)
        
        if not sample_data['net_visible']:
            continue
        
        segmentation = sample_data['segmentation']
        bbox = sample_data['bbox']
        
        # Get net pixel coordinates
        net_mask = segmentation == MATERIAL_IDS['net']
        net_pixels = np.sum(net_mask)
        
        if net_pixels == 0:
            continue
        
        # Check how many net pixels fall within bbox
        range_m = SONAR_CONFIG['range_m']
        fov_deg = SONAR_CONFIG['fov_deg']
        height, width = DATA_GEN_CONFIG['image_size']
        z_max = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
        
        # Get net pixel coordinates in meters
        net_coords = np.argwhere(net_mask)
        points_v = net_coords[:, 0]
        points_u = net_coords[:, 1]
        points_y = (points_v / height) * range_m
        points_z = (points_u / width) * (2 * z_max) - z_max
        
        # Compute bbox bounds (approximate - ignoring rotation)
        cy = bbox['center_y']
        cz = bbox['center_z']
        w = bbox['width']
        h = bbox['height']
        
        y_min = cy - h/2
        y_max = cy + h/2
        z_min = cz - w/2
        z_max = cz + w/2
        
        # Count pixels inside bbox bounds
        inside = ((points_y >= y_min) & (points_y <= y_max) & 
                  (points_z >= z_min) & (points_z <= z_max))
        
        accuracy = np.sum(inside) / len(points_y)
        accuracies.append(accuracy)
        
        print(f"  Sample {sample_id}: {accuracy*100:.1f}% net pixels inside bbox")
    
    if accuracies:
        print(f"\nAverage accuracy: {np.mean(accuracies)*100:.1f}% ± {np.std(accuracies)*100:.1f}%")
        print(f"(Note: This is approximate - ignores rotation)")


if __name__ == '__main__':
    test_bbox_generation()
    test_material_distribution()
    test_bbox_accuracy()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
