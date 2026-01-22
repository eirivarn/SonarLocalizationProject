"""
Data generator for neural network training.

Generates synthetic sonar images with ground truth labels using the voxel simulation.
"""
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import VoxelGrid, VoxelSonar, Material, EMPTY, NET, ROPE, FISH, BIOMASS
from config import (
    SONAR_CONFIG, SCENE_CONFIG, DATA_GEN_CONFIG, DATASET_DIR
)


def ray_segment_intersection(ray_origin, ray_dir, segment):
    """
    Calculate intersection of ray with line segment.
    
    Args:
        ray_origin: [x, y] ray starting point
        ray_dir: [dx, dy] ray direction (normalized)
        segment: (x1, y1, x2, y2) line segment
        
    Returns:
        distance: Distance along ray to intersection (None if no intersection)
        hit_point: [x, y] intersection point (None if no intersection)
    """
    x1, y1, x2, y2 = segment
    px, py = ray_origin
    dx, dy = ray_dir
    
    # Segment vector
    sx = x2 - x1
    sy = y2 - y1
    
    # Solve: P + t*D = S1 + u*(S2-S1)
    denominator = dx * (-sy) - dy * (-sx)
    
    if abs(denominator) < 1e-10:
        return None, None
    
    # Solve for t and u
    t = ((x1 - px) * (-sy) - (y1 - py) * (-sx)) / denominator
    u = ((x1 - px) * dy - (y1 - py) * dx) / denominator
    
    # Check if intersection is valid
    eps = 1e-6
    if t >= -eps and -eps <= u <= 1 + eps:
        hit_x = px + t * dx
        hit_y = py + t * dy
        return max(0, t), np.array([hit_x, hit_y])
    
    return None, None


def find_net_intersection(sonar_pos, sonar_dir, net_segments, max_range=20.0):
    """
    Find closest net intersection along sonar ray.
    
    Args:
        sonar_pos: [x, y] sonar position
        sonar_dir: [dx, dy] sonar direction (normalized)
        net_segments: List of (x1, y1, x2, y2) segments
        max_range: Maximum range to check
        
    Returns:
        distance: Distance to net in meters (None if not found)
        hit_point: [x, y] hit position (None if not found)
        segment: The segment that was hit (None if not found)
    """
    sonar_dir = np.array(sonar_dir)
    sonar_dir = sonar_dir / np.linalg.norm(sonar_dir)
    
    all_intersections = []
    
    for segment in net_segments:
        distance, hit_point = ray_segment_intersection(sonar_pos, sonar_dir, segment)
        
        if distance is not None and distance <= max_range:
            all_intersections.append((distance, hit_point, segment))
    
    if len(all_intersections) == 0:
        return None, None, None
    
    # Return the closest intersection
    all_intersections.sort(key=lambda x: x[0])
    return all_intersections[0]


def get_segment_orientation(segment):
    """
    Get orientation angle of a segment in degrees [0, 180).
    Since a line has no inherent direction, we normalize to half-circle.
    """
    x1, y1, x2, y2 = segment
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad) % 360
    
    # Normalize to [0, 180) - a line has no direction
    if angle_deg >= 180:
        angle_deg -= 180
    
    return angle_deg


def get_relative_orientation(segment, sonar_direction):
    """
    Get net orientation relative to sonar direction in degrees [-90, 90].
    
    Args:
        segment: (x1, y1, x2, y2) net segment
        sonar_direction: [dx, dy] sonar direction vector
        
    Returns:
        Relative angle in degrees [-90, 90]
        0° = net is perpendicular to sonar ray (broadside)
        ±90° = net is parallel to sonar ray (edge-on)
    """
    # Get absolute net orientation [0, 180)
    net_angle = get_segment_orientation(segment)
    
    # Get sonar direction angle
    sonar_angle = np.degrees(np.arctan2(sonar_direction[1], sonar_direction[0])) % 360
    
    # Calculate relative angle
    relative = net_angle - sonar_angle
    
    # Normalize to [-90, 90]
    # Since net has no inherent direction (0° = 180°), we fold the angle
    while relative > 90:
        relative -= 180
    while relative < -90:
        relative += 180
    
    return relative


def is_overlapping_net(grid, position, radius):
    """
    Check if a circular region overlaps with existing NET voxels.
    
    Args:
        grid: VoxelGrid to check
        position: [x, y] center position
        radius: Radius of region to check in meters
        
    Returns:
        True if any NET voxels are within the radius
    """
    voxel_size = grid.voxel_size
    check_radius_voxels = int(radius / voxel_size) + 2
    
    cx, cy = grid.world_to_voxel(position)
    
    for dx in range(-check_radius_voxels, check_radius_voxels + 1):
        for dy in range(-check_radius_voxels, check_radius_voxels + 1):
            vx = cx + dx
            vy = cy + dy
            
            if grid.is_inside(vx, vy):
                # Check if this is a NET voxel
                # NET has density=0.3, reflectivity=0.2
                if grid.density[vx, vy] > 0.25 and grid.density[vx, vy] < 0.4:
                    if grid.reflectivity[vx, vy] > 0.15 and grid.reflectivity[vx, vy] < 0.3:
                        # Check if within radius
                        world_x = (vx + 0.5) * voxel_size
                        world_y = (vy + 0.5) * voxel_size
                        dist = np.sqrt((world_x - position[0])**2 + (world_y - position[1])**2)
                        if dist < radius:
                            return True
    return False


def create_random_scene(seed=None):
    """
    Create a randomized fish farm scene.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        grid: VoxelGrid with scene
        fish_data: List of fish objects
        cage_center: Cage center position
        cage_radius: Cage radius
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create world
    world_size = int(SCENE_CONFIG['world_size_m'] / SONAR_CONFIG['voxel_size'])
    grid = VoxelGrid(world_size, world_size, voxel_size=SONAR_CONFIG['voxel_size'])
    
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
    
    # Create cage net
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
        start_idx = 1 if i > 0 else 0  # Skip first point except for first panel
        
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
            
            # Store point for segment creation
            all_net_points.append((x, y))
            
            # Set voxels for simulation
            grid.set_box(
                np.array([x - 0.08, y - 0.08]),
                np.array([x + 0.08, y + 0.08]),
                NET
            )
            
            if j % 3 == 0:  # Add rope at intervals
                grid.set_box(
                    np.array([x - 0.12, y - 0.12]),
                    np.array([x + 0.12, y + 0.12]),
                    ROPE
                )
    
    # Add biomass if enabled
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
            
            # Check if this overlaps with net
            if not is_overlapping_net(grid, np.array([x, y]), patch_size):
                grid.set_circle(np.array([x, y]), patch_size, BIOMASS)
                biomass_count += 1
    
    # Add fish if enabled
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
            
            # Use the larger dimension (length) for overlap check
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
    
    # Create segments from continuous loop of points
    for i in range(len(all_net_points)):
        x1, y1 = all_net_points[i]
        x2, y2 = all_net_points[(i + 1) % len(all_net_points)]
        net_segments.append((x1, y1, x2, y2))
    
    return grid, fish_data, cage_center, cage_radius, net_segments


def generate_random_sonar_position(cage_center, cage_radius, distance_range, angle_range, 
                                   prev_position=None, prev_direction=None):
    """
    Generate sonar position using random walk with bias towards optimal inspection distance.
    
    Args:
        cage_center: Cage center [x, y]
        cage_radius: Cage radius
        distance_range: [min_dist, max_dist] from net surface (1-5m for bias)
        angle_range: [min_angle, max_angle] in degrees (not used in random walk)
        prev_position: Previous sonar position (None for first sample in scene)
        prev_direction: Previous sonar direction (None for first sample in scene)
        
    Returns:
        position: [x, y] sonar position
        direction: [dx, dy] sonar direction (normalized)
        ground_truth: dict with actual distance and orientation
    """
    target_distance_range = distance_range  # [1, 5] meters from net
    
    if prev_position is None:
        # First sample in scene - initialize random position
        cage_angle = np.random.uniform(0, 360)
        cage_angle_rad = np.radians(cage_angle)
        
        distance_from_net = np.random.uniform(*target_distance_range)
        
        # 90% inside, 10% outside (mainly inside as requested)
        spawn_inside = np.random.rand() < 0.9
        
        # Calculate position based on distance from center
        if spawn_inside:
            # Inside cage: radius - distance
            distance_from_center = cage_radius - distance_from_net
        else:
            # Outside cage: radius + distance
            distance_from_center = cage_radius + distance_from_net
        
        sonar_x = cage_center[0] + distance_from_center * np.cos(cage_angle_rad)
        sonar_y = cage_center[1] + distance_from_center * np.sin(cage_angle_rad)
        
        position = np.array([sonar_x, sonar_y])
        
        # Point towards net surface (not center)
        to_center = cage_center - position
        distance_from_center = np.linalg.norm(to_center)
        to_center_norm = to_center / distance_from_center
        
        # If inside cage, point outward (away from center towards net)
        # If outside cage, point inward (towards center/net)
        if distance_from_center < cage_radius:
            # Inside cage - point away from center (towards net)
            to_net = -to_center_norm
        else:
            # Outside cage - point towards center (towards net)
            to_net = to_center_norm
        
        # Add random angular offset [-30, +30 degrees]
        angle_offset = np.random.uniform(-30, 30)
        rotation = np.radians(angle_offset)
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        direction = np.array([
            to_net[0] * cos_r - to_net[1] * sin_r,
            to_net[0] * sin_r + to_net[1] * cos_r
        ])
        
    else:
        # Random walk from previous position
        # Step size: 0.2 to 0.8 meters per sample
        step_size = np.random.uniform(0.2, 0.8)
        step_angle = np.random.uniform(0, 2 * np.pi)
        
        step = np.array([
            step_size * np.cos(step_angle),
            step_size * np.sin(step_angle)
        ])
        
        # Apply bias force to keep sonar in optimal range (1-5m from net)
        current_dist_to_center = np.linalg.norm(prev_position - cage_center)
        
        # Find closest point on cage
        to_center = cage_center - prev_position
        to_center_norm = to_center / np.linalg.norm(to_center)
        closest_net_point = cage_center + cage_radius * (-to_center_norm)
        distance_to_net = np.linalg.norm(prev_position - closest_net_point)
        
        # Bias force to maintain 1-5m from net
        bias_force = np.zeros(2)
        target_mid = (target_distance_range[0] + target_distance_range[1]) / 2  # 3m
        
        if distance_to_net < target_distance_range[0]:
            # Too close - push away from net
            bias_force = -(closest_net_point - prev_position) * 0.3
        elif distance_to_net > target_distance_range[1]:
            # Too far - pull towards net
            bias_force = (closest_net_point - prev_position) * 0.3
        
        # Combine random walk with bias
        new_position = prev_position + step + bias_force
        
        # Keep within world bounds
        world_size = SCENE_CONFIG['world_size_m']
        new_position = np.clip(new_position, [0.5, 0.5], [world_size - 0.5, world_size - 0.5])
        
        position = new_position
        
        # Update direction - point towards net surface
        to_center = cage_center - position
        distance_from_center = np.linalg.norm(to_center)
        to_center_norm = to_center / distance_from_center
        
        # If inside cage, point outward (towards net)
        # If outside cage, point inward (towards net)
        if distance_from_center < cage_radius:
            to_net = -to_center_norm
        else:
            to_net = to_center_norm
        
        # Add random angular offset [-45, +45 degrees] for variety
        angle_offset = np.random.uniform(-45, 45)
        rotation = np.radians(angle_offset)
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        target_direction = np.array([
            to_net[0] * cos_r - to_net[1] * sin_r,
            to_net[0] * sin_r + to_net[1] * cos_r
        ])
        
        # Smooth transition from previous direction (80% new, 20% old)
        if prev_direction is not None:
            direction = 0.8 * target_direction + 0.2 * prev_direction
            direction = direction / np.linalg.norm(direction)
        else:
            direction = target_direction
    
    # No ground truth yet - will be computed after raycasting
    return position, direction, None


# Global state for trajectory tracking (reset per scene)
_trajectory_state = {}


def generate_sample(scene_id, sample_id):
    """
    Generate a single training sample.
    
    Returns:
        sonar_image: (1024, 256) numpy array
        label: dict with distance_m, orientation_deg, and metadata
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
    
    # Generate sonar position using random walk
    position, direction, ground_truth = generate_random_sonar_position(
        cage_center, cage_radius,
        DATA_GEN_CONFIG['distance_range'],
        DATA_GEN_CONFIG['angle_range'],
        prev_position=scene_data['prev_position'],
        prev_direction=scene_data['prev_direction']
    )
    
    # Update trajectory state for next sample
    scene_data['prev_position'] = position
    scene_data['prev_direction'] = direction
    
    # Create sonar
    sonar = VoxelSonar(
        position=position,
        direction=direction,
        range_m=SONAR_CONFIG['range_m'],
        fov_deg=SONAR_CONFIG['fov_deg'],
        num_beams=SONAR_CONFIG['num_beams']
    )
    
    # Capture image
    sonar_image = sonar.scan(grid)
    
    # Process image if needed
    if DATA_GEN_CONFIG['log_scale']:
        sonar_image = 10 * np.log10(np.maximum(sonar_image, 1e-10))
    
    if DATA_GEN_CONFIG['normalize']:
        sonar_image = np.clip((sonar_image + 60) / 60, 0, 1)
    
    # Calculate ground truth using ray-segment intersection
    distance, hit_point, hit_segment = find_net_intersection(
        position, direction, net_segments, max_range=SONAR_CONFIG['range_m']
    )
    
    if distance is not None:
        orientation = get_relative_orientation(hit_segment, direction)
        net_visible = True
    else:
        orientation = None
        net_visible = False
    
    label = {
        'distance_m': float(distance) if distance is not None else None,
        'orientation_deg': float(orientation) if orientation is not None else None,
        'net_visible': net_visible,
        'scene_id': scene_id,
        'sample_id': sample_id,
        'sonar_position': position.tolist(),
        'sonar_direction': direction.tolist(),
        'cage_center': cage_center.tolist(),
        'cage_radius': float(cage_radius),
    }
    
    return sonar_image.astype(np.float32), label


def visualize_sample(sonar_image, label, output_path):
    """
    Visualize a sonar sample with ground truth overlay.
    
    Args:
        sonar_image: (1024, 256) sonar image
        label: Ground truth label dict
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': 'polar'})
    
    # Convert image to polar coordinates for display
    num_range_bins, num_beams = sonar_image.shape
    
    # Create polar mesh
    theta = np.linspace(-np.pi/3, np.pi/3, num_beams)  # 120° FOV
    r = np.linspace(0, 20, num_range_bins)  # 20m range
    theta_grid, r_grid = np.meshgrid(theta, r)
    
    # Display sonar image
    ax.pcolormesh(theta_grid, r_grid, sonar_image, cmap='gray', shading='auto', vmin=0, vmax=1)
    
    # Overlay ground truth (if net is visible)
    if label.get('net_visible', False) and label['distance_m'] is not None:
        distance = label['distance_m']
        orientation_deg = label['orientation_deg'] if label['orientation_deg'] is not None else 0
        
        # Draw hit point (small red dot)
        ax.plot(0, distance, 'ro', markersize=4, alpha=0.8)
        
        # Draw the net line at the detected distance and orientation
        # Net line indicates the orientation relative to sonar
        net_line_length = 4.0  # meters width of displayed net line
        angle_span = net_line_length / (2 * distance) if distance > 0 else 0.1
        
        # Draw net line at distance (perpendicular view if orientation=0)
        # Rotate based on relative orientation
        orient_rad = np.radians(orientation_deg)
        theta_left = -angle_span * np.cos(orient_rad)
        theta_right = angle_span * np.cos(orient_rad)
        
        ax.plot([theta_left, theta_right], [distance, distance], 'r-', 
               linewidth=2, alpha=0.8)
        
        title = f'Sample {label["sample_id"]} | d={distance:.2f}m | θ={orientation_deg:+.0f}°'
    else:
        title = f'Sample {label["sample_id"]} | NO NET'
    
    # Formatting
    ax.set_theta_zero_location('N')
    ax.set_ylim(0, 20)
    ax.set_title(title, fontsize=12, pad=15)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_theta_direction(-1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_dataset(split='train', num_samples=None):
    """
    Generate a dataset split.
    
    Args:
        split: 'train', 'val', or 'test'
        num_samples: Number of samples (uses config if None)
    """
    if num_samples is None:
        num_samples = DATA_GEN_CONFIG[f'num_{split}']
    
    output_dir = DATASET_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images subdirectory for visualizations
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating {num_samples} {split} samples...")
    print(f"Output directory: {output_dir}")
    print(f"Saving visualizations every 10th sample to: {images_dir}")
    
    for i in tqdm(range(num_samples)):
        # Generate sample
        sonar_image, label = generate_sample(scene_id=i, sample_id=i)
        
        # Save image
        image_path = output_dir / f"sample_{i:06d}.npy"
        np.save(image_path, sonar_image)
        
        # Save label
        label_path = output_dir / f"sample_{i:06d}.json"
        with open(label_path, 'w') as f:
            json.dump(label, f, indent=2)
        
        # Save visualization for every 10th sample
        if i % 10 == 0:
            viz_path = images_dir / f"sample_{i:06d}.png"
            visualize_sample(sonar_image, label, viz_path)
    
    print(f"✓ Generated {num_samples} samples")
    print(f"✓ Saved {num_samples // 10} visualization images")
    
    # Save dataset info
    info = {
        'split': split,
        'num_samples': num_samples,
        'sonar_config': SONAR_CONFIG,
        'scene_config': SCENE_CONFIG,
        'data_gen_config': DATA_GEN_CONFIG,
    }
    
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    """Generate all dataset splits."""
    print("=" * 80)
    print("SONAR NET DETECTION - DATASET GENERATION")
    print("=" * 80)
    
    # Generate datasets
    generate_dataset('train')
    generate_dataset('val')
    generate_dataset('test')
    
    print("\n" + "=" * 80)
    print("✓ Dataset generation complete!")
    print(f"Datasets saved to: {DATASET_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
