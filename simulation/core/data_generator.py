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
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .simulation import VoxelGrid, VoxelSonar, Material, EMPTY, NET, ROPE, FISH, BIOMASS
from .config import (
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


def meters_to_pixels(py, pz, range_m=None, fov_deg=None, image_size=None):
    """
    Convert meter coordinates to pixel coordinates.
    Uses config defaults if not specified.
    
    Args:
        py: Y coordinate in meters (forward)
        pz: Z coordinate in meters (lateral)
        range_m: Maximum range (default: from config)
        fov_deg: Field of view (default: from config)
        image_size: (height, width) of image (default: from config)
        
    Returns:
        u, v: Pixel coordinates (u=horizontal, v=vertical)
    """
    if range_m is None:
        range_m = SONAR_CONFIG['range_m']
    if fov_deg is None:
        fov_deg = SONAR_CONFIG['fov_deg']
    if image_size is None:
        image_size = DATA_GEN_CONFIG['image_size']
    
    height, width = image_size
    z_extent = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    
    # Convert meters to pixels
    u = (pz / (2 * z_extent) + 0.5) * width   # z -> horizontal pixel
    v = (py / range_m) * height                # y -> vertical pixel
    
    return u, v


def generate_heatmap_target(py, pz, image_size=None, sigma=None, 
                           range_m=None, fov_deg=None):
    """
    Generate Gaussian heatmap centered at hitpoint.
    Uses config defaults if not specified.
    
    Args:
        py, pz: Hitpoint in meters
        image_size: (H, W) (default: from config)
        sigma: Gaussian width in pixels (default: from config)
        range_m: Maximum range (default: from config)
        fov_deg: Field of view (default: from config)
        
    Returns:
        heatmap: (H, W) with Gaussian peak at hitpoint
    """
    if image_size is None:
        image_size = DATA_GEN_CONFIG['image_size']
    if sigma is None:
        sigma = DATA_GEN_CONFIG['heatmap_sigma_pixels']
    if range_m is None:
        range_m = SONAR_CONFIG['range_m']
    if fov_deg is None:
        fov_deg = SONAR_CONFIG['fov_deg']
    
    height, width = image_size
    u_star, v_star = meters_to_pixels(py, pz, range_m, fov_deg, image_size)
    
    # Create coordinate grids
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)
    
    # Gaussian heatmap
    heatmap = np.exp(-((uu - u_star)**2 + (vv - v_star)**2) / (2 * sigma**2))
    
    return heatmap.astype(np.float32)


def generate_direction_target(py, pz, ty, tz, image_size=None, 
                             radius=None, range_m=None, fov_deg=None):
    """
    Generate direction map with supervision only near hitpoint.
    Uses config defaults if not specified.
    
    Args:
        py, pz: Hitpoint in meters
        ty, tz: Direction unit vector
        image_size: (H, W) (default: from config)
        radius: Supervision radius in pixels (default: from config)
        range_m: Maximum range (default: from config)
        fov_deg: Field of view (default: from config)
        
    Returns:
        direction_map: (2, H, W) - direction at each pixel
        direction_mask: (H, W) - 1 where supervised, 0 elsewhere
    """
    if image_size is None:
        image_size = DATA_GEN_CONFIG['image_size']
    if radius is None:
        radius = DATA_GEN_CONFIG['direction_radius_pixels']
    if range_m is None:
        range_m = SONAR_CONFIG['range_m']
    if fov_deg is None:
        fov_deg = SONAR_CONFIG['fov_deg']
    
    height, width = image_size
    u_star, v_star = meters_to_pixels(py, pz, range_m, fov_deg, image_size)
    
    # Create coordinate grids
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)
    
    # Mask: circle around hitpoint
    dist_sq = (uu - u_star)**2 + (vv - v_star)**2
    direction_mask = (dist_sq <= radius**2).astype(np.float32)
    
    # Direction map (broadcast direction to all supervised pixels)
    direction_map = np.zeros((2, height, width), dtype=np.float32)
    direction_map[0] = ty  # y-direction
    direction_map[1] = tz  # z-direction
    
    return direction_map, direction_mask


def polar_to_cartesian(polar_image, range_m=None, fov_deg=None, output_size=None):
    """
    Convert polar sonar image to Cartesian representation with multiple channels.
    Uses config defaults if not specified.
    
    Args:
        polar_image: (range_bins, num_beams) polar sonar image
        range_m: Maximum range in meters (default: from config)
        fov_deg: Field of view in degrees (default: from config)
        output_size: (height, width) of output Cartesian image (default: from config)
        
    Returns:
        Multi-channel image (4, height, width):
            [0] intensity: sonar intensity
            [1] valid_mask: binary mask (1 where valid, 0 outside cone)
            [2] y_map: y-coordinate in meters for each pixel
            [3] z_map: z-coordinate in meters for each pixel (x in original frame)
    """
    if range_m is None:
        range_m = SONAR_CONFIG['range_m']
    if fov_deg is None:
        fov_deg = SONAR_CONFIG['fov_deg']
    if output_size is None:
        output_size = DATA_GEN_CONFIG['image_size']
    
    num_range_bins, num_beams = polar_image.shape
    height, width = output_size
    
    # Calculate extent
    x_extent = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    y_extent = range_m
    
    # Create coordinate grids for all pixels at once (vectorized)
    j_grid, i_grid = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixels to meters in sonar frame
    z_m = (j_grid - width/2) * (2 * x_extent / width)
    y_m = i_grid * (y_extent / height)  # No flip - row 0 is near, row N is far
    
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
    
    # Create multi-channel output
    intensity = np.zeros((height, width), dtype=np.float32)
    intensity[valid_mask] = polar_image[r_idx[valid_mask], theta_idx[valid_mask]]
    
    # Stack all channels: [intensity, valid_mask, y_map, z_map]
    multi_channel = np.stack([
        intensity,
        valid_mask.astype(np.float32),
        y_m,
        z_m
    ], axis=0)
    
    return multi_channel


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
    sonar_image_polar = sonar.scan(grid)
    
    # Process image if needed
    if DATA_GEN_CONFIG['log_scale']:
        sonar_image_polar = 10 * np.log10(np.maximum(sonar_image_polar, 1e-10))
    
    if DATA_GEN_CONFIG['normalize']:
        sonar_image_polar = np.clip((sonar_image_polar + 60) / 60, 0, 1)
    
    # Convert to Cartesian representation
    sonar_image = polar_to_cartesian(
        sonar_image_polar,
        range_m=SONAR_CONFIG['range_m'],
        fov_deg=SONAR_CONFIG['fov_deg'],
        output_size=(512, 512)
    )
    
    # Calculate ground truth using ray-segment intersection
    distance, hit_point, hit_segment = find_net_intersection(
        position, direction, net_segments, max_range=SONAR_CONFIG['range_m']
    )
    
    if distance is not None:
        orientation = get_relative_orientation(hit_segment, direction)
        net_visible = True
        
        # Convert to sonar's local coordinate frame
        # In sonar frame: Y-axis points forward (sonar direction), Z-axis points right
        relative_world = hit_point - position
        
        # Create rotation matrix from world to sonar frame
        # Sonar forward direction becomes Y-axis
        forward = direction / np.linalg.norm(direction)
        right = np.array([-forward[1], forward[0]])  # Perpendicular, 90° counter-clockwise
        
        # Transform hit point to sonar frame (py, pz)
        pz = np.dot(relative_world, right)
        py = np.dot(relative_world, forward)
        
        # Transform net direction to sonar frame (ty, tz) - unit vector
        x1, y1, x2, y2 = hit_segment
        net_dir_world = np.array([x2 - x1, y2 - y1])
        net_dir_world = net_dir_world / np.linalg.norm(net_dir_world)
        
        tz = np.dot(net_dir_world, right)
        ty = np.dot(net_dir_world, forward)
        
        # Normalize to ensure unit vector (should already be, but ensure numerical stability)
        net_norm = np.sqrt(ty**2 + tz**2)
        ty = ty / net_norm
        tz = tz / net_norm
    else:
        orientation = None
        net_visible = False
        py = None
        pz = None
        ty = None
        tz = None
    
    # Prepare output dictionary
    output = {
        'image': sonar_image[0].astype(np.float32),      # Intensity channel (H, W)
        'valid_mask': sonar_image[1].astype(np.float32), # Valid mask (H, W)
        'y_map': sonar_image[2].astype(np.float32),      # Y coordinate map (H, W)
        'z_map': sonar_image[3].astype(np.float32),      # Z coordinate map (H, W)
        'net_visible': net_visible,
        'p': np.array([py, pz], dtype=np.float32) if py is not None else None,
        't': np.array([ty, tz], dtype=np.float32) if ty is not None else None,
        # Derived metrics for evaluation
        'distance_m': float(distance) if distance is not None else None,
        'orientation_deg': float(orientation) if orientation is not None else None,
        # Metadata
        'scene_id': scene_id,
        'sample_id': sample_id,
    }
    
    return output


def apply_rotation_augmentation(output_dict, angle_deg):
    """
    Apply rotation augmentation to image and labels.
    
    Args:
        output_dict: Dictionary from generate_sample() with image, p, t, etc.
        angle_deg: Rotation angle in degrees (counter-clockwise)
        
    Returns:
        rotated_dict: Updated dictionary with rotated data
    """
    if angle_deg == 0:
        return output_dict.copy()
    
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotated = output_dict.copy()
    
    # Rotate image channels
    for key in ['image', 'valid_mask', 'y_map', 'z_map']:
        channel = output_dict[key]
        h, w = channel.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
        rotated[key] = cv2.warpAffine(channel, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    
    # Rotate coordinate map values
    y_map = rotated['y_map']
    z_map = rotated['z_map']
    rotated['y_map'] = y_map * cos_a - z_map * sin_a
    rotated['z_map'] = y_map * sin_a + z_map * cos_a
    
    # Rotate labels if net is visible
    if output_dict.get('net_visible', False) and output_dict.get('p') is not None:
        p = output_dict['p']  # [py, pz]
        t = output_dict['t']  # [ty, tz]
        
        py, pz = p[0], p[1]
        ty, tz = t[0], t[1]
        
        # Rotate hit point
        py_rot = py * cos_a - pz * sin_a
        pz_rot = py * sin_a + pz * cos_a
        rotated['p'] = np.array([py_rot, pz_rot], dtype=np.float32)
        
        # Rotate net direction
        ty_rot = ty * cos_a - tz * sin_a
        tz_rot = ty * sin_a + tz * cos_a
        rotated['t'] = np.array([ty_rot, tz_rot], dtype=np.float32)
        
        # Update derived metrics
        rotated['distance_m'] = float(np.sqrt(py_rot**2 + pz_rot**2))
        # Orientation relative to sonar
        net_normal_angle = np.arctan2(tz_rot, ty_rot)
        orientation_rad = net_normal_angle - np.pi/2
        orientation_deg = np.degrees(orientation_rad)
        while orientation_deg > 90:
            orientation_deg -= 180
        while orientation_deg < -90:
            orientation_deg += 180
        rotated['orientation_deg'] = float(orientation_deg)
    
    return rotated


def visualize_sample(sample_data, output_path):
    """
    Visualize a sonar sample with ground truth overlay in Cartesian coordinates.
    
    Args:
        sample_data: Dictionary or .npz file with image, p, t, etc.
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Handle both dict and npz formats
    if isinstance(sample_data, dict):
        intensity = sample_data.get('image')
        p = sample_data.get('p')
        t = sample_data.get('t')
        net_visible = sample_data.get('net_visible', False)
        distance_m = sample_data.get('distance_m')
        orientation_deg = sample_data.get('orientation_deg')
        sample_id = sample_data.get('sample_id', 0)
    else:
        # Assume npz file
        intensity = sample_data['image']
        p = sample_data['p'] if sample_data['net_visible'] else None
        t = sample_data['t'] if sample_data['net_visible'] else None
        net_visible = sample_data['net_visible']
        distance_m = sample_data.get('distance_m')
        orientation_deg = sample_data.get('orientation_deg')
        sample_id = sample_data.get('sample_id', 0)
    
    # Set dark grey background
    ax.set_facecolor('#2a2a2a')
    fig.patch.set_facecolor('#2a2a2a')
    
    # Calculate extent based on FOV (use config values)
    range_m = SONAR_CONFIG['range_m']
    fov_deg = SONAR_CONFIG['fov_deg']
    z_max = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    extent = [-z_max, z_max, 0, range_m]
    
    # Plot image with annotation
    ax.imshow(intensity, cmap='gray', vmin=0, vmax=1, 
              extent=extent, origin='lower', aspect='auto')
    
    # Overlay ground truth (if net is visible)
    if net_visible and p is not None:
        py, pz = p[0], p[1]
        
        # Draw hit point (small red dot)
        ax.plot(pz, py, 'ro', markersize=8, alpha=0.9, zorder=10)
        
        # Draw the net line segment
        if t is not None:
            ty, tz = t[0], t[1]
            net_line_length = 4.0  # meters
            half_length = net_line_length / 2
            
            # Compute endpoints of net line
            p1_y = py - half_length * ty
            p1_z = pz - half_length * tz
            p2_y = py + half_length * ty
            p2_z = pz + half_length * tz
            
            ax.plot([p1_z, p2_z], [p1_y, p2_y], 'r-', 
                   linewidth=2.5, alpha=0.9, zorder=10, label='Net line')
        
        title = f'Sample {sample_id} | d={distance_m:.2f}m | θ={orientation_deg:+.0f}°'
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
    viz_interval = DATA_GEN_CONFIG.get('viz_interval', 10)
    
    if viz_interval > 0:
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        print(f"\nGenerating {num_samples} {split} samples...")
        print(f"Output directory: {output_dir}")
        print(f"Saving visualizations every {viz_interval} samples to: {images_dir}")
    else:
        images_dir = None
        print(f"\nGenerating {num_samples} {split} samples...")
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
    """Save pixel-to-meters mapping and other constants."""
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
        'sonar_config': SONAR_CONFIG,
        'scene_config': SCENE_CONFIG,
        'data_gen_config': DATA_GEN_CONFIG,
    }
    
    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved dataset metadata to {metadata_path}")


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
