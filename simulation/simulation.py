"""Simple voxel-based sonar simulation with density field ray marching.

This is a clean reimplementation using volumetric ray marching:
- No surface raycasting
- Voxel grid with density/material properties
- Ray marching integrates scattering continuously
- Returns build up across volume
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dataclasses import dataclass
import cv2
import sys
from pathlib import Path

# Add utils to path for net detection
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG
from utils.image_enhancement import adaptive_linear_momentum_merge_fast
from utils.sonar_tracking import NetTracker


@dataclass
class Material:
    """Material properties for voxel grid."""
    name: str
    density: float          # 0-1: how much matter is here
    reflectivity: float     # 0-1: acoustic backscatter strength
    absorption: float       # 0-1: how much energy is absorbed per meter
    

# Material library
EMPTY = Material("empty", 0.0, 0.0, 0.0)
NET = Material("net", 0.3, 0.2, 0.15)  # Light absorption - nets are thin and sparse
ROPE = Material("rope", 0.8, 0.4, 0.8)  # Strong absorption
FISH = Material("fish", 0.7, 0.5, 0.6)  # Fish cast shadows
WALL = Material("wall", 1.0, 0.5, 1.0)
BIOMASS = Material("biomass", 0.9, 0.7, 0.4)  # Algae/fouling - very bright, moderate shadow
DEBRIS_LIGHT = Material("debris_light", 0.8, 0.6, 0.3)  # Light debris - bright
DEBRIS_MEDIUM = Material("debris_medium", 0.9, 0.7, 0.7)  # Medium debris - brighter
DEBRIS_HEAVY = Material("debris_heavy", 1.0, 0.8, 1.2)  # Heavy debris - very bright


class VoxelGrid:
    """2D voxel grid storing material density and properties."""
    
    def __init__(self, size_x: int, size_y: int, voxel_size: float = 0.1):
        """Initialize empty voxel grid.
        
        Args:
            size_x, size_y: Grid dimensions in voxels
            voxel_size: Size of each voxel in meters
        """
        self.size_x = size_x
        self.size_y = size_y
        self.voxel_size = voxel_size
        
        # Grid stores density and acoustic properties per voxel
        self.density = np.zeros((size_x, size_y), dtype=np.float32)
        self.reflectivity = np.zeros((size_x, size_y), dtype=np.float32)
        self.absorption = np.zeros((size_x, size_y), dtype=np.float32)
    
    def world_to_voxel(self, pos: np.ndarray) -> tuple:
        """Convert world position to voxel indices."""
        x = int(pos[0] / self.voxel_size)
        y = int(pos[1] / self.voxel_size)
        return x, y
    
    def is_inside(self, x: int, y: int) -> bool:
        """Check if voxel indices are inside grid."""
        return 0 <= x < self.size_x and 0 <= y < self.size_y
    
    def set_circle(self, center: np.ndarray, radius: float, material: Material):
        """Fill circle with material."""
        cx, cy = self.world_to_voxel(center)
        r_voxels = int(radius / self.voxel_size)
        
        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                if dx*dx + dy*dy <= r_voxels*r_voxels:
                    x, y = cx + dx, cy + dy
                    if self.is_inside(x, y):
                        self.density[x, y] = material.density
                        self.reflectivity[x, y] = material.reflectivity
                        self.absorption[x, y] = material.absorption
    
    def set_ellipse(self, center: np.ndarray, radii: np.ndarray, orientation: float, material: Material):
        """Fill ellipse with material (elongated along orientation angle).
        
        Args:
            center: Center position [x, y]
            radii: Semi-axes lengths [length_along_orientation, width]
            orientation: Rotation angle (radians)
            material: Material to fill with
        """
        cx, cy = self.world_to_voxel(center)
        # radii[0] is length (along orientation), radii[1] is width
        rx_vox = max(1, int(radii[0] / self.voxel_size))  # Length along orientation
        ry_vox = max(1, int(radii[1] / self.voxel_size))  # Width perpendicular
        
        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)
        
        max_r = max(rx_vox, ry_vox)
        
        for dx in range(-max_r, max_r + 1):
            for dy in range(-max_r, max_r + 1):
                # Rotate back to ellipse space
                dx_rot = dx * cos_a + dy * sin_a
                dy_rot = -dx * sin_a + dy * cos_a
                
                # Check if inside ellipse
                dist_sq = (dx_rot/rx_vox)**2 + (dy_rot/ry_vox)**2
                
                if dist_sq <= 1.0:
                    x, y = cx + dx, cy + dy
                    if self.is_inside(x, y):
                        self.density[x, y] = material.density
                        self.reflectivity[x, y] = material.reflectivity
                        self.absorption[x, y] = material.absorption
    
    def clear_fish(self):
        """Clear all fish material from grid (for dynamic updates)."""
        # Use reflectivity to identify fish (FISH has 0.5, NET has 0.2, ROPE has 0.4)
        mask = np.abs(self.reflectivity - FISH.reflectivity) < 0.01
        self.density[mask] = EMPTY.density
        self.reflectivity[mask] = EMPTY.reflectivity
        self.absorption[mask] = EMPTY.absorption
    
    def clear_debris(self):
        """Clear all debris material from grid (for dynamic updates)."""
        # Clear debris by checking for debris reflectivity values
        mask = (np.abs(self.reflectivity - DEBRIS_LIGHT.reflectivity) < 0.01) | \
               (np.abs(self.reflectivity - DEBRIS_MEDIUM.reflectivity) < 0.01) | \
               (np.abs(self.reflectivity - DEBRIS_HEAVY.reflectivity) < 0.01)
        self.density[mask] = EMPTY.density
        self.reflectivity[mask] = EMPTY.reflectivity
        self.absorption[mask] = EMPTY.absorption
    
    def set_box(self, min_pos: np.ndarray, max_pos: np.ndarray, material: Material):
        """Fill box region with material."""
        x1, y1 = self.world_to_voxel(min_pos)
        x2, y2 = self.world_to_voxel(max_pos)
        
        x1, x2 = max(0, min(x1, x2)), min(self.size_x, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(self.size_y, max(y1, y2))
        
        self.density[x1:x2, y1:y2] = material.density
        self.reflectivity[x1:x2, y1:y2] = material.reflectivity
        self.absorption[x1:x2, y1:y2] = material.absorption
    
    def set_net_plane(self, x_pos: float, y_range: tuple, z_range: tuple, 
                      mesh_size: float = 0.5, rope_thickness: float = 0.05):
        """Create a net plane (vertical mesh pattern)."""
        x_vox = int(x_pos / self.voxel_size)
        
        y_min, y_max = int(y_range[0] / self.voxel_size), int(y_range[1] / self.voxel_size)
        z_min, z_max = int(z_range[0] / self.voxel_size), int(z_range[1] / self.voxel_size)
        
        mesh_voxels = int(mesh_size / self.voxel_size)
        rope_voxels = int(rope_thickness / self.voxel_size)
        
        # Fill net pattern
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                if not self.is_inside(x_vox, y, z):
                    continue
                
                # Check if on rope line (grid pattern)
                y_mod = y % mesh_voxels
                z_mod = z % mesh_voxels
                
                on_rope = (y_mod < rope_voxels or z_mod < rope_voxels)
                
                if on_rope:
                    # Rope
                    self.density[x_vox, y, z] = ROPE.density
                    self.reflectivity[x_vox, y, z] = ROPE.reflectivity
                    self.absorption[x_vox, y, z] = ROPE.absorption
                else:
                    # Net mesh (thinner)
                    if np.random.rand() < 0.3:  # Sparse mesh
                        self.density[x_vox, y, z] = NET.density
                        self.reflectivity[x_vox, y, z] = NET.reflectivity
                        self.absorption[x_vox, y, z] = NET.absorption


class VoxelSonar:
    """Sonar using voxel ray marching (no surface raycasting)."""
    
    def __init__(self, position: np.ndarray, direction: np.ndarray,
                 range_m: float = 20.0, fov_deg: float = 120.0, num_beams: int = 256):
        """Initialize sonar.
        
        Args:
            position: Sonar position in world (meters)
            direction: Forward direction (normalized)
            range_m: Maximum range (20m matches SOLAQUA RANGE_MAX_M_DEFAULT)
            fov_deg: Field of view in degrees (120° matches SOLAQUA FOV_DEG_DEFAULT)
            num_beams: Number of beams (256 matches SOLAQUA dataset)
        """
        self.position = position.copy()
        self.direction = direction / (np.linalg.norm(direction) + 1e-9)
        self.range_m = range_m
        self.fov_deg = fov_deg
        self.num_beams = num_beams
        self.range_bins = 1024  # 1024 range bins matches SOLAQUA dataset (1024×256 frames)
    
    def scan(self, grid: VoxelGrid) -> np.ndarray:
        """Scan scene using volumetric ray marching.
        
        Returns:
            (range_bins, num_beams) array of accumulated returns
        """
        image = np.zeros((self.range_bins, self.num_beams), dtype=np.float32)
        
        fov_rad = np.deg2rad(self.fov_deg)
        
        for beam_idx in range(self.num_beams):
            # Beam direction
            t = beam_idx / (self.num_beams - 1) if self.num_beams > 1 else 0.5
            angle = (-fov_rad / 2) + t * fov_rad
            
            # BEAM PATTERN: Gaussian falloff toward edges
            # Center beams are strongest, edge beams are weaker
            beam_pattern = np.exp(-((angle / (fov_rad/2))**2) * 2.5)  # Increased from 1.5 to 2.5 for stronger falloff
            
            # Rotate direction by angle
            dir_angle = np.arctan2(self.direction[1], self.direction[0])
            beam_angle = dir_angle + angle
            beam_dir = np.array([np.cos(beam_angle), np.sin(beam_angle)])
            
            # Ray march through volume
            self._march_ray(grid, self.position, beam_dir, image[:, beam_idx], beam_pattern)
        
        # TEMPORAL DECORRELATION: Additional frame-to-frame variability on objects only
        # Small random fluctuations representing net sway, water movement, etc.
        # Only apply where there's actual signal (not background)
        signal_mask = image > 1e-8
        decorrelation_noise = np.random.gamma(shape=5.0, scale=1.0/5.0, size=image.shape)
        image[signal_mask] *= decorrelation_noise[signal_mask]
        
        return image
    
    def _march_ray(self, grid: VoxelGrid, origin: np.ndarray, direction: np.ndarray,
                   output_bins: np.ndarray, beam_strength: float = 1.0):
        """March ray through voxel grid, accumulating returns.
        
        Args:
            beam_strength: Beam pattern multiplier (1.0 at center, lower at edges)
        """
        # DDA-style voxel traversal
        step_size = grid.voxel_size * 0.5  # Sub-voxel steps for smooth integration
        num_steps = int(self.range_m / step_size)
        
        current_pos = origin.copy()
        energy = 1.0  # Energy budget
        
        for step in range(num_steps):
            distance = step * step_size
            if distance >= self.range_m or energy < 0.01:
                break
            
            # Get voxel properties at current position
            x, y = grid.world_to_voxel(current_pos)
            
            if not grid.is_inside(x, y):
                current_pos += direction * step_size
                continue
            
            density = grid.density[x, y]
            reflectivity = grid.reflectivity[x, y]
            absorption = grid.absorption[x, y]
            
            # VOLUME SCATTERING: Deposit return proportional to density and reflectivity
            if density > 0.01:
                # ACOUSTIC SPECKLE: Multiplicative noise from coherent interference
                # Real sonar sees interference between many sub-resolution scatterers
                # This creates Rayleigh/Rician distributed amplitude variations
                speckle = np.random.gamma(shape=1.2, scale=1.0/1.2)  # More speckle variation
                
                # ASPECT ANGLE VARIATION: Return strength varies with small angle changes
                # Model as random modulation representing micro-scale roughness/orientation
                aspect_variation = 0.5 + 0.8 * np.random.randn()  # Increased variation (~55% std dev)
                aspect_variation = np.clip(aspect_variation, 0.2, 2.0)  # Wider range
                
                # GEOMETRIC SHADOWING: Objects in front block energy to objects behind
                # Scatter is proportional to remaining energy budget
                scatter = energy * density * reflectivity * step_size * speckle * aspect_variation
                
                # Two-way propagation loss
                spreading_loss = 1.0 / (distance**2 + 1.0)
                water_absorption = np.exp(-0.05 * distance * 2 * 0.115)  # Two-way
                
                return_energy = scatter * spreading_loss * water_absorption
                
                # SPATIAL JITTER: Echo position uncertainty for freckled appearance
                # Real sonar has range uncertainty - echoes can appear in neighboring bins
                # This creates realistic "freckled" appearance on static objects like nets
                bin_idx = int((distance / self.range_m) * (len(output_bins) - 1))
                
                jitter_prob = 0.5  # 50% chance of spatial jitter
                if np.random.rand() < jitter_prob:
                    # Random Gaussian jitter - most small, occasionally large
                    # RANGE-DEPENDENT: jitter increases with distance (resolution degradation)
                    range_factor = 1.0 + (distance / self.range_m) * 3.0  # Increased from 2.0 to 3.0 - stronger degradation
                    jitter_offset = int(np.round(np.random.randn() * 1.5 * range_factor))  # Increased from 1.2 to 1.5
                    jitter_offset = np.clip(jitter_offset, -8, 8)  # Expanded from ±5 to ±8
                    bin_jitter = bin_idx + jitter_offset
                    bin_jitter = np.clip(bin_jitter, 0, len(output_bins) - 1)
                else:
                    bin_jitter = bin_idx
                
                # MULTI-BIN SPREADING: Sometimes deposit return across multiple adjacent bins
                # Simulates target extent, multipath, and processing artifacts
                spread_prob = 0.3  # 30% chance of multi-bin spreading
                if np.random.rand() < spread_prob:
                    # Spread across 2-4 adjacent bins
                    num_spread_bins = np.random.choice([2, 3, 4], p=[0.5, 0.35, 0.15])
                    spread_center = bin_jitter
                    
                    for offset in range(-(num_spread_bins//2), (num_spread_bins//2) + 1):
                        spread_bin = spread_center + offset
                        if 0 <= spread_bin < len(output_bins):
                            # Energy falls off from center of spread (Gaussian-ish)
                            spread_weight = np.exp(-0.5 * (offset / (num_spread_bins/3))**2)
                            range_quality = 1.0 / (1.0 + (distance / self.range_m) * 0.8)
                            output_bins[spread_bin] += return_energy * beam_strength * range_quality * spread_weight / num_spread_bins
                else:
                    # Single bin deposit
                    if 0 <= bin_jitter < len(output_bins):
                        range_quality = 1.0 / (1.0 + (distance / self.range_m) * 0.8)
                        output_bins[bin_jitter] += return_energy * beam_strength * range_quality
            
            # ABSORPTION: Reduce forward energy (creates shadows)
            if density > 0.01:
                # Both absorption and scattering reduce forward energy
                energy *= np.exp(-absorption * step_size * 2.0)  # Moderate absorption
                energy *= (1.0 - density * reflectivity * step_size * 3.0)  # Scattering loss
                energy = max(0.0, energy)
            
            # Move forward
            current_pos += direction * step_size
    
    def move(self, delta: np.ndarray):
        """Move sonar position."""
        self.position += delta
    
    def rotate(self, angle_deg: float):
        """Rotate sonar."""
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate in 2D
        new_x = self.direction[0] * cos_a - self.direction[1] * sin_a
        new_y = self.direction[0] * sin_a + self.direction[1] * cos_a
        self.direction = np.array([new_x, new_y])
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-9)


def create_demo_scene() -> VoxelGrid:
    """Create fish farm net cage with fish.
    
    World dimensions match realistic aquaculture cage:
    - 30m × 30m world (typical fish farm cage diameter ~20m)
    - 10cm voxel resolution for detailed net structure
    - Cage radius scaled to fit SOLAQUA operational distances (0.5-2m from net)
    """
    # Create 30m x 30m world at 10cm resolution (300×300 voxels)
    grid = VoxelGrid(300, 300, voxel_size=0.1)
    
    # Net cage parameters (centered at 15, 15)
    cage_center = np.array([15.0, 15.0])
    cage_radius = 12.0  # 12m radius (24m diameter) - typical fish farm cage
    num_sides = 24      # More sides for finer linear sections
    
    # Current effect: simulate southward current pulling on net
    # Current strength and direction
    current_direction = np.array([0.0, 1.0])  # Southward (positive Y)
    current_strength = 6.5  # Maximum displacement in meters (increased for more dramatic bending)
    
    # Create circular net panels with current deflection
    for i in range(num_sides):
        angle1 = (i / num_sides) * 2 * np.pi
        angle2 = ((i + 1) / num_sides) * 2 * np.pi
        
        # Panel corners (base positions)
        x1_base = cage_center[0] + cage_radius * np.cos(angle1)
        y1_base = cage_center[1] + cage_radius * np.sin(angle1)
        x2_base = cage_center[0] + cage_radius * np.cos(angle2)
        y2_base = cage_center[1] + cage_radius * np.sin(angle2)
        
        # Apply current deflection based on position
        # Panels on the south side are pushed more by current
        # Use a cosine-based deflection: maximum at south, zero at north
        deflection1 = current_strength * max(0, (y1_base - cage_center[1]) / cage_radius) * current_direction
        deflection2 = current_strength * max(0, (y2_base - cage_center[1]) / cage_radius) * current_direction
        
        # Add some lateral spreading based on angle from current direction
        lateral_factor1 = np.sin(angle1) * 0.4  # ±40% lateral deflection
        lateral_factor2 = np.sin(angle2) * 0.4
        deflection1 += np.array([lateral_factor1 * current_strength * 0.3, 0])
        deflection2 += np.array([lateral_factor2 * current_strength * 0.3, 0])
        
        # Final panel corners with deflection
        x1 = x1_base + deflection1[0]
        y1 = y1_base + deflection1[1]
        x2 = x2_base + deflection2[0]
        y2 = y2_base + deflection2[1]
        
        # Create net line for this panel - ensure we sample ALL segments
        # Use 100 samples instead of 80 to ensure no gaps
        for t in np.linspace(0, 1, 100):
            # Linear interpolation
            x_linear = x1 + t * (x2 - x1)
            y_linear = y1 + t * (y2 - y1)
            
            # Add catenary-like sag to each panel segment
            # Maximum sag at midpoint (t=0.5)
            sag = 0.25 * (1 - (2*t - 1)**2)  # Parabolic curve, max 0.25m at center (reduced for realism)
            
            # Direction perpendicular to panel (inward toward cage center)
            panel_dx = x2 - x1
            panel_dy = y2 - y1
            panel_length = np.sqrt(panel_dx**2 + panel_dy**2)
            if panel_length > 0:
                # Perpendicular vector pointing inward
                perp_x = -panel_dy / panel_length
                perp_y = panel_dx / panel_length
                # Determine if perpendicular points inward or outward
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                to_center_x = cage_center[0] - mid_x
                to_center_y = cage_center[1] - mid_y
                # If perpendicular is opposite to center direction, flip it
                if perp_x * to_center_x + perp_y * to_center_y < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y
                
                x = x_linear + sag * perp_x
                y = y_linear + sag * perp_y
            else:
                x = x_linear
                y = y_linear
            
            # Ensure coordinates are within grid bounds (with margin for box size)
            x = np.clip(x, 0.2, 29.8)
            y = np.clip(y, 0.2, 29.8)
            
            # Net mesh - always created
            grid.set_box(
                np.array([x - 0.08, y - 0.08]),
                np.array([x + 0.08, y + 0.08]),
                NET
            )
            
            # Add occasional rope structure - ensure this is also created
            if t % (1.0 / 7) < 0.025:  # Slightly increased threshold to ensure rope coverage
                grid.set_box(
                    np.array([x - 0.12, y - 0.12]),
                    np.array([x + 0.12, y + 0.12]),
                    ROPE
                )
    
    # Add random biomass patches (algae/fouling) to net
    # Simulate realistic net fouling in random locations
    np.random.seed(123)  # Different seed for biomass placement
    num_biomass_patches = 40  # Number of fouling patches
    
    for _ in range(num_biomass_patches):
        # Random position on cage perimeter
        angle = np.random.rand() * 2 * np.pi
        r = cage_radius + np.random.randn() * 0.5  # Slight radius variation
        
        x = cage_center[0] + r * np.cos(angle)
        y = cage_center[1] + r * np.sin(angle)
        
        # Biomass patch size: 20-60cm clusters
        patch_size = 0.2 + np.random.rand() * 0.4
        
        grid.set_circle(
            np.array([x, y]),
            patch_size,
            BIOMASS
        )
    
    np.random.seed(42)  # Reset seed for fish
    
    # Add fish scattered throughout cage - store as dynamic objects
    np.random.seed(42)  # Reproducible fish positions
    num_fish = 150  # Reduced from 500 for better performance and realism
    
    # Store fish data for animation
    fish_data = []
    for _ in range(num_fish):
        # Random position within cage (prefer near perimeter)
        angle = np.random.rand() * 2 * np.pi
        # Bias toward perimeter (0.6-0.95 of radius)
        r_fraction = 0.6 + 0.35 * np.random.rand()
        r = cage_radius * r_fraction
        
        x = cage_center[0] + r * np.cos(angle)
        y = cage_center[1] + r * np.sin(angle)
        
        # Random swimming direction and speed
        swim_angle = np.random.rand() * 2 * np.pi
        swim_speed = 0.08 + np.random.rand() * 0.12  # 8-20 cm/frame
        vx = swim_speed * np.cos(swim_angle)
        vy = swim_speed * np.sin(swim_angle)
        
        # Fish size: longer and slimmer
        fish_length = 0.4 + np.random.rand() * 0.1  # 40-60cm long
        fish_width = fish_length * 0.20  # Width is 20% of length (slimmer)
        
        # Assign species (A, B, C) evenly
        species = ['A', 'B', 'C'][_ % 3]
        
        fish_data.append({
            'pos': np.array([x, y]),
            'vel': np.array([vx, vy]),
            'orientation': swim_angle,
            'radii': np.array([fish_length, fish_width]),
            'turn_timer': np.random.rand() * 100,  # Random phase for turning
            'species': species
        })
    
    # Add floating debris - different sizes and materials
    num_debris = 0  
    debris_data = []
    debris_materials = [DEBRIS_LIGHT, DEBRIS_MEDIUM, DEBRIS_HEAVY]
    
    for i in range(num_debris):
        # Random position within cage
        angle = np.random.rand() * 2 * np.pi
        r = np.random.rand() * cage_radius * 0.9
        
        x = cage_center[0] + r * np.cos(angle)
        y = cage_center[1] + r * np.sin(angle)
        
        # Slow drift velocity
        drift_angle = np.random.rand() * 2 * np.pi
        drift_speed = 0.01 + np.random.rand() * 0.03  # 1-4 cm/frame
        vx = drift_speed * np.cos(drift_angle)
        vy = drift_speed * np.sin(drift_angle)
        
        # Random size (15-40cm) - increased for visibility
        size = 0.15 + np.random.rand() * 0.25
        
        # Random material type
        material = debris_materials[i % 3]
        
        debris_data.append({
            'pos': np.array([x, y]),
            'vel': np.array([vx, vy]),
            'size': size,
            'material': material
        })
    
    # Initial fish rendering
    for fish in fish_data:
        grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)
    
    # Initial debris rendering
    for debris in debris_data:
        grid.set_circle(debris['pos'], debris['size'], debris['material'])
    
    # Debug: Count how many fish voxels were created
    fish_voxel_count = np.sum(np.abs(grid.reflectivity - FISH.reflectivity) < 0.01)
    print(f"Fish voxels created: {fish_voxel_count}")
    
    # Add some larger objects (feed pipes, sensors, etc.)
    # Vertical feed pipe
    grid.set_box(
        np.array([cage_center[0] - 0.15, cage_center[1] - 0.15]),
        np.array([cage_center[0] + 0.15, cage_center[1] + 0.15]),
        ROPE
    )
    
    print(f"Created fish cage:")
    print(f"  Center: {cage_center}")
    print(f"  Radius: {cage_radius}m")
    print(f"  Fish: {len(fish_data)}")
    print(f"  Debris: {len(debris_data)}")
    
    return grid, fish_data, debris_data


def update_fish(grid: VoxelGrid, fish_data: list, cage_center: np.ndarray, cage_radius: float, sonar_pos: np.ndarray):
    """Update fish positions and redraw them in the grid."""
    # Clear existing fish
    grid.clear_fish()
    
    # Species behavior parameters
    # A: Schooling (strong same-species attraction)
    # B: Solitary (avoid all fish)
    # C: Mixed (moderate attraction to all)
    behavior = {
        'A': {'same_attract': 2.0, 'other_attract': 0.1, 'avoid': 0.8, 'sonar_avoid': 1.0},
        'B': {'same_attract': 0.0, 'other_attract': 0.0, 'avoid': 2.0, 'sonar_avoid': 2.5},
        'C': {'same_attract': 0.8, 'other_attract': 0.6, 'avoid': 1.0, 'sonar_avoid': 1.8}
    }
    
    # Update each fish
    for i, fish in enumerate(fish_data):
        # Update position
        fish['pos'] += fish['vel']
        
        # Flocking behavior (computed every frame for responsiveness)
        species = fish['species']
        avoid_vec = np.zeros(2)
        same_attract_vec = np.zeros(2)
        other_attract_vec = np.zeros(2)
        same_count = 0
        other_count = 0
        
        # Check nearby fish (within 3m for efficiency)
        for j, other in enumerate(fish_data):
            if i == j:
                continue
            
            diff = other['pos'] - fish['pos']
            dist = np.linalg.norm(diff)
            
            if dist < 3.0 and dist > 0.01:
                # Avoidance (short range)
                if dist < 0.8:
                    avoid_vec -= diff / (dist * dist + 0.1)
                
                # Attraction (medium range)
                if dist < 3.0:
                    if other['species'] == species:
                        same_attract_vec += diff / (dist + 0.1)
                        same_count += 1
                    else:
                        other_attract_vec += diff / (dist + 0.1)
                        other_count += 1
        
        # Apply behaviors based on species
        params = behavior[species]
        steer = np.zeros(2)
        
        if same_count > 0:
            steer += params['same_attract'] * same_attract_vec / same_count
        if other_count > 0:
            steer += params['other_attract'] * other_attract_vec / other_count
        steer += params['avoid'] * avoid_vec
        
        # SONAR AVOIDANCE: Fish flee from the robot/sonar
        diff_from_sonar = fish['pos'] - sonar_pos
        dist_from_sonar = np.linalg.norm(diff_from_sonar)
        
        if dist_from_sonar < 5.0 and dist_from_sonar > 0.01:  # Within 5m detection range
            # Flee away from sonar with inverse square law
            flee_strength = params['sonar_avoid'] / (dist_from_sonar**2 + 0.5)
            steer += flee_strength * diff_from_sonar / dist_from_sonar
        
        # Add small random component
        steer += (np.random.rand(2) - 0.5) * 0.3
        
        # Apply steering with momentum
        if np.linalg.norm(steer) > 0.01:
            fish['vel'] += steer * 0.03
            
            # Limit speed
            speed = np.linalg.norm(fish['vel'])
            max_speed = 0.2
            min_speed = 0.05
            if speed > max_speed:
                fish['vel'] = fish['vel'] / speed * max_speed
            elif speed < min_speed:
                fish['vel'] = fish['vel'] / speed * min_speed
            
            # Update orientation
            fish['orientation'] = np.arctan2(fish['vel'][1], fish['vel'][0])
        
        # Keep fish inside cage bounds
        dx = fish['pos'][0] - cage_center[0]
        dy = fish['pos'][1] - cage_center[1]
        dist_from_center = np.sqrt(dx*dx + dy*dy)
        
        if dist_from_center > cage_radius - 1.0:
            # Bounce back toward center
            angle_to_center = np.arctan2(-dy, -dx)
            swim_speed = np.linalg.norm(fish['vel'])
            fish['vel'][0] = swim_speed * np.cos(angle_to_center)
            fish['vel'][1] = swim_speed * np.sin(angle_to_center)
            fish['orientation'] = angle_to_center
        
        # Draw fish at new position
        grid.set_ellipse(fish['pos'], fish['radii'], fish['orientation'], FISH)


def update_debris(grid: VoxelGrid, debris_data: list, cage_center: np.ndarray, cage_radius: float):
    """Update debris positions and redraw them in the grid."""
    # Clear existing debris
    grid.clear_debris()
    
    # Update each debris piece
    for debris in debris_data:
        # Add random turbulence/drift
        turbulence = (np.random.rand(2) - 0.5) * 0.008  # Small random drift
        debris['vel'] += turbulence
        
        # Limit drift speed
        speed = np.linalg.norm(debris['vel'])
        max_speed = 0.05
        if speed > max_speed:
            debris['vel'] = debris['vel'] / speed * max_speed
        
        # Update position
        debris['pos'] += debris['vel']
        
        # Keep debris inside cage bounds (bounce off walls)
        dx = debris['pos'][0] - cage_center[0]
        dy = debris['pos'][1] - cage_center[1]
        dist_from_center = np.sqrt(dx*dx + dy*dy)
        
        if dist_from_center > cage_radius - 0.5:
            # Reflect velocity off wall
            normal = np.array([dx, dy]) / (dist_from_center + 0.001)
            debris['vel'] = debris['vel'] - 2 * np.dot(debris['vel'], normal) * normal
            # Also push back inside slightly
            debris['pos'] = cage_center + normal * (cage_radius - 0.5)
        
        # Draw debris at new position
        grid.set_circle(debris['pos'], debris['size'], debris['material'])


def main():
    """Run interactive voxel sonar viewer."""
    print("Building fish farm cage scene (this may take a moment)...")
    grid, fish_data, debris_data = create_demo_scene()
    
    # Cage parameters for fish updates (must match create_demo_scene)
    cage_center = np.array([15.0, 15.0])  # Center of 30m×30m world
    cage_radius = 12.0  # 12m radius cage
    
    print("Initializing sonar...")
    # Position sonar for typical net-following operation (1-2m from net)
    # This matches SOLAQUA operational distances: 0.5-2.0m from net
    sonar = VoxelSonar(
        position=np.array([15.0, 2.0]),  # 1m from southern net edge
        direction=np.array([0.0, 1.0]),  # Looking north toward cage
        range_m=20.0,     # SOLAQUA RANGE_MAX_M_DEFAULT
        fov_deg=120.0,    # SOLAQUA FOV_DEG_DEFAULT
        num_beams=256     # SOLAQUA dataset: 256 beams
    )
    
    # Create sonar figure
    fig_sonar, ax_sonar = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    plt.figure(fig_sonar.number)
    plt.subplots_adjust(bottom=0.15)
    
    # Create map figure
    fig_map, ax_map = plt.subplots(figsize=(8, 8))
    fig_map.canvas.manager.set_window_title('World Map')
    
    # Store cage parameters for map display
    num_sides = 12
    # Disable matplotlib default key bindings to avoid conflicts
    plt.rcParams['keymap.quit'] = []
    plt.rcParams['keymap.save'] = []
    plt.rcParams['keymap.fullscreen'] = []
    plt.rcParams['keymap.home'] = []
    plt.rcParams['keymap.back'] = []
    plt.rcParams['keymap.forward'] = []
    plt.rcParams['keymap.pan'] = []
    plt.rcParams['keymap.zoom'] = []
    plt.rcParams['keymap.grid'] = []
    
    
    # Initialize net tracker
    tracker = NetTracker(TRACKING_CONFIG)
    
    def update_display():
        """Update sonar and map displays."""
        # Update fish positions
        update_fish(grid, fish_data, cage_center, cage_radius, sonar.position)
        
        # Update debris positions
        update_debris(grid, debris_data, cage_center, cage_radius)
        
        # Extract current fish positions for map
        fish_positions = np.array([[f['pos'][0], f['pos'][1]] for f in fish_data])
        
        print(f"Scanning from position {sonar.position}...")
        image = sonar.scan(grid)
        
        # Verify dimensions match SOLAQUA dataset
        print(f"Sonar image shape: {image.shape} (should be 1024x256)")
        
        # Process image for net detection
        # Convert to 0-255 range
        image_db = 10 * np.log10(np.maximum(image, 1e-10))
        image_normalized = np.clip((image_db + 60) / 60, 0, 1)
        image_uint8 = (image_normalized * 255).astype(np.uint8)
        
        # Apply adaptive enhancement
        try:
            enhanced = adaptive_linear_momentum_merge_fast(
                image_uint8,
                angle_steps=IMAGE_PROCESSING_CONFIG.get('adaptive_angle_steps', 20),
                base_radius=IMAGE_PROCESSING_CONFIG.get('adaptive_base_radius', 3),
                max_elongation=IMAGE_PROCESSING_CONFIG.get('adaptive_max_elongation', 1.0),
                momentum_boost=IMAGE_PROCESSING_CONFIG.get('momentum_boost', 10.0),
                linearity_threshold=IMAGE_PROCESSING_CONFIG.get('adaptive_linearity_threshold', 0.75)
            )
        except Exception as e:
            print(f"Enhancement failed: {e}")
            enhanced = image_uint8
        
        # Binary threshold
        binary_threshold = IMAGE_PROCESSING_CONFIG.get('binary_threshold', 128)
        _, binary = cv2.threshold(enhanced, binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Find net contour
        best_contour = tracker.find_and_update(binary, image.shape)
        
        # Calculate distance
        distance_px, angle_deg = tracker.calculate_distance(image.shape[1], image.shape[0])
        
        if distance_px is not None:
            # Convert to meters (assuming SOLAQUA's 20m range)
            distance_m = (distance_px / image.shape[0]) * sonar.range_m
            print(f"Net detected at {distance_m:.2f}m, angle: {angle_deg:.1f}°")
        
        # Display polar plot WITHOUT GRID
        ax_sonar.clear()
        
        fov_rad = np.deg2rad(sonar.fov_deg)
        angles = np.linspace(-fov_rad/2, fov_rad/2, sonar.num_beams)
        ranges = np.linspace(0, sonar.range_m, image.shape[0])
        
        theta, r = np.meshgrid(angles, ranges)
        
        # Display processed image
        ax_sonar.contourf(theta, r, image_normalized, levels=20, cmap='gray')
        ax_sonar.set_theta_zero_location('N')
        ax_sonar.set_theta_direction(1)
        ax_sonar.set_thetamin(-sonar.fov_deg/2)
        ax_sonar.set_thetamax(sonar.fov_deg/2)
        ax_sonar.set_ylim(0, sonar.range_m)
        
        # Remove grid lines
        ax_sonar.grid(False)
        
        # Helper function to convert pixel coordinates to polar
        def pixel_to_polar(px_x, px_y, img_shape):
            """Convert pixel coordinates to polar (theta, r)"""
            # px_x is beam index (0 to num_beams-1)
            # px_y is range index (0 to range_bins-1)
            beam_frac = px_x / img_shape[1]  # 0 to 1
            range_frac = px_y / img_shape[0]  # 0 to 1
            
            theta = -fov_rad/2 + beam_frac * fov_rad
            r = range_frac * sonar.range_m
            return theta, r
        
        # Draw fitted ellipse (the actual ellipse fit to the contour)
        if best_contour is not None and len(best_contour) >= 5:
            try:
                (ex, ey), (ew, eh), eangle = cv2.fitEllipse(best_contour)
                
                # Generate ellipse points
                t = np.linspace(0, 2*np.pi, 100)
                cos_angle = np.cos(np.radians(eangle))
                sin_angle = np.sin(np.radians(eangle))
                
                ellipse_x = ex + (ew/2) * np.cos(t) * cos_angle - (eh/2) * np.sin(t) * sin_angle
                ellipse_y = ey + (ew/2) * np.cos(t) * sin_angle + (eh/2) * np.sin(t) * cos_angle
                
                # Convert to polar and plot
                ellipse_theta = []
                ellipse_r = []
                for ex_pt, ey_pt in zip(ellipse_x, ellipse_y):
                    theta_e, r_e = pixel_to_polar(ex_pt, ey_pt, image.shape)
                    ellipse_theta.append(theta_e)
                    ellipse_r.append(r_e)
                
                ax_sonar.plot(ellipse_theta, ellipse_r, 'm-', linewidth=1, alpha=0.8, label='Fitted Ellipse')
            except:
                pass
        
        # Draw AOI (Area of Interest) - the expanded elliptical search mask
        if tracker.center and tracker.size and tracker.angle is not None:
            expansion = TRACKING_CONFIG.get('ellipse_expansion_factor', 0.5)
            cx, cy = tracker.center
            w, h = tracker.size
            expanded_w = w * (1 + expansion)
            expanded_h = h * (1 + expansion)
            
            # Generate ellipse points with proper rotation
            t = np.linspace(0, 2*np.pi, 100)
            cos_angle = np.cos(np.radians(tracker.angle))
            sin_angle = np.sin(np.radians(tracker.angle))
            
            aoi_x = cx + (expanded_w/2) * np.cos(t) * cos_angle - (expanded_h/2) * np.sin(t) * sin_angle
            aoi_y = cy + (expanded_w/2) * np.cos(t) * sin_angle + (expanded_h/2) * np.sin(t) * cos_angle
            
            # Convert to polar and plot
            aoi_theta = []
            aoi_r = []
            for ax, ay in zip(aoi_x, aoi_y):
                theta_a, r_a = pixel_to_polar(ax, ay, image.shape)
                aoi_theta.append(theta_a)
                aoi_r.append(r_a)
            
            ax_sonar.plot(aoi_theta, aoi_r, 'c--', linewidth=1.5, alpha=0.6, label='AOI')
        
        # Draw all contours (in green for visibility)
        if best_contour is not None:
            # Draw the best contour
            contour_pts = best_contour.squeeze()
            if len(contour_pts.shape) == 2 and contour_pts.shape[0] > 1:
                cont_theta = []
                cont_r = []
                for pt in contour_pts:
                    theta_c, r_c = pixel_to_polar(pt[0], pt[1], image.shape)
                    cont_theta.append(theta_c)
                    cont_r.append(r_c)
                # Close the contour
                cont_theta.append(cont_theta[0])
                cont_r.append(cont_r[0])
                ax_sonar.plot(cont_theta, cont_r, 'g-', linewidth=1.5, alpha=0.8, label='Contour')
        
        # Draw detection overlay - show the RED LINE (perpendicular to major axis)
        if best_contour is not None and distance_px is not None and tracker.center and tracker.size and tracker.angle is not None:
            # Get ellipse parameters from tracker
            cx, cy = tracker.center
            w, h = tracker.size
            ellipse_angle = tracker.angle  # Major axis angle in degrees
            
            # The red line is perpendicular to major axis (angle_deg from calculate_distance)
            red_line_angle_deg = angle_deg  # This is already perpendicular
            
            # Calculate line endpoints in Cartesian space
            half_len = max(w, h) / 2
            ang_r = np.radians(red_line_angle_deg)
            
            # Endpoints in pixel coordinates
            p1x = cx + half_len * np.cos(ang_r)
            p1y = cy + half_len * np.sin(ang_r)
            p2x = cx - half_len * np.cos(ang_r)
            p2y = cy - half_len * np.sin(ang_r)
            
            # Convert line endpoints
            theta1, r1 = pixel_to_polar(p1x, p1y, image.shape)
            theta2, r2 = pixel_to_polar(p2x, p2y, image.shape)
            
            # Draw red line in polar coordinates
            ax_sonar.plot([theta1, theta2], [r1, r2], 'r-', linewidth=2.5, label='Net Line')
            
            # Also draw the distance point
            theta_center, r_center = pixel_to_polar(cx, distance_px, image.shape)
            ax_sonar.plot(theta_center, r_center, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        title = f'Sonar - Pos: [{sonar.position[0]:.1f}, {sonar.position[1]:.1f}]'
        if distance_px is not None:
            title += f' | Net: {distance_m:.2f}m'
        ax_sonar.set_title(title)
        if tracker.center or best_contour is not None:
            ax_sonar.legend(loc='upper right', fontsize=8)
        
        # Update map display
        ax_map.clear()
        
        # Draw bent cage outline with current deflection
        cage_x = []
        cage_y = []
        
        # Current parameters (must match create_demo_scene)
        current_strength = 6.5
        current_direction = np.array([0.0, 1.0])
        
        # Draw each panel segment with curvature
        for i in range(num_sides):
            # Get corner points
            angle1 = (i / num_sides) * 2 * np.pi
            angle2 = ((i + 1) / num_sides) * 2 * np.pi
            
            x1_base = cage_center[0] + cage_radius * np.cos(angle1)
            y1_base = cage_center[1] + cage_radius * np.sin(angle1)
            x2_base = cage_center[0] + cage_radius * np.cos(angle2)
            y2_base = cage_center[1] + cage_radius * np.sin(angle2)
            
            # Apply current deflection to corners
            deflection1 = current_strength * max(0, (y1_base - cage_center[1]) / cage_radius) * current_direction
            lateral_factor1 = np.sin(angle1) * 0.4
            deflection1 += np.array([lateral_factor1 * current_strength * 0.3, 0])
            
            deflection2 = current_strength * max(0, (y2_base - cage_center[1]) / cage_radius) * current_direction
            lateral_factor2 = np.sin(angle2) * 0.4
            deflection2 += np.array([lateral_factor2 * current_strength * 0.3, 0])
            
            x1 = x1_base + deflection1[0]
            y1 = y1_base + deflection1[1]
            x2 = x2_base + deflection2[0]
            y2 = y2_base + deflection2[1]
            
            # Draw curved segment
            segment_x = []
            segment_y = []
            for t in np.linspace(0, 1, 20):
                x_linear = x1 + t * (x2 - x1)
                y_linear = y1 + t * (y2 - y1)
                
                # Add catenary-like sag
                sag = 0.25 * (1 - (2*t - 1)**2)  # Matches reduced sag in scene creation
                
                # Perpendicular vector pointing inward
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
                    
                    segment_x.append(x_linear + sag * perp_x)
                    segment_y.append(y_linear + sag * perp_y)
                else:
                    segment_x.append(x_linear)
                    segment_y.append(y_linear)
            
            cage_x.extend(segment_x)
            cage_y.extend(segment_y)
        
        ax_map.plot(cage_x, cage_y, 'b-', linewidth=2, label='Bent Cage')
        
        # Draw fish
        ax_map.scatter(fish_positions[:, 0], fish_positions[:, 1], c='orange', s=3, alpha=0.6, label='Fish')
        
        # Draw sonar position and direction
        ax_map.scatter(sonar.position[0], sonar.position[1], c='red', s=100, marker='^', label='Sonar', zorder=5)
        
        # Draw sonar direction vector
        arrow_length = 3.0
        ax_map.arrow(sonar.position[0], sonar.position[1], 
                     sonar.direction[0] * arrow_length, sonar.direction[1] * arrow_length,
                     head_width=0.8, head_length=0.5, fc='red', ec='red', zorder=5)
        
        # Draw FOV cone
        fov_rad = np.deg2rad(sonar.fov_deg)
        # Rotate direction by +/- fov/2
        dir_angle = np.arctan2(sonar.direction[1], sonar.direction[0])
        left_angle = dir_angle + fov_rad / 2
        right_angle = dir_angle - fov_rad / 2
        
        cone_length = sonar.range_m * 0.5
        left_x = sonar.position[0] + cone_length * np.cos(left_angle)
        left_y = sonar.position[1] + cone_length * np.sin(left_angle)
        right_x = sonar.position[0] + cone_length * np.cos(right_angle)
        right_y = sonar.position[1] + cone_length * np.sin(right_angle)
        
        ax_map.plot([sonar.position[0], left_x], [sonar.position[1], left_y], 'r--', alpha=0.3, linewidth=1)
        ax_map.plot([sonar.position[0], right_x], [sonar.position[1], right_y], 'r--', alpha=0.3, linewidth=1)
        
        ax_map.set_xlim(0, 30)
        ax_map.set_ylim(0, 30)
        ax_map.set_aspect('equal')
        ax_map.set_xlabel('X (m)')
        ax_map.set_ylabel('Y (m)')
        ax_map.set_title('World Map (Top View)')
        ax_map.grid(True, alpha=0.3)
        ax_map.legend(loc='upper right')
        
        fig_sonar.canvas.draw()
        fig_map.canvas.draw()
    
    def on_key(event):
        """Handle keyboard input."""
        move_speed = 0.5
        rotate_speed = 10
        
        if event.key == 'w':
            sonar.move(sonar.direction * move_speed)
        elif event.key == 's':
            sonar.move(-sonar.direction * move_speed)
        elif event.key == 'a':
            # Move perpendicular (left)
            perp = np.array([-sonar.direction[1], sonar.direction[0]])
            sonar.move(-perp * move_speed)
        elif event.key == 'd':
            # Move perpendicular (right)
            perp = np.array([-sonar.direction[1], sonar.direction[0]])
            sonar.move(perp * move_speed)
        elif event.key == 'left':
            sonar.rotate(rotate_speed)
        elif event.key == 'right':
            sonar.rotate(-rotate_speed)
        else:
            return
        
        update_display()
    
    fig_sonar.canvas.mpl_connect('key_press_event', on_key)
    fig_map.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display()
    
    # Start continuous animation to show flickering
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig_sonar, lambda frame: update_display() or [], interval=100, cache_frame_data=False)
    
    print("\nControls:")
    print("  W/S: Move forward/back")
    print("  A/D: Move left/right")
    print("  Left/Right arrows: Rotate")
    print("\nNote: Image flickers continuously due to:")
    print("  - Acoustic speckle (coherent interference)")
    print("  - Aspect angle variations (micro-scale roughness)")
    print("  - Temporal decorrelation (net sway, water movement)")
    print("  - Receiver noise (electronic/thermal)")
    
    plt.show()


if __name__ == '__main__':
    main()
