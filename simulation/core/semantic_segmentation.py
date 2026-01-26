"""
Semantic segmentation visualization for sonar images.

Extends the simulator to track which material each sonar return came from,
then visualizes it with color-coded labels.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .simulation import VoxelGrid, VoxelSonar, Material, EMPTY, NET, ROPE, FISH, BIOMASS, DEBRIS_LIGHT, DEBRIS_MEDIUM, DEBRIS_HEAVY
from .config import SONAR_CONFIG, SCENE_CONFIG, DATA_GEN_CONFIG
from .data_generator import create_random_scene, generate_random_sonar_position, polar_to_cartesian, _trajectory_state


# Material ID mapping
MATERIAL_IDS = {
    'empty': 0,
    'net': 1,
    'rope': 2,
    'fish': 3,
    'biomass': 4,
    'debris_light': 5,
    'debris_medium': 6,
    'debris_heavy': 7,
    'wall': 8,
}

# Color map for visualization (RGB)
MATERIAL_COLORS = {
    0: (0.0, 0.0, 0.0),       # empty - black
    1: (1.0, 0.0, 0.0),       # net - red
    2: (1.0, 0.5, 0.0),       # rope - orange
    3: (0.0, 1.0, 0.0),       # fish - green
    4: (0.0, 1.0, 1.0),       # biomass - cyan
    5: (0.5, 0.5, 0.5),       # debris_light - light gray
    6: (0.7, 0.7, 0.7),       # debris_medium - medium gray
    7: (0.9, 0.9, 0.9),       # debris_heavy - white-ish
    8: (0.5, 0.0, 0.5),       # wall - purple
}


class VoxelGridWithMaterials(VoxelGrid):
    """Extended voxel grid that also tracks material IDs."""
    
    def __init__(self, size_x: int, size_y: int, voxel_size: float = 0.1):
        super().__init__(size_x, size_y, voxel_size)
        
        # Add material ID tracking
        self.material_id = np.zeros((size_x, size_y), dtype=np.uint8)
    
    def set_circle(self, center: np.ndarray, radius: float, material: Material):
        """Fill circle with material (override to track material ID)."""
        super().set_circle(center, radius, material)
        
        # Also set material ID
        material_id = MATERIAL_IDS.get(material.name, 0)
        cx, cy = self.world_to_voxel(center)
        r_voxels = int(radius / self.voxel_size)
        
        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                if dx*dx + dy*dy <= r_voxels*r_voxels:
                    x, y = cx + dx, cy + dy
                    if self.is_inside(x, y):
                        self.material_id[x, y] = material_id
    
    def set_ellipse(self, center: np.ndarray, radii: np.ndarray, orientation: float, material: Material):
        """Fill ellipse with material (override to track material ID)."""
        super().set_ellipse(center, radii, orientation, material)
        
        # Also set material ID
        material_id = MATERIAL_IDS.get(material.name, 0)
        cx, cy = self.world_to_voxel(center)
        rx_vox = max(1, int(radii[0] / self.voxel_size))
        ry_vox = max(1, int(radii[1] / self.voxel_size))
        
        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)
        max_r = max(rx_vox, ry_vox)
        
        for dx in range(-max_r, max_r + 1):
            for dy in range(-max_r, max_r + 1):
                dx_rot = dx * cos_a + dy * sin_a
                dy_rot = -dx * sin_a + dy * cos_a
                dist_sq = (dx_rot/rx_vox)**2 + (dy_rot/ry_vox)**2
                
                if dist_sq <= 1.0:
                    x, y = cx + dx, cy + dy
                    if self.is_inside(x, y):
                        self.material_id[x, y] = material_id
    
    def set_box(self, min_pos: np.ndarray, max_pos: np.ndarray, material: Material):
        """Fill box region with material (override to track material ID)."""
        super().set_box(min_pos, max_pos, material)
        
        # Also set material ID
        material_id = MATERIAL_IDS.get(material.name, 0)
        x1, y1 = self.world_to_voxel(min_pos)
        x2, y2 = self.world_to_voxel(max_pos)
        
        x1, x2 = max(0, min(x1, x2)), min(self.size_x, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(self.size_y, max(y1, y2))
        
        self.material_id[x1:x2, y1:y2] = material_id
    
    def clear_fish(self):
        """Clear all fish material from grid."""
        super().clear_fish()
        mask = self.material_id == MATERIAL_IDS['fish']
        self.material_id[mask] = MATERIAL_IDS['empty']
    
    def clear_debris(self):
        """Clear all debris material from grid."""
        super().clear_debris()
        mask = (self.material_id == MATERIAL_IDS['debris_light']) | \
               (self.material_id == MATERIAL_IDS['debris_medium']) | \
               (self.material_id == MATERIAL_IDS['debris_heavy'])
        self.material_id[mask] = MATERIAL_IDS['empty']


class VoxelSonarWithSegmentation(VoxelSonar):
    """Sonar that also tracks material ID for each return."""
    
    def scan_with_segmentation(self, grid: VoxelGridWithMaterials):
        """
        Scan scene and return both intensity and semantic segmentation.
        
        Returns:
            intensity: (range_bins, num_beams) array of sonar returns
            segmentation: (range_bins, num_beams) array of material IDs
        """
        intensity = np.zeros((self.range_bins, self.num_beams), dtype=np.float32)
        segmentation = np.zeros((self.range_bins, self.num_beams), dtype=np.uint8)
        
        fov_rad = np.deg2rad(self.fov_deg)
        
        for beam_idx in range(self.num_beams):
            # Beam direction
            t = beam_idx / (self.num_beams - 1) if self.num_beams > 1 else 0.5
            angle = (-fov_rad / 2) + t * fov_rad
            
            # Beam pattern
            beam_pattern = np.exp(-((angle / (fov_rad/2))**2) * 2.5)
            
            # Rotate direction by angle
            dir_angle = np.arctan2(self.direction[1], self.direction[0])
            beam_angle = dir_angle + angle
            beam_dir = np.array([np.cos(beam_angle), np.sin(beam_angle)])
            
            # Ray march through volume WITH material tracking
            self._march_ray_with_segmentation(
                grid, self.position, beam_dir,
                intensity[:, beam_idx], segmentation[:, beam_idx], beam_pattern
            )
        
        # Apply temporal decorrelation to intensity only
        signal_mask = intensity > 1e-8
        decorrelation_noise = np.random.gamma(shape=5.0, scale=1.0/5.0, size=intensity.shape)
        intensity[signal_mask] *= decorrelation_noise[signal_mask]
        
        return intensity, segmentation
    
    def _march_ray_with_segmentation(self, grid: VoxelGridWithMaterials, origin: np.ndarray, 
                                     direction: np.ndarray, output_bins: np.ndarray,
                                     output_segmentation: np.ndarray, beam_strength: float = 1.0):
        """March ray through voxel grid, accumulating returns AND tracking materials."""
        step_size = grid.voxel_size * 0.5
        num_steps = int(self.range_m / step_size)
        
        current_pos = origin.copy()
        energy = 1.0
        
        # Track strongest material contribution per bin
        bin_contributions = np.zeros((len(output_bins), 9), dtype=np.float32)  # 9 material types
        
        for step in range(num_steps):
            distance = step * step_size
            if distance >= self.range_m or energy < 0.01:
                break
            
            # Get voxel properties
            x, y = grid.world_to_voxel(current_pos)
            
            if not grid.is_inside(x, y):
                current_pos += direction * step_size
                continue
            
            density = grid.density[x, y]
            reflectivity = grid.reflectivity[x, y]
            absorption = grid.absorption[x, y]
            material_id = grid.material_id[x, y]
            
            # Accumulate return (same physics as original)
            if density > 0.01:
                speckle = np.random.gamma(shape=1.2, scale=1.0/1.2)
                aspect_variation = 0.5 + 0.8 * np.random.randn()
                aspect_variation = np.clip(aspect_variation, 0.2, 2.0)
                
                scatter = energy * density * reflectivity * step_size * speckle * aspect_variation
                spreading_loss = 1.0 / (distance**2 + 1.0)
                water_absorption = np.exp(-0.05 * distance * 2 * 0.115)
                
                return_energy = scatter * spreading_loss * water_absorption
                
                bin_idx = int((distance / self.range_m) * (len(output_bins) - 1))
                
                # Apply jitter
                jitter_prob = 0.5
                if np.random.rand() < jitter_prob:
                    range_factor = 1.0 + (distance / self.range_m) * 3.0
                    jitter_offset = int(np.round(np.random.randn() * 1.5 * range_factor))
                    jitter_offset = np.clip(jitter_offset, -8, 8)
                    bin_jitter = bin_idx + jitter_offset
                    bin_jitter = np.clip(bin_jitter, 0, len(output_bins) - 1)
                else:
                    bin_jitter = bin_idx
                
                # Add to output
                range_quality = 1.0 / (1.0 + (distance / self.range_m) * 0.8)
                contribution = return_energy * beam_strength * range_quality
                
                output_bins[bin_jitter] += contribution
                bin_contributions[bin_jitter, material_id] += contribution
                
                # Absorption
                energy *= np.exp(-absorption * step_size * 2.5)
            
            current_pos += direction * step_size
        
        # Assign material ID based on strongest contributor per bin
        for bin_idx in range(len(output_bins)):
            if output_bins[bin_idx] > 1e-8:
                strongest_material = bin_contributions[bin_idx].argmax()
                output_segmentation[bin_idx] = strongest_material


def create_semantic_visualization(intensity, segmentation, output_path):
    """
    Create a visualization showing intensity and semantic segmentation side-by-side.
    
    Args:
        intensity: (H, W) sonar intensity image
        segmentation: (H, W) material ID map
        output_path: Where to save the image
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Set dark background
    for ax in axes:
        ax.set_facecolor('#2a2a2a')
    fig.patch.set_facecolor('#2a2a2a')
    
    # Calculate extent
    range_m = SONAR_CONFIG['range_m']
    fov_deg = SONAR_CONFIG['fov_deg']
    z_max = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
    extent = [-z_max, z_max, 0, range_m]
    
    # 1. Intensity image (auto-scaled)
    vmin = np.percentile(intensity, 1)
    vmax = np.percentile(intensity, 99.5)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    
    axes[0].imshow(intensity, cmap='hot', vmin=vmin, vmax=vmax,
                   extent=extent, origin='lower', aspect='auto')
    axes[0].set_title('Sonar Intensity', fontsize=14, color='white')
    axes[0].set_xlabel('Z (m)', color='white')
    axes[0].set_ylabel('Y (m)', color='white')
    axes[0].grid(True, alpha=0.2, color='white')
    axes[0].tick_params(colors='white')
    
    # 2. Semantic segmentation (colored by material)
    # Create RGB image from segmentation
    h, w = segmentation.shape
    rgb_seg = np.zeros((h, w, 3), dtype=np.float32)
    
    for material_id, color in MATERIAL_COLORS.items():
        mask = segmentation == material_id
        rgb_seg[mask] = color
    
    axes[1].imshow(rgb_seg, extent=extent, origin='lower', aspect='auto')
    axes[1].set_title('Semantic Segmentation', fontsize=14, color='white')
    axes[1].set_xlabel('Z (m)', color='white')
    axes[1].set_ylabel('Y (m)', color='white')
    axes[1].grid(True, alpha=0.2, color='white')
    axes[1].tick_params(colors='white')
    
    # 3. Overlay (intensity with colored segmentation)
    # Normalize intensity to [0, 1] for blending
    intensity_norm = (intensity - vmin) / (vmax - vmin)
    intensity_norm = np.clip(intensity_norm, 0, 1)
    
    # Create grayscale RGB from intensity
    intensity_rgb = np.stack([intensity_norm] * 3, axis=-1)
    
    # Blend: 70% intensity, 30% segmentation where there's signal
    signal_mask = intensity > vmin + 0.1 * (vmax - vmin)
    overlay = intensity_rgb.copy()
    overlay[signal_mask] = 0.7 * intensity_rgb[signal_mask] + 0.3 * rgb_seg[signal_mask]
    
    axes[2].imshow(overlay, extent=extent, origin='lower', aspect='auto')
    axes[2].set_title('Overlay (Intensity + Segmentation)', fontsize=14, color='white')
    axes[2].set_xlabel('Z (m)', color='white')
    axes[2].set_ylabel('Y (m)', color='white')
    axes[2].grid(True, alpha=0.2, color='white')
    axes[2].tick_params(colors='white')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    for material_id in sorted(MATERIAL_COLORS.keys()):
        if np.any(segmentation == material_id):
            # Find material name
            material_name = [k for k, v in MATERIAL_IDS.items() if v == material_id][0]
            color = MATERIAL_COLORS[material_id]
            legend_elements.append(Patch(facecolor=color, edgecolor='white', label=material_name))
    
    if legend_elements:
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#2a2a2a')
    plt.close()
    
    print(f"Saved semantic visualization to: {output_path}")


def test_semantic_segmentation():
    """Test the semantic segmentation system."""
    print("=" * 80)
    print("SEMANTIC SEGMENTATION TEST")
    print("=" * 80)
    
    output_dir = Path("test_outputs/semantic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = 5
    
    for i in range(num_samples):
        print(f"\n--- Sample {i} ---")
        
        # Create scene with material tracking
        # NOTE: We need to modify create_random_scene to use VoxelGridWithMaterials
        # For now, manually create a simple scene
        from simulation import FISH, NET
        
        grid = VoxelGridWithMaterials(400, 400, voxel_size=0.1)
        
        # Add a net (circular cage)
        cage_center = np.array([20.0, 20.0])
        cage_radius = 8.0
        
        # Create net segments (circular approximation)
        num_segments = 32
        for seg_idx in range(num_segments):
            angle1 = 2 * np.pi * seg_idx / num_segments
            angle2 = 2 * np.pi * (seg_idx + 1) / num_segments
            
            p1 = cage_center + cage_radius * np.array([np.cos(angle1), np.sin(angle1)])
            p2 = cage_center + cage_radius * np.array([np.cos(angle2), np.sin(angle2)])
            
            # Draw line segment as circles
            num_points = 10
            for t in np.linspace(0, 1, num_points):
                pos = p1 + t * (p2 - p1)
                grid.set_circle(pos, 0.15, NET)
        
        # Add some fish
        for _ in range(3):
            fish_pos = cage_center + np.random.randn(2) * 3.0
            grid.set_circle(fish_pos, 0.3, FISH)
        
        # Position sonar looking at the net
        sonar_distance = 3.0 + np.random.rand() * 2.0
        sonar_angle = np.random.rand() * 2 * np.pi
        sonar_pos = cage_center + sonar_distance * np.array([np.cos(sonar_angle), np.sin(sonar_angle)])
        sonar_dir = cage_center - sonar_pos
        sonar_dir = sonar_dir / np.linalg.norm(sonar_dir)
        
        # Scan with segmentation
        sonar = VoxelSonarWithSegmentation(
            position=sonar_pos,
            direction=sonar_dir,
            range_m=SONAR_CONFIG['range_m'],
            fov_deg=SONAR_CONFIG['fov_deg'],
            num_beams=SONAR_CONFIG['num_beams']
        )
        
        polar_intensity, polar_segmentation = sonar.scan_with_segmentation(grid)
        
        # Convert to Cartesian
        cart_intensity = polar_to_cartesian(polar_intensity)
        cart_segmentation = polar_to_cartesian(polar_segmentation)
        
        # Visualize
        output_path = output_dir / f"semantic_sample_{i:03d}.png"
        create_semantic_visualization(cart_intensity[0], cart_segmentation[0], output_path)
        
        # Print material statistics
        unique_materials, counts = np.unique(cart_segmentation[0], return_counts=True)
        print(f"  Materials present:")
        for mat_id, count in zip(unique_materials, counts):
            mat_name = [k for k, v in MATERIAL_IDS.items() if v == mat_id][0]
            percentage = 100 * count / cart_segmentation[0].size
            print(f"    {mat_name}: {percentage:.1f}%")
    
    print(f"\n✓ Generated {num_samples} semantic segmentation samples")
    print(f"✓ Saved to: {output_dir}")


if __name__ == '__main__':
    test_semantic_segmentation()
