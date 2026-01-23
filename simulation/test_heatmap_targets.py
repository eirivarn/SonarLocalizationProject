"""Test heatmap and direction target generation."""
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_sample, generate_heatmap_target, generate_direction_target
import data_generator

# Clear cache
data_generator._trajectory_state = {}

# Generate one sample
print("Generating sample...")
sample_data = generate_sample(scene_id=100, sample_id=0)

if sample_data['net_visible']:
    print(f"Net visible: Yes")
    print(f"Position p: {sample_data['p']}")
    print(f"Direction t: {sample_data['t']}")
    print(f"Distance: {sample_data['distance_m']:.2f}m")
    print(f"Orientation: {sample_data['orientation_deg']:.1f}°")
    
    py, pz = sample_data['p']
    ty, tz = sample_data['t']
    
    # Generate targets
    heatmap = generate_heatmap_target(py, pz)
    direction_map, direction_mask = generate_direction_target(py, pz, ty, tz)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    ax = axes[0, 0]
    ax.imshow(sample_data['image'], cmap='gray', origin='lower')
    ax.set_title('Sonar Image')
    ax.axis('off')
    
    # Heatmap
    ax = axes[0, 1]
    ax.imshow(heatmap, cmap='hot', origin='lower')
    ax.set_title('Heatmap Target')
    ax.axis('off')
    
    # Heatmap overlay
    ax = axes[0, 2]
    ax.imshow(sample_data['image'], cmap='gray', origin='lower', alpha=0.7)
    ax.imshow(heatmap, cmap='hot', origin='lower', alpha=0.5)
    ax.set_title('Heatmap Overlay')
    ax.axis('off')
    
    # Direction Y component
    ax = axes[1, 0]
    ax.imshow(direction_map[0] * direction_mask, cmap='RdBu', origin='lower', vmin=-1, vmax=1)
    ax.set_title('Direction Y (masked)')
    ax.axis('off')
    
    # Direction Z component
    ax = axes[1, 1]
    ax.imshow(direction_map[1] * direction_mask, cmap='RdBu', origin='lower', vmin=-1, vmax=1)
    ax.set_title('Direction Z (masked)')
    ax.axis('off')
    
    # Direction mask
    ax = axes[1, 2]
    ax.imshow(direction_mask, cmap='gray', origin='lower')
    ax.set_title('Direction Supervision Mask')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_heatmap_targets.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to test_heatmap_targets.png")
    plt.close()
    
else:
    print("Net not visible in this sample. Try another scene_id.")
