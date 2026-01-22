"""Generate visual samples to test the trajectory-based data generation."""
import numpy as np
from data_generator import generate_sample, visualize_sample
import os

def main():
    """Generate and visualize samples from a trajectory."""
    # Clear any cached state
    import data_generator
    data_generator._trajectory_state = {}
    
    scene_id = 101
    num_samples = 20
    
    output_dir = "trajectory_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_samples} samples in scene {scene_id}...")
    
    inside_count = 0
    visible_count = 0
    distances = []
    
    for i in range(num_samples):
        img, label = generate_sample(scene_id, i)
        
        # Check if inside cage
        pos = np.array(label['sonar_position'])
        cage_center = np.array(label['cage_center'])
        cage_radius = label['cage_radius']
        dist_from_center = np.linalg.norm(pos - cage_center)
        is_inside = dist_from_center < cage_radius
        
        if is_inside:
            inside_count += 1
        
        if label['net_visible']:
            visible_count += 1
            distances.append(label['distance_m'])
        
        output_path = f"{output_dir}/sample_{i:03d}.png"
        visualize_sample(img, label, output_path)
        
        status = "✓" if label['net_visible'] else "✗"
        location = "IN" if is_inside else "OUT"
        if label['net_visible']:
            print(f"  Sample {i}: {status} [{location:3s}] Net @ {label['distance_m']:.2f}m, {label['orientation_deg']:+.1f}°")
        else:
            print(f"  Sample {i}: {status} [{location:3s}] No net visible")
    
    print(f"\nSamples saved to {output_dir}/")
    print(f"\nStatistics:")
    print(f"  Inside cage: {inside_count}/{num_samples} ({100*inside_count/num_samples:.0f}%)")
    print(f"  Net visible: {visible_count}/{num_samples} ({100*visible_count/num_samples:.0f}%)")
    if distances:
        in_range = sum(1 <= d <= 5 for d in distances)
        print(f"  Distance range: {min(distances):.2f}-{max(distances):.2f}m (mean: {np.mean(distances):.2f}m)")
        print(f"  In target 1-5m: {in_range}/{len(distances)} ({100*in_range/len(distances):.1f}%)")

if __name__ == '__main__':
    main()
