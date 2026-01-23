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
    
    visible_count = 0
    distances = []
    
    for i in range(num_samples):
        sample_data = generate_sample(scene_id, i)
        
        if sample_data['net_visible']:
            visible_count += 1
            distances.append(sample_data['distance_m'])
        
        output_path = f"{output_dir}/sample_{i:03d}.png"
        visualize_sample(sample_data, output_path)
        
        status = "✓" if sample_data['net_visible'] else "✗"
        if sample_data['net_visible']:
            py, pz = sample_data['p']
            print(f"  Sample {i}: {status} Net @ {sample_data['distance_m']:.2f}m, {sample_data['orientation_deg']:+.1f}° | p=({py:.2f}, {pz:.2f})")
        else:
            print(f"  Sample {i}: {status} No net visible")
    
    print(f"\nSamples saved to {output_dir}/")
    print(f"\nStatistics:")
    print(f"  Net visible: {visible_count}/{num_samples} ({100*visible_count/num_samples:.0f}%)")
    if distances:
        in_range = sum(1 <= d <= 5 for d in distances)
        print(f"  Distance range: {min(distances):.2f}-{max(distances):.2f}m (mean: {np.mean(distances):.2f}m)")
        print(f"  In target 1-5m: {in_range}/{len(distances)} ({100*in_range/len(distances):.1f}%)")

if __name__ == '__main__':
    main()
