"""Test random walk trajectory visualization."""
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_sample

def test_trajectory():
    """Generate a sequence of samples and visualize the trajectory."""
    # Clear any cached state
    import data_generator
    data_generator._trajectory_state = {}
    
    scene_id = 99  # Use different scene to avoid cache issues
    num_samples = 50
    
    positions = []
    distances = []
    orientations = []
    
    print(f"Generating {num_samples} samples in scene {scene_id}...")
    
    for i in range(num_samples):
        img, label = generate_sample(scene_id, i)
        
        if label['net_visible']:
            positions.append(label['sonar_position'])
            distances.append(label['distance_m'])
            orientations.append(label['orientation_deg'])
    
    positions = np.array(positions)
    cage_center = np.array(label['cage_center'])
    cage_radius = label['cage_radius']
    
    # Plot trajectory
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Trajectory in world coordinates
    ax = axes[0]
    
    # Draw cage
    circle = plt.Circle(cage_center, cage_radius, fill=False, 
                       color='blue', linestyle='--', linewidth=2, label='Cage')
    ax.add_patch(circle)
    
    # Draw target zone (1-5m from cage)
    inner_circle = plt.Circle(cage_center, cage_radius - 5, fill=False,
                             color='green', linestyle=':', linewidth=1, alpha=0.5, label='1-5m zone')
    outer_circle = plt.Circle(cage_center, cage_radius - 1, fill=False,
                             color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'r-', alpha=0.5, linewidth=1)
    ax.scatter(positions[:, 0], positions[:, 1], c=range(len(positions)), 
              cmap='plasma', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Mark start and end
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=15, 
           markeredgewidth=2, markeredgecolor='white', label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=15,
           markeredgewidth=2, markeredgecolor='white', label='End')
    
    # Cage center
    ax.plot(cage_center[0], cage_center[1], 'b+', markersize=20, 
           markeredgewidth=3, label='Cage center')
    
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'ROV Trajectory (Random Walk) - Scene {scene_id}')
    
    # Right plot: Distance and orientation over time
    ax = axes[1]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(range(len(distances)), distances, 'b-', linewidth=2, 
                    label='Distance to net')
    ax.axhline(1, color='green', linestyle=':', alpha=0.5, label='Target range')
    ax.axhline(5, color='green', linestyle=':', alpha=0.5)
    ax.fill_between(range(len(distances)), 1, 5, color='green', alpha=0.1)
    
    line2 = ax2.plot(range(len(orientations)), orientations, 'r-', linewidth=2,
                     label='Net orientation')
    
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Distance to net (m)', color='b')
    ax2.set_ylabel('Relative orientation (°)', color='r')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    
    ax.set_title('Distance and Orientation over Time')
    
    plt.tight_layout()
    plt.savefig('trajectory_test.png', dpi=150)
    print("\nVisualization saved to trajectory_test.png")
    
    # Statistics
    print(f"\n{'='*60}")
    print("TRAJECTORY STATISTICS")
    print('='*60)
    print(f"Samples generated: {len(positions)}")
    print(f"\nDistance to net:")
    print(f"  Min:  {min(distances):.2f} m")
    print(f"  Max:  {max(distances):.2f} m")
    print(f"  Mean: {np.mean(distances):.2f} m")
    print(f"  In target range (1-5m): {sum(1 <= d <= 5 for d in distances)}/{len(distances)} ({100*sum(1 <= d <= 5 for d in distances)/len(distances):.1f}%)")
    
    print(f"\nOrientation:")
    print(f"  Min:  {min(orientations):+.1f}°")
    print(f"  Max:  {max(orientations):+.1f}°")
    print(f"  Mean: {np.mean(orientations):+.1f}°")
    
    # Movement statistics
    movements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    print(f"\nMovement per sample:")
    print(f"  Min:  {min(movements):.2f} m")
    print(f"  Max:  {max(movements):.2f} m")
    print(f"  Mean: {np.mean(movements):.2f} m")
    
    plt.show()


if __name__ == '__main__':
    test_trajectory()
