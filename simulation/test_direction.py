"""Test that sonar is mainly inside cage and pointing towards net."""
import numpy as np
from data_generator import generate_sample
import data_generator

# Clear cache
data_generator._trajectory_state = {}

scene_id = 200
positions = []
inside_count = 0

print("Generating 100 samples...")
for i in range(100):
    img, label = generate_sample(scene_id, i)
    pos = np.array(label['sonar_position'])
    cage_center = np.array(label['cage_center'])
    cage_radius = label['cage_radius']
    
    dist_from_center = np.linalg.norm(pos - cage_center)
    if dist_from_center < cage_radius:
        inside_count += 1
    
    if label['net_visible']:
        positions.append((label['distance_m'], label['orientation_deg'], dist_from_center < cage_radius))

print(f'\nInside cage: {inside_count}/100 ({inside_count}%)')
print(f'Net visible: {len(positions)}/100 ({len(positions)}%)')

if positions:
    dists = [p[0] for p in positions]
    orientations = [p[1] for p in positions]
    inside_when_visible = sum(p[2] for p in positions)
    
    in_range = sum(1 <= d <= 5 for d in dists)
    print(f'\nDistance statistics:')
    print(f'  Range: {min(dists):.2f}-{max(dists):.2f}m')
    print(f'  Mean: {np.mean(dists):.2f}m')
    print(f'  In target 1-5m: {in_range}/{len(positions)} ({100*in_range/len(positions):.1f}%)')
    
    print(f'\nOrientation statistics:')
    print(f'  Range: {min(orientations):+.1f}° to {max(orientations):+.1f}°')
    print(f'  Mean: {np.mean(orientations):+.1f}°')
    
    print(f'\nInside when net visible: {inside_when_visible}/{len(positions)} ({100*inside_when_visible/len(positions):.1f}%)')
