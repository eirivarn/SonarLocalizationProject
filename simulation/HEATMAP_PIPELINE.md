# Heatmap-Based Training Pipeline - Implementation Summary

## Overview
Implemented a heatmap + direction-based training pipeline for net detection, following keypoint-style localization approach.

## Data Format Changes

### New .npz Structure
Each sample is now saved as a single `.npz` file containing:

```python
{
    'image': (512, 512) float32,        # Intensity channel
    'valid_mask': (512, 512) float32,   # Binary mask (1=valid, 0=outside cone)
    'y_map': (512, 512) float32,        # Y-coordinate map for CoordConv
    'z_map': (512, 512) float32,        # Z-coordinate map for CoordConv
    'p': (2,) float32,                  # Hitpoint [py, pz] in meters
    't': (2,) float32,                  # Direction [ty, tz] unit vector
    'net_visible': bool,                # Whether net is visible
    'distance_m': float,                # Derived: ||p||
    'orientation_deg': float,           # Derived: orientation angle
    'scene_id': int,
    'sample_id': int
}
```

### Removed Fields
- `cage_center`, `cage_radius` (constants, available in metadata)
- `sonar_position`, `sonar_direction` (not needed for training)

## New Utility Functions

### 1. `meters_to_pixels(py, pz, ...)`
Converts meter coordinates to pixel coordinates for target generation.

### 2. `generate_heatmap_target(py, pz, sigma=3.0, ...)`
Generates Gaussian heatmap centered at hitpoint:
```
H[v,u] = exp(-((u-u*)^2 + (v-v*)^2) / (2*sigma^2))
```

### 3. `generate_direction_target(py, pz, ty, tz, radius=10, ...)`
Generates direction map with supervision only in a disk around hitpoint:
- Returns `(2, H, W)` direction map
- Returns `(H, W)` mask indicating supervised region

### 4. `save_dataset_metadata(output_dir)`
Saves constant parameters:
- Pixel-to-meters conversion factors
- Image dimensions and extents
- Heatmap/direction parameters
- All config values

## Configuration Updates

Added to `config.py`:
```python
DATA_GEN_CONFIG = {
    ...
    'save_format': 'npz',              # Changed from 'npy'
    'image_size': (512, 512),
    'heatmap_sigma_pixels': 3.0,
    'direction_radius_pixels': 10,
}
```

## Augmentation Support

### `apply_rotation_augmentation(output_dict, angle_deg)`
- Rotates image channels (image, valid_mask, y_map, z_map)
- Rotates coordinate map values
- Rotates labels: p' = R(α)p, t' = R(α)t
- Updates derived metrics

## Usage Examples

### Generate Dataset
```python
from data_generator import generate_dataset

# Generate all splits with new format
generate_dataset('train')  # Saves as .npz files
generate_dataset('val')
generate_dataset('test')
```

### Load and Use Sample
```python
import numpy as np

# Load sample
data = np.load('sample_000000.npz')

# Extract data
image = data['image']           # (512, 512)
valid_mask = data['valid_mask']
y_map = data['y_map']
z_map = data['z_map']
p = data['p']                   # [py, pz]
t = data['t']                   # [ty, tz]

# Generate training targets
from data_generator import generate_heatmap_target, generate_direction_target

heatmap = generate_heatmap_target(p[0], p[1])
direction_map, direction_mask = generate_direction_target(p[0], p[1], t[0], t[1])
```

### Training Input Construction
```python
# Multi-channel input for network
input_tensor = np.stack([
    image,
    valid_mask,
    y_map,
    z_map
], axis=0)  # Shape: (4, 512, 512)
```

## Coordinate System

- **Y-axis**: Forward (sonar direction), 0-20m
- **Z-axis**: Lateral (perpendicular to sonar), ±18.2m
- **Origin**: At sonar position (bottom-center of image)
- **Pixel (u, v)**: u=horizontal (Z), v=vertical (Y)

## Model Architecture (Recommended)

```
Input: (4, 512, 512)
  ├─ Channel 0: Intensity
  ├─ Channel 1: Valid mask
  ├─ Channel 2: Y-map (CoordConv)
  └─ Channel 3: Z-map (CoordConv)

Encoder: U-Net / FPN backbone

Output Heads:
  ├─ Heatmap: (1, 512, 512) - BCE/Focal loss
  └─ Direction: (2, 512, 512) - Cosine loss (masked)
```

## Loss Functions

### Heatmap Loss
```python
# Binary cross-entropy or focal loss
loss_heatmap = F.binary_cross_entropy_with_logits(pred_heatmap, target_heatmap)
```

### Direction Loss (Masked)
```python
# Apply only in supervision region
loss_direction = (1 - cosine_similarity(pred_direction, target_direction)) * direction_mask
loss_direction = loss_direction.sum() / direction_mask.sum()
```

## Inference

```python
# Get predictions
pred_heatmap, pred_direction = model(input_image)

# Find peak
u_peak, v_peak = find_peak(pred_heatmap)

# Convert to meters
py, pz = pixels_to_meters(v_peak, u_peak)

# Get direction at peak
ty, tz = normalize(pred_direction[:, v_peak, u_peak])

# Derived metrics
distance = np.sqrt(py**2 + pz**2)
orientation = compute_orientation(ty, tz)
```

## Testing

Three test scripts provided:

1. `test_visual_samples.py` - Generate trajectory samples with visualization
2. `test_heatmap_targets.py` - Visualize heatmap and direction targets
3. `test_direction.py` - Statistical analysis of generated samples

## Files Modified

1. `data_generator.py` - Core changes
2. `config.py` - Added heatmap parameters
3. `test_visual_samples.py` - Updated for new format

## Files Created

1. `test_heatmap_targets.py` - Target visualization tool
2. `HEATMAP_PIPELINE.md` - This documentation

## Next Steps

1. **PyTorch Dataset class** - Load .npz, generate targets on-the-fly
2. **U-Net model** - Implement encoder-decoder with two heads
3. **Training script** - With proper loss masking
4. **Evaluation metrics** - Hitpoint error, distance/orientation MAE
5. **Augmentation pipeline** - Rotation, noise, intensity variations
