# Heatmap-Based Net Detection Training

This document describes the complete ML pipeline for sonar net detection using heatmap-based keypoint detection.

## Overview

The system uses a U-Net architecture with GroupNorm to predict:
1. **Heatmap**: Gaussian peak at net hitpoint location
2. **Direction**: Unit vector indicating net orientation
3. **Visibility**: Binary flag for net presence

This approach provides dense spatial supervision and is more robust than global regression.

## Data Format

### Input: 4-Channel Cartesian Image
- **Channel 0**: `intensity` - Sonar intensity values
- **Channel 1**: `valid_mask` - Binary mask of valid sonar region  
- **Channel 2**: `y_map` - Y-coordinate map (forward, 0-20m)
- **Channel 3**: `z_map` - Z-coordinate map (lateral, ±18.2m)

Shape: `(4, 512, 512)`, saved as `.npz` files

### Labels
- **Hitpoint**: `p = [py, pz]` in meters
- **Direction**: `t = [ty, tz]` unit vector
- **Visibility**: `v ∈ {0, 1}` net visible flag

## Training Pipeline

### 1. Generate Training Data

```bash
cd simulation
python data_generator.py --num_samples 10000 --output_dir datasets/train
```

This generates:
- 10,000 `.npz` files with 4-channel Cartesian images
- Varied net distances (0.5-20m), orientations, and scene configurations
- Automatic heatmap and direction targets computed on-the-fly during training

### 2. Train Model

```bash
python training/train.py \
  --epochs 200 \
  --batch_size 8 \
  --lr 1e-3 \
  --heatmap_weight 1.0 \
  --direction_weight 0.5 \
  --visibility_weight 0.2 \
  --rotation_range 20.0 \
  --mixed_precision
```

Key arguments:
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 8, use 4-16 depending on GPU)
- `--lr`: Learning rate (default: 1e-3)
- `--heatmap_weight`: Weight for heatmap loss (default: 1.0)
- `--direction_weight`: Weight for direction loss (default: 0.5)
- `--visibility_weight`: Weight for visibility loss (default: 0.2)
- `--rotation_range`: Rotation augmentation ±degrees (default: 20.0)
- `--gain_range`: Intensity gain augmentation (default: [0.5, 2.0])
- `--gamma_range`: Gamma correction range (default: [0.7, 1.4])
- `--noise_std`: Additive Gaussian noise std (default: 0.02)
- `--speckle_std`: Speckle noise std (default: 0.1)
- `--mixed_precision`: Enable mixed precision training (faster on modern GPUs)

Training outputs:
- `outputs/training/run_TIMESTAMP/checkpoints/` - Model checkpoints
- `outputs/training/run_TIMESTAMP/visualizations/` - Prediction visualizations
- `outputs/training/run_TIMESTAMP/args.json` - Training configuration

### 3. Augmentation Strategy

Applied during training:
1. **Rotation**: ±20° with proper coordinate transformation
2. **Intensity Augmentation**:
   - Gain: 0.5-2.0x
   - Gamma correction: 0.7-1.4
3. **Noise**:
   - Additive Gaussian: σ=0.02
   - Speckle noise: σ=0.1

All augmentations preserve coordinate maps and update targets accordingly.

## Model Architecture

### U-Net with GroupNorm

```
Input: (B, 4, 512, 512)
├── Encoder
│   ├── DoubleConv(4 -> 64) + GroupNorm + ReLU
│   ├── Down(64 -> 128)
│   ├── Down(128 -> 256)
│   ├── Down(256 -> 512)
│   └── Down(512 -> 1024) [bottleneck]
├── Decoder
│   ├── Up(1024 -> 512)
│   ├── Up(512 -> 256)
│   ├── Up(256 -> 128)
│   └── Up(128 -> 64)
└── Output Heads
    ├── Heatmap: Conv(64 -> 1) + Sigmoid → (B, 1, 512, 512)
    ├── Direction: Conv(64 -> 2) + Tanh → (B, 2, 512, 512)
    └── Visibility: GlobalAvgPool + FC → (B, 1)
```

**Why GroupNorm?** Stable with small batch sizes (8-16), unlike BatchNorm which degrades below batch size 32.

## Loss Functions

### 1. Focal Loss (Heatmap)

Addresses extreme class imbalance (1 positive pixel vs 262,143 negatives):

```
FL(p) = -α(1-p)^γ log(p)
```

- α = 0.25 (down-weight easy negatives)
- γ = 2.0 (focus on hard examples)

### 2. Cosine Direction Loss (Direction)

Rotation-invariant angular loss:

```
L_dir = 1 - cos(θ_pred, θ_target) + λ||pred||²
```

- Supervised only within 10px radius of hitpoint (masked)
- L2 penalty (λ=0.1) encourages unit norm

### 3. Binary Cross-Entropy (Visibility)

Simple BCE for net presence classification.

### Combined Loss

```
L_total = 1.0 * L_heatmap + 0.5 * L_direction + 0.2 * L_visibility
```

## Evaluation Metrics

### 1. Percentage of Correct Keypoints (PCK)

Fraction of predictions within threshold distance:
- **PCK@0.5m**: High precision requirement
- **PCK@1.0m**: Standard metric
- **PCK@2.0m**: Relaxed threshold
- **PCK@5.0m**: Coarse detection

### 2. Distance Error

- Mean/median Euclidean distance between predicted and ground truth hitpoints

### 3. Angular Error

- Mean/median angular error between predicted and target directions (degrees)

### 4. Detection Rate

- Fraction of samples with confident detection (heatmap > 0.1 threshold)

### 5. Stratified Analysis

Metrics broken down by:
- **Distance bins**: [0-5m, 5-10m, 10-15m, 15-20m]
- **Orientation bins**: [-60°, -30°, 0°, 30°, 60°]

## Inference

### Extract Predictions

```python
import torch
from models.unet import UNet
from training.metrics import extract_peak_from_heatmap, pixels_to_meters

# Load model
model = UNet(in_channels=4, ...)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    outputs = model(input_4channel)  # (1, 4, 512, 512)
    
    # Extract peak from heatmap
    peak_px, confidence, valid = extract_peak_from_heatmap(
        outputs['heatmap'], threshold=0.1
    )
    
    # Convert to meters
    hitpoint_m = pixels_to_meters(peak_px)  # [py, pz]
    
    # Get direction at peak
    direction = outputs['direction'][0, :, peak_px[0, 0], peak_px[0, 1]]
    direction = direction / direction.norm()  # Normalize to unit vector
    
    # Visibility
    visibility = outputs['visibility'].sigmoid() > 0.5
```

## Expected Performance

After 200 epochs with proper augmentation:

| Metric | Target |
|--------|--------|
| PCK@1m | >85% |
| PCK@2m | >95% |
| Mean Distance Error | <0.5m |
| Angular Error | <10° |
| Detection Rate | >98% |

Performance degrades gracefully with distance:
- **0-5m**: PCK@1m >95%, <0.3m error
- **5-10m**: PCK@1m >90%, <0.4m error
- **10-15m**: PCK@1m >80%, <0.6m error
- **15-20m**: PCK@1m >70%, <0.8m error

## Curriculum Learning (Optional)

For even better results, train in 3 phases:

### Phase 1: Heatmap Only (0-50 epochs)
```bash
python training/train.py --epochs 50 --direction_weight 0.0 --visibility_weight 0.0
```

### Phase 2: Add Direction (50-100 epochs)
```bash
python training/train.py --epochs 50 --resume checkpoints/phase1.pth \
  --direction_weight 0.5 --visibility_weight 0.0
```

### Phase 3: Full Multi-Task (100-200 epochs)
```bash
python training/train.py --epochs 100 --resume checkpoints/phase2.pth \
  --direction_weight 0.5 --visibility_weight 0.2
```

## Troubleshooting

### Low Detection Rate (<90%)
- Increase heatmap sigma: `heatmap_sigma_pixels = 5.0`
- Reduce focal gamma: `focal_gamma = 1.5`
- Check data quality (valid_mask coverage)

### Poor Localization (PCK@1m <70%)
- Reduce heatmap sigma for tighter peaks: `heatmap_sigma_pixels = 2.0`
- Increase heatmap weight: `heatmap_weight = 2.0`
- Check augmentation isn't too aggressive

### Large Angular Errors (>15°)
- Increase direction radius: `direction_radius_pixels = 15`
- Increase direction weight: `direction_weight = 1.0`
- Verify direction targets are correct in visualization

### Training Instability
- Reduce learning rate: `--lr 5e-4`
- Increase warmup: `--warmup_pct 0.2`
- Check for NaN losses (reduce augmentation)
- Use gradient clipping (add to train.py)

## File Structure

```
simulation/
├── config.py                   # Configuration parameters
├── data_generator.py           # Generate training data
├── models/
│   └── unet.py                 # U-Net architecture
├── training/
│   ├── dataset.py              # PyTorch Dataset with augmentation
│   ├── losses.py               # Focal, Cosine, combined losses
│   ├── metrics.py              # PCK, angular error, visualization
│   └── train.py                # Main training script
└── outputs/
    └── training/
        └── run_TIMESTAMP/
            ├── checkpoints/
            ├── visualizations/
            └── args.json
```

## References

- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet)
- **GroupNorm**: Wu & He, "Group Normalization"
- **Heatmap Keypoints**: Newell et al., "Stacked Hourglass Networks"
- **Medical Keypoints**: Similar to spine detection, cell tracking

## Next Steps

1. **Generate initial dataset**: `python data_generator.py --num_samples 10000`
2. **Overfit test**: Train on 256 samples to verify pipeline works
3. **Full training**: Run 200 epochs with augmentation
4. **Evaluation**: Compute PCK metrics, visualize worst-N cases
5. **Hyperparameter tuning**: Adjust loss weights based on results
6. **Deployment**: Export to ONNX/TorchScript for inference
