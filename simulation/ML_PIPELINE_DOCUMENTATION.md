# Sonar Net Detection - ML Pipeline Documentation

## Overview
This system trains a CNN to detect aquaculture nets in simulated sonar images and predict two key metrics:
1. **Distance** to the net (meters)
2. **Orientation** of the net relative to the sonar (degrees, -90° to +90°)

## 1. Data Generation

### 1.1 Simulation Environment
- **World**: 30m × 30m voxel grid (10cm resolution)
- **Cage**: Circular fish cage, 12m radius
- **Net Structure**: 721 line segments forming closed loop with realistic sag and current deflection
- **Scene Contents**:
  - 150 fish (avoid spawning on net)
  - 40 biomass patches (algae/organic matter)
  - All entities use voxel materials (FISH, BIOMASS, NET)

### 1.2 Sonar Specifications (SonoptixECHO)
- **Resolution**: 1024 range bins × 256 beams
- **Field of View**: 120° horizontal
- **Range**: 20 meters
- **Image Processing**:
  - Log scale: `10 * log10(max(intensity, 1e-10))`
  - Normalization: `clip((log_image + 60) / 60, 0, 1)`
  - Final shape: (1024, 256) normalized to [0, 1]

### 1.3 Trajectory-Based Sampling (Key Innovation)
Instead of random teleportation, sonar follows realistic ROV inspection paths:

**Initialization (first sample in scene)**:
```python
# Random spawn: 90% inside cage, 10% outside
distance_from_net = uniform(1.0, 5.0)  # Target inspection range
if inside_cage:
    distance_from_center = cage_radius - distance_from_net
else:
    distance_from_center = cage_radius + distance_from_net
    
# Direction: Point towards net surface ± 30° randomness
```

**Random Walk (subsequent samples)**:
```python
# Step: 0.2-0.8m per sample
step = random_unit_vector() * uniform(0.2, 0.8)

# Bias force to maintain 1-5m from net
if distance_to_net < 1m:
    bias_force = -0.3 * (net_point - position)  # Push away
elif distance_to_net > 5m:
    bias_force = 0.3 * (net_point - position)   # Pull towards

new_position = prev_position + step + bias_force

# Direction: Point towards net ± 45° with smooth transitions
target_direction = towards_net + random_rotation(±45°)
direction = 0.8 * target_direction + 0.2 * prev_direction
```

**Benefits**:
- Continuous trajectories mimic real ROV operations
- Distance bias keeps net in optimal detection range
- Smooth direction changes avoid unrealistic instant rotations
- 78.6% of samples in target 1-5m range

### 1.4 Ground Truth Calculation
Uses **geometric ray-segment intersection** (not voxel raycasting):

```python
# For each net segment (S1, S2):
# Solve: position + t*direction = S1 + u*(S2 - S1)
# Where: 0 ≤ t ≤ max_range, 0 ≤ u ≤ 1

# Returns:
distance_m = ||hit_point - sonar_position||
orientation_deg = relative_angle(segment, sonar_direction)
```

**Orientation Encoding**:
- Net segment has absolute angle θ
- Sonar looks in direction φ
- Relative orientation = normalize(θ - φ) to [-90°, +90°]
- 0° = perpendicular/broadside view (best detection)
- ±90° = parallel/edge-on view (worst detection)

**Advantages**:
- Perfect ground truth (no discretization errors)
- Handles net gaps correctly (closed loop)
- Fast computation (721 segments × 10k samples in seconds)

## 2. Dataset Structure

### 2.1 Organization
```
datasets/
├── train/       (10,000 samples)
├── val/         (2,000 samples)
└── test/        (1,000 samples)

Each sample:
- sample_XXXXXX.npy    # Sonar image (1024, 256) float32
- sample_XXXXXX.json   # Labels
- Every 10th: images/sample_XXXXXX.png  # Visualization
```

### 2.2 Label Format
```json
{
  "distance_m": 3.45,
  "orientation_deg": -23.7,
  "net_visible": true,
  "scene_id": 42,
  "sample_id": 5,
  "sonar_position": [14.2, 18.7],
  "sonar_direction": [0.87, -0.49],
  "cage_center": [15.0, 15.0],
  "cage_radius": 12.0
}
```

### 2.3 Filtering
Dataset loader filters to only include samples where:
- `net_visible == true`
- `distance_m is not None`
- `orientation_deg is not None`

**Current stats**:
- Train: ~10,000 valid samples (100% net visible)
- Val: ~2,000 valid samples
- Test: ~1,000 valid samples

## 3. Model Architecture

### 3.1 BaselineCNN
Simple convolutional architecture for regression:

```python
Input: (batch, 1, 1024, 256)

# Encoder
Conv2d(1 → 16, kernel=7, stride=2, padding=3)  # → (512, 128)
ReLU + MaxPool2d(2)                             # → (256, 64)

Conv2d(16 → 32, kernel=5, stride=2, padding=2) # → (128, 32)
ReLU + MaxPool2d(2)                             # → (64, 16)

Conv2d(32 → 64, kernel=3, stride=2, padding=1) # → (32, 8)
ReLU + MaxPool2d(2)                             # → (16, 4)

Conv2d(64 → 128, kernel=3, stride=2, padding=1)# → (8, 2)
ReLU + AdaptiveAvgPool2d((4, 2))                # → (4, 2)

Flatten()                                        # → 1024

# Regression heads
FC(1024 → 256) + ReLU + Dropout(0.3)
FC(256 → 128) + ReLU + Dropout(0.3)

# Outputs
FC(128 → 1) → distance (meters)
FC(128 → 2) → orientation (sin, cos)
```

**Parameters**: 33,685,123 total

**Output format**:
```python
{
  'distance': tensor([3.45]),           # Meters
  'orientation_sin': tensor([-0.402]),  # sin(θ)
  'orientation_cos': tensor([0.916])    # cos(θ)
}
```

### 3.2 Design Rationale
- **Simple baseline**: Easy to train, debug, and compare against
- **Fully convolutional encoder**: Preserves spatial information
- **Separate regression heads**: Distance and orientation are independent
- **Sin/cos encoding**: Handles angle wrap-around (89° and -89° are close)
- **Dropout**: Prevents overfitting on synthetic data

## 4. Loss Function

### 4.1 NetDetectionLoss
Combined loss with two components:

```python
class NetDetectionLoss(nn.Module):
    def __init__(self, distance_weight=1.0, orientation_weight=1.0):
        # Distance: MSE on meter values
        distance_loss = F.mse_loss(pred['distance'], target['distance'])
        
        # Orientation: MSE on sin/cos components
        orientation_loss = (
            F.mse_loss(pred['orientation_sin'], target['orientation_sin']) +
            F.mse_loss(pred['orientation_cos'], target['orientation_cos'])
        )
        
        total_loss = (
            distance_weight * distance_loss +
            orientation_weight * orientation_loss
        )
```

### 4.2 Why Sin/Cos Encoding?
**Problem**: Direct angle regression fails at boundaries:
- Model predicts 89° and 91°
- MSE sees these as close (2° error)
- But 91° wraps to -89°, actually 178° apart!

**Solution**: Encode angle θ as (sin(θ), cos(θ)):
- 89° → (sin=1.0, cos=0.017)
- -89° → (sin=-1.0, cos=0.017)
- MSE correctly measures angular distance
- No discontinuity at ±90°

### 4.3 Metrics
**Distance MAE**: Mean absolute error in meters
```python
distance_mae = |pred_distance - true_distance|
```

**Orientation MAE**: Degrees error from sin/cos
```python
pred_angle = atan2(pred_sin, pred_cos)
true_angle = atan2(true_sin, true_cos)
orientation_mae = |pred_angle - true_angle|
```

## 5. Training Configuration

### 5.1 Hyperparameters
```python
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'optimizer': Adam,
    'scheduler': ReduceLROnPlateau(
        mode='min',
        factor=0.5,
        patience=10
    ),
    'num_workers': 4,
    'distance_weight': 1.0,
    'orientation_weight': 1.0
}
```

### 5.2 Training Loop
```python
for epoch in range(num_epochs):
    # Train
    for images, labels in train_loader:
        pred = model(images)
        loss, metrics = criterion(pred, labels)
        loss.backward()
        optimizer.step()
    
    # Validate
    val_metrics = validate(model, val_loader)
    scheduler.step(val_metrics['loss'])
    
    # Save best models
    if val_loss < best_loss:
        save('best_loss.pth')
    if val_distance_mae < best_distance:
        save('best_distance.pth')
```

### 5.3 Logging
- **TensorBoard**: Loss curves, learning rate, metrics
- **Checkpoints**: Latest, best loss, best distance MAE
- **Console**: Real-time progress bars with metrics

## 6. Current Status & Known Issues

### 6.1 Dataset Quality
✓ **Strengths**:
- Perfect ground truth (geometric intersection)
- Realistic trajectories (random walk)
- Good distance distribution (78.6% in 1-5m range)
- Diverse orientations (-90° to +90°)
- 100% net visibility in filtered dataset

⚠ **Potential Issues**:
- Synthetic data only (no real sonar images)
- Single cage geometry (12m radius circle)
- Limited environmental variation
- Net always present (no "net absent" negative samples)

### 6.2 Model Concerns
⚠ **Baseline CNN**:
- Very large (33M params for 1024×256 input)
- May overfit on synthetic patterns
- No attention mechanism for focusing on net
- Doesn't explicitly model sonar physics

### 6.3 Training Challenges
⚠ **Current blockers**:
- PyTorch installation issues (Python 3.13 compatibility)
- Some samples still have None orientation despite filtering
- No GPU available (CPU training will be slow)

## 7. Questions for ML Expert

### 7.1 Data & Augmentation
1. Should we add samples with `net_visible=false` as negative examples?
2. Is 90/10 inside/outside cage distribution appropriate?
3. Data augmentation strategies for sonar images?
   - Random noise injection?
   - Intensity scaling?
   - Geometric transforms (rotation/flip)?

### 7.2 Architecture
4. Is 33M params too large for this task?
5. Should we use a smaller backbone (ResNet18, EfficientNet)?
6. Would attention mechanisms help (focus on net region)?
7. Multi-task learning: predict visibility as classification + regression?
8. Use pretrained weights (ImageNet) despite domain mismatch?

### 7.3 Loss Function
9. Are loss weights (1.0, 1.0) appropriate?
10. Should distance and orientation have different scales?
11. Alternative orientation encoding (von Mises distribution)?
12. Add uncertainty estimation (heteroscedastic loss)?

### 7.4 Training Strategy
13. Current LR schedule optimal?
14. Should we use warmup?
15. Batch size impact (32 vs 64 vs 128)?
16. Early stopping criteria?
17. Curriculum learning (easy → hard samples)?

### 7.5 Evaluation
18. What metrics beyond MAE should we track?
19. How to measure sim-to-real transfer potential?
20. Should we split test set by distance ranges?
21. Analyze failure modes (which orientations fail)?

### 7.6 Deployment
22. Model compression strategies (quantization, pruning)?
23. Real-time inference requirements?
24. How to update model with real data when available?

## 8. File Locations

```
simulation/
├── data_generator.py          # Dataset generation with trajectory system
├── config.py                  # All configuration constants
├── models/
│   └── baseline.py           # BaselineCNN architecture
├── training/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── dataset.py            # PyTorch Dataset class
│   └── losses.py             # Loss functions and metrics
└── datasets/
    ├── train/                # 10k samples
    ├── val/                  # 2k samples
    └── test/                 # 1k samples
```

## 9. Running Training

```bash
cd simulation

# Generate dataset (if not done)
python3 data_generator.py --train 10000 --val 2000 --test 1000

# Train model
python3 training/train.py \
    --epochs 50 \
    --batch-size 32 \
    --output-dir checkpoints/baseline_run1

# Evaluate
python3 training/evaluate.py \
    --checkpoint checkpoints/baseline_run1/best_distance.pth \
    --split test
```

## 10. Key Design Decisions Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Ground Truth** | Ray-segment intersection | Perfect accuracy, no voxel discretization |
| **Trajectory** | Random walk with bias | Realistic ROV behavior, smooth paths |
| **Orientation** | Sin/cos encoding | Handles angle wrap-around correctly |
| **Architecture** | Simple CNN baseline | Easy to train, debug, establish baseline |
| **Loss** | MSE for both tasks | Standard regression, equal weights initially |
| **Distance Range** | 1-5 meters | Typical inspection distance for aquaculture |
| **Spawn Distribution** | 90% inside cage | Reflects real ROV operations |
| **Dataset Size** | 10k/2k/1k split | Standard proportions, sufficient for baseline |

---

**Last Updated**: January 22, 2026  
**Status**: Dataset generated, training pipeline ready, awaiting successful training run
