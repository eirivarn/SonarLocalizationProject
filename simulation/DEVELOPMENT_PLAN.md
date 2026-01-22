# Neural Network Net Detection - Development Plan

## Goal
Develop a neural network model that can detect net distance and orientation from sonar images, using the voxel-based simulation to generate ground truth training data.

## Architecture Overview

```
simulation/
├── DEVELOPMENT_PLAN.md          # This file
├── simulation.py                # Interactive viewer (existing)
├── config.py                    # Centralized configuration
├── data_generator.py            # Generate training/validation datasets
├── models/
│   ├── __init__.py
│   ├── baseline.py              # Simple baseline model
│   ├── cnn_regressor.py         # CNN-based regression model
│   └── attention_net.py         # Advanced: attention-based model
├── training/
│   ├── __init__.py
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation utilities
│   └── losses.py                # Custom loss functions
├── utils/
│   ├── __init__.py
│   ├── scene.py                 # Scene generation (extracted from simulation.py)
│   ├── sonar.py                 # Sonar simulation (extracted from simulation.py)
│   └── visualization.py         # Plotting and visualization
└── datasets/                    # Generated datasets (gitignored)
    ├── train/
    ├── val/
    └── test/
```

## Development Phases

### Phase 1: Data Generation & Baseline (Week 1)
**Goal:** Generate dataset and create simplest possible working model

#### 1.1 Data Generation
- Extract scene generation and sonar simulation into separate modules
- Create `data_generator.py` to automatically:
  - Generate random sonar positions around cage
  - Vary distances (0.5-5m from net)
  - Vary orientations (different approach angles)
  - Save: sonar images (1024×256 numpy arrays)
  - Save: ground truth (distance_m, orientation_deg, net_visible: bool)
- Target: 10,000 training samples, 2,000 validation, 1,000 test

#### 1.2 Baseline Model
- Simple CNN architecture:
  - Input: 1024×256 grayscale image
  - 3-4 convolutional layers
  - Global average pooling
  - 2 fully connected layers
  - Output: [distance, orientation] (2 values)
- Loss: MSE for both outputs
- Metric: Mean Absolute Error (MAE) in meters and degrees

#### 1.3 Training Pipeline
- Basic PyTorch training loop
- Simple data augmentation (noise, rotation)
- Learning rate scheduling
- Checkpointing best model

**Success Criteria:**
- Model achieves < 0.5m distance error
- Model achieves < 15° orientation error
- Training runs without errors

---

### Phase 2: Improved Architecture (Week 2)
**Goal:** Improve model accuracy with better architecture

#### 2.1 Enhanced CNN
- Residual connections
- Batch normalization
- Dropout for regularization
- Separate heads for distance and orientation

#### 2.2 Data Augmentation
- Intensity variations (simulating different sonar gains)
- Geometric transformations
- Synthetic noise patterns
- Occlusions (simulating fish)

#### 2.3 Loss Functions
- Weighted loss (distance more important than orientation)
- Huber loss for robustness to outliers
- Circular loss for orientation (handling 0°/360° wraparound)

**Success Criteria:**
- Model achieves < 0.2m distance error
- Model achieves < 10° orientation error
- Model generalizes well to test set

---

### Phase 3: Advanced Features (Week 3)
**Goal:** Add uncertainty estimation and handle edge cases

#### 3.1 Confidence Estimation
- Add confidence output (3rd output value)
- Predicts when net is not visible
- Monte Carlo dropout for uncertainty

#### 3.2 Handle Complex Scenarios
- Multiple nets/objects
- Heavy occlusions (fish schools)
- Various cage geometries
- Different current effects (net bending)

#### 3.3 Attention Mechanism
- Spatial attention to focus on relevant regions
- Channel attention for feature selection
- Visualize attention maps for interpretability

**Success Criteria:**
- Model predicts confidence accurately
- Reduces false positives when net not visible
- Attention maps highlight net regions

---

### Phase 4: Real-Time Optimization (Week 4)
**Goal:** Optimize for deployment

#### 4.1 Model Compression
- Quantization (FP32 → FP16/INT8)
- Pruning unnecessary connections
- Knowledge distillation to smaller model

#### 4.2 Speed Optimization
- Batch inference
- ONNX export for deployment
- Benchmark inference time

#### 4.3 Comparison with Algorithm
- Compare accuracy: NN vs traditional algorithm
- Compare speed: NN vs traditional algorithm
- Compare robustness to noise/artifacts

**Success Criteria:**
- Inference < 20ms per frame
- Accuracy comparable or better than algorithm
- Model size < 50MB

---

## Immediate Next Steps (Phase 1)

### Step 1: Modularize Simulation Code
1. Extract `VoxelGrid` and `VoxelSonar` classes to `utils/sonar.py`
2. Extract scene creation to `utils/scene.py`
3. Keep `simulation.py` as interactive viewer only

### Step 2: Create Configuration
1. Centralize all parameters in `config.py`
2. Include sonar specs, scene parameters, training params

### Step 3: Data Generator
1. Create `data_generator.py` with:
   - `generate_random_scene()` - creates varied scenarios
   - `capture_sonar_image()` - gets image + ground truth
   - `save_dataset()` - saves to disk in organized format

### Step 4: Simple Baseline Model
1. Create `models/baseline.py` with minimal CNN
2. Create `training/train.py` for basic training loop
3. Test on small dataset (100 samples) to verify pipeline

### Step 5: Initial Training Run
1. Generate 1000 sample dataset
2. Train baseline model for 10 epochs
3. Evaluate and visualize results
4. Iterate on issues

---

## Key Design Decisions

### Data Format
- **Input:** `.npy` files (1024×256 float32)
- **Labels:** `.json` files with `{distance_m, orientation_deg, net_visible, metadata}`
- **Organization:** `dataset/split/scene_XXX/sonar_YYY.npy`

### Model Output
- **Distance:** Regression in meters (0-20m range)
- **Orientation:** Regression in degrees (0-360°, or use sin/cos encoding)
- **Alternative:** Classification + regression hybrid for robustness

### Validation Strategy
- **Spatial split:** Different cage positions for train/val/test
- **Scenario split:** Different fish densities, net conditions
- **Metrics:** MAE, RMSE, R² for both outputs

### Augmentation Strategy
- **Intensity:** Gamma correction, noise injection
- **Geometric:** Small rotations, translations (careful with labels!)
- **Physical:** Simulate different water conditions, fish occlusions

---

## Success Metrics

### Phase 1 (Baseline)
- [x] Dataset generated: 10k+ samples
- [ ] Model trains without errors
- [ ] Distance MAE < 0.5m
- [ ] Orientation MAE < 15°

### Phase 2 (Improved)
- [ ] Distance MAE < 0.2m
- [ ] Orientation MAE < 10°
- [ ] Good generalization (test close to val)

### Phase 3 (Advanced)
- [ ] Confidence calibrated (when net visible)
- [ ] Attention maps interpretable
- [ ] Robust to occlusions

### Phase 4 (Deployment)
- [ ] Inference < 20ms
- [ ] Model < 50MB
- [ ] Accuracy ≥ traditional algorithm

---

## Notes

- Start simple, iterate quickly
- Use simulation for infinite data generation
- Compare against traditional algorithm as baseline
- Focus on robustness and interpretability
- Document findings and experiments
