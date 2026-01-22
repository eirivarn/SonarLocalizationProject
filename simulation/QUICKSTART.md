# Quick Start Guide - Neural Network Net Detection

## Overview
This guide walks you through training a neural network to detect net distance and orientation from sonar images using the voxel simulation for ground truth data generation.

## 📋 Prerequisites

```bash
# Ensure you have PyTorch installed
pip install torch torchvision torchaudio
pip install tensorboard tqdm
```

## 🚀 Step-by-Step Workflow

### 1. Generate Training Data (Required First!)

```bash
cd /Users/eirikvarnes/code/SOLAQUA/simulation
python data_generator.py
```

**Expected Output:**
```
Generating 10000 train samples...
Generating 2000 val samples...
Generating 1000 test samples...
✓ Dataset generation complete!
```

**What This Does:**
- Creates 10,000 training samples (random sonar positions around cage)
- Creates 2,000 validation samples
- Creates 1,000 test samples
- Each sample = sonar image (`.npy`) + ground truth label (`.json`)
- Saves to `simulation/datasets/train/`, `val/`, `test/`

**Time:** ~30-60 minutes depending on your CPU

---

### 2. Test Dataset Loading

```bash
python training/dataset.py
```

**Expected Output:**
```
Loaded 10000 samples from datasets/train
Dataset size: 10000

Sample 0:
  Image shape: torch.Size([1, 1024, 256])
  Distance: 2.34m
  Orientation: 127.5°
  
✓ Dataset loading works!
```

---

### 3. Test Loss Functions

```bash
python training/losses.py
```

**Expected Output:**
```
Loss Metrics:
  Total loss: 0.0234
  Distance loss: 0.0156
  Orientation loss: 0.0078
  
✓ Loss functions work!
```

---

### 4. Test Model Architecture

```bash
python models/baseline.py
```

**Expected Output:**
```
BaselineCNN Summary:
  Parameters: 485,123
  
Forward pass test:
  Input: torch.Size([4, 1, 1024, 256])
  Distance output: torch.Size([4])
  Orientation output: torch.Size([4, 2])
  
✓ Model architecture works!
```

---

### 5. Train Model

```bash
python training/train.py --epochs 100 --batch-size 32
```

**What This Does:**
- Trains baseline CNN for 100 epochs
- Batch size 32 (adjust based on GPU memory)
- Saves checkpoints to `simulation/checkpoints/`
- Logs training curves to TensorBoard

**Expected Output:**
```
Epoch 1/100
Train - Loss: 1.234, Dist MAE: 0.654m, Orient MAE: 32.1°
Val   - Loss: 1.123, Dist MAE: 0.612m, Orient MAE: 28.4°
→ Saved best loss model (1.123)

Epoch 2/100
...
```

**Time:** ~1-2 hours on GPU, ~5-10 hours on CPU

**Monitor Training:**
```bash
tensorboard --logdir=checkpoints/runs
# Open http://localhost:6006 in browser
```

---

### 6. Evaluate Model

```bash
python training/evaluate.py --checkpoint checkpoints/best_distance.pth --split test
```

**Expected Output:**
```
EVALUATION RESULTS
==================
Dataset: test (1000 samples)

Distance Metrics:
  MAE:    0.234m
  RMSE:   0.312m
  Median: 0.189m
  Max:    1.234m

Orientation Metrics:
  MAE:    12.3°
  RMSE:   16.8°
  Median: 9.4°
  Max:    45.6°
```

**What This Does:**
- Loads best model checkpoint
- Evaluates on test set
- Generates plots (scatter, histograms, error analysis)
- Saves results to JSON and NPZ files

---

## 📊 Understanding the Results

### Phase 1 Success Criteria
- ✅ **Distance MAE < 0.5m** (Target: competitive with algorithm)
- ✅ **Orientation MAE < 15°** (Target: usable for navigation)
- ✅ **Training time < 2 hours** (On GPU)

### Interpreting Plots

1. **Predicted vs True**: Should be close to diagonal line
2. **Error Distribution**: Should be narrow and centered at 0
3. **Error vs True Value**: Should show no systematic bias

---

## 🔧 Configuration

All parameters are in [`config.py`](config.py):

### Adjust Data Generation
```python
DATA_GEN_CONFIG = {
    'num_train': 10000,      # More data = better model
    'num_val': 2000,
    'num_test': 1000,
    'distance_range': [0.5, 5.0],  # Meters from net
    'angle_range': [-45, 45],      # Approach angles
}
```

### Adjust Training
```python
TRAINING_CONFIG = {
    'batch_size': 32,        # Larger = faster but needs more GPU memory
    'learning_rate': 1e-3,   # Lower = more stable, higher = faster
    'epochs': 100,           # More = better fit (watch for overfitting)
}
```

### Adjust Model
```python
MODEL_CONFIG = {
    'conv_channels': [16, 32, 64, 128],  # Wider = more capacity
    'fc_features': [256, 128],           # Deeper = more expressive
}
```

---

## 🐛 Troubleshooting

### Problem: "Dataset not found"
**Solution:** Run `python data_generator.py` first!

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size in training command:
```bash
python training/train.py --batch-size 16
```

### Problem: "Model overfitting (train loss << val loss)"
**Solution:** 
- Generate more data (increase `num_train`)
- Add data augmentation
- Reduce model size
- Add dropout/regularization

### Problem: "Training too slow"
**Solution:**
- Use GPU if available
- Reduce dataset size for testing
- Increase `num_workers` in data loading

### Problem: "Poor distance accuracy"
**Solution:**
- Check distance_range matches operational range
- Increase `distance_weight` in loss
- Generate more samples at challenging distances

### Problem: "Poor orientation accuracy"
**Solution:**
- Check angle_range covers expected scenarios
- Increase `orientation_weight` in loss
- Verify sin/cos encoding is working

---

## 📈 Next Steps After Phase 1

Once baseline achieves < 0.5m MAE and < 15° MAE:

### Phase 2: Enhanced Architecture
1. Add deeper CNN (ResNet-style)
2. Add data augmentation (noise, blur, rotation)
3. Target: < 0.3m MAE, < 12° MAE

### Phase 3: Advanced Features
1. Add uncertainty estimation
2. Add attention mechanisms
3. Multi-task learning (detect + segment)

### Phase 4: Deployment
1. Optimize for real-time inference
2. Test on real SOLAQUA data
3. Deploy to ROV

---

## 📝 File Structure Recap

```
simulation/
├── data_generator.py      # Generate datasets
├── training/
│   ├── train.py          # Main training script
│   ├── evaluate.py       # Evaluation script
│   ├── dataset.py        # PyTorch Dataset
│   └── losses.py         # Loss functions
├── models/
│   └── baseline.py       # CNN model
├── config.py             # All configuration
└── datasets/             # Generated data (created by data_generator.py)
    ├── train/
    ├── val/
    └── test/
```

---

## 🎯 Expected Timeline

| Task | Time | Output |
|------|------|--------|
| Generate dataset | 30-60 min | 13,000 samples |
| Test pipeline | 5 min | Verify everything works |
| Train baseline | 1-2 hours | Trained model |
| Evaluate | 5 min | Performance metrics |
| **Total** | **2-3 hours** | Working NN detector |

---

## 💡 Tips

1. **Start Small**: Test with 100 samples first to verify pipeline
   ```bash
   # Modify data_generator.py temporarily:
   DATA_GEN_CONFIG['num_train'] = 100
   DATA_GEN_CONFIG['num_val'] = 20
   DATA_GEN_CONFIG['num_test'] = 20
   ```

2. **Monitor Training**: Watch TensorBoard to catch issues early

3. **Save Everything**: Checkpoints are cheap, experiments are expensive

4. **Compare with Algorithm**: Run evaluation on same test set as traditional algorithm

5. **Iterate Quickly**: Start with fast experiments, scale up when confident

---

**Ready to train your first model? Start with Step 1!** 🚀
