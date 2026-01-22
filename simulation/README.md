# Neural Network Net Detection - Simulation

This simulation environment is designed to train and evaluate neural network models for detecting fishing net distance and orientation from sonar images.

## 📁 Project Structure

```
simulation/
├── README.md                    # This file
├── DEVELOPMENT_PLAN.md          # Detailed development roadmap
├── config.py                    # Centralized configuration
├── simulation.py                # Interactive viewer (existing)
├── data_generator.py            # Generate training datasets
├── models/                      # Neural network architectures
│   ├── baseline.py             # Simple CNN baseline
│   └── ...                     # More models to come
├── training/                    # Training scripts
│   ├── train.py               # Main training loop
│   ├── evaluate.py            # Evaluation utilities
│   └── losses.py              # Custom loss functions
├── utils/                       # Utility functions
│   ├── sonar.py              # Sonar simulation
│   ├── scene.py              # Scene generation
│   └── visualization.py       # Plotting tools
└── datasets/                    # Generated data (gitignored)
    ├── train/
    ├── val/
    └── test/
```

## 🚀 Quick Start

### 1. Generate Dataset
```bash
cd simulation
python data_generator.py
```

This will generate:
- 10,000 training samples
- 2,000 validation samples
- 1,000 test samples

Each sample consists of:
- **Input**: `sample_XXXXXX.npy` - (1024, 256) sonar image
- **Label**: `sample_XXXXXX.json` - {distance_m, orientation_deg, metadata}

### 2. Test Baseline Model
```bash
python models/baseline.py
```

This runs a forward pass test to verify the model architecture.

### 3. Train Model (Coming Soon)
```bash
python training/train.py --model baseline --epochs 100
```

### 4. Evaluate Model (Coming Soon)
```bash
python training/evaluate.py --checkpoint best_model.pth --split test
```

## 📊 Data Format

### Input (Sonar Image)
- **Shape**: (1024, 256)
- **Type**: float32
- **Range**: [0, 1] (normalized)
- **Format**: `.npy` file

### Output (Labels)
```json
{
  "distance_m": 1.234,           // Distance to net in meters
  "orientation_deg": 45.6,       // Net orientation in degrees [0, 360)
  "scene_id": 123,               // Scene identifier
  "sample_id": 456,              // Sample identifier
  "sonar_position": [15.2, 3.4], // Sonar XY position
  "cage_center": [15.0, 15.0],   // Cage center
  "cage_radius": 12.0            // Cage radius
}
```

## 🎯 Current Status

✅ **Phase 1.1: Data Generation**
- [x] Modular simulation structure
- [x] Centralized configuration
- [x] Random scene generation
- [x] Dataset generator script
- [x] Baseline CNN model

⏳ **Phase 1.2: Training Pipeline** (Next Steps)
- [ ] PyTorch Dataset class
- [ ] Training loop
- [ ] Evaluation metrics
- [ ] Checkpointing

🔄 **Phase 1.3: Initial Results**
- [ ] Train on 10k samples
- [ ] Evaluate accuracy
- [ ] Visualize predictions
- [ ] Compare with algorithm

## 🧠 Model Architecture

### Baseline CNN
- **Input**: (1, 1024, 256) grayscale image
- **Architecture**:
  - 4 Conv layers [16, 32, 64, 128 channels]
  - MaxPool after each conv
  - 2 FC layers [256, 128]
  - 2 output heads:
    - Distance: 1 value (meters)
    - Orientation: 2 values [sin(θ), cos(θ)]
- **Loss**: MSE for distance + Circular loss for orientation
- **Parameters**: ~500K

## 📈 Success Metrics

### Phase 1 Goals
- **Distance MAE** < 0.5m
- **Orientation MAE** < 15°
- **Training time** < 2 hours on GPU

### Ultimate Goals (Phase 4)
- **Distance MAE** < 0.2m
- **Orientation MAE** < 10°
- **Inference time** < 20ms per frame
- **Model size** < 50MB

## 🛠️ Configuration

All parameters are centralized in `config.py`:

- **SONAR_CONFIG**: Sonar specifications (range, FOV, resolution)
- **SCENE_CONFIG**: World and cage parameters
- **DATA_GEN_CONFIG**: Dataset generation settings
- **MODEL_CONFIG**: Neural network architecture
- **TRAINING_CONFIG**: Optimization and training parameters
- **AUGMENTATION_CONFIG**: Data augmentation settings

## 📝 Development Plan

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the complete 4-phase development roadmap.

## 🤝 Contributing

This is a research project. To add new features:

1. Update `config.py` with new parameters
2. Implement in appropriate module (models/, training/, utils/)
3. Test thoroughly before committing
4. Update this README and DEVELOPMENT_PLAN.md

## 📚 Next Steps

1. **Implement PyTorch Dataset** (`training/dataset.py`)
2. **Create training script** (`training/train.py`)
3. **Add evaluation metrics** (`training/evaluate.py`)
4. **Generate initial dataset** (run `data_generator.py`)
5. **Train baseline model** (10k samples, 100 epochs)
6. **Analyze results** and iterate

---

**Last Updated**: January 22, 2026  
**Status**: Phase 1.1 Complete - Ready for Training Pipeline
