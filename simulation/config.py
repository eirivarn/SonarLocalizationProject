"""
Centralized configuration for neural network net detection simulation.
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
SIMULATION_DIR = Path(__file__).parent
DATASET_DIR = SIMULATION_DIR / "datasets"
DATASET_PATH = DATASET_DIR  # Alias for compatibility
MODELS_DIR = SIMULATION_DIR / "models_saved"
RESULTS_DIR = SIMULATION_DIR / "results"

# ============================================================================
# SONAR SPECIFICATIONS (Match SOLAQUA dataset)
# ============================================================================
SONAR_CONFIG = {
    'range_m': 20.0,           # Maximum range in meters
    'fov_deg': 120.0,          # Field of view in degrees
    'num_beams': 256,          # Number of beams (width)
    'range_bins': 1024,        # Number of range bins (height)
    'voxel_size': 0.1,         # Voxel size in meters
}

# ============================================================================
# WORLD/SCENE PARAMETERS
# ============================================================================
SCENE_CONFIG = {
    'world_size_m': 30.0,      # World is 30m x 30m
    'cage_center': [15.0, 15.0],  # Center of cage
    'cage_radius': 12.0,       # Cage radius in meters
    'num_sides': 24,           # Number of cage panel segments
    'current_strength': 6.5,   # Current deflection strength
    'net_sag': 0.25,          # Maximum net sag in meters
    
    # Fish parameters
    'num_fish': 150,           # Number of fish in cage
    'fish_length_range': [0.4, 0.6],  # Fish length range [min, max]
    'fish_width_ratio': 0.20,  # Width as fraction of length
    
    # Biomass/fouling
    'num_biomass_patches': 40,
    'biomass_size_range': [0.2, 0.6],  # Patch size range
}

# ============================================================================
# DATA GENERATION PARAMETERS
# ============================================================================
DATA_GEN_CONFIG = {
    # Sampling ranges
    'distance_range': [0.5, 5.0],      # Distance from net (meters)
    'angle_range': [-60, 60],          # Approach angle range (degrees)
    'height_range': [0.5, 3.0],        # Height variation (meters)
    
    # Dataset sizes
    'num_train': 10000,
    'num_val': 2000,
    'num_test': 1000,
    
    # Randomization
    'randomize_fish': True,
    'randomize_biomass': True,
    'vary_current': True,
    'current_range': [3.0, 10.0],     # Current strength variation
    
    # Output format
    'save_format': 'npz',              # 'npz' for heatmap training
    'normalize': True,                 # Normalize images to [0, 1]
    'log_scale': True,                 # Apply log scaling
    
    # Heatmap training parameters
    'image_size': (512, 512),          # Cartesian image size
    'heatmap_sigma_pixels': 3.0,       # Gaussian width for heatmap
    'direction_radius_pixels': 10,     # Supervision radius for direction
}

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================
MODEL_CONFIG = {
    # Input/Output
    'input_shape': (1, 1024, 256),     # (C, H, W)
    'output_dim': 2,                   # [distance, orientation]
    
    # Baseline CNN
    'baseline': {
        'conv_channels': [16, 32, 64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'dropout': 0.2,
        'fc_dims': [256, 128],
    },
    
    # Advanced CNN
    'advanced': {
        'use_residual': True,
        'use_batch_norm': True,
        'use_attention': False,
        'conv_channels': [32, 64, 128, 256, 512],
        'dropout': 0.3,
        'fc_dims': [512, 256],
    },
}

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
TRAINING_CONFIG = {
    # Optimization
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'patience': 15,                    # Early stopping patience
    
    # Loss weights
    'distance_weight': 1.0,
    'orientation_weight': 0.5,         # Less critical than distance
    
    # Loss function
    'distance_loss': 'mse',            # 'mse', 'mae', or 'huber'
    'orientation_loss': 'circular',    # 'mse', 'mae', or 'circular'
    
    # Scheduler
    'scheduler': 'plateau',            # 'plateau', 'step', or 'cosine'
    'scheduler_params': {
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-6,
    },
    
    # Checkpointing
    'save_best': True,
    'save_every': 10,                  # Save every N epochs
    'metric': 'val_loss',              # Metric for best model
}

# ============================================================================
# AUGMENTATION PARAMETERS
# ============================================================================
AUGMENTATION_CONFIG = {
    'enabled': True,
    
    # Intensity augmentation
    'gamma_range': [0.8, 1.2],         # Gamma correction
    'noise_std': 0.02,                 # Gaussian noise
    'salt_pepper_prob': 0.01,          # Salt & pepper noise
    
    # Geometric augmentation (careful with labels!)
    'horizontal_flip': False,          # Don't flip - breaks orientation
    'vertical_shift': 0.05,            # Small vertical shifts OK
    'rotation_degrees': 0,             # Don't rotate - polar image
    
    # Occlusion simulation
    'random_occlusion': True,
    'occlusion_prob': 0.1,
    'occlusion_size': [50, 200],       # Size range in pixels
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVAL_CONFIG = {
    'metrics': ['mae', 'rmse', 'r2'],
    
    # Thresholds for "good" predictions
    'distance_threshold_m': 0.3,      # < 30cm is good
    'orientation_threshold_deg': 10.0,  # < 10° is good
    
    # Visualization
    'plot_predictions': True,
    'plot_errors': True,
    'plot_attention': True,            # If model has attention
    'num_vis_samples': 20,
}

# ============================================================================
# LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'use_tensorboard': True,
    'use_wandb': False,                # Set to True if using Weights & Biases
    'log_dir': RESULTS_DIR / 'logs',
    'log_every': 10,                   # Log every N batches
    'save_model_graph': True,
}

# ============================================================================
# GROUND TRUTH ALGORITHM (for comparison)
# ============================================================================
ALGORITHM_CONFIG = {
    'binary_threshold': 128,
    'adaptive_angle_steps': 20,
    'adaptive_base_radius': 3,
    'adaptive_max_elongation': 1.0,
    'momentum_boost': 10.0,
    'adaptive_linearity_threshold': 0.75,
    'ellipse_expansion_factor': 0.5,
    'center_smoothing_alpha': 0.8,
    'ellipse_size_smoothing_alpha': 0.01,
}
