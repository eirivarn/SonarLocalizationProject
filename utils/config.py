# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""Centralized sonar configuration for SOLAQUA.

Place common display, enhancement and image-analysis defaults here so all
sonar-related modules use the same canonical values.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import hashlib
import json

# --- Geometry / display defaults ---
FOV_DEG_DEFAULT: float = 120.0
RANGE_MIN_M_DEFAULT: float = 0.0
RANGE_MAX_M_DEFAULT: float = 20.0
DISPLAY_RANGE_MAX_M_DEFAULT: float = 5.0

CONE_W_DEFAULT: int = 900
CONE_H_DEFAULT: int = 700
CONE_FLIP_VERTICAL_DEFAULT: bool = True

CMAP_NAME_DEFAULT: str = "viridis"

# A dict used by visualizer classes
SONAR_VIS_DEFAULTS: Dict = {
    "fov_deg": FOV_DEG_DEFAULT,
    "range_min_m": RANGE_MIN_M_DEFAULT,
    "range_max_m": RANGE_MAX_M_DEFAULT,
    "display_range_max_m": DISPLAY_RANGE_MAX_M_DEFAULT,
    "swap_hw": False,
    "flip_beams": False,
    "flip_range": False,
    # enhancement
    "enh_scale": "db",
    "enh_tvg": "amplitude",
    "enh_alpha_db_per_m": 0.0,
    "enh_eps_log": 1e-5,
    "enh_r0": 1e-6,
    "enh_p_low": 1.0,
    "enh_p_high": 99.5,
    "enh_gamma": 0.9,
    "enh_zero_aware": True,
    # visualization
    "cmap_raw": CMAP_NAME_DEFAULT,
    "cmap_enh": CMAP_NAME_DEFAULT,
    "figsize": (12, 5.6),
    # cone view
    "img_w": CONE_W_DEFAULT,
    "img_h": CONE_H_DEFAULT,
    "bg_color": "black",
    "n_spokes": 5,
    "rotate_deg": 0.0,
    "display_range_max_m": DISPLAY_RANGE_MAX_M_DEFAULT,
}

# --- Enhancement defaults (used by enhance_intensity) ---
ENHANCE_DEFAULTS: Dict = {
    "scale": "db",
    "tvg": "amplitude",
    "alpha_db_per_m": 0.0,
    "r0": 1e-3,
    "p_low": 1.0,
    "p_high": 99.5,
    "gamma": 0.6,
    "zero_aware": True,
    "eps_log": 1e-3,
}

# Note: ConeGridSpec is defined in utils.rendering and imported by utils.sonar_utils
# It's re-exported from sonar_utils for backward compatibility

# --- Image-analysis / contour-detection defaults ---
IMAGE_PROCESSING_CONFIG: Dict = {
    
    # === BINARY CONVERSION ===
    'binary_threshold': 1,              # Threshold for binary conversion (0-255)
                                        # Pixels > threshold become 255, others 0
                                        # Low threshold (1) means any non-zero signal is detected
    
    # === ADAPTIVE LINEAR MOMENTUM MERGING ===
    'use_advanced_momentum_merging': True,  # Use structure tensor analysis vs basic Gaussian
    
    # Advanced momentum parameters
    'adaptive_angle_steps': 20,         # Number of angles tested for linearity
    'adaptive_base_radius': 3,          # Base circular kernel radius
    'adaptive_max_elongation': 1.0,     # Maximum kernel elongation (1.0 = circular)
    'momentum_boost': 10.0,           # Enhancement strength multiplier
    'adaptive_linearity_threshold': 0.75, # Minimum linearity to trigger elongation
    'downscale_factor': 2,              # Downscaling for faster processing
    'top_k_bins': 8,                    # Top bins for orientation histogram
    'min_coverage_percent': 0.3,        # Minimum coverage for valid bins
    'gaussian_sigma': 5.0,              # Sigma for Gaussian smoothing in tensor
    
    # Basic Gaussian parameters (when use_advanced_momentum_merging=False)
    'basic_gaussian_kernel_size': 3,     # Kernel size for basic mode (odd integer)
    'basic_gaussian_sigma': 1.0,        # Sigma for basic Gaussian blur
    'basic_use_dilation': True,          # Use morphological dilation instead of blur
    'basic_dilation_kernel_size': 3,     # Kernel size for dilation (grows non-zero pixels)
    'basic_dilation_iterations': 1,      # Number of dilation iterations
    
    # === MORPHOLOGICAL POST-PROCESSING ===
    'morph_close_kernel': 0,            # Kernel size for morphological closing
                                        # 0 = disabled, 3-5 = light, >5 = aggressive
    'edge_dilation_iterations': 0,      # Edge dilation iterations (0 = disabled)
    
    # === CONTOUR FILTERING ===
    'min_contour_area': 200,            # Minimum contour area (pixels)
                                        # Reduced by 70% when tracking (0.3x multiplier)
    'aoi_boost_factor': 10.0,           # Score boost for contours inside AOI
    
    # === DISTANCE TRACKING STABILITY ===
    'max_distance_change_pixels': 20,   # Max allowed distance change between frames
}

# Tracking and AOI configuration
TRACKING_CONFIG: Dict = {
    # === ELLIPTICAL AOI SETTINGS ===
    'use_elliptical_aoi': True,
    'ellipse_expansion_factor': 0.5,  # AOI expansion (0.5 = 50% larger)
    
    # === SMOOTHING PARAMETERS ===
    # CRITICAL: Lower alpha = more smoothing
    # Formula: new = old * (1 - alpha) + measured * alpha
    # 
    # Examples:
    #   alpha = 0.01 → new = old * 0.99 + measured * 0.01  (99% old, 1% new - VERY SMOOTH)
    #   alpha = 0.30 → new = old * 0.70 + measured * 0.30  (70% old, 30% new)
    #   alpha = 1.00 → new = measured                      (0% old, 100% new - NO SMOOTHING)

    'center_smoothing_alpha': 0.8,                # 80% old, 20% new
    'ellipse_size_smoothing_alpha': 0.01,         # 99% old, 1% new (VERY SMOOTH)
    'ellipse_orientation_smoothing_alpha': 0.1,   # 80% old, 20% new
    'ellipse_max_movement_pixels': 30.0,          # Max center jump per frame
    
    # === CORRIDOR EXTENSION (LEGACY - mostly unused) ===
    'corridor_band_k': 1.0,            # Corridor width relative to minor axis (used in some legacy code)
    'corridor_length_factor': 2.0,     # Length = major axis * factor (used in some legacy code)
    'corridor_widen': 1.0,             # Legacy parameter
    'corridor_both_directions': True,  # Legacy parameter
    
    # Note: The pipeline now primarily uses elliptical AOI instead of complex corridor splitting
    
    # === PERSISTENCE ===
    'max_frames_without_detection': 30,  # Keep tracking for 30 frames without detection
    'aoi_decay_factor': 0.98,            # Grow AOI by 2% per frame when losing track
}

# Video output configuration
VIDEO_CONFIG: Dict = {
    # === VIDEO OVERLAY ===
    'enable_video_overlay': True,       # Auto-find and overlay camera video
    'video_topic_preference': [         # Preferred camera topics (in order)
        'image_compressed_image_data',
        'ted_image',
        'camera_image',
    ],
    
    # === VIDEO OUTPUT SETTINGS ===
    'fps': 15,                          # Output framerate
    'show_all_contours': True,          # Draw all detected contours
    'show_ellipse': True,               # Draw ellipse fit
    'show_bounding_box': False,         # Draw bounding box
    'text_scale': 0.6,                  # Text size
    
    # === AOI VISUALIZATION ===
    'show_aoi_corridor': True,          # Show AOI and corridor masks
    'aoi_mask_color': (0, 255, 0),      # BGR: Green
    'corridor_mask_color': (0, 128, 255), # BGR: Orange
    'aoi_mask_alpha': 0.25,             # Transparency for AOI overlay
    'corridor_mask_alpha': 0.25,        # Transparency for corridor overlay
    
    # === SYNCHRONIZATION ===
    'max_sync_tolerance_seconds': 5.0,  # Max time diff for camera/sonar sync
}

# --- Navigation analysis defaults ---
NAVIGATION_ANALYSIS_CONFIG: Dict = {
    'export_summary': False,  # Export data summary to CSV
    'export_plots': False,   # Save plots to files
    'bag_selection': None,   # None for all bags, or specify like "2024-08-22_14-06-43"
    'sensor_selection': None, # None for all sensors, or specify like ["bottomtrack", "ins"]
    'plot_style': 'static',   # 'static' or 'interactive' for plots
}

# --- Repository / on-disk layout defaults ---
# Centralize the default exports directory name so all tools use the same
# path (can be overridden by CLI args or environment-specific settings).
EXPORTS_DIR_DEFAULT: str = "/Volumes/LaCie/SOLAQUA/exports"

# Standard subfolders created/used under the exports dir. Tools should join
# these with the configured exports dir to read/write outputs.
EXPORTS_SUBDIRS: Dict = {
    'by_bag': 'by_bag',
    'videos': 'videos',
    'frames': 'frames',
    'camera_info': 'camera_info',
    'outputs': 'outputs',
    'index': ''  # index files live in the root exports dir
}

def validate_config_consistency():
    """Validate that all config values are consistent and print warnings if not."""
    required_image_keys = {
        'binary_threshold', 'adaptive_angle_steps', 'adaptive_base_radius',
        'adaptive_max_elongation', 'momentum_boost', 'adaptive_linearity_threshold',
        'downscale_factor', 'top_k_bins', 'min_coverage_percent', 'gaussian_sigma',
        'morph_close_kernel', 'edge_dilation_iterations', 'min_contour_area',
        'max_distance_change_pixels', 'use_advanced_momentum_merging', 'aoi_boost_factor'
    }
    
    required_tracking_keys = {
        'ellipse_expansion_factor', 'center_smoothing_alpha', 'ellipse_size_smoothing_alpha',
        'ellipse_orientation_smoothing_alpha', 'ellipse_max_movement_pixels',
        'corridor_band_k', 'corridor_length_factor', 'corridor_widen', 'corridor_both_directions',
        'use_corridor_splitting', 'use_elliptical_aoi'
    }
    
    missing_image = required_image_keys - set(IMAGE_PROCESSING_CONFIG.keys())
    missing_tracking = required_tracking_keys - set(TRACKING_CONFIG.keys())
    
    if missing_image or missing_tracking:
        print(" CONFIG VALIDATION WARNING:")
        if missing_image:
            print(f"   Missing IMAGE_PROCESSING_CONFIG keys: {missing_image}")
        if missing_tracking:
            print(f"   Missing TRACKING_CONFIG keys: {missing_tracking}")
        return False
    
    return True

# Auto-validate on import
validate_config_consistency()

def get_config_hash():
    """Get hash of current config for change detection."""
    config_data = {
        'tracking': TRACKING_CONFIG,
        'image_processing': IMAGE_PROCESSING_CONFIG,
        'EXPORTS_DIR_DEFAULT': EXPORTS_DIR_DEFAULT,
        'EXPORTS_SUBDIRS': EXPORTS_SUBDIRS
    }
    config_str = json.dumps(config_data, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

_CONFIG_HASH = get_config_hash()  # Initial hash

def validate_config_unchanged():
    """Check if config has changed since last validation."""
    current_hash = get_config_hash()
    if current_hash != _CONFIG_HASH:
        print("⚠️  CONFIG HAS CHANGED! Results may not be comparable.")
        print(f"   Previous hash: {_CONFIG_HASH[:8]}...")
        print(f"   Current hash:  {current_hash[:8]}...")
    return current_hash == _CONFIG_HASH

class ConfigManager:
    """Centralized config access to ensure consistency."""
    
    def __init__(self):
        self._image_config = IMAGE_PROCESSING_CONFIG.copy()
        self._tracking_config = TRACKING_CONFIG.copy()
        self._hash = self._calculate_hash()
    
    def _calculate_hash(self):
        import hashlib
        import json
        data = {'image': self._image_config, 'tracking': self._tracking_config}
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def get_image_param(self, key, default=None):
        """Get image processing parameter with validation."""
        if key not in self._image_config and default is None:
            raise KeyError(f"Required IMAGE_PROCESSING_CONFIG key '{key}' not found")
        return self._image_config.get(key, default)
    
    def get_tracking_param(self, key, default=None):
        """Get tracking parameter with validation."""
        if key not in self._tracking_config and default is None:
            raise KeyError(f"Required TRACKING_CONFIG key '{key}' not found")
        return self._tracking_config.get(key, default)
    
    def validate_consistency(self):
        """Validate config hasn't changed."""
        current_hash = self._calculate_hash()
        if current_hash != self._hash:
            raise RuntimeError("Config has been modified! Create new ConfigManager instance.")
        return True

# Global instance
config_manager = ConfigManager()

__all__ = [
    'SONAR_VIS_DEFAULTS', 'ENHANCE_DEFAULTS', 'ConeGridSpec',
    'CONE_W_DEFAULT', 'CONE_H_DEFAULT', 'CONE_FLIP_VERTICAL_DEFAULT', 'CMAP_NAME_DEFAULT',
    'FOV_DEG_DEFAULT', 'RANGE_MIN_M_DEFAULT', 'RANGE_MAX_M_DEFAULT', 'DISPLAY_RANGE_MAX_M_DEFAULT',
    'IMAGE_PROCESSING_CONFIG', 'TRACKING_CONFIG', 'VIDEO_CONFIG',
    'EXPORTS_DIR_DEFAULT', 'EXPORTS_SUBDIRS',
    'validate_config_consistency', 'get_config_hash', 'validate_config_unchanged',
    'config_manager'
]