# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""Centralized sonar configuration for SOLAQUA.

Place common display, enhancement and image-analysis defaults here so all
sonar-related modules use the same canonical values.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

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

# Cone grid default (for rasterization)
@dataclass
class ConeGridSpec:
    img_w: int = CONE_W_DEFAULT
    img_h: int = CONE_H_DEFAULT

# --- Image-analysis / contour-detection defaults (moved from sonar_image_analysis) ---
IMAGE_PROCESSING_CONFIG: Dict = {
    
    # === BINARY CONVERSION (NEW: SIGNAL-STRENGTH INDEPENDENT) ===
    'binary_threshold': 90,            # Threshold for converting frame to binary (0-255)
                                        # Pixels > threshold become white (255), others black (0)
                                        # Makes pipeline completely signal-strength independent
                                        # Focus on structural patterns only
    
    # === ADAPTIVE LINEAR MERGING ===

    'adaptive_base_radius':2,           # Base circular merging radius 
                                        # Starting radius for circular kernel before elongation
                                        # Larger values = more aggressive base merging

    'adaptive_max_elongation': 2,        # Maximum elongation factor
                                         # 1.0 = always circular, 4.0 = ellipse can be 4x longer than wide
                                         # Higher values = stronger linear feature enhancement

    'adaptive_linearity_threshold': 0.5, # Minimum linearity to trigger elongation 
                                         # Lower values = more pixels get elliptical kernels (more sensitive)
                                         # Higher values = only very linear patterns get elongated (selective)

    'adaptive_angle_steps': 20,          # Number of angles tested for linearity 
                                         # Uses 20° increments for optimal speed/quality balance
                                         # More steps = better angle resolution but slower processing

    'momentum_boost': 20.0,              # Enhancement strength multiplier 
                                         # Higher values = stronger directional feature enhancement
                                         # Lower values = more subtle enhancement, preserves original intensities

    'use_advanced_momentum_merging': False,  # Toggle between advanced momentum merging and basic Gaussian kernel
                                             # True = use advanced structure tensor analysis with elliptical kernels
                                             # False = use simple Gaussian blur for faster processing
                                             # Advanced mode provides better linear feature detection but slower

    # === BASIC GAUSSIAN KERNEL PARAMETERS (when use_advanced_momentum_merging=False) ===
    'basic_gaussian_kernel_size': 3,        # Kernel size for basic Gaussian blur (odd integer)
                                             # Larger values = more smoothing but slower processing
                                             # Typical range: 3-9
    'basic_gaussian_sigma': 5.0,            # Sigma for basic Gaussian blur
                                             # Higher values = more aggressive smoothing
                                             # Lower values = sharper enhancement
    'basic_momentum_boost': 5.0,            # Enhancement strength for basic mode
                                             # Multiplier for how much enhanced signal to add back
                                             # Higher values = stronger enhancement
    
    # === EDGE DETECTION (SIMPLIFIED FOR BINARY DATA) ===
    # Binary edge detection using gradient operators - no Canny parameters needed
    # Binary frames use direct edge operators for cleaner, faster edge detection
    
    # === CONTOUR FILTERING ===
    'min_contour_area': 1000,            # Minimum contour area in pixels to be considered valid
                                        # Filters out small noise artifacts and debris
                                        
    
    # === MORPHOLOGICAL POST-PROCESSING ===
    'morph_close_kernel': 3,            # Kernel size for morphological closing (connects nearby edges)
                                        # 0 = no closing, 3-5 = light closing, >5 = aggressive closing
                                        # Helps connect broken parts of net structures
    
    # === DISTANCE TRACKING STABILITY ===
    'max_distance_change_pixels': 5,   # Maximum allowed distance change between frames (pixels)
                                        # Prevents tracking jumps when contour detection fails
                                        # Large jumps often indicate false positives or tracking loss
    'distance_change_smoothing': 0.05,   # Smoothing factor when distance change exceeds threshold
                                        # 0.0 = reject change completely, 1.0 = accept change fully
                                        # Intermediate values blend new and previous distance
    
    'edge_dilation_iterations': 0,      # Number of dilation iterations on final edges (0-3 typical)
                                        # Makes detected edges slightly thicker for better contour detection
                                        # 0 = no dilation, 1 = thin edges, 2+ = thick edges
}# Tracking and AOI configuration
TRACKING_CONFIG: Dict = {
    'aoi_boost_factor': 20.0,            # Reasonable boost for contours inside AOI
    'aoi_expansion_pixels': 1,           # Expand AOI by this many pixels (was 2)
    'ellipse_smoothing_alpha': 0.6,      # Ellipse temporal smoothing: 0.0=no smoothing (jittery), 0.8=very smooth, 1.0=no updates (frozen)
                                         # Controls how much the ellipse parameters (size, orientation) change between frames
                                         # Higher values = smoother tracking but slower adaptation to real changes
                                         # Lower values = faster adaptation but jittery tracking
    'ellipse_shape_smoothing_alpha': 0.1, # Shape resistance smoothing: very low value makes ellipse resist shape changes
                                          # 0.0 = no resistance (snaps to new shape immediately)
                                          # 1.0 = maximum resistance (shape never changes)
                                          # 0.1 = gradual shape adaptation over many frames
    'ellipse_max_movement_pixels': 5.0,  # Maximum pixels ellipse center can move per frame (was 4.0)
    'max_frames_outside_aoi': 3,         # Max consecutive frames to allow best contour outside ellipse AOI
    'ellipse_expansion_factor': 0.15,     # Factor to expand detected contour ellipse for AOI (was 0.2, now 0.1 = 10% expansion) (was 0.2)
    'center_smoothing_alpha': 0.3,       # Smoothing factor for center tracking (was 0.2, lower = smoother)
    'use_elliptical_aoi': True,          # Use elliptical AOI instead of rectangular
}

# Video output configuration
VIDEO_CONFIG: Dict = {
    # === VIDEO GENERATION TOGGLE ===
    'enable_video_overlay': True,      # Auto-find and overlay camera video with sonar analysis
                                        # True = automatically locate video files based on bag name
                                        # False = skip video overlay functionality
                                        # Requires both CSV (sonar data) and MP4 (camera video) files
    
    'video_topic_preference': [         # Preferred video topics (in order of preference)
        'image_compressed_image_data',  # Main camera feed (most common)
        'ted_image',                    # Auxiliary camera
        'camera_image',                 # Generic camera topic
    ],
    
    # === VIDEO OUTPUT SETTINGS ===
    'fps': 15,
    'show_all_contours': True,
    'show_ellipse': True,
    'show_bounding_box': False,
    'text_scale': 0.6,
    
    # === VIDEO SYNCHRONIZATION ===
    'max_sync_tolerance_seconds': 5.0,     # Maximum time difference for camera/sonar sync
                                            # If camera is more than this many seconds away from
                                            # sonar timestamp, fall back to sonar-only frame
                                            # Prevents frozen camera frames when video ends early
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

__all__ = [
    'FOV_DEG_DEFAULT', 'RANGE_MIN_M_DEFAULT', 'RANGE_MAX_M_DEFAULT', 'DISPLAY_RANGE_MAX_M_DEFAULT',
    'CONE_W_DEFAULT', 'CONE_H_DEFAULT', 'CONE_FLIP_VERTICAL_DEFAULT', 'CMAP_NAME_DEFAULT',
    'SONAR_VIS_DEFAULTS', 'ENHANCE_DEFAULTS', 'ConeGridSpec',
    'IMAGE_PROCESSING_CONFIG', 'TRACKING_CONFIG', 'VIDEO_CONFIG',
    'NAVIGATION_ANALYSIS_CONFIG',
    'EXPORTS_DIR_DEFAULT', 'EXPORTS_SUBDIRS',
]
