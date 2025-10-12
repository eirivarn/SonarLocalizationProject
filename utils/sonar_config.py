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

# A small dict used by visualizer classes
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
# Core image processing configuration using adaptive linear merging
IMAGE_PROCESSING_CONFIG: Dict = {
    
    # === CV2 ENHANCEMENT ===
    # Fast OpenCV enhancement methods as alternative to custom adaptive kernels
    'use_cv2_enhancement': False,       # Toggle: False=custom adaptive, True=OpenCV methods
    'cv2_method': 'morphological',      # Method: 'morphological', 'bilateral', 'gabor'
    'cv2_kernel_size': 3,               # Kernel/filter size (3, 5, 7)

    # === ADAPTIVE LINEAR MERGING ===
    # Revolutionary approach: adapts merging kernel from circular to elliptical based on detected linearity
    # This is the core enhancement system for detecting nets, ropes, and other linear structures

    'adaptive_base_radius': 2,          # Base circular merging radius 
                                        # Starting radius for circular kernel before elongation
                                        # Larger values = more aggressive base merging

    'adaptive_max_elongation': 4,        # Maximum elongation factor
                                        # 1.0 = always circular, 4.0 = ellipse can be 4x longer than wide
                                        # Higher values = stronger linear feature enhancement

    'adaptive_linearity_threshold': 0.3, # Minimum linearity to trigger elongation 
                                        # Lower values = more pixels get elliptical kernels (more sensitive)
                                        # Higher values = only very linear patterns get elongated (selective)

    'adaptive_angle_steps': 10,           # Number of angles tested for linearity 
                                        # Uses 20° increments for optimal speed/quality balance
                                        # More steps = better angle resolution but slower processing

    'momentum_boost': 10.0,             # Enhancement strength multiplier 
                                        # Higher values = stronger directional feature enhancement
                                        # Lower values = more subtle enhancement, preserves original intensities
    
    # === CANNY EDGE DETECTION ===
    # Sharp edge detection parameters - NO GAUSSIAN BLUR applied before Canny
    'canny_low_threshold': 60,          # Lower threshold for Canny edge detection (0-255)
                                        # Lower values = more edges detected (including noise)
                                        # Higher values = only strong edges detected
    
    'canny_high_threshold': 180,        # Upper threshold for Canny edge detection (0-255)  
                                        # Ratio with low_threshold should be 2:1 to 3:1
                                        # Edges above high_threshold = definitely edges
                                        # Edges between low/high = edges if connected to strong edges
    
    # === CONTOUR FILTERING ===
    'min_contour_area': 200,            # Minimum contour area in pixels to be considered valid
                                        # Filters out small noise artifacts and debris
                                        # Typical net sections are >>200 pixels
    
    # === MORPHOLOGICAL POST-PROCESSING ===
    'morph_close_kernel': 3,            # Kernel size for morphological closing (connects nearby edges)
                                        # 0 = no closing, 3-5 = light closing, >5 = aggressive closing
                                        # Helps connect broken parts of net structures
    
    'edge_dilation_iterations': 1,      # Number of dilation iterations on final edges (0-3 typical)
                                        # Makes detected edges slightly thicker for better contour detection
                                        # 0 = no dilation, 1 = thin edges, 2+ = thick edges
}# Tracking and AOI configuration
TRACKING_CONFIG: Dict = {
    'aoi_boost_factor': 2.0,  # Reasonable boost for contours inside AOI
    'aoi_expansion_pixels': 1,
    'ellipse_smoothing_alpha': 0.2,  # Smoothing factor (0.0 = no smoothing, 1.0 = instant jump)
    'ellipse_max_movement_pixels': 4.0,  # Maximum pixels ellipse center can move per frame
    'max_frames_outside_aoi': 5,  # Max consecutive frames to allow best contour outside ellipse AOI
    'ellipse_expansion_factor': 0.2,  # Factor to expand detected contour ellipse for AOI 
    'center_smoothing_alpha': 0.2,  # Smoothing factor for center tracking (0.0 = no smoothing, 1.0 = instant jump)
    'use_elliptical_aoi': True,  # Use elliptical AOI instead of rectangular
}

# Video output configuration
VIDEO_CONFIG: Dict = {
    'fps': 15,
    'show_all_contours': True,
    'show_ellipse': True,
    'show_bounding_box': False,
    'text_scale': 0.6,
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
# Default set to the external Lacie disk exports folder per user request
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
