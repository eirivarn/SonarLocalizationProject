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
# Core image processing configuration - NO BLURRING, sharp objects
IMAGE_PROCESSING_CONFIG: Dict = {
    # Momentum merging (core feature) - connects net parts
    'use_momentum_merging': True,
    'momentum_search_radius': 1,
    'momentum_threshold': 0.1,
    'momentum_decay': 0.9,
    'momentum_boost': 10.0,
    
    # Sharp edge detection - NO BLUR
    'canny_low_threshold': 60,
    'canny_high_threshold': 180,
    'min_contour_area': 200,
    'morph_close_kernel': 3,       # Minimal closing to connect very close edges
    'edge_dilation_iterations': 1,
    
    # === PIXEL OWNERSHIP TRACKING TOGGLE ===
    'use_pixel_ownership': False,   # MASTER SWITCH: Enable/disable pixel ownership tracking system
                                   # Set to False to disable object separation and speed up processing
                                   # 
                                   # TRUE  = Prevents fish/debris from merging with net (slower, more accurate)
                                   # FALSE = Faster processing but objects may merge together
                                   #
                                   # You can change this setting dynamically:
                                   # IMAGE_PROCESSING_CONFIG['use_pixel_ownership'] = False  # for speed
                                   # IMAGE_PROCESSING_CONFIG['use_pixel_ownership'] = True   # for accuracy
}

# Tracking and AOI configuration
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
