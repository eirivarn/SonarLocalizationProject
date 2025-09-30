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
IMAGE_PROCESSING_CONFIG: Dict = {
    'use_momentum_merging': True,
    'momentum_search_radius': 3,
    'momentum_threshold': 0.1,
    'momentum_decay': 0.9,
    'momentum_boost': 5.0,
    'blur_kernel_size': (31, 31),
    'canny_low_threshold': 40,
    'canny_high_threshold': 120,
    'min_contour_area': 100,
    'morph_close_kernel': 5,
    'edge_dilation_iterations': 2,
}

ELONGATION_CONFIG: Dict = {
    'aspect_ratio_weight': 0.4,
    'ellipse_elongation_weight': 0.7,
    'solidity_weight': 0.1,
    'extent_weight': 0.0,
    'perimeter_weight': 0.0,
}

TRACKING_CONFIG: Dict = {
    'aoi_boost_factor': 1000.0,
    'aoi_expansion_pixels': 10,
}

VIDEO_CONFIG: Dict = {
    'fps': 15,
    'show_all_contours': True,
    'show_ellipse': True,
    'show_bounding_box': True,
    'text_scale': 0.6,
}

__all__ = [
    'FOV_DEG_DEFAULT', 'RANGE_MIN_M_DEFAULT', 'RANGE_MAX_M_DEFAULT', 'DISPLAY_RANGE_MAX_M_DEFAULT',
    'CONE_W_DEFAULT', 'CONE_H_DEFAULT', 'CONE_FLIP_VERTICAL_DEFAULT', 'CMAP_NAME_DEFAULT',
    'SONAR_VIS_DEFAULTS', 'ENHANCE_DEFAULTS', 'ConeGridSpec',
    'IMAGE_PROCESSING_CONFIG', 'ELONGATION_CONFIG', 'TRACKING_CONFIG', 'VIDEO_CONFIG',
]
