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

# === SONAR VISUALIZATION DEFAULTS ===
# Used by sonar visualization classes for consistent display settings
SONAR_VIS_DEFAULTS: Dict = {
    # Geometry settings
    "fov_deg": FOV_DEG_DEFAULT,                    # Field of view in degrees (120°)
    "range_min_m": RANGE_MIN_M_DEFAULT,            # Minimum range in meters (0m)
    "range_max_m": RANGE_MAX_M_DEFAULT,            # Maximum range in meters (20m)
    "display_range_max_m": DISPLAY_RANGE_MAX_M_DEFAULT,  # Display range limit (5m)
    
    # Data orientation flags
    "swap_hw": False,                              # Swap height/width dimensions
    "flip_beams": False,                           # Flip beam direction
    "flip_range": False,                           # Flip range direction
    
    # Enhancement parameters for sonar data processing
    "enh_scale": "db",                             # Enhancement scale type (decibels)
    "enh_tvg": "amplitude",                        # Time-varying gain type
    "enh_alpha_db_per_m": 0.0,                     # Attenuation coefficient
    "enh_eps_log": 1e-5,                           # Small value for log calculations
    "enh_r0": 1e-6,                                # Reference range
    "enh_p_low": 1.0,                              # Lower percentile for enhancement
    "enh_p_high": 99.5,                            # Upper percentile for enhancement
    "enh_gamma": 0.9,                              # Gamma correction factor
    "enh_zero_aware": True,                        # Handle zero values specially
    
    # Visualization settings
    "cmap_raw": CMAP_NAME_DEFAULT,                 # Colormap for raw data (viridis)
    "cmap_enh": CMAP_NAME_DEFAULT,                 # Colormap for enhanced data (viridis)
    "figsize": (12, 5.6),                          # Figure size for plots
    
    # Cone view specific settings
    "img_w": CONE_W_DEFAULT,                       # Image width (900 pixels)
    "img_h": CONE_H_DEFAULT,                       # Image height (700 pixels)
    "bg_color": "black",                           # Background color
    "n_spokes": 5,                                 # Number of range spokes
    "rotate_deg": 0.0,                             # Rotation angle
    "display_range_max_m": DISPLAY_RANGE_MAX_M_DEFAULT,  # Max display range (5m)
}

# === ENHANCEMENT DEFAULTS ===
# Used by enhance_intensity function for sonar data enhancement
ENHANCE_DEFAULTS: Dict = {
    "scale": "db",           # Enhancement scale: "db" (decibels) or "linear"
    "tvg": "amplitude",      # Time-varying gain: "amplitude" or "power"
    "alpha_db_per_m": 0.0,   # Attenuation coefficient in dB per meter
    "r0": 1e-3,              # Reference range in meters
    "p_low": 1.0,            # Lower percentile for contrast stretching
    "p_high": 99.5,          # Upper percentile for contrast stretching
    "gamma": 0.6,            # Gamma correction factor (0.6 = slight brightening)
    "zero_aware": True,      # Handle zero/negative values specially
    "eps_log": 1e-3,         # Small epsilon for log calculations
}

# === CONE GRID SPECIFICATION ===
# Default image dimensions for sonar cone rasterization
@dataclass
class ConeGridSpec:
    img_w: int = CONE_W_DEFAULT   # Image width in pixels (900)
    img_h: int = CONE_H_DEFAULT   # Image height in pixels (700)

# --- Image-analysis / contour-detection defaults (moved from sonar_image_analysis) ---
# Core image processing configuration for sonar net detection
IMAGE_PROCESSING_CONFIG: Dict = {
    # === MOMENTUM MERGING PARAMETERS ===
    # Controls how broken net segments are connected together
    'use_momentum_merging': True,     # Enable momentum-based pixel merging (CORE FEATURE)
    'momentum_search_radius': 10,      # Pixel radius to search for connections (1-10 range)
                                      # ↑ Higher = stronger connections, slower processing
                                      # ↓ Lower = faster but may miss broken net segments
    'momentum_threshold': 0.1,        # Minimum intensity difference for merging (0.0-1.0)
                                      # ↑ Higher = more selective merging, cleaner results
                                      # ↓ Lower = more aggressive merging, may connect noise
    'momentum_decay': 0.9,            # How momentum fades with distance (0.0-1.0)
                                      # ↑ Higher = connections reach further pixels
                                      # ↓ Lower = connections stay more localized
    'momentum_boost': 10.0,           # Strength multiplier for momentum connections
                                      # ↑ Higher = stronger net connectivity, may over-connect
                                      # ↓ Lower = weaker connections, may miss net parts
    
    # === EDGE DETECTION PARAMETERS ===
    # Controls how edges are detected in sonar images - optimized for sharp net edges
    'canny_low_threshold': 60,        # Lower Canny threshold (increase to reduce noise)
                                      # ↑ Higher = cleaner edges, may miss weak net parts
                                      # ↓ Lower = more edges detected, more noise included
    'canny_high_threshold': 180,      # Upper Canny threshold (increase for cleaner edges)
                                      # ↑ Higher = only strongest edges, very clean but may miss net
                                      # ↓ Lower = more edge candidates, noisier but catches weak nets
    'min_contour_area': 200,          # Minimum pixel area to consider as valid contour
                                      # ↑ Higher = filters out small noise, may miss small net parts
                                      # ↓ Lower = detects smaller objects, more noise/false positives
    'morph_close_kernel': 1,          # Kernel size for closing gaps in contours (3-7 range)
                                      # ↑ Higher = closes larger gaps, may connect separate objects
                                      # ↓ Lower = preserves gaps, may leave net broken
    'edge_dilation_iterations': 1,    # Additional edge strengthening iterations
                                      # ↑ Higher = thicker/stronger edges, may merge nearby objects
                                      # ↓ Lower (0) = thinner edges, may lose weak net connections
    
    # === PIXEL OWNERSHIP TRACKING TOGGLE ===
    'use_pixel_ownership': False,     # MASTER SWITCH: Enable/disable pixel ownership tracking
                                      # TRUE  = Prevents fish/debris merging with net (slower, accurate)
                                      # FALSE = Faster processing, but objects may merge together
                                      # Toggle dynamically: IMAGE_PROCESSING_CONFIG['use_pixel_ownership'] = True/False
    
    # === DISTANCE VALIDATION FILTER ===
    'use_distance_validation': True,  # Filter invalid distance measurements
                                      # TRUE  = Uses last valid distance when current is negative/out of bounds
                                      # FALSE = No filtering, raw distance values used (may have jumps/errors)
}

# === OBJECT TRACKING CONFIGURATION ===
# Controls Area of Interest (AOI) expansion around detected nets
TRACKING_CONFIG: Dict = {
    'aoi_expansion_pixels': 15,       # Pixels to expand AOI around detected net 
                                      # ↑ Higher = larger search area, more stable but slower
                                      # ↓ Lower = tighter tracking, faster but may lose net
    'aoi_boost_factor': 2.0,          # Score boost for contours inside AOI 
                                      # ↑ Higher = strongly prefers previous net location
                                      # ↓ Lower = more willing to jump to new net locations
    'ellipse_smoothing_alpha': 0.3,   # Smoothing for ellipse center movement (ACTIVELY USED) 
                                      # ↑ Higher (→1.0) = instant jumps, very responsive but jittery
                                      # ↓ Lower (→0.0) = very smooth movement, slow to adapt
                                      # RECOMMENDED: 0.1-0.3 for stable tracking
    'ellipse_max_movement_pixels': 8.0,  # Max ellipse center movement per frame (ACTIVELY USED)
                                         # ↑ Higher = allows faster net movement, may jump to noise
                                         # ↓ Lower = prevents jumping/flickering, very stable tracking  
                                         # RECOMMENDED: 2.0-6.0 pixels for smooth movement
    'max_frames_outside_aoi': 5,      # Max frames to track contour outside AOI 
                                      # ↑ Higher = more persistent tracking of lost nets
                                      # ↓ Lower = quickly abandons tracking when net moves away
    'ellipse_expansion_factor': 0.1,  # Factor to expand ellipse for AOI 
                                      # ↑ Higher = larger elliptical AOI, more stable tracking
                                      # ↓ Lower = tighter elliptical AOI, more precise but may lose net
    'ellipse_aspect_ratio': 5.0,     # Aspect ratio modification for ellipse (major/minor axis ratio)
                                      # ↑ Higher (>1.0) = elongated ellipse, thinner and longer 
                                      # ↓ Lower (<1.0) = wider ellipse, shorter and fatter
                                      # 1.0 = no modification, uses natural contour ellipse
                                      # RECOMMENDED: 1.5-3.0 for fishing net tracking
}

# === EXCLUSION ZONE CONFIGURATION ===
# Prevents fish/debris from being included in net detection 
EXCLUSION_CONFIG: Dict = {
    'enable_exclusions': False,        # Master switch for exclusion system 
    'min_secondary_area': 100,        # Min area for objects to create exclusion zones 
                                      # ↑ Higher = only large fish/debris create exclusions
                                      # ↓ Lower = small objects also create exclusions, may over-exclude
    'exclusion_radius': 5,            # Pixel radius around excluded objects 
                                      # ↑ Higher = larger exclusion zones, may exclude valid net parts
                                      # ↓ Lower = smaller exclusions, fish may still merge with net
    'max_exclusion_zones': 5,         # Max number of exclusion zones to track 
                                      # ↑ Higher = tracks more objects, more computation
                                      # ↓ Lower = fewer tracked objects, may miss some fish/debris
    'zone_decay_frames': 3,           # Frames to keep exclusion zones active 
                                      # ↑ Higher = longer memory of fish locations, more stable
                                      # ↓ Lower = shorter memory, adapts faster to moving objects
}

# === VIDEO OUTPUT CONFIGURATION ===
# Controls visualization in generated analysis videos 
VIDEO_CONFIG: Dict = {
    'fps': 15,                        # Video frame rate for output files 
                                      # ↑ Higher = smoother video, larger file size
                                      # ↓ Lower = choppier video, smaller file size
    'show_all_contours': True,        # Display all detected contours 
                                      # True = shows all objects for debugging
                                      # False = cleaner display, only shows best net
    'show_ellipse': True,             # Show fitted ellipse for elongated objects 
                                      # True = visualizes net shape and orientation
                                      # False = cleaner display without ellipse overlay
    'show_bounding_box': False,       # Show bounding boxes 
                                      # True = shows rectangular bounds around objects
                                      # False = cleaner display without boxes
    'text_scale': 0.6,                # Scale factor for overlay text 
                                      # ↑ Higher = larger text, more readable but may obscure image
                                      # ↓ Lower = smaller text, less intrusive but harder to read
}

# === NAVIGATION ANALYSIS CONFIGURATION ===
# Controls navigation data analysis and export settings 
NAVIGATION_ANALYSIS_CONFIG: Dict = {
    'export_summary': False,   # Export data summary to CSV files
    'export_plots': False,     # Save analysis plots to image files
    'bag_selection': None,     # None for all bags, or specify bag ID like "2024-08-22_14-06-43"
    'sensor_selection': None,  # None for all sensors, or list like ["bottomtrack", "ins"]
    'plot_style': 'static',    # 'static' for matplotlib or 'interactive' for plotly
}

# === FILE SYSTEM LAYOUT CONFIGURATION ===
# Centralized paths for consistent file organization across all tools
EXPORTS_DIR_DEFAULT: str = "/Volumes/LaCie/SOLAQUA/exports"  # Main exports directory (external drive)

# Standard subdirectories created under exports directory
EXPORTS_SUBDIRS: Dict = {
    'by_bag': 'by_bag',         # Per-bag analysis outputs (CSV files, etc.)
    'videos': 'videos',         # Generated analysis videos  
    'frames': 'frames',         # Extracted frame images
    'camera_info': 'camera_info',  # Camera calibration data
    'outputs': 'outputs',       # NPZ files and processed data
    'index': ''                 # Index files (live in root exports directory)
}

# === EXPORTS FOR OTHER MODULES ===
# Public API - only export actively used configurations
__all__ = [
    # Geometry constants
    'FOV_DEG_DEFAULT', 'RANGE_MIN_M_DEFAULT', 'RANGE_MAX_M_DEFAULT', 'DISPLAY_RANGE_MAX_M_DEFAULT',
    'CONE_W_DEFAULT', 'CONE_H_DEFAULT', 'CONE_FLIP_VERTICAL_DEFAULT', 'CMAP_NAME_DEFAULT',
    
    # Main configuration dictionaries (ALL ACTIVELY USED)
    'SONAR_VIS_DEFAULTS',           # Sonar visualization settings
    'ENHANCE_DEFAULTS',             # Data enhancement parameters
    'IMAGE_PROCESSING_CONFIG',      # Core image analysis settings
    'TRACKING_CONFIG',              # Object tracking parameters
    'EXCLUSION_CONFIG',             # Exclusion zone settings
    'VIDEO_CONFIG',                 # Video output settings
    'NAVIGATION_ANALYSIS_CONFIG',   # Navigation analysis settings
    
    # Utility classes and paths
    'ConeGridSpec',                 # Image grid specification
    'EXPORTS_DIR_DEFAULT',          # File system paths
    'EXPORTS_SUBDIRS',
]
