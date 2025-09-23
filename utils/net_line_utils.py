"""
Net Line Overlay Utilities for SOLAQUA Video Generation

This module provides utilities for adding net line overlays to sonar video frames,
including net position and angle calculations from navigation data.
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import utils.sonar_distance_analysis as sda


def load_net_data_for_bag(target_bag: str, exports_folder: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """
    Load navigation and guidance data for net line calculations.
    
    Args:
        target_bag: Bag identifier (e.g., "2024-08-22_14-06-43")
        exports_folder: Path to exports folder
        
    Returns:
        Tuple of (nav_data, guidance_data, distance_measurements)
    """
    try:
        raw_data, distance_measurements = sda.load_all_distance_data_for_bag(target_bag, exports_folder)
        nav_data = raw_data['navigation']
        guidance_data = raw_data['guidance']
        return nav_data, guidance_data, distance_measurements
    except Exception as e:
        print(f"Warning: Could not load net data for bag {target_bag}: {e}")
        return None, None, {}


def get_net_line_info(sonar_timestamp: pd.Timestamp, 
                      nav_data: Optional[pd.DataFrame],
                      guidance_data: Optional[pd.DataFrame], 
                      distance_measurements: Dict,
                      exports_folder: str,
                      target_bag: str,
                      max_distance: float = 10.0) -> Optional[Dict[str, Any]]:
    """
    Calculate net line position and angle for a given timestamp.
    
    Args:
        sonar_timestamp: Target timestamp for measurements
        nav_data: Navigation DataFrame
        guidance_data: Guidance DataFrame  
        distance_measurements: Distance measurement data
        exports_folder: Path to exports folder
        target_bag: Bag identifier
        max_distance: Maximum display distance
        
    Returns:
        Dictionary with net line info or None if not available
    """
    if nav_data is None:
        return None
    
    # Get distance measurements for this timestamp
    distance_data = sda.collect_distance_measurements_at_timestamp(
        sonar_timestamp, nav_data, guidance_data, distance_measurements
    )
    
    if not distance_data or 'Navigation NetDistance' not in distance_data:
        return None
    
    distance = distance_data['Navigation NetDistance']['value']
    if distance > max_distance:
        return None
    
    # Get net pitch angle
    net_angle_rad = 0.0
    net_angle_deg = 0.0
    
    # Try navigation data (NetPitch) - LOCAL ORIENTATION
    if nav_data is not None and 'NetPitch' in nav_data.columns:
        nav_time_diffs = abs(nav_data['timestamp'] - sonar_timestamp)
        min_time_diff = nav_time_diffs.min()
        
        if min_time_diff <= pd.Timedelta('5s'):
            nav_idx = nav_time_diffs.idxmin()
            net_angle_rad = nav_data.loc[nav_idx, 'NetPitch']
            net_angle_deg = np.degrees(net_angle_rad)
    
    # Direct file load if still no angle
    if abs(net_angle_deg) < 0.1:
        nav_file = Path(exports_folder) / "by_bag" / f"navigation_plane_approximation__{target_bag}_data.csv"
        if nav_file.exists():
            direct_nav_df = pd.read_csv(nav_file)
            direct_nav_df['timestamp'] = pd.to_datetime(direct_nav_df['ts_utc'])
            
            direct_time_diffs = abs(direct_nav_df['timestamp'] - sonar_timestamp)
            direct_min_diff = direct_time_diffs.min()
            
            if direct_min_diff <= pd.Timedelta('5s'):
                direct_idx = direct_time_diffs.idxmin()
                net_angle_rad = direct_nav_df.loc[direct_idx, 'NetPitch']
                net_angle_deg = np.degrees(net_angle_rad)
    
    return {
        'distance': distance,
        'angle_rad': net_angle_rad,
        'angle_deg': net_angle_deg,
        'data': distance_data
    }


def calculate_net_line_coordinates(net_info: Dict[str, Any], 
                                   net_half_width: float = 2.0) -> Tuple[float, float, float, float]:
    """
    Calculate rotated net line coordinates in meters.
    
    Args:
        net_info: Net information dictionary from get_net_line_info
        net_half_width: Half width of net line in meters
        
    Returns:
        Tuple of (x1, y1, x2, y2) coordinates in meters
    """
    distance = net_info['distance']
    net_angle_rad = net_info['angle_rad']
    
    # Original cross-track oriented line points (0° reference = parallel to x-axis)
    original_x1 = -net_half_width
    original_y1 = 0  # Start at origin level
    original_x2 = net_half_width  
    original_y2 = 0  # Start at origin level
    
    # Rotate by NEGATIVE net pitch angle (0° = parallel to x-axis)
    cos_angle = np.cos(-net_angle_rad)  # Use negative angle
    sin_angle = np.sin(-net_angle_rad)  # Use negative angle
    
    # Apply rotation matrix to both endpoints
    rotated_x1 = original_x1 * cos_angle - original_y1 * sin_angle
    rotated_y1 = original_x1 * sin_angle + original_y1 * cos_angle
    rotated_x2 = original_x2 * cos_angle - original_y2 * sin_angle
    rotated_y2 = original_x2 * sin_angle + original_y2 * cos_angle
    
    # Translate the rotated line to the net distance
    rotated_x1 += 0  # No x offset
    rotated_y1 += distance  # Move to net distance
    rotated_x2 += 0  # No x offset  
    rotated_y2 += distance  # Move to net distance
    
    return rotated_x1, rotated_y1, rotated_x2, rotated_y2


def draw_net_line_on_cone(cone_bgr: np.ndarray, 
                          net_info: Optional[Dict[str, Any]], 
                          x_px_func, 
                          y_px_func,
                          x_offset: int = 0,
                          line_color: Tuple[int, int, int] = (0, 255, 255),  # Bright yellow in BGR
                          outline_color: Tuple[int, int, int] = (0, 0, 0),  # Black outline
                          line_thickness: int = 8,  # Even thicker for better visibility
                          outline_thickness: int = 4,  # Thicker outline
                          endpoint_radius: int = 10) -> np.ndarray:  # Larger endpoints
    """
    Draw the net line on the cone image.
    
    Args:
        cone_bgr: Cone image in BGR format
        net_info: Net information dictionary or None
        x_px_func: Function to convert x meters to pixels
        y_px_func: Function to convert y meters to pixels
        x_offset: X pixel offset for drawing
        line_color: BGR color for net line
        outline_color: BGR color for outline
        line_thickness: Thickness of main line
        outline_thickness: Thickness of outline
        endpoint_radius: Radius of endpoint circles
        
    Returns:
        Modified cone image with net line overlay
    """
    if net_info is None:
        return cone_bgr
    
    # Calculate net line coordinates in meters
    x1_m, y1_m, x2_m, y2_m = calculate_net_line_coordinates(net_info)
    
    # Convert to pixel coordinates
    x1_px = x_px_func(x1_m) + x_offset
    y1_px = y_px_func(y1_m)
    x2_px = x_px_func(x2_m) + x_offset
    y2_px = y_px_func(y2_m)
    
    # Draw black outline first for better visibility
    cv2.line(cone_bgr, (x1_px, y1_px), (x2_px, y2_px), outline_color, line_thickness + outline_thickness, cv2.LINE_AA)
    cv2.circle(cone_bgr, (x1_px, y1_px), endpoint_radius + outline_thickness, outline_color, -1, cv2.LINE_AA)
    cv2.circle(cone_bgr, (x2_px, y2_px), endpoint_radius + outline_thickness, outline_color, -1, cv2.LINE_AA)
    
    # Draw the main net line on top (bright yellow)
    cv2.line(cone_bgr, (x1_px, y1_px), (x2_px, y2_px), line_color, line_thickness, cv2.LINE_AA)
    
    # Draw endpoints (bright yellow circles)
    cv2.circle(cone_bgr, (x1_px, y1_px), endpoint_radius, line_color, -1, cv2.LINE_AA)
    cv2.circle(cone_bgr, (x2_px, y2_px), endpoint_radius, line_color, -1, cv2.LINE_AA)
    
    return cone_bgr


def format_net_info_text(net_info: Optional[Dict[str, Any]]) -> str:
    """
    Format net information for display in video text overlay.
    
    Args:
        net_info: Net information dictionary or None
        
    Returns:
        Formatted string for display
    """
    if net_info is None:
        return ""
    
    return f"  |  Net: {net_info['distance']:.2f}m @ {net_info['angle_deg']:.1f}°"