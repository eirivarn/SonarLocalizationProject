"""
FFT Net Position Integration Patch for Video Generation

This module provides FFT overlay functionality that can be integrated 
into the existing export_optimized_sonar_video function.
"""

import pandas as pd
import numpy as np
import cv2

def add_fft_overlay_to_frame(overlay, fft_data, current_time, W, H, pixels_per_meter, 
                           tolerance_seconds=0.5):
    """
    Add FFT net position overlay to video frame.
    
    Args:
        overlay: OpenCV image array to draw on
        fft_data: DataFrame with FFT data (timestamp, distance_m, pitch_deg)
        current_time: Current frame timestamp
        W, H: Frame dimensions
        pixels_per_meter: Conversion factor
        tolerance_seconds: Time tolerance for matching FFT data
    """
    if fft_data is None or current_time is None:
        return
    
    # Find closest FFT measurement in time
    time_diffs = abs(fft_data['timestamp'] - current_time)
    closest_idx = time_diffs.idxmin()
    
    if time_diffs.loc[closest_idx] <= pd.Timedelta(seconds=tolerance_seconds):
        fft_row = fft_data.loc[closest_idx]
        fft_distance = fft_row.get('distance_m', np.nan)
        fft_pitch_deg = fft_row.get('pitch_deg', np.nan)
        
        if not pd.isna(fft_distance) and not pd.isna(fft_pitch_deg):
            # Convert FFT net position to sonar image coordinates
            fft_pitch_rad = np.radians(fft_pitch_deg)
            
            # Convert to sonar image coordinates
            fft_x_sonar = fft_distance * np.cos(fft_pitch_rad)
            fft_y_sonar = fft_distance * np.sin(fft_pitch_rad)
            
            # Convert to pixel coordinates
            fft_x_px = int(W/2 + fft_x_sonar * pixels_per_meter)
            fft_y_px = int(H/2 - fft_y_sonar * pixels_per_meter)  # Flip Y for image coordinates
            
            # Draw FFT net position line (cyan color)
            if 0 <= fft_x_px < W and 0 <= fft_y_px < H:
                # Draw line from center to FFT position
                cv2.line(overlay, (W//2, H//2), (fft_x_px, fft_y_px), 
                        (255, 255, 0), 2)  # Cyan color (BGR format)
                
                # Draw FFT position marker
                cv2.circle(overlay, (fft_x_px, fft_y_px), 5, (255, 255, 0), -1)
                
                # Add FFT label
                cv2.putText(overlay, f'FFT: {fft_distance:.1f}m@{fft_pitch_deg:.1f}°',
                          (fft_x_px + 10, fft_y_px - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def prepare_fft_data_for_video(fft_net_data):
    """
    Prepare FFT data for video integration with proper timestamp handling.
    
    Args:
        fft_net_data: Raw FFT data DataFrame
        
    Returns:
        Processed FFT data ready for video overlay
    """
    if fft_net_data is None:
        return None
    
    fft_data = fft_net_data.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in fft_data.columns:
        fft_data['timestamp'] = pd.to_datetime(fft_data['timestamp'])
    
    # Filter out invalid measurements
    valid_mask = (
        fft_data['distance_m'].notna() & 
        fft_data['pitch_deg'].notna() &
        (abs(fft_data['distance_m']) < 1000)  # Reasonable distance limit
    )
    
    fft_data = fft_data[valid_mask].copy()
    
    return fft_data
