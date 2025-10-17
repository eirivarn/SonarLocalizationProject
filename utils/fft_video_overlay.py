"""
FFT Video Overlay Module

Provides functions to add FFT net position overlays to sonar video frames.
"""

import pandas as pd
import numpy as np
import cv2

def add_fft_overlay(frame, overlay, fft_data, current_time, W, H, pixels_per_meter, tolerance_seconds=0.5):
    """
    Add FFT net position overlay to a video frame.
    
    Args:
        frame: Original frame image
        overlay: Overlay image to draw on
        fft_data: DataFrame with FFT data (timestamp, distance_m, pitch_deg)
        current_time: Current frame timestamp
        W, H: Frame dimensions
        pixels_per_meter: Pixel to meter conversion factor
        tolerance_seconds: Time tolerance for finding matching FFT data
    
    Returns:
        Modified overlay with FFT data drawn on it
    """
    if fft_data is None or current_time is None:
        return overlay
    
    # Find closest FFT measurement in time
    time_diffs = abs(fft_data['timestamp'] - current_time)
    if len(time_diffs) == 0:
        return overlay
        
    closest_idx = time_diffs.idxmin()
    
    if time_diffs.loc[closest_idx] <= pd.Timedelta(seconds=tolerance_seconds):
        fft_row = fft_data.loc[closest_idx]
        fft_distance = fft_row.get('distance_m', np.nan)
        fft_pitch_deg = fft_row.get('pitch_deg', np.nan)
        
        if not pd.isna(fft_distance) and not pd.isna(fft_pitch_deg):
            # Convert FFT net position to sonar image coordinates
            fft_pitch_rad = np.radians(fft_pitch_deg)
            
            # Convert to sonar coordinate system
            fft_x_sonar = fft_distance * np.cos(fft_pitch_rad)
            fft_y_sonar = fft_distance * np.sin(fft_pitch_rad)
            
            # Convert to pixel coordinates
            fft_x_px = int(W/2 + fft_x_sonar * pixels_per_meter)
            fft_y_px = int(H/2 - fft_y_sonar * pixels_per_meter)  # Flip Y for image coordinates
            
            # Draw FFT net position line (CYAN color in BGR)
            if 0 <= fft_x_px < W and 0 <= fft_y_px < H:
                # Draw line from center to FFT position
                cv2.line(overlay, (W//2, H//2), (fft_x_px, fft_y_px), 
                        (255, 255, 0), 3)  # Cyan color in BGR format
                
                # Draw FFT position marker (circle)
                cv2.circle(overlay, (fft_x_px, fft_y_px), 6, (255, 255, 0), -1)
                
                # Add FFT text label
                label_text = f'FFT: {fft_distance:.1f}m@{fft_pitch_deg:.1f}°'
                cv2.putText(overlay, label_text,
                          (fft_x_px + 10, fft_y_px - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return overlay

def prepare_fft_data(fft_net_data):
    """
    Prepare FFT data for video overlay use.
    
    Args:
        fft_net_data: Raw FFT DataFrame
        
    Returns:
        Processed FFT DataFrame ready for overlay
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
        (abs(fft_data['distance_m']) < 100)  # Reasonable distance limit
    )
    
    return fft_data[valid_mask].copy()
