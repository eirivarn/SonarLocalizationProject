"""
Relative FFT Pose Analysis Utilities

This module provides functions to load, process, and analyze data from the relative
FFT pose system that provides distance (in cm) and pitch (in radians).
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings

def load_relative_fft_data(csv_path: Path) -> pd.DataFrame:
    """
    Load relative FFT pose data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing measurements
        
    Returns:
        DataFrame with processed relative FFT data
    """
    try:
        # Load the CSV data
        df = pd.read_csv(csv_path)
        
        # Verify expected columns exist
        expected_cols = ['time', 'distance', 'heading', 'pitch']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert time to datetime
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        
        # Convert distance from cm to meters for consistency with other systems
        df['distance_m'] = df['distance'] / 100.0
        
        # Keep pitch in radians (already correct unit)
        df['pitch_rad'] = df['pitch']
        df['pitch_deg'] = np.degrees(df['pitch'])
        
        # Keep heading for reference
        df['heading_rad'] = df['heading']
        df['heading_deg'] = np.degrees(df['heading'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Loaded {len(df)} relative FFT pose records")
        return df
        
    except Exception as e:
        print(f"✗ Error loading relative FFT data: {e}")
        raise

def analyze_fft_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the quality and characteristics of relative FFT data.
    
    Args:
        df: DataFrame with relative FFT data
        
    Returns:
        Dictionary with quality statistics
    """
    stats = {}
    
    # Basic statistics
    stats['total_measurements'] = len(df)
    stats['time_range_seconds'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    stats['sampling_rate_hz'] = len(df) / stats['time_range_seconds'] if stats['time_range_seconds'] > 0 else 0
    
    # Distance statistics (in meters)
    valid_distances = df['distance_m'].dropna()
    stats['distance_count'] = len(valid_distances)
    stats['distance_min_m'] = valid_distances.min() if len(valid_distances) > 0 else np.nan
    stats['distance_max_m'] = valid_distances.max() if len(valid_distances) > 0 else np.nan
    stats['distance_mean_m'] = valid_distances.mean() if len(valid_distances) > 0 else np.nan
    stats['distance_std_m'] = valid_distances.std() if len(valid_distances) > 0 else np.nan
    
    # Pitch statistics (in degrees for readability)
    valid_pitch = df['pitch_deg'].dropna()
    stats['pitch_count'] = len(valid_pitch)
    stats['pitch_min_deg'] = valid_pitch.min() if len(valid_pitch) > 0 else np.nan
    stats['pitch_max_deg'] = valid_pitch.max() if len(valid_pitch) > 0 else np.nan
    stats['pitch_mean_deg'] = valid_pitch.mean() if len(valid_pitch) > 0 else np.nan
    stats['pitch_std_deg'] = valid_pitch.std() if len(valid_pitch) > 0 else np.nan
    
    # Heading statistics (in degrees)
    valid_heading = df['heading_deg'].dropna()
    stats['heading_count'] = len(valid_heading)
    stats['heading_min_deg'] = valid_heading.min() if len(valid_heading) > 0 else np.nan
    stats['heading_max_deg'] = valid_heading.max() if len(valid_heading) > 0 else np.nan
    stats['heading_mean_deg'] = valid_heading.mean() if len(valid_heading) > 0 else np.nan
    stats['heading_std_deg'] = valid_heading.std() if len(valid_heading) > 0 else np.nan
    
    return stats

def convert_fft_to_xy_coordinates(df: pd.DataFrame, reference_heading: float = 0.0) -> pd.DataFrame:
    """
    Convert relative FFT measurements to XY coordinates.
    
    Args:
        df: DataFrame with relative FFT data
        reference_heading: Reference heading in degrees (default: 0.0)
        
    Returns:
        DataFrame with added XY coordinate columns
    """
    df_result = df.copy()
    
    # Calculate XY coordinates using distance and pitch
    # Assuming pitch represents the angle to the target relative to straight ahead
    # X: forward/backward component
    # Y: left/right component
    
    df_result['fft_x'] = df_result['distance_m'] * np.cos(df_result['pitch_rad'])
    df_result['fft_y'] = df_result['distance_m'] * np.sin(df_result['pitch_rad'])
    
    return df_result

def synchronize_fft_with_other_systems(df_fft: pd.DataFrame, df_sonar: pd.DataFrame, 
                                     df_nav: pd.DataFrame, tolerance_seconds: float = 0.5) -> pd.DataFrame:
    """
    Synchronize relative FFT data with sonar and navigation data by timestamp.
    
    Args:
        df_fft: DataFrame with relative FFT data
        df_sonar: DataFrame with sonar data
        df_nav: DataFrame with navigation data
        tolerance_seconds: Maximum time difference for matching (seconds)
        
    Returns:
        DataFrame with synchronized measurements from all three systems
    """
    # Ensure all dataframes have proper timestamp columns
    df_fft_work = df_fft.copy()
    df_sonar_work = df_sonar.copy()
    df_nav_work = df_nav.copy()
    
    # Convert sonar timestamps if needed
    if 'timestamp' not in df_sonar_work.columns:
        if 'frame_timestamp' in df_sonar_work.columns:
            df_sonar_work['timestamp'] = pd.to_datetime(df_sonar_work['frame_timestamp'])
        elif 'ts_utc' in df_sonar_work.columns:
            df_sonar_work['timestamp'] = pd.to_datetime(df_sonar_work['ts_utc'])
        else:
            print("Warning: No timestamp column found in sonar data")
            df_sonar_work['timestamp'] = pd.NaT
    else:
        # Ensure timestamp is datetime
        df_sonar_work['timestamp'] = pd.to_datetime(df_sonar_work['timestamp'])
    
    # Convert navigation timestamps if needed
    if 'timestamp' not in df_nav_work.columns:
        if 'ts_utc' in df_nav_work.columns:
            df_nav_work['timestamp'] = pd.to_datetime(df_nav_work['ts_utc'])
        else:
            print("Warning: No timestamp column found in navigation data")
            df_nav_work['timestamp'] = pd.NaT
    else:
        # Ensure timestamp is datetime
        df_nav_work['timestamp'] = pd.to_datetime(df_nav_work['timestamp'])
    
    # Ensure FFT timestamp is datetime
    df_fft_work['timestamp'] = pd.to_datetime(df_fft_work['timestamp'])
    
    synchronized_data = []
    
    for _, fft_row in df_fft_work.iterrows():
        fft_time = fft_row['timestamp']
        
        # Skip if FFT timestamp is invalid
        if pd.isna(fft_time):
            continue
        
        # Find closest sonar measurement
        sonar_time_diff = float('inf')
        closest_sonar_idx = None
        if 'timestamp' in df_sonar_work.columns and not df_sonar_work['timestamp'].isna().all():
            valid_sonar = df_sonar_work.dropna(subset=['timestamp'])
            if len(valid_sonar) > 0:
                sonar_time_diffs = abs((valid_sonar['timestamp'] - fft_time).dt.total_seconds())
                closest_sonar_idx = sonar_time_diffs.idxmin()
                sonar_time_diff = sonar_time_diffs.loc[closest_sonar_idx]
        
        # Find closest navigation measurement
        nav_time_diff = float('inf')
        closest_nav_idx = None
        if 'timestamp' in df_nav_work.columns and not df_nav_work['timestamp'].isna().all():
            valid_nav = df_nav_work.dropna(subset=['timestamp'])
            if len(valid_nav) > 0:
                nav_time_diffs = abs((valid_nav['timestamp'] - fft_time).dt.total_seconds())
                closest_nav_idx = nav_time_diffs.idxmin()
                nav_time_diff = nav_time_diffs.loc[closest_nav_idx]
        
        # Create synchronized row if within tolerance
        if (sonar_time_diff <= tolerance_seconds or nav_time_diff <= tolerance_seconds):
            sync_row = {
                'timestamp': fft_time,
                'fft_distance_m': fft_row['distance_m'],
                'fft_pitch_deg': fft_row['pitch_deg'],
                'fft_heading_deg': fft_row['heading_deg'],
                'fft_x': fft_row.get('fft_x', np.nan),
                'fft_y': fft_row.get('fft_y', np.nan),
            }
            
            # Add sonar data if within tolerance
            if sonar_time_diff <= tolerance_seconds and closest_sonar_idx is not None:
                sonar_row = df_sonar_work.loc[closest_sonar_idx]
                sync_row.update({
                    'sonar_time_diff': sonar_time_diff,
                    'sonar_distance_m': sonar_row.get('distance_meters', np.nan),
                    'sonar_angle_deg': sonar_row.get('angle_degrees', np.nan),
                    'sonar_detection_success': sonar_row.get('detection_success', False)
                })
            else:
                sync_row.update({
                    'sonar_time_diff': np.nan,
                    'sonar_distance_m': np.nan,
                    'sonar_angle_deg': np.nan,
                    'sonar_detection_success': False
                })
            
            # Add navigation data if within tolerance
            if nav_time_diff <= tolerance_seconds and closest_nav_idx is not None:
                nav_row = df_nav_work.loc[closest_nav_idx]
                sync_row.update({
                    'nav_time_diff': nav_time_diff,
                    'nav_distance_m': nav_row.get('NetDistance', np.nan),
                    'nav_pitch_deg': np.degrees(nav_row.get('NetPitch', np.nan)) if not pd.isna(nav_row.get('NetPitch')) else np.nan
                })
            else:
                sync_row.update({
                    'nav_time_diff': np.nan,
                    'nav_distance_m': np.nan,
                    'nav_pitch_deg': np.nan
                })
            
            synchronized_data.append(sync_row)
    
    result_df = pd.DataFrame(synchronized_data)
    
    print(f"Synchronization completed:")
    print(f"  FFT records: {len(df_fft_work)}")
    print(f"  Sonar records: {len(df_sonar_work)}")
    print(f"  Navigation records: {len(df_nav_work)}")
    print(f"  Synchronized records: {len(result_df)}")
    
    return result_df

def create_three_system_comparison_plot(sync_df: pd.DataFrame) -> go.Figure:
    """
    Create visualization comparing all three measurement systems.
    
    Args:
        sync_df: DataFrame with synchronized data from all systems
        
    Returns:
        Plotly figure with comparison plots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Distance Measurements Over Time',
            'XY Position Comparison', 
            'Pitch/Angle Measurements',
            'System Availability'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Distance measurements over time
    if 'fft_distance_m' in sync_df.columns:
        fig.add_trace(
            go.Scatter(x=sync_df['timestamp'], y=sync_df['fft_distance_m'],
                      mode='lines+markers', name='FFT Distance',
                      line=dict(color='blue'), marker=dict(size=4)),
            row=1, col=1
        )
    
    if 'sonar_distance_m' in sync_df.columns:
        valid_sonar = sync_df[sync_df['sonar_detection_success'] == True]
        fig.add_trace(
            go.Scatter(x=valid_sonar['timestamp'], y=valid_sonar['sonar_distance_m'],
                      mode='markers', name='Sonar Distance',
                      marker=dict(color='red', size=6)),
            row=1, col=1
        )
    
    if 'nav_distance_m' in sync_df.columns:
        valid_nav = sync_df.dropna(subset=['nav_distance_m'])
        fig.add_trace(
            go.Scatter(x=valid_nav['timestamp'], y=valid_nav['nav_distance_m'],
                      mode='lines', name='DVL Distance',
                      line=dict(color='green', width=2)),
            row=1, col=1
        )
    
    # Plot 2: XY Position comparison
    if 'fft_x' in sync_df.columns and 'fft_y' in sync_df.columns:
        fig.add_trace(
            go.Scatter(x=sync_df['fft_x'], y=sync_df['fft_y'],
                      mode='markers', name='FFT XY',
                      marker=dict(color='blue', size=6)),
            row=1, col=2
        )
    
    # Add sonar XY if available (would need calculation)
    # Add DVL XY if available (would need calculation)
    
    # Plot 3: Pitch/Angle measurements
    if 'fft_pitch_deg' in sync_df.columns:
        fig.add_trace(
            go.Scatter(x=sync_df['timestamp'], y=sync_df['fft_pitch_deg'],
                      mode='lines+markers', name='FFT Pitch',
                      line=dict(color='blue'), marker=dict(size=4)),
            row=2, col=1
        )
    
    if 'sonar_angle_deg' in sync_df.columns:
        valid_sonar = sync_df[sync_df['sonar_detection_success'] == True]
        fig.add_trace(
            go.Scatter(x=valid_sonar['timestamp'], y=valid_sonar['sonar_angle_deg'],
                      mode='markers', name='Sonar Angle',
                      marker=dict(color='red', size=6)),
            row=2, col=1
        )
    
    if 'nav_pitch_deg' in sync_df.columns:
        valid_nav = sync_df.dropna(subset=['nav_pitch_deg'])
        fig.add_trace(
            go.Scatter(x=valid_nav['timestamp'], y=valid_nav['nav_pitch_deg'],
                      mode='lines', name='DVL Pitch',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
    
    # Plot 4: System availability
    availability_data = []
    for _, row in sync_df.iterrows():
        availability_data.append({
            'timestamp': row['timestamp'],
            'FFT': 1 if not pd.isna(row.get('fft_distance_m')) else 0,
            'Sonar': 1 if row.get('sonar_detection_success', False) else 0,
            'DVL': 1 if not pd.isna(row.get('nav_distance_m')) else 0
        })
    
    avail_df = pd.DataFrame(availability_data)
    
    for system, color in [('FFT', 'blue'), ('Sonar', 'red'), ('DVL', 'green')]:
        if system in avail_df.columns:
            fig.add_trace(
                go.Scatter(x=avail_df['timestamp'], y=avail_df[system],
                          mode='lines', name=f'{system} Available',
                          line=dict(color=color, width=2)),
                row=2, col=2
            )
    
    # Update layout
    fig.update_layout(
        title='Three-System Measurement Comparison',
        height=800,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Distance (m)", row=1, col=1)
    
    fig.update_xaxes(title_text="X Position (m)", row=1, col=2)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Pitch/Angle (deg)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="System Available", row=2, col=2)
    
    return fig
