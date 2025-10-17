"""
Net Relative Position Utilities

This module provides classes and functions for:
1. Converting distance/pitch measurements to net relative positions
2. Synchronizing measurements across multiple systems (Sonar, DVL, FFT)
3. Creating visualizations for multi-system comparisons
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
from typing import Tuple, Dict, List, Optional


class NetRelativePositionCalculator:
    """Calculate net relative positions from distance/pitch measurements for different systems."""
    
    def __init__(self):
        """Initialize the calculator."""
        pass
    
    def distance_pitch_to_xy(self, distance: float, pitch_degrees: float) -> Tuple[float, float]:
        """
        Convert distance and pitch to X,Y coordinates.
        
        Args:
            distance: Distance in meters
            pitch_degrees: Pitch angle in degrees
            
        Returns:
            Tuple of (x, y) coordinates in meters
        """
        if pd.isna(distance) or pd.isna(pitch_degrees):
            return np.nan, np.nan
        
        pitch_rad = np.radians(pitch_degrees)
        x = distance * np.cos(pitch_rad)
        y = distance * np.sin(pitch_rad)
        return x, y
    
    def calculate_sonar_net_position(self, df: pd.DataFrame, 
                                   distance_col: str = 'distance_meters',
                                   angle_col: str = 'angle_degrees',
                                   apply_angle_correction: bool = True) -> pd.DataFrame:
        """
        Calculate net relative position from sonar measurements.
        
        Args:
            df: DataFrame with sonar measurements
            distance_col: Column name for distance measurements
            angle_col: Column name for angle measurements
            apply_angle_correction: Whether to apply -180° correction to angles
            
        Returns:
            DataFrame with added sonar_x_m and sonar_y_m columns
        """
        result_df = df.copy()
        
        # Apply angle correction if requested
        if apply_angle_correction:
            corrected_angles = result_df[angle_col] - 180
        else:
            corrected_angles = result_df[angle_col]
        
        # Calculate X,Y positions
        positions = []
        for _, row in result_df.iterrows():
            distance = row.get(distance_col, np.nan)
            angle = corrected_angles.loc[row.name] if not pd.isna(corrected_angles.loc[row.name]) else np.nan
            x, y = self.distance_pitch_to_xy(distance, angle)
            positions.append({'sonar_x_m': x, 'sonar_y_m': y, 'sonar_pitch_corrected': angle})
        
        position_df = pd.DataFrame(positions, index=result_df.index)
        return pd.concat([result_df, position_df], axis=1)
    
    def calculate_dvl_net_position(self, df: pd.DataFrame,
                                 distance_col: str = 'NetDistance',
                                 pitch_col: str = 'NetPitch') -> pd.DataFrame:
        """
        Calculate net relative position from DVL measurements.
        
        Args:
            df: DataFrame with DVL measurements
            distance_col: Column name for distance measurements
            pitch_col: Column name for pitch measurements (in radians)
            
        Returns:
            DataFrame with added nav_x_m and nav_y_m columns
        """
        result_df = df.copy()
        
        # Calculate X,Y positions
        positions = []
        for _, row in result_df.iterrows():
            distance = row.get(distance_col, np.nan)
            pitch_rad = row.get(pitch_col, np.nan)
            
            if not pd.isna(distance) and not pd.isna(pitch_rad):
                x = distance * np.cos(pitch_rad)
                y = distance * np.sin(pitch_rad)
                pitch_deg = np.degrees(pitch_rad)
            else:
                x, y, pitch_deg = np.nan, np.nan, np.nan
            
            positions.append({'nav_x_m': x, 'nav_y_m': y, 'nav_pitch_deg': pitch_deg})
        
        position_df = pd.DataFrame(positions, index=result_df.index)
        return pd.concat([result_df, position_df], axis=1)
    
    def calculate_fft_net_position(self, df: pd.DataFrame,
                                 distance_col: str = 'distance',
                                 pitch_col: str = 'pitch',
                                 x_col: str = 'x_m',
                                 y_col: str = 'y_m') -> pd.DataFrame:
        """
        Calculate net relative position from FFT measurements.
        
        Args:
            df: DataFrame with FFT measurements
            distance_col: Column name for distance measurements (default: 'distance')
            pitch_col: Column name for pitch measurements in radians (default: 'pitch')
            x_col: Column name for existing X coordinates (if available)
            y_col: Column name for existing Y coordinates (if available)
            
        Returns:
            DataFrame with added/verified fft_x_m and fft_y_m columns
        """
        result_df = df.copy()
        
        # Handle timestamp conversion for Unix timestamps
        if 'time' in result_df.columns:
            result_df['timestamp'] = pd.to_datetime(result_df['time'], unit='s')
        
        # Check if X,Y coordinates already exist
        if x_col in result_df.columns and y_col in result_df.columns:
            # Use existing coordinates, just rename for consistency
            result_df['fft_x_m'] = result_df[x_col]
            result_df['fft_y_m'] = result_df[y_col]
        else:
            # Calculate from distance and pitch
            # Note: pitch in FFT data appears to be in radians, not degrees
            positions = []
            for _, row in result_df.iterrows():
                distance = row.get(distance_col, np.nan)
                pitch_rad = row.get(pitch_col, np.nan)  # Already in radians
                
                if not pd.isna(distance) and not pd.isna(pitch_rad):
                    x = distance * np.cos(pitch_rad)
                    y = distance * np.sin(pitch_rad)
                    pitch_deg = np.degrees(pitch_rad)  # Convert to degrees for consistency
                else:
                    x, y, pitch_deg = np.nan, np.nan, np.nan
                
                positions.append({
                    'fft_x_m': x, 
                    'fft_y_m': y,
                    'distance_m': distance,
                    'pitch_deg': pitch_deg
                })
            
            position_df = pd.DataFrame(positions, index=result_df.index)
            result_df = pd.concat([result_df, position_df], axis=1)
        
        return result_df


class MultiSystemSynchronizer:
    """Synchronize measurements across multiple systems with timestamp alignment."""
    
    def __init__(self, tolerance_seconds: float = 0.5):
        """
        Initialize the synchronizer.
        
        Args:
            tolerance_seconds: Maximum time difference for synchronization
        """
        self.tolerance = pd.Timedelta(seconds=tolerance_seconds)
    
    def standardize_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Standardize timestamps to UTC timezone-aware format.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataFrame with standardized timestamps
        """
        result_df = df.copy()
        
        if timestamp_col in result_df.columns:
            # Convert to datetime - handle Unix timestamps if needed
            if result_df[timestamp_col].dtype in ['float64', 'int64']:
                # Unix timestamp
                result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col], unit='s')
            else:
                # String timestamp
                result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
            
            # Make timezone-aware if needed
            if result_df[timestamp_col].dt.tz is None:
                result_df[timestamp_col] = result_df[timestamp_col].dt.tz_localize('UTC')
            else:
                result_df[timestamp_col] = result_df[timestamp_col].dt.tz_convert('UTC')
        
        return result_df
    
    def synchronize_three_systems(self, df_fft: pd.DataFrame, df_sonar: pd.DataFrame, 
                                df_nav: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronize measurements from three systems based on timestamps.
        
        Args:
            df_fft: FFT data with timestamp column
            df_sonar: Sonar data with timestamp column
            df_nav: Navigation data with timestamp column
            
        Returns:
            Synchronized DataFrame with measurements from all systems
        """
        # Standardize timestamps - handle different column names
        df_fft_std = df_fft.copy()
        if 'time' in df_fft_std.columns and 'timestamp' not in df_fft_std.columns:
            df_fft_std['timestamp'] = pd.to_datetime(df_fft_std['time'], unit='s')
        df_fft_std = self.standardize_timestamps(df_fft_std)
        
        df_sonar_std = self.standardize_timestamps(df_sonar)
        df_nav_std = self.standardize_timestamps(df_nav)
        
        # Create unified time grid
        all_times = pd.concat([
            df_fft_std['timestamp'].dropna(),
            df_sonar_std['timestamp'].dropna(),
            df_nav_std['timestamp'].dropna()
        ]).sort_values().drop_duplicates()
        
        sync_results = []
        
        for target_time in all_times:
            # Find closest matches in each system
            fft_match = df_fft_std[abs(df_fft_std['timestamp'] - target_time) <= self.tolerance]
            sonar_match = df_sonar_std[abs(df_sonar_std['timestamp'] - target_time) <= self.tolerance]
            nav_match = df_nav_std[abs(df_nav_std['timestamp'] - target_time) <= self.tolerance]
            
            if len(fft_match) > 0 or len(sonar_match) > 0 or len(nav_match) > 0:
                sync_row = {'sync_timestamp': target_time}
                
                # Add FFT data with proper column mapping for your actual data
                if len(fft_match) > 0:
                    fft_row = fft_match.iloc[0]
                    
                    # Map your actual FFT columns
                    if 'distance_m' in fft_row:
                        sync_row['fft_distance_m'] = fft_row['distance_m']
                    elif 'distance' in fft_row:
                        sync_row['fft_distance_m'] = fft_row['distance']
                    
                    if 'pitch_deg' in fft_row:
                        sync_row['fft_pitch_deg'] = fft_row['pitch_deg']
                    elif 'pitch' in fft_row:
                        sync_row['fft_pitch_deg'] = np.degrees(fft_row['pitch'])  # Convert radians to degrees
                    
                    if 'fft_x_m' in fft_row:
                        sync_row['fft_x_m'] = fft_row['fft_x_m']
                    if 'fft_y_m' in fft_row:
                        sync_row['fft_y_m'] = fft_row['fft_y_m']
                    
                    if 'heading' in fft_row:
                        sync_row['fft_heading'] = fft_row['heading']
                
                # Add Sonar data with proper column mapping
                if len(sonar_match) > 0:
                    sonar_row = sonar_match.iloc[0]
                    
                    # Map sonar columns to standardized names
                    if 'distance_meters' in sonar_row:
                        sync_row['sonar_distance_m'] = sonar_row['distance_meters']
                    if 'sonar_pitch_corrected' in sonar_row:
                        sync_row['sonar_pitch_deg'] = sonar_row['sonar_pitch_corrected']
                    if 'sonar_x_m' in sonar_row:
                        sync_row['sonar_x_m'] = sonar_row['sonar_x_m']
                    if 'sonar_y_m' in sonar_row:
                        sync_row['sonar_y_m'] = sonar_row['sonar_y_m']
                    if 'detection_success' in sonar_row:
                        sync_row['sonar_detection_success'] = sonar_row['detection_success']
                
                # Add Navigation data with proper column mapping
                if len(nav_match) > 0:
                    nav_row = nav_match.iloc[0]
                    
                    # Map navigation columns to standardized names
                    if 'NetDistance' in nav_row:
                        sync_row['nav_distance_m'] = nav_row['NetDistance']
                    if 'nav_pitch_deg' in nav_row:
                        sync_row['nav_pitch_deg'] = nav_row['nav_pitch_deg']
                    if 'nav_x_m' in nav_row:
                        sync_row['nav_x_m'] = nav_row['nav_x_m']
                    if 'nav_y_m' in nav_row:
                        sync_row['nav_y_m'] = nav_row['nav_y_m']
                
                sync_results.append(sync_row)
        
        return pd.DataFrame(sync_results)


class NetRelativeVisualizer:
    """Create visualizations for multi-system net relative position analysis."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def create_distance_comparison(self, sync_df: pd.DataFrame, target_bag: str) -> go.Figure:
        """
        Create distance comparison plot.
        
        Args:
            sync_df: Synchronized data from multiple systems
            target_bag: Target bag identifier for title
            
        Returns:
            Plotly figure with distance comparison
        """
        fig = go.Figure()
        
        # Distance Over Time
        systems = [
            ('FFT', 'fft_distance_m', 'red'),
            ('Sonar', 'sonar_distance_m', 'blue'),
            ('DVL', 'nav_distance_m', 'green')
        ]
        
        for name, col, color in systems:
            if col in sync_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sync_df['sync_timestamp'],
                        y=sync_df[col],
                        mode='lines+markers',
                        name=f'{name} Distance',
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    )
                )
        
        fig.update_layout(
            title=f"Distance Comparison Over Time: {target_bag}",
            xaxis_title="Time",
            yaxis_title="Distance (m)",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_pitch_comparison(self, sync_df: pd.DataFrame, target_bag: str) -> go.Figure:
        """
        Create pitch comparison plot.
        
        Args:
            sync_df: Synchronized data from multiple systems
            target_bag: Target bag identifier for title
            
        Returns:
            Plotly figure with pitch comparison
        """
        fig = go.Figure()
        
        # Pitch Over Time
        pitch_cols = [
            ('FFT', 'fft_pitch_deg', 'red'),
            ('Sonar', 'sonar_pitch_deg', 'blue'),
            ('DVL', 'nav_pitch_deg', 'green')
        ]
        
        for name, col, color in pitch_cols:
            if col in sync_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sync_df['sync_timestamp'],
                        y=sync_df[col],
                        mode='lines+markers',
                        name=f'{name} Pitch',
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    )
                )
        
        fig.update_layout(
            title=f"Pitch Comparison Over Time: {target_bag}",
            xaxis_title="Time",
            yaxis_title="Pitch (°)",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_x_position_comparison(self, sync_df: pd.DataFrame, target_bag: str) -> go.Figure:
        """
        Create X position comparison plot.
        
        Args:
            sync_df: Synchronized data from multiple systems
            target_bag: Target bag identifier for title
            
        Returns:
            Plotly figure with X position comparison
        """
        fig = go.Figure()
        
        # X Position Over Time
        position_systems = [
            ('FFT', 'fft_x_m', 'red'),
            ('Sonar', 'sonar_x_m', 'blue'),
            ('DVL', 'nav_x_m', 'green')
        ]
        
        for name, x_col, color in position_systems:
            if x_col in sync_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sync_df['sync_timestamp'],
                        y=sync_df[x_col],
                        mode='lines+markers',
                        name=f'{name} X Position',
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    )
                )
        
        fig.update_layout(
            title=f"X Position Comparison Over Time: {target_bag}",
            xaxis_title="Time",
            yaxis_title="X Position (m)",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_y_position_comparison(self, sync_df: pd.DataFrame, target_bag: str) -> go.Figure:
        """
        Create Y position comparison plot.
        
        Args:
            sync_df: Synchronized data from multiple systems
            target_bag: Target bag identifier for title
            
        Returns:
            Plotly figure with Y position comparison
        """
        fig = go.Figure()
        
        # Y Position Over Time
        position_systems = [
            ('FFT', 'fft_y_m', 'red'),
            ('Sonar', 'sonar_y_m', 'blue'),
            ('DVL', 'nav_y_m', 'green')
        ]
        
        for name, y_col, color in position_systems:
            if y_col in sync_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=sync_df['sync_timestamp'],
                        y=sync_df[y_col],
                        mode='lines+markers',
                        name=f'{name} Y Position',
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    )
                )
        
        fig.update_layout(
            title=f"Y Position Comparison Over Time: {target_bag}",
            xaxis_title="Time",
            yaxis_title="Y Position (m)",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def print_position_summary(self, sync_df: pd.DataFrame):
        """
        Print summary statistics for position data from all systems.
        
        Args:
            sync_df: Synchronized data from multiple systems
        """
        print("Position data summary:")
        
        systems = [
            ('FFT', 'fft_x_m', 'fft_y_m'),
            ('Sonar', 'sonar_x_m', 'sonar_y_m'),
            ('DVL', 'nav_x_m', 'nav_y_m')
        ]
        
        for name, x_col, y_col in systems:
            if x_col in sync_df.columns and y_col in sync_df.columns:
                valid_mask = sync_df[x_col].notna() & sync_df[y_col].notna()
                if valid_mask.sum() > 0:
                    x_data = sync_df.loc[valid_mask, x_col]
                    y_data = sync_df.loc[valid_mask, y_col]
                    print(f"  {name}: {len(x_data)} points, "
                          f"X: {x_data.mean():.2f}±{x_data.std():.2f}, "
                          f"Y: {y_data.mean():.2f}±{y_data.std():.2f}")


def run_complete_three_system_analysis(df_sonar: pd.DataFrame, df_nav: pd.DataFrame, 
                                      df_fft: pd.DataFrame, target_bag: str,
                                      exports_folder: Path) -> Tuple[pd.DataFrame, Dict, go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    Run complete three-system analysis with position calculations and visualizations.
    
    Args:
        df_sonar: Sonar measurements DataFrame
        df_nav: Navigation measurements DataFrame  
        df_fft: FFT measurements DataFrame
        target_bag: Target bag identifier
        exports_folder: Path to exports folder for saving results
        
    Returns:
        Tuple of (synchronized_data, statistics, distance_fig, pitch_fig, x_position_fig, y_position_fig)
    """
    # Initialize components
    calculator = NetRelativePositionCalculator()
    synchronizer = MultiSystemSynchronizer(tolerance_seconds=0.5)
    visualizer = NetRelativeVisualizer()
    
    print("Step 1: Calculating net relative positions")
    
    # Calculate net positions for each system
    df_sonar_with_pos = calculator.calculate_sonar_net_position(df_sonar, apply_angle_correction=True)
    df_nav_with_pos = calculator.calculate_dvl_net_position(df_nav)
    
    # Handle actual FFT data format - CHECK FOR UNIT CONVERSION
    df_fft_processed = df_fft.copy()
    
    # Convert Unix timestamp to datetime if needed
    if 'time' in df_fft_processed.columns:
        df_fft_processed['timestamp'] = pd.to_datetime(df_fft_processed['time'], unit='s')
    
    # Handle distance units - check if conversion from cm to m is needed
    if 'distance' in df_fft_processed.columns:
        fft_distances = pd.to_numeric(df_fft_processed['distance'], errors='coerce')
        max_distance = fft_distances.abs().max()
        
        if max_distance > 50:  # Likely in centimeters
            print(f"FFT distances appear to be in cm (max: {max_distance:.1f}), converting to meters")
            df_fft_processed['distance_m'] = fft_distances / 100.0  # Convert cm to m
        elif max_distance > 0.1:  # Already in reasonable meter range
            print(f"FFT distances appear to be in meters already (max: {max_distance:.3f})")
            df_fft_processed['distance_m'] = fft_distances
        else:  # Very small values, might be in different units
            print(f"FFT distances are very small (max: {max_distance:.3f}), using as-is")
            df_fft_processed['distance_m'] = fft_distances
            
        print(f"Final FFT distance range: {df_fft_processed['distance_m'].min():.3f} to {df_fft_processed['distance_m'].max():.3f} m")
    
    # Use original pitch values (already in radians)
    if 'pitch' in df_fft_processed.columns:
        df_fft_processed['pitch_rad'] = pd.to_numeric(df_fft_processed['pitch'], errors='coerce')
        df_fft_processed['pitch_deg'] = df_fft_processed['pitch_rad'] * 180 / np.pi
        print(f"FFT pitch range: {df_fft_processed['pitch_deg'].min():.1f}° to {df_fft_processed['pitch_deg'].max():.1f}°")
    
    # Calculate X,Y positions from distance and pitch (using radians)
    if 'distance_m' in df_fft_processed.columns and 'pitch_rad' in df_fft_processed.columns:
        df_fft_processed['fft_x_m'] = df_fft_processed['distance_m'] * np.cos(df_fft_processed['pitch_rad'])
        df_fft_processed['fft_y_m'] = df_fft_processed['distance_m'] * np.sin(df_fft_processed['pitch_rad'])
        
        print(f"FFT X range: {df_fft_processed['fft_x_m'].min():.3f} to {df_fft_processed['fft_x_m'].max():.3f} m")
        print(f"FFT Y range: {df_fft_processed['fft_y_m'].min():.3f} to {df_fft_processed['fft_y_m'].max():.3f} m")
    
    df_fft_with_pos = df_fft_processed
    
    print("Step 2: Synchronizing all three systems")
    
    # Synchronize all systems
    sync_df = synchronizer.synchronize_three_systems(df_fft_with_pos, df_sonar_with_pos, df_nav_with_pos)
    
    print(f"Synchronized {len(sync_df)} time points")
    
    # Debug: Print available columns
    print(f"Available columns in sync_df: {list(sync_df.columns)}")
    
    # Calculate comparison statistics with robust error handling
    stats = {'total_sync_points': len(sync_df)}
    
    # Find distance columns more robustly
    distance_cols = [col for col in sync_df.columns if 'distance' in col.lower()]
    print(f"Available distance columns: {distance_cols}")
    
    # Try to find the actual column names
    fft_dist_col = None
    sonar_dist_col = None
    nav_dist_col = None
    
    for col in sync_df.columns:
        if 'fft' in col and 'distance' in col:
            fft_dist_col = col
        elif 'sonar' in col and 'distance' in col:
            sonar_dist_col = col
        elif 'nav' in col and 'distance' in col:
            nav_dist_col = col
    
    print(f"Found distance columns: FFT={fft_dist_col}, Sonar={sonar_dist_col}, Nav={nav_dist_col}")
    
    if fft_dist_col and sonar_dist_col and nav_dist_col:
        # Ensure all distance columns are numeric
        try:
            sync_df[fft_dist_col] = pd.to_numeric(sync_df[fft_dist_col], errors='coerce')
            sync_df[sonar_dist_col] = pd.to_numeric(sync_df[sonar_dist_col], errors='coerce')
            sync_df[nav_dist_col] = pd.to_numeric(sync_df[nav_dist_col], errors='coerce')
            
            valid_distance_mask = (
                sync_df[fft_dist_col].notna() & 
                sync_df[sonar_dist_col].notna() & 
                sync_df[nav_dist_col].notna()
            )
            
            if valid_distance_mask.sum() > 0:
                valid_data = sync_df[valid_distance_mask].copy()
                
                # Calculate correlations with error handling
                try:
                    fft_sonar_corr = valid_data[fft_dist_col].corr(valid_data[sonar_dist_col])
                    fft_nav_corr = valid_data[fft_dist_col].corr(valid_data[nav_dist_col])
                    sonar_nav_corr = valid_data[sonar_dist_col].corr(valid_data[nav_dist_col])
                    
                    stats.update({
                        'valid_distance_points': len(valid_data),
                        'fft_sonar_distance_corr': fft_sonar_corr if not pd.isna(fft_sonar_corr) else 0.0,
                        'fft_nav_distance_corr': fft_nav_corr if not pd.isna(fft_nav_corr) else 0.0,
                        'sonar_nav_distance_corr': sonar_nav_corr if not pd.isna(sonar_nav_corr) else 0.0,
                    })
                    
                except Exception as corr_error:
                    print(f"Error calculating correlations: {corr_error}")
                    stats.update({
                        'valid_distance_points': len(valid_data),
                        'fft_sonar_distance_corr': 0.0,
                        'fft_nav_distance_corr': 0.0,
                        'sonar_nav_distance_corr': 0.0,
                    })
            else:
                print("No overlapping valid distance measurements found")
                
        except Exception as numeric_error:
            print(f"Error converting distance columns to numeric: {numeric_error}")
    else:
        print("Could not find all required distance columns")
    
    print("Step 3: Creating visualizations")
    
    # Create separate visualizations
    fig_distance = visualizer.create_distance_comparison(sync_df, target_bag)
    fig_pitch = visualizer.create_pitch_comparison(sync_df, target_bag)
    fig_x_position = visualizer.create_x_position_comparison(sync_df, target_bag)
    fig_y_position = visualizer.create_y_position_comparison(sync_df, target_bag)
    
    # Print summary
    visualizer.print_position_summary(sync_df)
    
    print("Step 4: Saving results")
    
    # Save results
    output_file = exports_folder / "outputs" / f"{target_bag}_three_system_net_relative_analysis.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sync_df.to_csv(output_file, index=False)
    
    stats_file = exports_folder / "outputs" / f"{target_bag}_three_system_net_relative_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    print(f"Statistics saved to: {stats_file}")
    
    return sync_df, stats, fig_distance, fig_pitch, fig_x_position, fig_y_position
