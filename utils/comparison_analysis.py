"""
Comparison analysis utilities for multi-system net tracking performance assessment.

This module provides functions for:
- Loading and preparing comparison data
- Computing pairwise differences
- Detecting stable baseline segments
- Baseline statistics and noise analysis
- Outlier detection using robust statistics
- Failure correlation analysis
- Generating comprehensive visualizations
- Creating markdown summary reports
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from itertools import combinations
import datetime


def load_and_prepare_data(data_path, smoothing_alpha=None):
    """
    Load comparison data and optionally apply smoothing.
    
    Args:
        data_path: Path to comparison CSV file
        smoothing_alpha: Exponential smoothing factor (0-1), None = no smoothing
        
    Returns:
        tuple: (df, sampling_rate, available_systems)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Comparison data not found: {data_path}")
    
    print(f"Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['sync_timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Apply smoothing if specified
    if smoothing_alpha is not None and smoothing_alpha < 1.0:
        print(f"\nApplying exponential smoothing (α={smoothing_alpha:.2f})...")
        
        smooth_cols = [
            'fft_distance_m', 'fft_pitch_deg', 'fft_x', 'fft_y',
            'sonar_distance_m', 'sonar_pitch_deg', 'sonar_x', 'sonar_y',
            'nav_distance_m', 'nav_pitch_deg', 'dvl_x', 'dvl_y'
        ]
        
        for col in smooth_cols:
            if col in df.columns:
                df[col] = df[col].ewm(alpha=smoothing_alpha, adjust=False).mean()
        
        print(f"  ✓ Smoothed {sum(1 for c in smooth_cols if c in df.columns)} columns")
    
    # Calculate sampling rate
    time_diffs = df.index.to_series().diff().dt.total_seconds()
    median_dt = time_diffs.median()
    sampling_rate = 1.0 / median_dt if median_dt > 0 else None
    
    print(f"\nData loaded: {len(df)} samples")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Duration: {(df.index[-1] - df.index[0]).total_seconds():.1f} seconds")
    print(f"Median sampling interval: {median_dt:.3f} s ({sampling_rate:.1f} Hz)")
    
    # Check for available systems
    available_systems = []
    if 'fft_distance_m' in df.columns and df['fft_distance_m'].notna().any():
        available_systems.append('FFT')
    if 'sonar_distance_m' in df.columns and df['sonar_distance_m'].notna().any():
        available_systems.append('Sonar')
    if 'nav_distance_m' in df.columns and df['nav_distance_m'].notna().any():
        available_systems.append('DVL')
    
    print(f"Available systems: {', '.join(available_systems)}")
    
    return df, sampling_rate, available_systems


def compute_pairwise_differences(df):
    """
    Compute pairwise differences for distance, pitch, X, and Y.
    
    Args:
        df: DataFrame with comparison data
        
    Returns:
        DataFrame with added difference columns
    """
    # Distance differences
    if 'fft_distance_m' in df.columns and 'nav_distance_m' in df.columns:
        df['diff_fft_nav'] = df['fft_distance_m'] - df['nav_distance_m']
    
    if 'sonar_distance_m' in df.columns and 'nav_distance_m' in df.columns:
        df['diff_sonar_nav'] = df['sonar_distance_m'] - df['nav_distance_m']
    
    if 'fft_distance_m' in df.columns and 'sonar_distance_m' in df.columns:
        df['diff_fft_sonar'] = df['fft_distance_m'] - df['sonar_distance_m']
    
    # Pitch differences
    if 'fft_pitch_deg' in df.columns and 'nav_pitch_deg' in df.columns:
        df['diff_pitch_fft_nav'] = df['fft_pitch_deg'] - df['nav_pitch_deg']
    
    if 'sonar_pitch_deg' in df.columns and 'nav_pitch_deg' in df.columns:
        df['diff_pitch_sonar_nav'] = df['sonar_pitch_deg'] - df['nav_pitch_deg']
    
    if 'fft_pitch_deg' in df.columns and 'sonar_pitch_deg' in df.columns:
        df['diff_pitch_fft_sonar'] = df['fft_pitch_deg'] - df['sonar_pitch_deg']
    
    # X position differences
    if 'fft_x' in df.columns and 'dvl_x' in df.columns:
        df['diff_x_fft_nav'] = df['fft_x'] - df['dvl_x']
    
    if 'sonar_x' in df.columns and 'dvl_x' in df.columns:
        df['diff_x_sonar_nav'] = df['sonar_x'] - df['dvl_x']
    
    if 'fft_x' in df.columns and 'sonar_x' in df.columns:
        df['diff_x_fft_sonar'] = df['fft_x'] - df['sonar_x']
    
    # Y position differences
    if 'fft_y' in df.columns and 'dvl_y' in df.columns:
        df['diff_y_fft_nav'] = df['fft_y'] - df['dvl_y']
    
    if 'sonar_y' in df.columns and 'dvl_y' in df.columns:
        df['diff_y_sonar_nav'] = df['sonar_y'] - df['dvl_y']
    
    if 'fft_y' in df.columns and 'sonar_y' in df.columns:
        df['diff_y_fft_sonar'] = df['fft_y'] - df['sonar_y']
    
    print("Pairwise differences computed:")
    diff_cols = [c for c in df.columns if c.startswith('diff_')]
    for col in diff_cols:
        valid_count = df[col].notna().sum()
        if valid_count > 0:
            print(f"  {col}: {valid_count} valid samples, "
                  f"mean={df[col].mean():.3f}, std={df[col].std():.3f}")
    
    return df


def detect_stable_segments(df, sampling_rate, rolling_window_sec=3.0, 
                          sigma_thresh=0.15, delta_thresh=0.15, min_segment_sec=1.0):
    """
    Detect stable baseline segments in the data.
    
    Args:
        df: DataFrame with comparison data and differences
        sampling_rate: Sampling rate in Hz
        rolling_window_sec: Window size for rolling statistics
        sigma_thresh: Threshold for rolling std deviation
        delta_thresh: Threshold for mean differences
        min_segment_sec: Minimum segment duration
        
    Returns:
        tuple: (df with stability columns, segments list, window_samples)
    """
    window_samples = int(rolling_window_sec * sampling_rate) if sampling_rate else 30
    print(f"Rolling window: {window_samples} samples ({rolling_window_sec} s)\n")
    
    print("Computing rolling statistics...")
    
    # New: prioritize X/Y stability per system
    position_axes = {
        'fft': ('fft_x', 'fft_y'),
        'sonar': ('sonar_x', 'sonar_y'),
        'dvl': ('dvl_x', 'dvl_y'),
        'nav': ('dvl_x', 'dvl_y'),  # alias for legacy naming
    }
    for label, (col_x, col_y) in position_axes.items():
        if col_x in df.columns:
            df[f'rolling_std_{col_x}'] = df[col_x].rolling(window_samples, center=True).std()
        if col_y in df.columns:
            df[f'rolling_std_{col_y}'] = df[col_y].rolling(window_samples, center=True).std()

    # Existing distance/pitch rolling stats now optional
    if 'fft_distance_m' in df.columns:
        df['rolling_mean_fft'] = df['fft_distance_m'].rolling(window_samples, center=True).mean()
        df['rolling_std_fft'] = df['fft_distance_m'].rolling(window_samples, center=True).std()
    if 'sonar_distance_m' in df.columns:
        df['rolling_mean_sonar'] = df['sonar_distance_m'].rolling(window_samples, center=True).mean()
        df['rolling_std_sonar'] = df['sonar_distance_m'].rolling(window_samples, center=True).std()
    if 'nav_distance_m' in df.columns:
        df['rolling_mean_dvl'] = df['nav_distance_m'].rolling(window_samples, center=True).mean()
        df['rolling_std_dvl'] = df['nav_distance_m'].rolling(window_samples, center=True).std()
    
    # Rolling mean and std of differences
    if 'diff_fft_nav' in df.columns:
        df['rolling_mean_diff_fft_nav'] = df['diff_fft_nav'].rolling(window_samples, center=True).mean()
        df['rolling_std_diff_fft_nav'] = df['diff_fft_nav'].rolling(window_samples, center=True).std()
    
    if 'diff_sonar_nav' in df.columns:
        df['rolling_mean_diff_sonar_nav'] = df['diff_sonar_nav'].rolling(window_samples, center=True).mean()
        df['rolling_std_diff_sonar_nav'] = df['diff_sonar_nav'].rolling(window_samples, center=True).std()
    
    # Define stability criteria (X/Y driven)
    print("Applying stability criteria...")
    stable_mask = pd.Series(True, index=df.index)

    for label, (col_x, col_y) in position_axes.items():
        if f'rolling_std_{col_x}' in df.columns:
            stable_mask &= (df[f'rolling_std_{col_x}'] < sigma_thresh)
        if f'rolling_std_{col_y}' in df.columns:
            stable_mask &= (df[f'rolling_std_{col_y}'] < sigma_thresh)

    # X/Y inter-system deltas
    xy_diff_pairs = [
        ('diff_x_fft_nav', 'diff_y_fft_nav'),
        ('diff_x_fft_sonar', 'diff_y_fft_sonar'),
        ('diff_x_sonar_nav', 'diff_y_sonar_nav'),
    ]
    for diff_x_col, diff_y_col in xy_diff_pairs:
        if diff_x_col in df.columns:
            df[f'rolling_mean_{diff_x_col}'] = df[diff_x_col].rolling(window_samples, center=True).mean()
            stable_mask &= df[f'rolling_mean_{diff_x_col}'].abs() < delta_thresh
        if diff_y_col in df.columns:
            df[f'rolling_mean_{diff_y_col}'] = df[diff_y_col].rolling(window_samples, center=True).mean()
            stable_mask &= df[f'rolling_mean_{diff_y_col}'].abs() < delta_thresh

    # No NaNs in distance measurements
    for col in ['fft_distance_m', 'sonar_distance_m', 'nav_distance_m']:
        if col in df.columns:
            stable_mask &= df[col].notna()
    
    df['is_stable'] = stable_mask
    
    print(f"Stable samples: {stable_mask.sum()} / {len(df)} ({100*stable_mask.sum()/len(df):.1f}%)")
    
    # Identify connected stable segments
    print("\nIdentifying stable baseline segments...")
    
    stable_changes = df['is_stable'].astype(int).diff()
    segment_starts = df.index[stable_changes == 1]
    segment_ends = df.index[stable_changes == -1]
    
    # Handle edge cases
    if df['is_stable'].iloc[0]:
        segment_starts = pd.DatetimeIndex([df.index[0]]).append(segment_starts)
    if df['is_stable'].iloc[-1]:
        segment_ends = segment_ends.append(pd.DatetimeIndex([df.index[-1]]))
    
    # Create segment list
    segments = []
    min_duration = pd.Timedelta(seconds=min_segment_sec)
    
    for start, end in zip(segment_starts, segment_ends):
        duration = end - start
        if duration >= min_duration:
            segment_data = df.loc[start:end]
            segments.append({
                'start': start,
                'end': end,
                'duration_sec': duration.total_seconds(),
                'n_samples': len(segment_data),
                'mean_dvl_dist': segment_data['nav_distance_m'].mean() if 'nav_distance_m' in df.columns else np.nan
            })
    
    print(f"Found {len(segments)} stable baseline segments (≥ {min_segment_sec} s):\n")
    for i, seg in enumerate(segments[:10]):
        print(f"  Segment {i+1}: {seg['start'].strftime('%H:%M:%S')} - {seg['end'].strftime('%H:%M:%S')} "
              f"({seg['duration_sec']:.1f} s, {seg['n_samples']} samples)")
    if len(segments) > 10:
        print(f"  ... and {len(segments)-10} more segments")
    
    # Add segment ID to dataframe
    df['segment_id'] = -1
    for i, seg in enumerate(segments):
        df.loc[seg['start']:seg['end'], 'segment_id'] = i
    
    return df, segments, window_samples


def compute_baseline_statistics(df):
    """
    Compute comprehensive baseline statistics for all measurements.
    
    Args:
        df: DataFrame with stability information
        
    Returns:
        dict: Statistics organized by measurement type
    """
    baseline_df = df[df['is_stable']].copy()
    
    print(f"=== BASELINE STATISTICS ({len(baseline_df)} samples) ===\n")
    
    stats_dict = {}
    
    # Distance noise
    print("1. PER-METHOD NOISE - DISTANCE")
    stats_dict['distance_noise'] = {}
    for method, col in [('FFT', 'fft_distance_m'), ('Sonar', 'sonar_distance_m'), ('DVL', 'nav_distance_m')]:
        if col in baseline_df.columns:
            data = baseline_df[col].dropna()
            if len(data) > 0:
                std = data.std()
                mad = stats.median_abs_deviation(data, scale='normal')
                stats_dict['distance_noise'][method] = {'std': std, 'mad': mad}
                print(f"  {method}: σ = {std:.4f} m, MAD = {mad:.4f} m")
    
    # Pitch noise
    print("\n2. PER-METHOD NOISE - PITCH")
    stats_dict['pitch_noise'] = {}
    for method, col in [('FFT', 'fft_pitch_deg'), ('Sonar', 'sonar_pitch_deg'), ('DVL', 'nav_pitch_deg')]:
        if col in baseline_df.columns:
            data = baseline_df[col].dropna()
            if len(data) > 0:
                std = data.std()
                mad = stats.median_abs_deviation(data, scale='normal')
                stats_dict['pitch_noise'][method] = {'std': std, 'mad': mad}
                print(f"  {method}: σ = {std:.4f}°, MAD = {mad:.4f}°")
    
    # X position noise
    print("\n3. PER-METHOD NOISE - X POSITION")
    stats_dict['x_noise'] = {}
    for method, col in [('FFT', 'fft_x'), ('Sonar', 'sonar_x'), ('DVL', 'dvl_x')]:
        if col in baseline_df.columns:
            data = baseline_df[col].dropna()
            if len(data) > 0:
                std = data.std()
                mad = stats.median_abs_deviation(data, scale='normal')
                stats_dict['x_noise'][method] = {'std': std, 'mad': mad}
                print(f"  {method}: σ = {std:.4f} m, MAD = {mad:.4f} m")
    
    # Y position noise
    print("\n4. PER-METHOD NOISE - Y POSITION")
    stats_dict['y_noise'] = {}
    for method, col in [('FFT', 'fft_y'), ('Sonar', 'sonar_y'), ('DVL', 'dvl_y')]:
        if col in baseline_df.columns:
            data = baseline_df[col].dropna()
            if len(data) > 0:
                std = data.std()
                mad = stats.median_abs_deviation(data, scale='normal')
                stats_dict['y_noise'][method] = {'std': std, 'mad': mad}
                print(f"  {method}: σ = {std:.4f} m, MAD = {mad:.4f} m")
    
    # Pairwise biases for each measurement type
    measurement_types = [
        ('DISTANCE', [('FFT', 'DVL', 'diff_fft_nav'), ('Sonar', 'DVL', 'diff_sonar_nav'), ('FFT', 'Sonar', 'diff_fft_sonar')]),
        ('PITCH', [('FFT', 'DVL', 'diff_pitch_fft_nav'), ('Sonar', 'DVL', 'diff_pitch_sonar_nav'), ('FFT', 'Sonar', 'diff_pitch_fft_sonar')]),
        ('X POSITION', [('FFT', 'DVL', 'diff_x_fft_nav'), ('Sonar', 'DVL', 'diff_x_sonar_nav'), ('FFT', 'Sonar', 'diff_x_fft_sonar')]),
        ('Y POSITION', [('FFT', 'DVL', 'diff_y_fft_nav'), ('Sonar', 'DVL', 'diff_y_sonar_nav'), ('FFT', 'Sonar', 'diff_y_fft_sonar')])
    ]
    
    for i, (meas_type, pairs) in enumerate(measurement_types, start=5):
        print(f"\n{i}. PAIRWISE BIASES - {meas_type}")
        stats_dict[f'{meas_type.lower().replace(" ", "_")}_biases'] = {}
        
        for method1, method2, diff_col in pairs:
            if diff_col in baseline_df.columns:
                data = baseline_df[diff_col].dropna()
                if len(data) > 0:
                    bias = data.mean()
                    std = data.std()
                    unit = '°' if 'pitch' in diff_col else 'm'
                    stats_dict[f'{meas_type.lower().replace(" ", "_")}_biases'][f'{method1}_vs_{method2}'] = {
                        'bias': bias, 'std': std, 'agreement_95': 1.96 * std
                    }
                    print(f"  {method1} vs {method2}: Bias = {bias:+.4f} {unit}, Std = {std:.4f} {unit}")
    
    # Linear relationships
    print("\n9. LINEAR RELATIONSHIPS (in baseline)")
    stats_dict['linear_relationships'] = {}
    
    relationships = [
        ('Distance - FFT vs DVL', 'fft_distance_m', 'nav_distance_m'),
        ('Distance - Sonar vs DVL', 'sonar_distance_m', 'nav_distance_m'),
        ('Pitch - FFT vs DVL', 'fft_pitch_deg', 'nav_pitch_deg'),
        ('Pitch - Sonar vs DVL', 'sonar_pitch_deg', 'nav_pitch_deg')
    ]
    
    for name, col1, col2 in relationships:
        if col1 in baseline_df.columns and col2 in baseline_df.columns:
            valid = baseline_df[[col1, col2]].dropna()
            if len(valid) > 10:
                X = valid[col2].values.reshape(-1, 1)
                y = valid[col1].values
                reg = LinearRegression().fit(X, y)
                r2 = reg.score(X, y)
                stats_dict['linear_relationships'][name] = {
                    'intercept': reg.intercept_,
                    'slope': reg.coef_[0],
                    'r2': r2
                }
                print(f"  {name}: R² = {r2:.4f}")
    
    return stats_dict


def detect_outliers(df, window_samples, outlier_k=3.5):
    """
    Detect outliers using MAD-based thresholds and analyze their characteristics.
    
    Args:
        df: DataFrame with measurements
        window_samples: Window size for baseline computation
        outlier_k: MAD multiplier for threshold
        
    Returns:
        tuple: (DataFrame with outlier columns added, outlier_stats dict)
    """
    print("=== OUTLIER DETECTION ===\n")
    
    smooth_window = int(2 * window_samples)
    
    for method, col in [('FFT', 'fft_distance_m'), ('Sonar', 'sonar_distance_m'), ('DVL', 'nav_distance_m')]:
        if col in df.columns:
            df[f'{col}_baseline'] = df[col].rolling(smooth_window, center=True, min_periods=1).median()
            df[f'{col}_residual'] = df[col] - df[f'{col}_baseline']
            
            residuals = df[f'{col}_residual'].dropna()
            if len(residuals) > 0:
                mad = stats.median_abs_deviation(residuals, scale='normal')
                threshold = outlier_k * mad
                df[f'{col}_outlier'] = (df[f'{col}_residual'].abs() > threshold)
                
                n_outliers = df[f'{col}_outlier'].sum()
                outlier_rate = 100 * n_outliers / len(df)
                
                print(f"{method}: {n_outliers} outliers ({outlier_rate:.2f}%), MAD = {mad:.4f} m")
    
    # Analyze outlier characteristics
    outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
    outlier_stats = analyze_outlier_characteristics(df, outlier_cols)
    
    return df, outlier_stats


def analyze_outlier_characteristics(df, outlier_cols):
    """
    Analyze outlier characteristics including magnitude, clustering, and persistence.
    
    Args:
        df: DataFrame with outlier columns and residuals
        outlier_cols: List of outlier column names
        
    Returns:
        dict: Comprehensive outlier statistics
    """
    print("\n=== OUTLIER CHARACTERISTICS ===\n")
    
    outlier_stats = {}
    
    for col in outlier_cols:
        # Extract method name
        method_raw = col.replace('_distance_m_outlier', '').replace('_pitch_deg_outlier', '').replace('_x_outlier', '').replace('_y_outlier', '')
        method = 'DVL' if method_raw == 'nav' else method_raw.upper()
        
        # Get corresponding residual column
        residual_col = col.replace('_outlier', '_residual')
        
        if col not in df.columns or residual_col not in df.columns:
            continue
        
        outliers = df[df[col] == True]
        
        if len(outliers) == 0:
            print(f"{method}: No outliers detected")
            outlier_stats[method] = {
                'count': 0,
                'rate_pct': 0,
                'mean_magnitude': 0,
                'median_magnitude': 0,
                'max_magnitude': 0,
                'cluster_count': 0,
                'avg_cluster_size': 0,
                'max_cluster_size': 0,
                'isolated_count': 0,
                'isolated_pct': 0,
                'small_count': 0,
                'small_pct': 0,
                'small_threshold': 0,
                'medium_count': 0,
                'medium_pct': 0,
                'medium_threshold': 0,
                'large_count': 0,
                'large_pct': 0
            }
            continue
        
        # Magnitude analysis
        residuals = outliers[residual_col].abs()
        mean_mag = residuals.mean()
        median_mag = residuals.median()
        max_mag = residuals.max()
        
        # Categorize outliers by magnitude (based on percentiles)
        p33 = residuals.quantile(0.33)
        p67 = residuals.quantile(0.67)
        
        small_outliers = residuals[residuals <= p33]
        medium_outliers = residuals[(residuals > p33) & (residuals <= p67)]
        large_outliers = residuals[residuals > p67]
        
        small_count = len(small_outliers)
        medium_count = len(medium_outliers)
        large_count = len(large_outliers)
        
        small_pct = 100 * small_count / len(residuals) if len(residuals) > 0 else 0
        medium_pct = 100 * medium_count / len(residuals) if len(residuals) > 0 else 0
        large_pct = 100 * large_count / len(residuals) if len(residuals) > 0 else 0
        
        # Clustering analysis: identify consecutive outlier groups
        outlier_mask = df[col].astype(int)
        outlier_diff = outlier_mask.diff()
        
        # Start of clusters (0 -> 1 transition)
        cluster_starts = outlier_diff == 1
        # End of clusters (1 -> 0 transition)
        cluster_ends = outlier_diff == -1
        
        # Handle edge cases
        if outlier_mask.iloc[0] == 1:
            cluster_starts.iloc[0] = True
        if outlier_mask.iloc[-1] == 1:
            cluster_ends.iloc[-1] = True
        
        # Count clusters and their sizes
        start_indices = df.index[cluster_starts].tolist()
        end_indices = df.index[cluster_ends].tolist()
        
        cluster_sizes = []
        for start, end in zip(start_indices, end_indices):
            cluster_data = df.loc[start:end]
            cluster_size = len(cluster_data)
            cluster_sizes.append(cluster_size)
        
        num_clusters = len(cluster_sizes)
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
        
        # Isolated outliers (cluster size = 1)
        isolated_count = sum(1 for size in cluster_sizes if size == 1)
        isolated_pct = 100 * isolated_count / num_clusters if num_clusters > 0 else 0
        
        # Store statistics
        outlier_stats[method] = {
            'count': len(outliers),
            'rate_pct': 100 * len(outliers) / len(df),
            'mean_magnitude': mean_mag,
            'median_magnitude': median_mag,
            'max_magnitude': max_mag,
            'cluster_count': num_clusters,
            'avg_cluster_size': avg_cluster_size,
            'max_cluster_size': max_cluster_size,
            'isolated_count': isolated_count,
            'isolated_pct': isolated_pct,
            'small_count': small_count,
            'small_pct': small_pct,
            'small_threshold': p33,
            'medium_count': medium_count,
            'medium_pct': medium_pct,
            'medium_threshold': p67,
            'large_count': large_count,
            'large_pct': large_pct
        }
        
        # Print detailed analysis
        print(f"{method}:")
        print(f"  Magnitude:")
        print(f"    Mean: {mean_mag:.4f} m, Median: {median_mag:.4f} m, Max: {max_mag:.4f} m")
        print(f"  Magnitude Categories:")
        print(f"    Small (≤{p33:.4f}m): {small_count} ({small_pct:.1f}%)")
        print(f"    Medium ({p33:.4f}-{p67:.4f}m): {medium_count} ({medium_pct:.1f}%)")
        print(f"    Large (>{p67:.4f}m): {large_count} ({large_pct:.1f}%)")
        print(f"  Clustering:")
        print(f"    Total clusters: {num_clusters}")
        print(f"    Avg cluster size: {avg_cluster_size:.1f} frames")
        print(f"    Max cluster size: {max_cluster_size} frames")
        print(f"    Isolated outliers: {isolated_count} ({isolated_pct:.1f}%)")
        
        if num_clusters > 0:
            multi_frame_clusters = sum(1 for size in cluster_sizes if size > 1)
            print(f"    Multi-frame clusters: {multi_frame_clusters} ({100*multi_frame_clusters/num_clusters:.1f}%)")
        print()
    
    return outlier_stats


# Update generate_multi_bag_summary_csv to include outlier characteristics
def generate_multi_bag_summary_csv(
    comparison_data_dir,
    output_path=None,
    smoothing_alpha=None,
    rolling_window_sec=3.0,
    sigma_thresh=0.15,
    delta_thresh=0.15,
    min_segment_sec=1.0,
    outlier_k=3.5,
):
    """
    Generate a comprehensive CSV summary of all bags in the comparison data directory.
    
    Each row represents one bag, with columns for all available statistics.
    """
    import pandas as pd
    from pathlib import Path
    import re
    
    comparison_data_dir = Path(comparison_data_dir)
    
    # Find all comparison CSV files
    comparison_files = sorted(comparison_data_dir.glob("*_raw_comparison.csv"))
    
    if not comparison_files:
        print(f"No comparison CSV files found in {comparison_data_dir}")
        return None
    
    print(f"\n=== GENERATING MULTI-BAG SUMMARY CSV ===\n")
    print(f"Found {len(comparison_files)} bags to process\n")
    
    # Set default output path
    if output_path is None:
        output_path = comparison_data_dir / "bag_summary.csv"
    else:
        output_path = Path(output_path)
    
    all_rows = []
    
    for csv_path in comparison_files:
        # Extract bag ID from filename
        bag_id = csv_path.stem.replace('_raw_comparison', '')
        print(f"Processing: {bag_id}")
        
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['sync_timestamp'])
            df = df.set_index('timestamp').sort_index()

            if smoothing_alpha is not None and smoothing_alpha < 1.0:
                print(f"\nApplying exponential smoothing (α={smoothing_alpha:.2f})...")
                
                smooth_cols = [
                    'fft_distance_m', 'fft_pitch_deg', 'fft_x', 'fft_y',
                    'sonar_distance_m', 'sonar_pitch_deg', 'sonar_x', 'sonar_y',
                    'nav_distance_m', 'nav_pitch_deg', 'dvl_x', 'dvl_y'
                ]
                
                for col in smooth_cols:
                    if col in df.columns:
                        df[col] = df[col].ewm(alpha=smoothing_alpha, adjust=False).mean()
                
                print(f"  ✓ Smoothed {sum(1 for c in smooth_cols if c in df.columns)} columns")
            
            # Calculate sampling rate
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            median_dt = time_diffs.median()
            sampling_rate = 1.0 / median_dt if median_dt > 0 else None
            
            # Check available systems
            available_systems = []
            if 'fft_distance_m' in df.columns and df['fft_distance_m'].notna().any():
                available_systems.append('FFT')
            if 'sonar_distance_m' in df.columns and df['sonar_distance_m'].notna().any():
                available_systems.append('Sonar')
            if 'nav_distance_m' in df.columns and df['nav_distance_m'].notna().any():
                available_systems.append('DVL')
            
            # Initialize row with metadata
            row = {
                'bag_id': bag_id,
                'total_samples': len(df),
                'duration_sec': (df.index[-1] - df.index[0]).total_seconds(),
                'sampling_rate_hz': sampling_rate,
                'systems_available': ','.join(available_systems),
                'start_time': df.index[0].isoformat(),
                'end_time': df.index[-1].isoformat(),
            }
            
            df = compute_pairwise_differences(df)

            df, segments, window_samples = detect_stable_segments(
                df,
                sampling_rate,
                rolling_window_sec=rolling_window_sec,
                sigma_thresh=sigma_thresh,
                delta_thresh=delta_thresh,
                min_segment_sec=min_segment_sec,
            )
            
            # Stable segment statistics
            row['stable_segments'] = len(segments)
            if len(segments) > 0:
                total_baseline_time = sum(s['duration_sec'] for s in segments)
                row['baseline_time_sec'] = total_baseline_time
                row['baseline_percentage'] = 100 * total_baseline_time / row['duration_sec']
                row['avg_segment_duration_sec'] = total_baseline_time / len(segments)
            else:
                row['baseline_time_sec'] = 0
                row['baseline_percentage'] = 0
                row['avg_segment_duration_sec'] = 0
            
            # Baseline statistics
            baseline_df = df[df['is_stable']].copy()
            row['baseline_samples'] = len(baseline_df)
            
            # Distance noise levels (baseline)
            for method, col in [('fft', 'fft_distance_m'), ('sonar', 'sonar_distance_m'), ('dvl', 'nav_distance_m')]:
                if col in baseline_df.columns:
                    data = baseline_df[col].dropna()
                    if len(data) > 0:
                        row[f'{method}_distance_std_m'] = data.std()
                        row[f'{method}_distance_mad_m'] = stats.median_abs_deviation(data, scale='normal')
                        row[f'{method}_distance_mean_m'] = data.mean()
            
            # Pitch noise levels (baseline)
            for method, col in [('fft', 'fft_pitch_deg'), ('sonar', 'sonar_pitch_deg'), ('dvl', 'nav_pitch_deg')]:
                if col in baseline_df.columns:
                    data = baseline_df[col].dropna()
                    if len(data) > 0:
                        row[f'{method}_pitch_std_deg'] = data.std()
                        row[f'{method}_pitch_mad_deg'] = stats.median_abs_deviation(data, scale='normal')
                        row[f'{method}_pitch_mean_deg'] = data.mean()
            
            # X position noise levels (baseline)
            for method, col in [('fft', 'fft_x'), ('sonar', 'sonar_x'), ('dvl', 'dvl_x')]:
                if col in baseline_df.columns:
                    data = baseline_df[col].dropna()
                    if len(data) > 0:
                        row[f'{method}_x_std_m'] = data.std()
                        row[f'{method}_x_mad_m'] = stats.median_abs_deviation(data, scale='normal')
            
            # Y position noise levels (baseline)
            for method, col in [('fft', 'fft_y'), ('sonar', 'sonar_y'), ('dvl', 'dvl_y')]:
                if col in baseline_df.columns:
                    data = baseline_df[col].dropna()
                    if len(data) > 0:
                        row[f'{method}_y_std_m'] = data.std()
                        row[f'{method}_y_mad_m'] = stats.median_abs_deviation(data, scale='normal')
            
            # Pairwise biases - Distance
            distance_pairs = [
                ('fft_dvl', 'diff_fft_nav'),
                ('sonar_dvl', 'diff_sonar_nav'),
                ('fft_sonar', 'diff_fft_sonar')
            ]
            
            for pair_name, diff_col in distance_pairs:
                if diff_col in baseline_df.columns:
                    data = baseline_df[diff_col].dropna()
                    if len(data) > 0:
                        row[f'distance_bias_{pair_name}_m'] = data.mean()
                        row[f'distance_std_{pair_name}_m'] = data.std()
                        row[f'distance_agreement95_{pair_name}_m'] = 1.96 * data.std()
            
            # Pairwise biases - Pitch
            pitch_pairs = [
                ('fft_dvl', 'diff_pitch_fft_nav'),
                ('sonar_dvl', 'diff_pitch_sonar_nav'),
                ('fft_sonar', 'diff_pitch_fft_sonar')
            ]
            
            for pair_name, diff_col in pitch_pairs:
                if diff_col in baseline_df.columns:
                    data = baseline_df[diff_col].dropna()
                    if len(data) > 0:
                        row[f'pitch_bias_{pair_name}_deg'] = data.mean()
                        row[f'pitch_std_{pair_name}_deg'] = data.std()
                        row[f'pitch_agreement95_{pair_name}_deg'] = 1.96 * data.std()
            
            # Pairwise biases - X position
            x_pairs = [
                ('fft_dvl', 'diff_x_fft_nav'),
                ('sonar_dvl', 'diff_x_sonar_nav'),
                ('fft_sonar', 'diff_x_fft_sonar')
            ]
            
            for pair_name, diff_col in x_pairs:
                if diff_col in baseline_df.columns:
                    data = baseline_df[diff_col].dropna()
                    if len(data) > 0:
                        row[f'x_bias_{pair_name}_m'] = data.mean()
                        row[f'x_std_{pair_name}_m'] = data.std()
            
            # Pairwise biases - Y position
            y_pairs = [
                ('fft_dvl', 'diff_y_fft_nav'),
                ('sonar_dvl', 'diff_y_sonar_nav'),
                ('fft_sonar', 'diff_y_fft_sonar')
            ]
            
            for pair_name, diff_col in y_pairs:
                if diff_col in baseline_df.columns:
                    data = baseline_df[diff_col].dropna()
                    if len(data) > 0:
                        row[f'y_bias_{pair_name}_m'] = data.mean()
                        row[f'y_std_{pair_name}_m'] = data.std()
            
            # Linear relationships
            relationships = [
                ('distance_fft_dvl', 'fft_distance_m', 'nav_distance_m'),
                ('distance_sonar_dvl', 'sonar_distance_m', 'nav_distance_m'),
                ('pitch_fft_dvl', 'fft_pitch_deg', 'nav_pitch_deg'),
                ('pitch_sonar_dvl', 'sonar_pitch_deg', 'nav_pitch_deg')
            ]
            
            for name, col1, col2 in relationships:
                if col1 in baseline_df.columns and col2 in baseline_df.columns:
                    valid = baseline_df[[col1, col2]].dropna()
                    if len(valid) > 10:
                        X = valid[col2].values.reshape(-1, 1)
                        y = valid[col1].values
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(X, y)
                        r2 = reg.score(X, y)
                        row[f'{name}_r2'] = r2
                        row[f'{name}_slope'] = reg.coef_[0]
                        row[f'{name}_intercept'] = reg.intercept_
            
            df, outlier_stats = detect_outliers(df, window_samples, outlier_k=outlier_k)
            
            outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
            for col in outlier_cols:
                method_raw = col.replace('_distance_m_outlier', '').replace('_pitch_deg_outlier', '').replace('_x_outlier', '').replace('_y_outlier', '')
                method = 'dvl' if method_raw == 'nav' else method_raw
                
                count = df[col].sum()
                rate = 100 * count / len(df)
                row[f'{method}_outlier_count'] = count
                row[f'{method}_outlier_rate_pct'] = rate
                
                # Add outlier characteristics
                method_upper = 'DVL' if method_raw == 'nav' else method_raw.upper()
                if method_upper in outlier_stats:
                    stats_dict = outlier_stats[method_upper]
                    row[f'{method}_outlier_mean_magnitude_m'] = stats_dict['mean_magnitude']
                    row[f'{method}_outlier_median_magnitude_m'] = stats_dict['median_magnitude']
                    row[f'{method}_outlier_max_magnitude_m'] = stats_dict['max_magnitude']
                    row[f'{method}_outlier_cluster_count'] = stats_dict['cluster_count']
                    row[f'{method}_outlier_avg_cluster_size'] = stats_dict['avg_cluster_size']
                    row[f'{method}_outlier_max_cluster_size'] = stats_dict['max_cluster_size']
                    row[f'{method}_outlier_isolated_count'] = stats_dict['isolated_count']
                    row[f'{method}_outlier_isolated_pct'] = stats_dict['isolated_pct']
                    # Add magnitude categories
                    row[f'{method}_outlier_small_count'] = stats_dict['small_count']
                    row[f'{method}_outlier_small_pct'] = stats_dict['small_pct']
                    row[f'{method}_outlier_small_threshold_m'] = stats_dict['small_threshold']
                    row[f'{method}_outlier_medium_count'] = stats_dict['medium_count']
                    row[f'{method}_outlier_medium_pct'] = stats_dict['medium_pct']
                    row[f'{method}_outlier_medium_threshold_m'] = stats_dict['medium_threshold']
                    row[f'{method}_outlier_large_count'] = stats_dict['large_count']
                    row[f'{method}_outlier_large_pct'] = stats_dict['large_pct']
    
            all_rows.append(row)
            print(f"  ✓ Processed {len(baseline_df)} baseline samples, {len(segments)} stable segments")
            
        except Exception as e:
            print(f"  ✗ Error processing {bag_id}: {e}")
            # Add minimal row with error info
            all_rows.append({
                'bag_id': bag_id,
                'error': str(e)
            })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(all_rows)
    
    # Sort columns: metadata first, then by name
    metadata_cols = ['bag_id', 'total_samples', 'duration_sec', 'sampling_rate_hz', 
                     'systems_available', 'start_time', 'end_time', 'stable_segments',
                     'baseline_time_sec', 'baseline_percentage', 'baseline_samples']
    
    # Add missing metadata columns to all rows
    for col in metadata_cols:
        if col not in summary_df.columns:
            summary_df[col] = None
    
    # Reorder columns
    sorted_columns = metadata_cols + [col for col in summary_df.columns if col not in metadata_cols]
    summary_df = summary_df[sorted_columns]
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Summary CSV saved: {output_path.name}")
    print(f"  Location: {output_path}")
    
    return output_path


def generate_summary_report(df, segments, baseline_stats, available_systems, 
                           target_bag, output_path, config, outlier_stats=None):
    """
    Generate comprehensive markdown summary report.
    
    Args:
        df: DataFrame with all data
        segments: List of stable segments
        baseline_stats: Dictionary of baseline statistics
        available_systems: List of available systems
        target_bag: Bag identifier
        output_path: Path to save report
        config: Configuration dictionary
        outlier_stats: Dictionary of outlier characteristics (optional)
        
    Returns:
        Path: Path to saved report
    """
    print("\n=== GENERATING SUMMARY REPORT ===\n")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df = df[df['is_stable']].copy()
    outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
    
    with open(output_path, 'w') as f:
        f.write(f"# Net Tracking Analysis Summary: {target_bag}\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Dataset info
        f.write("## Dataset Information\n\n")
        f.write(f"- **Bag ID:** {target_bag}\n")
        f.write(f"- **Duration:** {(df.index[-1] - df.index[0]).total_seconds():.1f} seconds\n")
        f.write(f"- **Total Samples:** {len(df)}\n")
        f.write(f"- **Available Systems:** {', '.join(available_systems)}\n\n")
        
        # Configuration
        f.write("## Analysis Configuration\n\n")
        for key, value in config.items():
            f.write(f"- **{key}:** {value}\n")
        f.write("\n")
        
        # Stable segments
        f.write("## Stable Baseline Segments\n\n")
        if len(segments) > 0:
            total_baseline_time = sum(s['duration_sec'] for s in segments)
            baseline_pct = 100 * total_baseline_time / (df.index[-1]-df.index[0]).total_seconds()
            f.write(f"- **Total Segments:** {len(segments)}\n")
            f.write(f"- **Total Baseline Time:** {total_baseline_time:.1f} s ({baseline_pct:.1f}%)\n\n")
        
        # Baseline noise levels
        f.write("## Baseline Noise Levels\n\n")
        for meas_type in ['distance', 'pitch', 'x', 'y']:
            if f'{meas_type}_noise' in baseline_stats:
                f.write(f"### {meas_type.title()}\n\n")
                f.write("| Method | σ | MAD |\n|--------|---|-----|\n")
                for method, values in baseline_stats[f'{meas_type}_noise'].items():
                    unit = '°' if meas_type == 'pitch' else 'm'
                    f.write(f"| {method} | {values['std']:.4f} {unit} | {values['mad']:.4f} {unit} |\n")
                f.write("\n")
        
        # Pairwise biases
        f.write("## Pairwise Biases\n\n")
        for meas_type in ['distance', 'pitch', 'x', 'y']:
            if f'{meas_type}_biases' in baseline_stats:
                f.write(f"### {meas_type.title()}\n\n")
                f.write("| Comparison | Bias | Std Dev | Agreement (95%) |\n|-------------|------|---------|-----------------|\n")
                for pair, values in baseline_stats[f'{meas_type}_biases'].items():
                    f.write(f"| {pair} | {values['bias']:.4f} | {values['std']:.4f} | {values['agreement_95']:.4f} |\n")
                f.write("\n")
        
        # Linear relationships
        f.write("## Linear Relationships (in Baseline)\n\n")
        f.write("| Relationship | Intercept | Slope | R² |\n|--------------|------------|-------|----|\n")
        for name, values in baseline_stats['linear_relationships'].items():
            f.write(f"| {name} | {values['intercept']:.4f} | {values['slope']:.4f} | {values['r2']:.4f} |\n")
        f.write("\n")
        
        # Outlier rates and characteristics
        f.write("\n## Outlier Detection Results\n\n")
        f.write("### Outlier Rates\n\n")
        f.write("| Method | Count | Rate (%) |\n")
        f.write("|--------|-------|----------|\n")
        for col in outlier_cols:
            method = col.split('_')[0].upper()
            method = 'DVL' if method == 'NAV' else method
            count = df[col].sum()
            rate = 100 * count / len(df)
            f.write(f"| {method} | {count} | {rate:.2f} |\n")
        
        # Outlier characteristics
        if outlier_stats:
            f.write("\n### Outlier Characteristics\n\n")
            f.write("| Method | Mean Mag (m) | Max Mag (m) | Clusters | Avg Size | Max Size | Isolated (%) |\n|--------|--------------|-------------|----------|----------|----------|---------------|\n")
            for method, stats_dict in outlier_stats.items():
                if stats_dict['count'] > 0:
                    f.write(f"| {method} | {stats_dict['mean_magnitude']:.4f} | "
                           f"{stats_dict['max_magnitude']:.4f} | {stats_dict['cluster_count']} | "
                           f"{stats_dict['avg_cluster_size']:.1f} | {stats_dict['max_cluster_size']} | "
                           f"{stats_dict['isolated_pct']:.1f} |\n")
            
            f.write("\n### Outlier Magnitude Categories\n\n")
            f.write("| Method | Small | Small (%) | Medium | Medium (%) | Large | Large (%) |\n")
            f.write("|--------|-------|-----------|--------|------------|-------|------------|\n")
            for method, stats_dict in outlier_stats.items():
                if stats_dict['count'] > 0:
                    f.write(f"| {method} | {stats_dict['small_count']} | {stats_dict['small_pct']:.1f} | "
                           f"{stats_dict['medium_count']} | {stats_dict['medium_pct']:.1f} | "
                           f"{stats_dict['large_count']} | {stats_dict['large_pct']:.1f} |\n")
            
            f.write("\n**Interpretation:**\n")
            f.write("- **Mean/Max Magnitude**: Size of outlier deviations from baseline\n")
            f.write("- **Clusters**: Number of contiguous outlier groups\n")
            f.write("- **Avg/Max Size**: Duration of outlier events in frames\n")
            f.write("- **Isolated**: Percentage of single-frame outliers vs. multi-frame events\n")
            f.write("- **Small/Medium/Large**: Outliers categorized by magnitude (33rd/67th percentiles)\n")
        
        # Failure correlation
        if len(segments) > 0 and 'jaccard_index' in df.columns:
            f.write("\n## Failure Correlation Analysis\n\n")
            f.write("| Segment | Jaccard Index |\n|---------|----------------|\n")
            for seg in segments:
                if seg['n_samples'] > 1:
                    start, end = seg['start'], seg['end']
                    segment_data = df.loc[start:end]
                    failure_corr = compute_failure_correlation(segment_data)
                    f.write(f"| {start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')} | {failure_corr:.4f} |\n")
            f.write("\n")
        
        # Plots
        f.write("## Visualizations\n\n")
        # Time series plots
        f.write("### Time Series Overview\n\n")
        f.write("```python\n")
        f.write("import matplotlib.pyplot as plt\n")
        f.write("plt.figure(figsize=(12, 6))\n")
        for method in available_systems:
            if f'{method.lower()}_distance_m' in df.columns:
                f.write(f"plt.plot(df.index, df['{method.lower()}_distance_m'], label='{method} Distance')\n")
        f.write("plt.xlabel('Time')\n")
        f.write("plt.ylabel('Distance (m)')\n")
        f.write("plt.title('Time Series of Distance Measurements')\n")
        f.write("plt.legend()\n")
        f.write("plt.show()\n")
        f.write("```\n\n")
        
        # Baseline segment plot
        f.write("### Stable Baseline Segments\n\n")
        f.write("```python\n")
        f.write("plt.figure(figsize=(12, 6))\n")
        for method in available_systems:
            if f'{method.lower()}_distance_m' in df.columns:
                f.write(f"plt.plot(df.index, df['{method.lower()}_distance_m'], label='{method} Distance')\n")
        f.write("plt.xlabel('Time')\n")
        f.write("plt.ylabel('Distance (m)')\n")
        f.write("plt.title('Stable Baseline Segments')\n")
        f.write("plt.axvspan(segments[0]['start'], segments[0]['end'], color='green', alpha=0.3, label='Stable Segment')\n")
        f.write("plt.legend()\n")
        f.write("plt.show()\n")
        f.write("```\n\n")
        
        # Outlier detection plot
        f.write("### Outlier Detection Example\n\n")
        f.write("```python\n")
        f.write("plt.figure(figsize=(12, 6))\n")
        for method in available_systems:
            if f'{method.lower()}_distance_m' in df.columns:
                f.write(f"plt.plot(df.index, df['{method.lower()}_distance_m'], label='{method} Distance')\n")
        f.write("plt.xlabel('Time')\n")
        f.write("plt.ylabel('Distance (m)')\n")
        f.write("plt.title('Outlier Detection Example')\n")
        f.write("plt.scatter(outliers.index, outliers['fft_distance_m'], color='red', label='Detected Outliers')\n")
        f.write("plt.legend()\n")
        f.write("plt.show()\n")
        f.write("```\n\n")
        
        # Footer
        f.write("---\n")
        f.write("**Note:** This report is auto-generated. Please refer to the analysis scripts for details.\n")
    
    print(f"✓ Summary report saved: {output_path.name}")
    print(f"  Location: {output_path}")
    
    return output_path


def print_multi_bag_summary(summary_csv_path):
    """
    Load and print comprehensive summary statistics across all analyzed bags.
    
    Args:
        summary_csv_path: Path to the bag_summary.csv file
        
    Returns:
        DataFrame: The loaded summary data, or None if file not found
    """
    import pandas as pd
    from pathlib import Path
    
    summary_csv_path = Path(summary_csv_path)
    
    if not summary_csv_path.exists():
        print(f"✗ Summary CSV not found: {summary_csv_path}")
        print("\nTo generate it, run:")
        print("  python scripts/generate_bag_summary.py")
        return None
    
    print(f"Loading multi-bag summary from: {summary_csv_path.name}\n")
    summary_df = pd.read_csv(summary_csv_path)
    
    print("="*70)
    print("MULTI-BAG SUMMARY STATISTICS")
    print("="*70)
    
    # Dataset overview
    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total bags analyzed: {len(summary_df)}")
    print(f"  Total recording time: {summary_df['duration_sec'].sum() / 3600:.1f} hours")
    print(f"  Total samples: {summary_df['total_samples'].sum():,}")
    
    # System availability
    print(f"\n🔧 SYSTEM AVAILABILITY")
    for system in ['FFT', 'Sonar', 'DVL']:
        count = summary_df['systems_available'].str.contains(system).sum()
        pct = 100 * count / len(summary_df)
        print(f"  {system}: {count}/{len(summary_df)} bags ({pct:.1f}%)")
    
    # Baseline segment statistics
    print(f"\n📈 BASELINE SEGMENTS (Stable Data)")
    print(f"  Avg baseline percentage: {summary_df['baseline_percentage'].mean():.1f}%")
    print(f"  Median baseline percentage: {summary_df['baseline_percentage'].median():.1f}%")
    print(f"  Avg segments per bag: {summary_df['stable_segments'].mean():.1f}")
    print(f"  Total baseline time: {summary_df['baseline_time_sec'].sum() / 3600:.1f} hours")
    
    # Distance noise levels (across all bags)
    print(f"\n📏 DISTANCE MEASUREMENT NOISE (Baseline STD)")
    for system in ['fft', 'sonar', 'dvl']:
        col = f'{system}_distance_std_m'
        if col in summary_df.columns:
            data = summary_df[col].dropna()
            if len(data) > 0:
                print(f"  {system.upper()}: mean={data.mean():.4f}m, median={data.median():.4f}m, "
                      f"min={data.min():.4f}m, max={data.max():.4f}m")
    
    # Pitch noise levels
    print(f"\n📐 PITCH MEASUREMENT NOISE (Baseline STD)")
    for system in ['fft', 'sonar', 'dvl']:
        col = f'{system}_pitch_std_deg'
        if col in summary_df.columns:
            data = summary_df[col].dropna()
            if len(data) > 0:
                print(f"  {system.upper()}: mean={data.mean():.4f}°, median={data.median():.4f}°, "
                      f"min={data.min():.4f}°, max={data.max():.4f}°")
    
    # Position (X/Y) noise levels
    print(f"\n🧭 POSITION NOISE (Baseline STD)")
    for axis_label, axis in [("X", "x"), ("Y", "y")]:
        for system in ['fft', 'sonar', 'dvl']:
            col = f'{system}_{axis}_std_m'
            if col in summary_df.columns:
                data = summary_df[col].dropna()
                if len(data) > 0:
                    print(f"  {axis_label} {system.upper()}: mean={data.mean():.4f}m, "
                          f"median={data.median():.4f}m, min={data.min():.4f}m, max={data.max():.4f}m")

    # Pairwise bias (systematic differences)
    print(f"\n⚖️  PAIRWISE BIASES (Systematic Differences)")
    distance_pairs = [
        ('FFT vs DVL', 'distance_bias_fft_dvl_m'),
        ('Sonar vs DVL', 'distance_bias_sonar_dvl_m'),
        ('FFT vs Sonar', 'distance_bias_fft_sonar_m')
    ]
    
    for name, col in distance_pairs:
        if col in summary_df.columns:
            data = summary_df[col].dropna()
            if len(data) > 0:
                print(f"  {name}: mean={data.mean():+.4f}m, median={data.median():+.4f}m, "
                      f"std={data.std():.4f}m")
    
    # Linear relationship quality
    print(f"\n📉 LINEAR RELATIONSHIP QUALITY (R² scores)")
    r2_pairs = [
        ('Distance: FFT vs DVL', 'distance_fft_dvl_r2'),
        ('Distance: Sonar vs DVL', 'distance_sonar_dvl_r2'),
        ('Pitch: FFT vs DVL', 'pitch_fft_dvl_r2'),
        ('Pitch: Sonar vs DVL', 'pitch_sonar_dvl_r2')
    ]
    
    for name, col in r2_pairs:
        if col in summary_df.columns:
            data = summary_df[col].dropna()
            if len(data) > 0:
                print(f"  {name}: mean={data.mean():.4f}, median={data.median():.4f}, "
                      f"min={data.min():.4f}")
    
    # Outlier rates
    print(f"\n⚠️  OUTLIER RATES")
    for system in ['fft', 'sonar', 'dvl']:
        col = f'{system}_outlier_rate_pct'
        if col in summary_df.columns:
            data = summary_df[col].dropna()
            if len(data) > 0:
                print(f"  {system.upper()}: mean={data.mean():.2f}%, median={data.median():.2f}%, "
                      f"max={data.max():.2f}%")
    
    # Outlier magnitude
    print(f"\n📏 OUTLIER MAGNITUDE")
    for system in ['fft', 'sonar', 'dvl']:
        col = f'{system}_outlier_mean_magnitude_m'
        if col in summary_df.columns:
            data = summary_df[col].dropna()
            if len(data) > 0:
                print(f"  {system.upper()}: mean={data.mean():.4f}m, median={data.median():.4f}m, "
                      f"max={data.max():.4f}m")
    
    # Outlier clustering
    print(f"\n🔗 OUTLIER CLUSTERING (Multi-frame events)")
    for system in ['fft', 'sonar', 'dvl']:
        avg_col = f'{system}_outlier_avg_cluster_size'
        isolated_col = f'{system}_outlier_isolated_pct'
        if avg_col in summary_df.columns and isolated_col in summary_df.columns:
            avg_data = summary_df[avg_col].dropna()
            isolated_data = summary_df[isolated_col].dropna()
            if len(avg_data) > 0 and len(isolated_data) > 0:
                print(f"  {system.upper()}: avg_cluster_size={avg_data.mean():.1f} frames, "
                      f"isolated={isolated_data.mean():.1f}%")
    
    # Outlier magnitude categories
    print(f"\n📊 OUTLIER MAGNITUDE CATEGORIES")
    for system in ['fft', 'sonar', 'dvl']:
        small_col = f'{system}_outlier_small_pct'
        medium_col = f'{system}_outlier_medium_pct'
        large_col = f'{system}_outlier_large_pct'
        if all(col in summary_df.columns for col in [small_col, medium_col, large_col]):
            small_data = summary_df[small_col].dropna()
            medium_data = summary_df[medium_col].dropna()
            large_data = summary_df[large_col].dropna()
            if len(small_data) > 0:
                print(f"  {system.upper()}: small={small_data.mean():.1f}%, "
                      f"medium={medium_data.mean():.1f}%, large={large_data.mean():.1f}%")
    
    # Best and worst performing bags
    print(f"\n🏆 TOP 5 BAGS (by baseline percentage)")
    top_bags = summary_df.nlargest(5, 'baseline_percentage')[['bag_id', 'baseline_percentage', 'stable_segments']]
    for idx, row in top_bags.iterrows():
        print(f"  {row['bag_id']}: {row['baseline_percentage']:.1f}% baseline, "
              f"{row['stable_segments']:.0f} segments")
    
    print(f"\n⚠️  BOTTOM 5 BAGS (by baseline percentage)")
    bottom_bags = summary_df.nsmallest(5, 'baseline_percentage')[['bag_id', 'baseline_percentage', 'stable_segments']]
    for idx, row in bottom_bags.iterrows():
        print(f"  {row['bag_id']}: {row['baseline_percentage']:.1f}% baseline, "
              f"{row['stable_segments']:.0f} segments")
    
    print("\n" + "="*70)
    print(f"✓ Multi-bag summary complete")
    print(f"  Full data available in: {summary_csv_path.name}")
    print(f"  Columns: {len(summary_df.columns)}")
    
    # Define metadata columns
    metadata_cols = ['bag_id', 'total_samples', 'duration_sec', 'sampling_rate_hz', 
                    'systems_available', 'start_time', 'end_time', 'stable_segments',
                    'baseline_time_sec', 'baseline_percentage', 'baseline_samples']
    
    # Print column summary
    print(f"\n  Column categories:")
    print(f"    Metadata: {len([c for c in summary_df.columns if c in metadata_cols])}")
    print(f"    Distance metrics: {len([c for c in summary_df.columns if 'distance' in c])}")
    print(f"    Pitch metrics: {len([c for c in summary_df.columns if 'pitch' in c])}")
    print(f"    X/Y position metrics: {len([c for c in summary_df.columns if '_x_' in c or '_y_' in c])}")
    print(f"    Outlier metrics: {len([c for c in summary_df.columns if 'outlier' in c])}")
    print(f"    Correlation metrics: {len([c for c in summary_df.columns if 'jaccard' in c])}")
    
    return summary_df


def compute_failure_correlation(df):
    """
    Compute failure correlation metrics for stable segments.
    
    Args:
        df: DataFrame with stable segment data
        
    Returns:
        dict: Correlation metrics (Jaccard index, etc.)
    """
    print("\n=== FAILURE CORRELATION ANALYSIS ===\n")
    
    # Jaccard index for failure correlation
    if 'jaccard_index' in df.columns:
        jaccard_index = df['jaccard_index'].mean()
        print(f"Jaccard index (mean): {jaccard_index:.4f}")
    else:
        jaccard_index = np.nan
        print("Jaccard index not available")
    
    return {
        'jaccard_index': jaccard_index
    }


def analyze_failure_correlation(df):
    """
    Analyze correlation between system failures.
    
    Args:
        df: DataFrame with outlier columns
        
    Returns:
        dict: Correlation statistics
    """
    print("\n=== FAILURE CORRELATION ===\n")
    
    outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
    
    if len(outlier_cols) < 2:
        print("Not enough systems for correlation analysis")
        return {}
    
    print("1. INDIVIDUAL OUTLIER RATES")
    for col in outlier_cols:
        method_raw = col.replace('_distance_m_outlier', '').replace('_pitch_deg_outlier', '').replace('_x_outlier', '').replace('_y_outlier', '')
        method = 'DVL' if method_raw == 'nav' else method_raw.upper()
        rate = 100 * df[col].sum() / len(df)
        print(f"  P({method} outlier) = {rate:.2f}%")
    
    print("\n2. CO-OCCURRENCE ANALYSIS")
    
    correlation_stats = {}
    from itertools import combinations
    for col1, col2 in combinations(outlier_cols, 2):
        method1_raw = col1.split('_')[0]
        method2_raw = col2.split('_')[0]
        method1 = 'DVL' if method1_raw == 'nav' else method1_raw.upper()
        method2 = 'DVL' if method2_raw == 'nav' else method2_raw.upper()
        
        both = (df[col1] & df[col2]).sum()
        either = (df[col1] | df[col2]).sum()
        
        if either > 0:
            jaccard = both / either
            p_2_given_1 = both / df[col1].sum() if df[col1].sum() > 0 else 0
            p_1_given_2 = both / df[col2].sum() if df[col2].sum() > 0 else 0
            
            correlation_stats[f'{method1}-{method2}'] = {
                'jaccard': jaccard,
                'p_2_given_1': p_2_given_1,
                'p_1_given_2': p_1_given_2
            }
            
            print(f"  Jaccard({method1}, {method2}) = {jaccard:.3f}")
            if df[col1].sum() > 0:
                print(f"    P({method2} outlier | {method1} outlier) = {p_2_given_1:.3f}")
            if df[col2].sum() > 0:
                print(f"    P({method1} outlier | {method2} outlier) = {p_1_given_2:.3f}")
    
    return correlation_stats


def create_visualizations(df, segments, target_bag, plots_dir, sigma_thresh, outlier_k):
    """
    Create all visualization plots.
    
    Args:
        df: DataFrame with all computed metrics
        segments: List of stable segments
        target_bag: Bag identifier
        plots_dir: Output directory for plots
        sigma_thresh: Stability threshold
        outlier_k: Outlier threshold multiplier
        
    Returns:
        dict: Created figures
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    figs = {}
    
    # Plot 1: Time series with stable segments
    fig1 = go.Figure()
    for method, col, color in [('FFT', 'fft_distance_m', 'red'), 
                               ('Sonar', 'sonar_distance_m', 'blue'), 
                               ('DVL', 'nav_distance_m', 'green')]:
        if col in df.columns:
            fig1.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=method,
                                     line=dict(color=color, width=1)))
    
    for seg in segments[:20]:
        fig1.add_vrect(x0=seg['start'], x1=seg['end'], fillcolor="lightgray", 
                      opacity=0.3, layer="below", line_width=0)
    
    fig1.update_layout(title=f"Distance Time Series: {target_bag}",
                      xaxis_title="Time", yaxis_title="Distance (m)", height=500,
                      hovermode='x unified')
    figs['timeseries'] = fig1
    
    # Plot 2: Stability analysis
    fig2 = make_subplots(rows=2, cols=1, 
                         subplot_titles=('Distance with Stable Regions', 'Rolling Standard Deviation'),
                         vertical_spacing=0.15, row_heights=[0.6, 0.4])
    
    for method, col, color in [('FFT', 'fft_distance_m', 'red'), 
                               ('Sonar', 'sonar_distance_m', 'blue'), 
                               ('DVL', 'nav_distance_m', 'green')]:
        if col in df.columns:
            fig2.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=method,
                                     line=dict(color=color, width=1.5)), row=1, col=1)
    
    for seg in segments:
        fig2.add_vrect(x0=seg['start'], x1=seg['end'], fillcolor="lightgreen",
                      opacity=0.2, layer="below", line_width=0, row=1, col=1)
    
    for method, col, color in [('FFT', 'rolling_std_fft', 'red'), 
                               ('Sonar', 'rolling_std_sonar', 'blue'), 
                               ('DVL', 'rolling_std_dvl', 'green')]:
        if col in df.columns:
            fig2.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'{method} σ',
                                     line=dict(color=color, width=1.5, dash='dot')), row=2, col=1)
    
    fig2.add_hline(y=sigma_thresh, line_dash="dash", line_color="black", 
                  annotation_text=f"σ threshold = {sigma_thresh} m",
                  annotation_position="right", row=2, col=1)
    
    for seg in segments:
        fig2.add_vrect(x0=seg['start'], x1=seg['end'], fillcolor="lightgreen",
                      opacity=0.2, layer="below", line_width=0, row=2, col=1)
    
    fig2.update_xaxes(title_text="Time", row=2, col=1)
    fig2.update_yaxes(title_text="Distance (m)", row=1, col=1)
    fig2.update_yaxes(title_text="Rolling Std (m)", row=2, col=1)
    fig2.update_layout(title=f"Stability Analysis: {target_bag}", height=800,
                      hovermode='x unified')
    figs['stability'] = fig2
    
    # Plot 3: Outliers
    fig3 = go.Figure()
    for method, col, outlier_col, color in [
        ('FFT', 'fft_distance_m', 'fft_distance_m_outlier', 'red'),
        ('Sonar', 'sonar_distance_m', 'sonar_distance_m_outlier', 'blue'),
        ('DVL', 'nav_distance_m', 'nav_distance_m_outlier', 'green')
    ]:
        if col in df.columns and outlier_col in df.columns:
            normal_data = df[~df[outlier_col]]
            fig3.add_trace(go.Scatter(x=normal_data.index, y=normal_data[col],
                                     mode='lines', name=method, line=dict(color=color, width=1)))
            
            outlier_data = df[df[outlier_col]]
            if len(outlier_data) > 0:
                fig3.add_trace(go.Scatter(x=outlier_data.index, y=outlier_data[col],
                                         mode='markers', name=f'{method} outliers',
                                         marker=dict(color=color, size=4, symbol='circle', 
                                                   opacity=0.7, line=dict(width=1, color='white'))))
    
    fig3.update_layout(title=f"Outlier Detection ({outlier_k}× MAD): {target_bag}",
                      xaxis_title="Time", yaxis_title="Distance (m)", height=500,
                      hovermode='x unified')
    figs['outliers'] = fig3
    
    # New XY trajectory visualization
    if any(col in df.columns for col in ['fft_x', 'sonar_x', 'dvl_x']):
        fig_xy = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("X Position vs Time", "Y Position vs Time"),
            vertical_spacing=0.12,
            shared_xaxes=True,
        )

        xy_traces = [
            ('FFT', 'fft_x', 'fft_y', 'red'),
            ('Sonar', 'sonar_x', 'sonar_y', 'blue'),
            ('DVL', 'dvl_x', 'dvl_y', 'green'),
        ]

        for label, col_x, col_y, color in xy_traces:
            if col_x in df.columns:
                fig_xy.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col_x],
                        mode='lines',
                        name=f'{label} X',
                        line=dict(color=color)  # solid line
                    ),
                    row=1, col=1
                )
            if col_y in df.columns:
                fig_xy.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col_y],
                        mode='lines',
                        name=f'{label} Y',
                        line=dict(color=color)  # solid line (no dash)
                    ),
                    row=2, col=1
                )

        for seg in segments:
            fig_xy.add_vrect(x0=seg['start'], x1=seg['end'], fillcolor="LightGreen", opacity=0.15, line_width=0)

        fig_xy.update_layout(
            title=f"XY Position Stability: {target_bag}",
            height=700,
            xaxis2_title="Time",
            yaxis_title="X (m)",
            yaxis2_title="Y (m)",
            hovermode='x unified'
        )

        figs['xy_positions'] = fig_xy
        save_path = plots_dir / f"{target_bag}_xy_positions.html"
        fig_xy.write_html(str(save_path))
        print(f"Saved: {save_path.name}")

    # Save all plots
    for name, fig in figs.items():
        save_path = plots_dir / f"{target_bag}_{name}.html"
        fig.write_html(str(save_path))
        print(f"Saved: {save_path.name}")
    
    return figs
