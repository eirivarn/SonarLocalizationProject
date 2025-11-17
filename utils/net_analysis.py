from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import os
import numpy as np
import pandas as pd

from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS

def _resolve_export_paths(exports_dir: str | Path | None = None) -> Tuple[Path, Path, Path]:
    root = Path(exports_dir) if exports_dir else Path(EXPORTS_DIR_DEFAULT)
    outputs_dir = root / EXPORTS_SUBDIRS.get('outputs', 'outputs')
    by_bag_dir = root / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    return root, outputs_dir, by_bag_dir

def load_sonar_analysis_results(
    target_bag: str,
    exports_dir: str | Path | None = None,
    filenames: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    _, outputs_dir, _ = _resolve_export_paths(exports_dir)
    candidates = filenames or [
        f"{target_bag}_analysis.csv",
        f"{target_bag}_data_cones_analysis.csv",
        f"{target_bag}_video_cones_analysis.csv",
    ]
    tried_paths = [outputs_dir / name for name in candidates]
    for path in tried_paths:
        if path.exists():
            df = pd.read_csv(path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp']).sort_values('timestamp')
            return df, {
                "resolved_path": str(path),
                "rows": len(df),
                "paths_checked": [str(p) for p in tried_paths],
            }

    return None, {
        "resolved_path": None,
        "rows": 0,
        "paths_checked": [str(p) for p in tried_paths],
    }

def load_navigation_dataset(
    target_bag: str,
    exports_dir: str | Path | None = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    _, _, by_bag_dir = _resolve_export_paths(exports_dir)
    nav_path = by_bag_dir / f"navigation_plane_approximation__{target_bag}_data.csv"

    if nav_path.exists():
        df = pd.read_csv(nav_path)
        timestamp_source = 'timestamp' if 'timestamp' in df.columns else 'ts_utc'
        df['timestamp'] = pd.to_datetime(df[timestamp_source], errors='coerce')
        df = df.sort_values('timestamp')
        return df, {"resolved_path": str(nav_path), "rows": len(df)}

    try:
        import utils.distance_measurement as distance_api

        raw_data, _ = distance_api.load_all_distance_data_for_bag(target_bag, by_bag_dir)
        df = raw_data.get('navigation') if raw_data else None
        if df is not None:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp')
            return df, {"resolved_path": "distance_measurement_fallback", "rows": len(df)}
    except Exception:
        pass

    return None, {"resolved_path": str(nav_path), "rows": 0}

def load_fft_dataset(
    target_bag: str,
    fft_root: str | Path | None = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    candidate_dirs = []
    if fft_root is not None:
        candidate_dirs.append(Path(fft_root))
    env_dir = os.getenv("SOLAQUA_RELATIVE_FFT_DIR")
    if env_dir:
        candidate_dirs.append(Path(env_dir))
    candidate_dirs.append(Path(EXPORTS_DIR_DEFAULT).parent / "relative_fft_pose")

    tried_paths: List[str] = []
    for base in candidate_dirs:
        path = base / f"{target_bag}_relative_pose_fft.csv"
        tried_paths.append(str(path))
        if path.exists():
            df = pd.read_csv(path)
            prepared = _prepare_fft_dataframe(df)
            return prepared, {
                "resolved_path": str(path),
                "rows": len(prepared),
                "paths_checked": tried_paths,
            }

    return None, {
        "resolved_path": None,
        "rows": 0,
        "paths_checked": tried_paths,
    }

def summarize_sonar_results(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": 0,
        "detections": 0,
        "detection_rate": 0.0,
        "distance_range_m": None,
        "angle_range_deg": None,
    }
    if df is None or df.empty:
        return summary

    summary["rows"] = len(df)
    detection_mask = df.get('detection_success', pd.Series(dtype=bool)).fillna(False)
    summary["detections"] = int(detection_mask.sum())
    summary["detection_rate"] = float(detection_mask.mean()) * 100.0

    if 'distance_meters' in df.columns:
        distances = df['distance_meters'].dropna()
        if not distances.empty:
            summary["distance_range_m"] = (float(distances.min()), float(distances.max()))

    if 'angle_degrees' in df.columns:
        angles = df['angle_degrees'].dropna()
        if not angles.empty:
            summary["angle_range_deg"] = (float(angles.min()), float(angles.max()))

    return summary

def summarize_navigation_results(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": 0,
        "distance_range_m": None,
        "distance_mean_m": None,
        "distance_std_m": None,
        "pitch_range_deg": None,
    }
    if df is None or df.empty:
        return summary

    summary["rows"] = len(df)
    if 'NetDistance' in df.columns:
        dist = df['NetDistance'].dropna()
        if not dist.empty:
            summary["distance_range_m"] = (float(dist.min()), float(dist.max()))
            summary["distance_mean_m"] = float(dist.mean())
            summary["distance_std_m"] = float(dist.std())

    if 'NetPitch' in df.columns:
        pitch = np.degrees(df['NetPitch'].dropna())
        if not pitch.empty:
            summary["pitch_range_deg"] = (float(pitch.min()), float(pitch.max()))

    return summary

def summarize_fft_results(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": 0,
        "distance_range_m": None,
        "distance_mean_m": None,
        "distance_std_m": None,
        "pitch_range_deg": None,
    }
    if df is None or df.empty:
        return summary

    summary["rows"] = len(df)
    if 'distance_m' in df.columns:
        distances = pd.to_numeric(df['distance_m'], errors='coerce').dropna()
        if not distances.empty:
            summary["distance_range_m"] = (float(distances.min()), float(distances.max()))
            summary["distance_mean_m"] = float(distances.mean())
            summary["distance_std_m"] = float(distances.std())

    if 'pitch_deg' in df.columns:
        pitch = pd.to_numeric(df['pitch_deg'], errors='coerce').dropna()
        if not pitch.empty:
            summary["pitch_range_deg"] = (float(pitch.min()), float(pitch.max()))

    return summary

def synchronize_sonar_and_dvl(
    df_sonar: Optional[pd.DataFrame],
    df_nav: Optional[pd.DataFrame],
    tolerance_seconds: float = 1.0,
) -> pd.DataFrame:
    if df_sonar is None or df_nav is None or df_sonar.empty or df_nav.empty:
        return pd.DataFrame()

    sonar = df_sonar.copy()
    sonar['timestamp'] = pd.to_datetime(sonar['timestamp'], errors='coerce')
    sonar = sonar.dropna(subset=['timestamp']).sort_values('timestamp')

    nav = df_nav.copy()
    timestamp_col = 'timestamp' if 'timestamp' in nav.columns else 'ts_utc'
    nav['timestamp'] = pd.to_datetime(nav[timestamp_col], errors='coerce')
    nav = nav.dropna(subset=['timestamp']).sort_values('timestamp')

    if nav.empty or sonar.empty:
        return pd.DataFrame()

    nav_times = nav['timestamp'].to_numpy(dtype='datetime64[ns]')
    rows: List[Dict[str, Any]] = []

    for sonar_row in sonar.itertuples(index=False):
        sonar_ts = np.datetime64(sonar_row.timestamp)
        deltas = np.abs(nav_times - sonar_ts)
        idx = int(deltas.argmin())
        diff_sec = float(deltas[idx] / np.timedelta64(1, 's'))
        if diff_sec <= tolerance_seconds:
            nav_row = nav.iloc[idx]
            rows.append({
                'timestamp': pd.Timestamp(sonar_row.timestamp),
                'sonar_distance_m': getattr(sonar_row, 'distance_meters', None),
                'sonar_angle_deg': getattr(sonar_row, 'angle_degrees', None),
                'sonar_detection': bool(getattr(sonar_row, 'detection_success', False)),
                'dvl_distance_m': nav_row.get('NetDistance'),
                'dvl_pitch_deg': float(np.degrees(nav_row['NetPitch'])) if 'NetPitch' in nav_row and pd.notna(nav_row['NetPitch']) else None,
                'time_diff_s': diff_sec,
            })

    return pd.DataFrame(rows)

def summarize_distance_alignment(sync_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "pairs": int(len(sync_df)) if not sync_df.empty else 0,
        "valid_pairs": 0,
        "difference_stats": None,
    }
    if sync_df.empty:
        return summary

    valid = sync_df['sonar_distance_m'].notna() & sync_df['dvl_distance_m'].notna()
    summary["valid_pairs"] = int(valid.sum())
    if not valid.any():
        return summary

    diff = sync_df.loc[valid, 'sonar_distance_m'] - sync_df.loc[valid, 'dvl_distance_m']
    stats = {
        "mean": float(diff.mean()),
        "std": float(diff.std()),
        "min": float(diff.min()),
        "max": float(diff.max()),
        "p95": float(np.percentile(diff, 95)),
    }
    if len(diff) > 10:
        stats["correlation"] = float(sync_df.loc[valid, 'sonar_distance_m'].corr(sync_df.loc[valid, 'dvl_distance_m']))
    summary["difference_stats"] = stats
    return summary

def summarize_xy_positions(sync_df: pd.DataFrame) -> Dict:
    """
    Summarize XY position data from synchronized DataFrame.
    
    Args:
        sync_df: Synchronized DataFrame with XY position columns
        
    Returns:
        Dictionary of statistics per system
    """
    stats = {}
    
    # Support both naming conventions: _x_m/_y_m and _x/_y
    systems = [
        ('FFT', ['fft_x_m', 'fft_y_m', 'fft_x', 'fft_y']),
        ('Sonar', ['sonar_x_m', 'sonar_y_m', 'sonar_x', 'sonar_y']),
        ('DVL', ['nav_x_m', 'nav_y_m', 'dvl_x', 'dvl_y'])
    ]
    
    for system_name, possible_cols in systems:
        # Find which column names actually exist
        x_col = None
        y_col = None
        
        for col in possible_cols:
            if col in sync_df.columns:
                if '_x' in col:
                    x_col = col
                elif '_y' in col:
                    y_col = col
        
        if x_col and y_col:
            valid_mask = sync_df[x_col].notna() & sync_df[y_col].notna()
            
            if valid_mask.sum() > 0:
                x_data = sync_df.loc[valid_mask, x_col]
                y_data = sync_df.loc[valid_mask, y_col]
                
                # Calculate distance from origin
                distances = np.sqrt(x_data**2 + y_data**2)
                
                stats[system_name] = {
                    'count': len(x_data),
                    'x_mean': float(x_data.mean()),
                    'x_std': float(x_data.std()),
                    'x_min': float(x_data.min()),
                    'x_max': float(x_data.max()),
                    'y_mean': float(y_data.mean()),
                    'y_std': float(y_data.std()),
                    'y_min': float(y_data.min()),
                    'y_max': float(y_data.max()),
                    'distance_mean': float(distances.mean()),
                    'distance_std': float(distances.std()),
                    'distance_min': float(distances.min()),
                    'distance_max': float(distances.max())
                }
    
    return stats

def prepare_three_system_comparison(
    target_bag: str,
    df_sonar: pd.DataFrame,
    df_nav: pd.DataFrame, 
    df_fft: pd.DataFrame,
    tolerance_seconds: float = 0.5
) -> Tuple[pd.DataFrame, Dict, 'NetRelativeVisualizer']:
    """
    Prepare synchronized data and create comparison visualizations.
    
    Returns:
        Tuple of (sync_df, figs_dict, visualizer_instance)
    """
    from utils.multi_system_sync import (
        NetRelativePositionCalculator,
        MultiSystemSynchronizer,
        NetRelativeVisualizer
    )
    
    # Initialize components
    calculator = NetRelativePositionCalculator()
    synchronizer = MultiSystemSynchronizer(tolerance_seconds=tolerance_seconds)
    visualizer = NetRelativeVisualizer()
    
    # Calculate positions for each system
    df_sonar_pos = calculator.calculate_sonar_net_position(df_sonar) if df_sonar is not None else None
    df_nav_pos = calculator.calculate_dvl_net_position(df_nav) if df_nav is not None else None
    
    # Process FFT data (already has positions typically)
    df_fft_pos = df_fft.copy() if df_fft is not None else None
    
    # Synchronize systems
    sync_df = synchronizer.synchronize_three_systems(df_fft_pos, df_sonar_pos, df_nav_pos)
    
    # Create comparison plots
    figs = {}
    if not sync_df.empty:
        figs['distance_comparison'] = visualizer.create_distance_comparison(sync_df, target_bag)
        figs['pitch_comparison'] = visualizer.create_pitch_comparison(sync_df, target_bag)
    
    return sync_df, figs, visualizer  # Return visualizer instance!

def _prepare_fft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if 'timestamp' not in prepared.columns:
        if 'time' in prepared.columns:
            prepared['timestamp'] = pd.to_datetime(pd.to_numeric(prepared['time'], errors='coerce'), unit='s', errors='coerce')
        else:
            prepared['timestamp'] = pd.NaT
    else:
        prepared['timestamp'] = pd.to_datetime(prepared['timestamp'], errors='coerce')

    if 'distance' in prepared.columns and 'distance_m' not in prepared.columns:
        distances = pd.to_numeric(prepared['distance'], errors='coerce')
        max_abs = float(distances.abs().max()) if distances.notna().any() else 0.0
        if max_abs > 50.0:
            prepared['distance_m'] = distances / 100.0
        else:
            prepared['distance_m'] = distances
    elif 'distance_m' in prepared.columns:
        prepared['distance_m'] = pd.to_numeric(prepared['distance_m'], errors='coerce')

    if 'pitch' in prepared.columns and 'pitch_rad' not in prepared.columns:
        prepared['pitch_rad'] = pd.to_numeric(prepared['pitch'], errors='coerce')
    elif 'pitch_rad' in prepared.columns:
        prepared['pitch_rad'] = pd.to_numeric(prepared['pitch_rad'], errors='coerce')

    if 'pitch_rad' in prepared.columns:
        prepared['pitch_deg'] = np.degrees(prepared['pitch_rad'])
        prepared['fft_x_m'] = prepared['distance_m'] * np.cos(prepared['pitch_rad'])
        prepared['fft_y_m'] = prepared['distance_m'] * np.sin(prepared['pitch_rad'])

    return prepared

def compute_distance_pitch_statistics(sync_df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive distance and pitch statistics for all systems.
    
    Args:
        sync_df: Synchronized DataFrame with distance and pitch columns
        
    Returns:
        Dictionary containing statistics for each system and cross-system comparisons
    """
    stats = {
        'distance': {},
        'pitch': {},
        'distance_differences': {},
        'pitch_differences': {}
    }
    
    # Define column mappings
    distance_cols = {
        'FFT': 'fft_distance_m',
        'Sonar': 'sonar_distance_m',
        'DVL': 'nav_distance_m'
    }
    
    pitch_cols = {
        'FFT': 'fft_pitch_deg',
        'Sonar': 'sonar_pitch_deg',
        'DVL': 'nav_pitch_deg'
    }
    
    # Compute per-system statistics
    for system, col in distance_cols.items():
        if col in sync_df.columns:
            valid_data = sync_df[col].dropna()
            if len(valid_data) > 0:
                stats['distance'][system] = {
                    'count': len(valid_data),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'range': float(valid_data.max() - valid_data.min())
                }
    
    for system, col in pitch_cols.items():
        if col in sync_df.columns:
            valid_data = sync_df[col].dropna()
            if len(valid_data) > 0:
                stats['pitch'][system] = {
                    'count': len(valid_data),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'range': float(valid_data.max() - valid_data.min())
                }
    
    # Compute cross-system comparisons
    system_pairs = [
        ('FFT', 'Sonar', 'fft_distance_m', 'sonar_distance_m'),
        ('FFT', 'DVL', 'fft_distance_m', 'nav_distance_m'),
        ('Sonar', 'DVL', 'sonar_distance_m', 'nav_distance_m')
    ]
    
    for sys1, sys2, col1, col2 in system_pairs:
        if col1 in sync_df.columns and col2 in sync_df.columns:
            valid_mask = sync_df[col1].notna() & sync_df[col2].notna()
            if valid_mask.sum() > 0:
                diff = sync_df.loc[valid_mask, col1] - sync_df.loc[valid_mask, col2]
                pair_name = f"{sys1}-{sys2}"
                stats['distance_differences'][pair_name] = {
                    'count': len(diff),
                    'mean': float(diff.mean()),
                    'std': float(diff.std()),
                    'rmse': float(np.sqrt((diff**2).mean())),
                    'min': float(diff.min()),
                    'max': float(diff.max())
                }
    
    # Pitch differences
    pitch_pairs = [
        ('FFT', 'Sonar', 'fft_pitch_deg', 'sonar_pitch_deg'),
        ('FFT', 'DVL', 'fft_pitch_deg', 'nav_pitch_deg'),
        ('Sonar', 'DVL', 'sonar_pitch_deg', 'nav_pitch_deg')
    ]
    
    for sys1, sys2, col1, col2 in pitch_pairs:
        if col1 in sync_df.columns and col2 in sync_df.columns:
            valid_mask = sync_df[col1].notna() & sync_df[col2].notna()
            if valid_mask.sum() > 0:
                diff = sync_df.loc[valid_mask, col1] - sync_df.loc[valid_mask, col2]
                pair_name = f"{sys1}-{sys2}"
                stats['pitch_differences'][pair_name] = {
                    'count': len(diff),
                    'mean': float(diff.mean()),
                    'std': float(diff.std()),
                    'rmse': float(np.sqrt((diff**2).mean())),
                    'min': float(diff.min()),
                    'max': float(diff.max())
                }
    
    return stats


def print_distance_pitch_statistics(stats: Dict):
    """
    Pretty-print distance and pitch statistics.
    
    Args:
        stats: Statistics dictionary from compute_distance_pitch_statistics
    """
    print("=== DISTANCE & PITCH STATISTICS ===\n")
    
    # Distance measurements
    print("Distance Measurements:")
    for system, data in stats['distance'].items():
        print(f"  {system}:")
        print(f"    Mean: {data['mean']:.3f} m")
        print(f"    Std: {data['std']:.3f} m")
        print(f"    Min: {data['min']:.3f} m")
        print(f"    Max: {data['max']:.3f} m")
        print(f"    Range: {data['range']:.3f} m")
    
    # Pitch measurements
    print("\nPitch/Angle Measurements:")
    for system, data in stats['pitch'].items():
        print(f"  {system}:")
        print(f"    Mean: {data['mean']:.1f}°")
        print(f"    Std: {data['std']:.1f}°")
        print(f"    Min: {data['min']:.1f}°")
        print(f"    Max: {data['max']:.1f}°")
        print(f"    Range: {data['range']:.1f}°")
    
    # Cross-system comparisons
    print("\n=== CROSS-SYSTEM COMPARISONS ===\n")
    
    print("Distance Differences:")
    for pair, data in stats['distance_differences'].items():
        print(f"  {pair}:")
        print(f"    Mean: {data['mean']:.3f} m")
        print(f"    Std: {data['std']:.3f} m")
        print(f"    RMSE: {data['rmse']:.3f} m")
    
    print("\nPitch/Angle Differences:")
    for pair, data in stats['pitch_differences'].items():
        print(f"  {pair}:")
        print(f"    Mean: {data['mean']:.2f}°")
        print(f"    Std: {data['std']:.2f}°")
        print(f"    RMSE: {data['rmse']:.2f}°")

def apply_frame_range_filter(df_sonar: pd.DataFrame, df_nav: pd.DataFrame, df_fft: pd.DataFrame,
                            frame_start: Optional[int] = None, frame_end: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply frame range filtering to all dataframes.
    
    Args:
        df_sonar: Sonar analysis DataFrame
        df_nav: Navigation DataFrame
        df_fft: FFT DataFrame
        frame_start: Start frame index (None = from beginning)
        frame_end: End frame index (None = to end)
        
    Returns:
        Tuple of filtered (df_sonar, df_nav, df_fft)
    """
    if frame_start is None and frame_end is None:
        return df_sonar, df_nav, df_fft
    
    print(f"Applying frame range filter: {frame_start or 'start'} to {frame_end or 'end'}\n")
    
    if df_sonar is not None and not df_sonar.empty:
        original_len = len(df_sonar)
        df_sonar = df_sonar.iloc[frame_start:frame_end].copy()
        print(f"  Sonar: {original_len} → {len(df_sonar)} records")
    
    if df_nav is not None and not df_nav.empty:
        original_len = len(df_nav)
        df_nav = df_nav.iloc[frame_start:frame_end].copy()
        print(f"  DVL: {original_len} → {len(df_nav)} records")
    
    if df_fft is not None and not df_fft.empty:
        original_len = len(df_fft)
        df_fft = df_fft.iloc[frame_start:frame_end].copy()
        print(f"  FFT: {original_len} → {len(df_fft)} records")
    
    print()
    return df_sonar, df_nav, df_fft


def print_data_summaries(df_sonar: pd.DataFrame, df_nav: pd.DataFrame, df_fft: pd.DataFrame):
    """Print data summaries for all three systems."""
    print("\n=== DATA SUMMARIES ===\n")
    
    summaries = {
        "Sonar": summarize_sonar_results(df_sonar),
        "DVL": summarize_navigation_results(df_nav),
        "FFT": summarize_fft_results(df_fft),
    }
    
    for label, stats in summaries.items():
        print(f"{label}:")
        for key, value in stats.items():
            if value is not None:
                if isinstance(value, tuple):
                    print(f"  {key}: {value[0]:.3f} to {value[1]:.3f}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        print()


def print_sample_data(df_sonar: pd.DataFrame, df_nav: pd.DataFrame, df_fft: pd.DataFrame):
    """Print sample data from all three systems."""
    if df_sonar is not None and not df_sonar.empty:
        print("Sonar Analysis Sample:")
        display_cols = ['timestamp', 'distance_meters', 'angle_degrees', 'detection_success']
        display_cols = [c for c in display_cols if c in df_sonar.columns]
        print(df_sonar[display_cols].head(5))
        print()
    
    if df_nav is not None and not df_nav.empty:
        print("DVL Navigation Sample:")
        display_cols = ['timestamp', 'NetDistance', 'NetPitch']
        display_cols = [c for c in display_cols if c in df_nav.columns]
        print(df_nav[display_cols].head(5))
        print()
    
    if df_fft is not None and not df_fft.empty:
        print("FFT Data Sample:")
        display_cols = ['timestamp', 'distance_m', 'pitch_deg']
        display_cols = [c for c in display_cols if c in df_fft.columns]
        print(df_fft[display_cols].head(5))


def ensure_xy_columns(sync_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure XY coordinate columns exist in synchronized DataFrame.
    Handles column renaming and computation if needed.
    
    Args:
        sync_df: Synchronized DataFrame
        
    Returns:
        DataFrame with standardized XY column names
    """
    # Check if XY columns already exist (with _m suffix)
    existing_xy = [c for c in sync_df.columns if ('_x_m' in c or '_y_m' in c or c.endswith('_x') or c.endswith('_y'))]
    
    if existing_xy:
        print(f"✓ XY coordinates already exist: {existing_xy}")
        
        # Rename _m suffix columns to match visualizer expectations
        rename_map = {}
        if 'fft_x_m' in sync_df.columns:
            rename_map['fft_x_m'] = 'fft_x'
            rename_map['fft_y_m'] = 'fft_y'
        if 'sonar_x_m' in sync_df.columns:
            rename_map['sonar_x_m'] = 'sonar_x'
            rename_map['sonar_y_m'] = 'sonar_y'
        if 'nav_x_m' in sync_df.columns:
            rename_map['nav_x_m'] = 'dvl_x'  # Note: nav → dvl
            rename_map['nav_y_m'] = 'dvl_y'
        
        if rename_map:
            sync_df = sync_df.rename(columns=rename_map)
            print(f"   Renamed columns: {list(rename_map.keys())} → {list(rename_map.values())}")
    else:
        # Compute XY if they don't exist at all
        print("⚠️  XY coordinates not found, computing...")
        
        # Compute XY for each system that has distance and pitch
        if 'sonar_distance_m' in sync_df.columns and 'sonar_pitch_deg' in sync_df.columns:
            print("  Computing Sonar XY...")
            sync_df['sonar_x'] = sync_df['sonar_distance_m'] * np.sin(np.radians(sync_df['sonar_pitch_deg']))
            sync_df['sonar_y'] = sync_df['sonar_distance_m'] * np.cos(np.radians(sync_df['sonar_pitch_deg']))
        
        if 'nav_distance_m' in sync_df.columns and 'nav_pitch_deg' in sync_df.columns:
            print("  Computing DVL XY...")
            sync_df['dvl_x'] = sync_df['nav_distance_m'] * np.sin(np.radians(sync_df['nav_pitch_deg']))
            sync_df['dvl_y'] = sync_df['nav_distance_m'] * np.cos(np.radians(sync_df['nav_pitch_deg']))
        
        if 'fft_distance_m' in sync_df.columns and 'fft_pitch_deg' in sync_df.columns:
            print("  Computing FFT XY...")
            sync_df['fft_x'] = sync_df['fft_distance_m'] * np.sin(np.radians(sync_df['fft_pitch_deg']))
            sync_df['fft_y'] = sync_df['fft_distance_m'] * np.cos(np.radians(sync_df['fft_pitch_deg']))
        
        # Report what was computed
        xy_cols = [c for c in sync_df.columns if c.endswith('_x') or c.endswith('_y')]
        print(f"  ✓ Computed XY columns: {xy_cols}")
    
    return sync_df


def print_xy_position_statistics(sync_df: pd.DataFrame):
    """Print XY position statistics for all systems."""
    print("=== POSITION STATISTICS ===\n")
    
    xy_stats = summarize_xy_positions(sync_df)
    
    if xy_stats:
        for system, stats in xy_stats.items():
            print(f"{system}:")
            print(f"  Data points: {stats['count']}")
            print(f"  X position: {stats['x_mean']:.3f} ± {stats['x_std']:.3f} m")
            print(f"  Y position: {stats['y_mean']:.3f} ± {stats['y_std']:.3f} m")
            print(f"  Distance from origin: {stats['distance_mean']:.3f} ± {stats['distance_std']:.3f} m")
            print()
    else:
        print("No XY position data available\n")

def generate_and_save_xy_plots(sync_df: pd.DataFrame, visualizer, target_bag: str, 
                               plots_dir: Path, lateral_speed_m_s: float = 0.2) -> Dict[str, bool]:
    """
    Generate and save all XY position plots.
    
    Args:
        sync_df: Synchronized DataFrame with XY coordinates
        visualizer: NetRelativeVisualizer instance
        target_bag: Target bag identifier for filenames
        plots_dir: Directory to save plots
        lateral_speed_m_s: Lateral speed for 3D trajectory plot
        
    Returns:
        Dictionary with success status for each plot type
    """
    results = {
        'xy_trajectory': False,
        'x_comparison': False,
        'y_comparison': False
    }
    
    print("=== DISPLAYING XY POSITION PLOTS ===\n")
    
    # Check which XY columns are available
    has_sonar_xy = 'sonar_x' in sync_df.columns and 'sonar_y' in sync_df.columns
    has_dvl_xy = 'dvl_x' in sync_df.columns and 'dvl_y' in sync_df.columns
    has_fft_xy = 'fft_x' in sync_df.columns and 'fft_y' in sync_df.columns
    
    available_xy_systems = []
    if has_sonar_xy:
        available_xy_systems.append("Sonar")
    if has_dvl_xy:
        available_xy_systems.append("DVL")
    if has_fft_xy:
        available_xy_systems.append("FFT")
    
    print(f"Systems with XY coordinates: {', '.join(available_xy_systems)}")
    
    if len(available_xy_systems) == 0:
        print("✗ No XY position data available")
        print("  XY coordinates require distance and pitch measurements")
        return results
    
    # Generate 3D XY trajectory plot
    print(f"\nGenerating 3D XY trajectory plot...")
    print(f"  Lateral speed: {lateral_speed_m_s} m/s")
    try:
        fig_xy = visualizer.plot_xy_trajectories(sync_df, lateral_speed_m_s=lateral_speed_m_s)
        
        if fig_xy is not None:
            # Display the plot inline
            try:
                from IPython.display import display
                display(fig_xy)
            except:
                fig_xy.show()
            
            # Save plot
            save_path = plots_dir / f"{target_bag}_xy_trajectories_3d.html"
            fig_xy.write_html(str(save_path))
            print(f"✓ Saved: {save_path.name}\n")
            results['xy_trajectory'] = True
        else:
            print("✗ Could not generate XY trajectory plot\n")
            
    except Exception as e:
        print(f"✗ Error generating XY plot: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Generate individual XY component comparison plots if multiple systems available
    if len(available_xy_systems) >= 2:
        print("\nGenerating XY component comparison plots...")
        try:
            # X position comparison (perpendicular distance to net)
            print("\n--- Perpendicular Distance to Net Over Time ---")
            fig_x = visualizer.plot_xy_component_comparison(sync_df, component='x')
            if fig_x is not None:
                try:
                    from IPython.display import display
                    display(fig_x)
                except:
                    fig_x.show()
                save_path = plots_dir / f"{target_bag}_x_position_comparison.html"
                fig_x.write_html(str(save_path))
                print(f"✓ X position comparison saved: {save_path.name}\n")
                results['x_comparison'] = True
            
            # Y position comparison (lateral position along net)
            print("--- Lateral Position Along Net Over Time ---")
            fig_y = visualizer.plot_xy_component_comparison(sync_df, component='y')
            if fig_y is not None:
                try:
                    from IPython.display import display
                    display(fig_y)
                except:
                    fig_y.show()
                save_path = plots_dir / f"{target_bag}_y_position_comparison.html"
                fig_y.write_html(str(save_path))
                print(f"✓ Y position comparison saved: {save_path.name}\n")
                results['y_comparison'] = True
            
        except Exception as e:
            print(f"✗ Error generating component plots: {e}")
            import traceback
            traceback.print_exc()
    
    print("✓ XY position plots complete")
    return results