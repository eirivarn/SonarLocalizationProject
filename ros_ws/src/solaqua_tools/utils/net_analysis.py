from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from matplotlib.colors import to_rgb

from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS


def _infer_unix_unit(values: pd.Series) -> Optional[str]:
    """Guess time unit from numeric magnitude."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return None
    max_abs = float(numeric.abs().max())
    if max_abs > 1e16:
        return "ns"
    if max_abs > 1e13:
        return "us"
    if max_abs > 1e11:
        return "ms"
    return "s"


def _parse_timestamp_col(series: pd.Series) -> pd.Series:
    """
    Parse a timestamp column that may be ISO strings or Unix epochs (s/ms/us/ns).
    """
    unit = _infer_unix_unit(series) if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_integer_dtype(series) else None
    if unit:
        return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def _resolve_export_paths(exports_dir: str | Path | None = None) -> Tuple[Path, Path, Path]:
    root = Path(exports_dir) if exports_dir else Path(EXPORTS_DIR_DEFAULT)
    outputs_dir = root / EXPORTS_SUBDIRS.get('outputs', 'outputs')
    by_bag_dir = root / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    return root, outputs_dir, by_bag_dir

def load_sonar_analysis_results(
    target_bag: str,
    exports_dir: str | Path | None = None,
    filenames: Optional[List[str]] = None,
    filename_suffix: str = "_analysis.csv"
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    base_dir = (
        Path(exports_dir).expanduser().resolve()
        if exports_dir is not None
        else Path(SONAR_ANALYSIS_DIR).expanduser().resolve()
        if "SONAR_ANALYSIS_DIR" in globals() and SONAR_ANALYSIS_DIR is not None
        else (Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get("outputs", "outputs")).expanduser().resolve()
    )
    candidate_files = [
        base_dir / f"{target_bag}_analysis.csv",
        base_dir / f"{target_bag}_data_cones_analysis.csv",
    ]
    csv_path = next((p for p in candidate_files if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"Sonar analysis CSV not found. Tried: {', '.join(str(p) for p in candidate_files)}"
        )

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    if "timestamp" in df.columns:
        df["timestamp"] = _parse_timestamp_col(df["timestamp"])
    else:
        print(f"[load_sonar_analysis_results] ⚠️  'timestamp' column missing in {csv_path.name}")

    metadata = {
        "rows": len(df),
        "resolved_path": str(csv_path),
        "paths_checked": [str(csv_path)],
    }
    
    if "timestamp" in df.columns:
        df["timestamp"] = _parse_timestamp_col(df["timestamp"])
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    
    return df, metadata

def load_navigation_dataset(
    target_bag: str,
    exports_dir: str | Path | None = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    _, _, by_bag_dir = _resolve_export_paths(exports_dir)
    nav_path = by_bag_dir / f"navigation_plane_approximation__{target_bag}_data.csv"

    if nav_path.exists():
        df = pd.read_csv(nav_path)
        timestamp_source = "timestamp" if "timestamp" in df.columns else "ts_utc"
        df["timestamp"] = _parse_timestamp_col(df[timestamp_source])
        df = df.sort_values("timestamp")
        return df, {"resolved_path": str(nav_path), "rows": len(df)}

    try:
        import utils.distance_measurement as distance_api

        raw_data, _ = distance_api.load_all_distance_data_for_bag(target_bag, by_bag_dir)
        df = raw_data.get("navigation") if raw_data else None
        if df is not None:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")
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
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="ISO-8859-1")
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

    if "distance_meters" in df.columns:
        distances = df["distance_meters"].dropna()
        if not distances.empty:
            summary["distance_range_m"] = (float(distances.min()), float(distances.max()))

    if "angle_degrees" in df.columns:
        angles = df["angle_degrees"].dropna()
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
    if "NetDistance" in df.columns:
        dist = df["NetDistance"].dropna()
        if not dist.empty:
            summary["distance_range_m"] = (float(dist.min()), float(dist.max()))
            summary["distance_mean_m"] = float(dist.mean())
            summary["distance_std_m"] = float(dist.std())

    if "NetPitch" in df.columns:
        pitch = np.degrees(df["NetPitch"].dropna())
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
    if "distance_m" in df.columns:
        distances = pd.to_numeric(df["distance_m"], errors="coerce").dropna()
        if not distances.empty:
            summary["distance_range_m"] = (float(distances.min()), float(distances.max()))
            summary["distance_mean_m"] = float(distances.mean())
            summary["distance_std_m"] = float(distances.std())

    if "pitch_deg" in df.columns:
        pitch = pd.to_numeric(df["pitch_deg"], errors="coerce").dropna()
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
    sonar["timestamp"] = _parse_timestamp_col(sonar["timestamp"])
    sonar = sonar.dropna(subset=["timestamp"]).sort_values("timestamp")

    nav = df_nav.copy()
    timestamp_col = "timestamp" if "timestamp" in nav.columns else "ts_utc"
    nav["timestamp"] = _parse_timestamp_col(nav[timestamp_col])
    nav = nav.dropna(subset=["timestamp"]).sort_values("timestamp")

    if nav.empty or sonar.empty:
        return pd.DataFrame()

    nav_times = nav["timestamp"].to_numpy(dtype="datetime64[ns]")
    rows: List[Dict[str, Any]] = []

    for sonar_row in sonar.itertuples(index=False):
        sonar_ts = np.datetime64(sonar_row.timestamp)
        deltas = np.abs(nav_times - sonar_ts)
        idx = int(deltas.argmin())
        diff_sec = float(deltas[idx] / np.timedelta64(1, "s"))
        if diff_sec <= tolerance_seconds:
            nav_row = nav.iloc[idx]
            rows.append({
                "timestamp": pd.Timestamp(sonar_row.timestamp),
                "sonar_distance_m": getattr(sonar_row, "distance_meters", None),
                "sonar_angle_deg": getattr(sonar_row, "angle_degrees", None),
                "sonar_detection": bool(getattr(sonar_row, "detection_success", False)),
                "dvl_distance_m": nav_row.get("NetDistance"),
                "dvl_pitch_deg": float(np.degrees(nav_row["NetPitch"])) if "NetPitch" in nav_row and pd.notna(nav_row["NetPitch"]) else None,
                "time_diff_s": diff_sec,
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

    valid = sync_df["sonar_distance_m"].notna() & sync_df["dvl_distance_m"].notna()
    summary["valid_pairs"] = int(valid.sum())
    if not valid.any():
        return summary

    diff = sync_df.loc[valid, "sonar_distance_m"] - sync_df.loc[valid, "dvl_distance_m"]
    stats = {
        "mean": float(diff.mean()),
        "std": float(diff.std()),
        "min": float(diff.min()),
        "max": float(diff.max()),
        "p95": float(np.percentile(diff, 95)),
    }
    if len(diff) > 10:
        stats["correlation"] = float(sync_df.loc[valid, "sonar_distance_m"].corr(sync_df.loc[valid, "dvl_distance_m"]))
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
        ("FFT", ["fft_x_m", "fft_y_m", "fft_x", "fft_y"]),
        ("Sonar", ["sonar_x_m", "sonar_y_m", "sonar_x", "sonar_y"]),
        ("DVL", ["nav_x_m", "nav_y_m", "dvl_x", "dvl_y"])
    ]
    
    for system_name, possible_cols in systems:
        # Find which column names actually exist
        x_col = None
        y_col = None
        
        for col in possible_cols:
            if col in sync_df.columns:
                if "_x" in col:
                    x_col = col
                elif "_y" in col:
                    y_col = col
        
        if x_col and y_col:
            valid_mask = sync_df[x_col].notna() & sync_df[y_col].notna()
            
            if valid_mask.sum() > 0:
                x_data = sync_df.loc[valid_mask, x_col]
                y_data = sync_df.loc[valid_mask, y_col]
                
                # Calculate distance from origin
                distances = np.sqrt(x_data**2 + y_data**2)
                
                stats[system_name] = {
                    "count": len(x_data),
                    "x_mean": float(x_data.mean()),
                    "x_std": float(x_data.std()),
                    "x_min": float(x_data.min()),
                    "x_max": float(x_data.max()),
                    "y_mean": float(y_data.mean()),
                    "y_std": float(y_data.std()),
                    "y_min": float(y_data.min()),
                    "y_max": float(y_data.max()),
                    "distance_mean": float(distances.mean()),
                    "distance_std": float(distances.std()),
                    "distance_min": float(distances.min()),
                    "distance_max": float(distances.max())
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
        figs["distance_comparison"] = visualizer.create_distance_comparison(sync_df, target_bag)
        figs["pitch_comparison"] = visualizer.create_pitch_comparison(sync_df, target_bag)
    
    return sync_df, figs, visualizer  # Return visualizer instance!

def _prepare_fft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "timestamp" not in prepared.columns:
        if "time" in prepared.columns:
            prepared["timestamp"] = pd.to_datetime(pd.to_numeric(prepared["time"], errors="coerce"), unit="s", errors="coerce")
        else:
            prepared["timestamp"] = pd.NaT
    else:
        prepared["timestamp"] = _parse_timestamp_col(prepared["timestamp"])

    if "distance" in prepared.columns and "distance_m" not in prepared.columns:
        distances = pd.to_numeric(prepared["distance"], errors="coerce")
        max_abs = float(distances.abs().max()) if distances.notna().any() else 0.0
        if max_abs > 50.0:
            prepared["distance_m"] = distances / 100.0
        else:
            prepared["distance_m"] = distances
    elif "distance_m" in prepared.columns:
        prepared["distance_m"] = pd.to_numeric(prepared["distance_m"], errors="coerce")

    if "pitch" in prepared.columns and "pitch_rad" not in prepared.columns:
        prepared["pitch_rad"] = pd.to_numeric(prepared["pitch"], errors="coerce")
    elif "pitch_rad" in prepared.columns:
        prepared["pitch_rad"] = pd.to_numeric(prepared["pitch_rad"], errors="coerce")

    if "pitch_rad" in prepared.columns:
        prepared["pitch_deg"] = np.degrees(prepared["pitch_rad"])
        prepared["fft_x_m"] = prepared["distance_m"] * np.cos(prepared["pitch_rad"])
        prepared["fft_y_m"] = prepared["distance_m"] * np.sin(prepared["pitch_rad"])

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
        "distance": {},
        "pitch": {},
        "distance_differences": {},
        "pitch_differences": {}
    }
    
    # Define column mappings
    distance_cols = {
        "FFT": "fft_distance_m",
        "Sonar": "sonar_distance_m",
        "DVL": "nav_distance_m"
    }
    
    pitch_cols = {
        "FFT": "fft_pitch_deg",
        "Sonar": "sonar_pitch_deg",
        "DVL": "nav_pitch_deg"
    }
    
    # Compute per-system statistics
    for system, col in distance_cols.items():
        if col in sync_df.columns:
            valid_data = sync_df[col].dropna()
            if len(valid_data) > 0:
                stats["distance"][system] = {
                    "count": len(valid_data),
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "range": float(valid_data.max() - valid_data.min())
                }
    
    for system, col in pitch_cols.items():
        if col in sync_df.columns:
            valid_data = sync_df[col].dropna()
            if len(valid_data) > 0:
                stats["pitch"][system] = {
                    "count": len(valid_data),
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "range": float(valid_data.max() - valid_data.min())
                }
    
    # Compute cross-system comparisons
    system_pairs = [
        ("FFT", "Sonar", "fft_distance_m", "sonar_distance_m"),
        ("FFT", "DVL", "fft_distance_m", "nav_distance_m"),
        ("Sonar", "DVL", "sonar_distance_m", "nav_distance_m")
    ]
    
    for sys1, sys2, col1, col2 in system_pairs:
        if col1 in sync_df.columns and col2 in sync_df.columns:
            valid_mask = sync_df[col1].notna() & sync_df[col2].notna()
            if valid_mask.sum() > 0:
                diff = sync_df.loc[valid_mask, col1] - sync_df.loc[valid_mask, col2]
                pair_name = f"{sys1}-{sys2}"
                stats["distance_differences"][pair_name] = {
                    "count": len(diff),
                    "mean": float(diff.mean()),
                    "std": float(diff.std()),
                    "rmse": float(np.sqrt((diff**2).mean())),
                    "min": float(diff.min()),
                    "max": float(diff.max())
                }
    
    # Pitch differences
    pitch_pairs = [
        ("FFT", "Sonar", "fft_pitch_deg", "sonar_pitch_deg"),
        ("FFT", "DVL", "fft_pitch_deg", "nav_pitch_deg"),
        ("Sonar", "DVL", "sonar_pitch_deg", "nav_pitch_deg")
    ]
    
    for sys1, sys2, col1, col2 in pitch_pairs:
        if col1 in sync_df.columns and col2 in sync_df.columns:
            valid_mask = sync_df[col1].notna() & sync_df[col2].notna()
            if valid_mask.sum() > 0:
                diff = sync_df.loc[valid_mask, col1] - sync_df.loc[valid_mask, col2]
                pair_name = f"{sys1}-{sys2}"
                stats["pitch_differences"][pair_name] = {
                    "count": len(diff),
                    "mean": float(diff.mean()),
                    "std": float(diff.std()),
                    "rmse": float(np.sqrt((diff**2).mean())),
                    "min": float(diff.min()),
                    "max": float(diff.max())
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
    for system, data in stats["distance"].items():
        print(f"  {system}:")
        print(f"    Mean: {data['mean']:.3f} m")
        print(f"    Std: {data['std']:.3f} m")
        print(f"    Min: {data['min']:.3f} m")
        print(f"    Max: {data['max']:.3f} m")
        print(f"    Range: {data['range']:.3f} m")
    
    # Pitch measurements
    print("\nPitch/Angle Measurements:")
    for system, data in stats["pitch"].items():
        print(f"  {system}:")
        print(f"    Mean: {data['mean']:.1f}°")
        print(f"    Std: {data['std']:.1f}°")
        print(f"    Min: {data['min']:.1f}°")
        print(f"    Max: {data['max']:.1f}°")
        print(f"    Range: {data['range']:.1f}°")
    
    # Cross-system comparisons
    print("\n=== CROSS-SYSTEM COMPARISONS ===\n")
    
    print("Distance Differences:")
    for pair, data in stats["distance_differences"].items():
        print(f"  {pair}:")
        print(f"    Mean: {data['mean']:.3f} m")
        print(f"    Std: {data['std']:.3f} m")
        print(f"    RMSE: {data['rmse']:.3f} m")
    
    print("\nPitch/Angle Differences:")
    for pair, data in stats["pitch_differences"].items():
        print(f"  {pair}:")
        print(f"    Mean: {data['mean']:.2f}°")
        print(f"    Std: {data['std']:.2f}°")
        print(f"    RMSE: {data['rmse']:.2f}°")

def apply_time_range_filter(df_sonar: pd.DataFrame, df_nav: pd.DataFrame, df_fft: pd.DataFrame,
                           time_start: Optional[str] = None, time_end: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply time range filtering to all dataframes using timestamps.
    
    Args:
        df_sonar: Sonar analysis DataFrame
        df_nav: Navigation DataFrame
        df_fft: FFT DataFrame
        time_start: Start time (ISO format string or seconds, None = from beginning)
        time_end: End time (ISO format string or seconds, None = to end)
        
    Returns:
        Tuple of filtered (df_sonar, df_nav, df_fft)
    """
    if time_start is None and time_end is None:
        return df_sonar, df_nav, df_fft
    
    print(f"Applying time range filter: {time_start or 'start'} to {time_end or 'end'}\n")
    
    # Convert time strings to datetime if needed
    if time_start is not None:
        if isinstance(time_start, (int, float)):
            # Relative seconds from start
            time_start_dt = None  # Will be handled per-dataframe
            time_start_seconds = time_start
        else:
            time_start_dt = pd.to_datetime(time_start, utc=True)
            time_start_seconds = None
    else:
        time_start_dt = None
        time_start_seconds = None
    
    if time_end is not None:
        if isinstance(time_end, (int, float)):
            time_end_dt = None
            time_end_seconds = time_end
        else:
            time_end_dt = pd.to_datetime(time_end, utc=True)
            time_end_seconds = None
    else:
        time_end_dt = None
        time_end_seconds = None
    
    # Filter sonar
    if df_sonar is not None and not df_sonar.empty and "timestamp" in df_sonar.columns:
        original_len = len(df_sonar)
        df_sonar_ts = pd.to_datetime(df_sonar["timestamp"], utc=True)
        
        if time_start_seconds is not None:
            # Relative time from first timestamp
            first_time = df_sonar_ts.min()
            time_start_dt = first_time + pd.Timedelta(seconds=time_start_seconds)
        
        if time_end_seconds is not None:
            first_time = df_sonar_ts.min()
            time_end_dt = first_time + pd.Timedelta(seconds=time_end_seconds)
        
        mask = pd.Series(True, index=df_sonar.index)
        if time_start_dt is not None:
            mask &= (df_sonar_ts >= time_start_dt)
        if time_end_dt is not None:
            mask &= (df_sonar_ts <= time_end_dt)
        
        df_sonar = df_sonar[mask].copy()
        print(f"  Sonar: {original_len} → {len(df_sonar)} records")
    
    # Filter DVL
    if df_nav is not None and not df_nav.empty and "timestamp" in df_nav.columns:
        original_len = len(df_nav)
        df_nav_ts = pd.to_datetime(df_nav["timestamp"], utc=True)
        
        if time_start_seconds is not None:
            first_time = df_nav_ts.min()
            time_start_dt = first_time + pd.Timedelta(seconds=time_start_seconds)
        
        if time_end_seconds is not None:
            first_time = df_nav_ts.min()
            time_end_dt = first_time + pd.Timedelta(seconds=time_end_seconds)
        
        mask = pd.Series(True, index=df_nav.index)
        if time_start_dt is not None:
            mask &= (df_nav_ts >= time_start_dt)
        if time_end_dt is not None:
            mask &= (df_nav_ts <= time_end_dt)
        
        df_nav = df_nav[mask].copy()
        print(f"  DVL: {original_len} → {len(df_nav)} records")
    
    # Filter FFT
    if df_fft is not None and not df_fft.empty and "timestamp" in df_fft.columns:
        original_len = len(df_fft)
        df_fft_ts = pd.to_datetime(df_fft["timestamp"], utc=True)
        
        if time_start_seconds is not None:
            first_time = df_fft_ts.min()
            time_start_dt = first_time + pd.Timedelta(seconds=time_start_seconds)
        
        if time_end_seconds is not None:
            first_time = df_fft_ts.min()
            time_end_dt = first_time + pd.Timedelta(seconds=time_end_seconds)
        
        mask = pd.Series(True, index=df_fft.index)
        if time_start_dt is not None:
            mask &= (df_fft_ts >= time_start_dt)
        if time_end_dt is not None:
            mask &= (df_fft_ts <= time_end_dt)
        
        df_fft = df_fft[mask].copy()
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
    };
    
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
        display_cols = ["timestamp", "distance_meters", "angle_degrees", "detection_success"]
        display_cols = [c for c in display_cols if c in df_sonar.columns]
        print(df_sonar[display_cols].head(5))
        print()
    
    if df_nav is not None and not df_nav.empty:
        print("DVL Navigation Sample:")
        display_cols = ["timestamp", "NetDistance", "NetPitch"]
        display_cols = [c for c in display_cols if c in df_nav.columns]
        print(df_nav[display_cols].head(5))
        print()
    
    if df_fft is not None and not df_fft.empty:
        print("FFT Data Sample:")
        display_cols = ["timestamp", "distance_m", "pitch_deg"]
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
    existing_xy = [c for c in sync_df.columns if ("_x_m" in c or "_y_m" in c or c.endswith("_x") or c.endswith("_y"))]
    
    if existing_xy:
        print(f"✓ XY coordinates already exist: {existing_xy}")
        
        # Rename _m suffix columns to match visualizer expectations
        rename_map = {}
        if "fft_x_m" in sync_df.columns:
            rename_map["fft_x_m"] = "fft_x"
            rename_map["fft_y_m"] = "fft_y"
        if "sonar_x_m" in sync_df.columns:
            rename_map["sonar_x_m"] = "sonar_x"
            rename_map["sonar_y_m"] = "sonar_y"
        if "nav_x_m" in sync_df.columns:
            rename_map["nav_x_m"] = "dvl_x"  # Note: nav → dvl
            rename_map["nav_y_m"] = "dvl_y"
        
        if rename_map:
            sync_df = sync_df.rename(columns=rename_map)
            print(f"   Renamed columns: {list(rename_map.keys())} → {list(rename_map.values())}")
    else:
        # Compute XY if they don't exist at all
        print("⚠️  XY coordinates not found, computing...")
        
        # Compute XY for each system that has distance and pitch
        if "sonar_distance_m" in sync_df.columns and "sonar_pitch_deg" in sync_df.columns:
            print("  Computing Sonar XY...")
            sync_df["sonar_x"] = sync_df["sonar_distance_m"] * np.sin(np.radians(sync_df["sonar_pitch_deg"]))
            sync_df["sonar_y"] = sync_df["sonar_distance_m"] * np.cos(np.radians(sync_df["sonar_pitch_deg"]))
        
        if "nav_distance_m" in sync_df.columns and "nav_pitch_deg" in sync_df.columns:
            print("  Computing DVL XY...")
            sync_df["dvl_x"] = sync_df["nav_distance_m"] * np.sin(np.radians(sync_df["nav_pitch_deg"]))
            sync_df["dvl_y"] = sync_df["nav_distance_m"] * np.cos(np.radians(sync_df["nav_pitch_deg"]))
        
        if "fft_distance_m" in sync_df.columns and "fft_pitch_deg" in sync_df.columns:
            print("  Computing FFT XY...")
            sync_df["fft_x"] = sync_df["fft_distance_m"] * np.sin(np.radians(sync_df["fft_pitch_deg"]))
            sync_df["fft_y"] = sync_df["fft_distance_m"] * np.cos(np.radians(sync_df["fft_pitch_deg"]))
        
        # Report what was computed
        xy_cols = [c for c in sync_df.columns if c.endswith("_x") or c.endswith("_y")]
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
        "xy_trajectory": False,
        "x_comparison": False,
        "y_comparison": False
    }
    
    print("=== DISPLAYING XY POSITION PLOTS ===\n")
    
    # Check which XY columns are available
    has_sonar_xy = "sonar_x" in sync_df.columns and "sonar_y" in sync_df.columns
    has_dvl_xy = "dvl_x" in sync_df.columns and "dvl_y" in sync_df.columns
    has_fft_xy = "fft_x" in sync_df.columns and "fft_y" in sync_df.columns
    
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
            results["xy_trajectory"] = True
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
            fig_x = visualizer.plot_xy_component_comparison(sync_df, component="x")
            if fig_x is not None:
                try:
                    from IPython.display import display
                    display(fig_x)
                except:
                    fig_x.show()
                save_path = plots_dir / f"{target_bag}_x_position_comparison.html"
                fig_x.write_html(str(save_path))
                print(f"✓ X position comparison saved: {save_path.name}\n")
                results["x_comparison"] = True
            
            # Y position comparison (lateral position along net)
            print("--- Lateral Position Along Net Over Time ---")
            fig_y = visualizer.plot_xy_component_comparison(sync_df, component="y")
            if fig_y is not None:
                try:
                    from IPython.display import display
                    display(fig_y)
                except:
                    fig_y.show()
                save_path = plots_dir / f"{target_bag}_y_position_comparison.html"
                fig_y.write_html(str(save_path))
                print(f"✓ Y position comparison saved: {save_path.name}\n")
                results["y_comparison"] = True
            
        except Exception as e:
            print(f"✗ Error generating component plots: {e}")
            import traceback
            traceback.print_exc()
    
    print("✓ XY position plots complete")
    return results

def apply_exponential_smoothing(df: pd.DataFrame, columns: List[str], alpha: float = 0.3) -> pd.DataFrame:
    """
    Apply exponential moving average smoothing to specified columns.
    This is a causal filter suitable for real-time processing.
    
    Args:
        df: DataFrame to smooth
        columns: List of column names to smooth
        alpha: Smoothing factor (0-1). Higher = less smoothing, more responsive
               0.1 = heavy smoothing (90% history, 10% current)
               0.5 = balanced (50% history, 50% current)
               0.9 = light smoothing (10% history, 90% current)
        
    Returns:
        DataFrame with smoothed columns
    """
    if df is None or df.empty:
        return df
    
    result_df = df.copy()
    
    for col in columns:
        if col in result_df.columns:
            # Apply exponential weighted moving average
            # This only uses past and current values (causal/real-time compatible)
            result_df[col] = result_df[col].ewm(alpha=alpha, adjust=False).mean()
    
    return result_df


def apply_smoothing_to_all_systems(df_sonar: pd.DataFrame, df_nav: pd.DataFrame, 
                                   df_fft: pd.DataFrame, alpha: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply exponential smoothing to distance and pitch measurements for all systems.
    
    Args:
        df_sonar: Sonar DataFrame
        df_nav: Navigation DataFrame
        df_fft: FFT DataFrame
        alpha: Smoothing factor (0-1)
        
    Returns:
        Tuple of smoothed (df_sonar, df_nav, df_fft)
    """
    if alpha is None or alpha >= 1.0:
        # No smoothing
        return df_sonar, df_nav, df_fft
    
    print(f"\n=== APPLYING EXPONENTIAL SMOOTHING (α={alpha:.2f}) ===")
    print(f"  Lower α = more smoothing (e.g., 0.1 = heavy smoothing)")
    print(f"  Higher α = less smoothing (e.g., 0.9 = light smoothing)\n")
    
    # Sonar columns to smooth
    sonar_cols = ["distance_meters", "angle_degrees"]
    if df_sonar is not None and not df_sonar.empty:
        before_dist = df_sonar["distance_meters"].mean() if "distance_meters" in df_sonar.columns else None
        df_sonar = apply_exponential_smoothing(df_sonar, sonar_cols, alpha)
        after_dist = df_sonar["distance_meters"].mean() if "distance_meters" in df_sonar.columns else None
        if before_dist and after_dist:
            print(f"  Sonar distance: {before_dist:.3f} → {after_dist:.3f} m (mean)")
    
    # DVL/Navigation columns to smooth
    nav_cols = ["NetDistance", "NetPitch"]
    if df_nav is not None and not df_nav.empty:
        before_dist = df_nav["NetDistance"].mean() if "NetDistance" in df_nav.columns else None
        df_nav = apply_exponential_smoothing(df_nav, nav_cols, alpha)
        after_dist = df_nav["NetDistance"].mean() if "NetDistance" in df_nav.columns else None
        if before_dist and after_dist:
            print(f"  DVL distance: {before_dist:.3f} → {after_dist:.3f} m (mean)")
    
    # FFT columns to smooth
    fft_cols = ["distance_m", "pitch_deg"]
    if df_fft is not None and not df_fft.empty:
        before_dist = df_fft["distance_m"].mean() if "distance_m" in df_fft.columns else None
        df_fft = apply_exponential_smoothing(df_fft, fft_cols, alpha)
        after_dist = df_fft["distance_m"].mean() if "distance_m" in df_fft.columns else None
        if before_dist and after_dist:
            print(f"  FFT distance: {before_dist:.3f} → {after_dist:.3f} m (mean)")
    
    print("✓ Smoothing applied\n")
    
    return df_sonar, df_nav, df_fft

def compute_temporal_stability_metrics(sync_df: pd.DataFrame) -> Dict:
    """
    Compute temporal stability metrics for quality assessment without ground truth.
    
    Args:
        sync_df: Synchronized DataFrame
        
    Returns:
        Dictionary with stability metrics for each system
    """
    metrics = {}
    
    systems = {
        "FFT": ("fft_distance_m", "fft_pitch_deg"),
        "Sonar": ("sonar_distance_m", "sonar_pitch_deg"),
        "DVL": ("nav_distance_m", "nav_pitch_deg")
    }
    
    for system, (dist_col, pitch_col) in systems.items():
        if dist_col in sync_df.columns and pitch_col in sync_df.columns:
            # Compute first-order differences (rate of change)
            dist_diff = sync_df[dist_col].diff().dropna()
            pitch_diff = sync_df[pitch_col].diff().dropna()
            
            # Temporal stability = inverse of standard deviation of changes
            # Lower change variance = more stable
            metrics[system] = {
                "distance_change_std": float(dist_diff.std()),
                "distance_change_mean_abs": float(dist_diff.abs().mean()),
                "pitch_change_std": float(pitch_diff.std()),
                "pitch_change_mean_abs": float(pitch_diff.abs().mean()),
                "distance_smoothness": float(1.0 / (dist_diff.std() + 1e-6)),  # Inverse of variance
                "pitch_smoothness": float(1.0 / (pitch_diff.std() + 1e-6))
            }
    
    return metrics


def compute_inter_system_agreement(sync_df: pd.DataFrame) -> Dict:
    """
    Compute agreement metrics between systems (correlation, MAE, consistency).
    
    Args:
        sync_df: Synchronized DataFrame
        
    """
    agreement = {}
    
    pairs = [
        ("FFT", "Sonar", "fft_distance_m", "sonar_distance_m", "fft_pitch_deg", "sonar_pitch_deg"),
        ("FFT", "DVL", "fft_distance_m", "nav_distance_m", "fft_pitch_deg", "nav_pitch_deg"),
        ("Sonar", "DVL", "sonar_distance_m", "nav_distance_m", "sonar_pitch_deg", "nav_pitch_deg")
    ]
    
    for sys1, sys2, d1, d2, p1, p2 in pairs:
        pair_name = f"{sys1}-{sys2}"
        
        # Distance agreement
        if d1 in sync_df.columns and d2 in sync_df.columns:
            valid = sync_df[[d1, d2]].dropna()
            if len(valid) > 1:
                corr = valid[d1].corr(valid[d2])
                mae = (valid[d1] - valid[d2]).abs().mean()
                agreement[pair_name] = {
                    "distance_correlation": float(corr),
                    "distance_mae": float(mae),
                    "distance_agreement_score": float(corr * (1.0 / (mae + 0.1)))  # Combined metric
                }
        
        # Pitch agreement
        if p1 in sync_df.columns and p2 in sync_df.columns:
            valid = sync_df[[p1, p2]].dropna()
            if len(valid) > 1:
                corr = valid[p1].corr(valid[p2])
                mae = (valid[p1] - valid[p2]).abs().mean()
                if pair_name not in agreement:
                    agreement[pair_name] = {}
                agreement[pair_name].update({
                    "pitch_correlation": float(corr),
                    "pitch_mae_deg": float(mae),
                    "pitch_agreement_score": float(corr * (1.0 / (mae + 1.0)))
                })
    
    return agreement


def create_statistics_visualizations(sync_df: pd.DataFrame, target_bag: str) -> Dict[str, go.Figure]:
    """
    Create comprehensive statistical visualization plots.
    
    Args:
        sync_df: Synchronized DataFrame
        target_bag: Target bag identifier
        
    Returns:
        Dictionary of plotly figures
    """
    figs = {}
    
    # 1. Distribution plots (histograms + box plots)
    fig_dist = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Distance Distributions", "Distance Box Plots",
                       "Pitch Distributions", "Pitch Box Plots"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    systems = [
        ("FFT", "fft_distance_m", "fft_pitch_deg", "red"),
        ("Sonar", "sonar_distance_m", "sonar_pitch_deg", "blue"),
        ("DVL", "nav_distance_m", "nav_pitch_deg", "green")
    ]
    
    for name, dist_col, pitch_col, color in systems:
        if dist_col in sync_df.columns:
            # Distance histogram
            fig_dist.add_trace(
                go.Histogram(x=sync_df[dist_col].dropna(), name=f"{name}",
                           marker_color=color, opacity=0.6, nbinsx=30),
                row=1, col=1
            )
            # Distance box plot
            fig_dist.add_trace(
                go.Box(y=sync_df[dist_col].dropna(), name=f"{name}",
                      marker_color=color),
                row=1, col=2
            )
        
        if pitch_col in sync_df.columns:
            # Pitch histogram
            fig_dist.add_trace(
                go.Histogram(x=sync_df[pitch_col].dropna(), name=f"{name}",
                           marker_color=color, opacity=0.6, nbinsx=30,
                           showlegend=False),
                row=2, col=1
            )
            # Pitch box plot
            fig_dist.add_trace(
                go.Box(y=sync_df[pitch_col].dropna(), name=f"{name}",
                      marker_color=color, showlegend=False),
                row=2, col=2
            )
    
    fig_dist.update_xaxes(title_text="Distance (m)", row=1, col=1)
    fig_dist.update_xaxes(title_text="Distance (m)", row=1, col=2)
    fig_dist.update_xaxes(title_text="Pitch (°)", row=2, col=1)
    fig_dist.update_xaxes(title_text="Pitch (°)", row=2, col=2)
    fig_dist.update_layout(height=800, title_text=f"Measurement Distributions: {target_bag}")
    figs["distributions"] = fig_dist
    
    # 2. Temporal stability plot (rate of change)
    fig_stability = go.Figure()
    
    for name, dist_col, pitch_col, color in systems:
        if dist_col in sync_df.columns:
            dist_changes = sync_df[dist_col].diff().abs()
            fig_stability.add_trace(
                go.Scatter(x=sync_df["sync_timestamp"], y=dist_changes,
                          mode="lines", name=f"{name} Distance Change",
                          line=dict(color=color, width=1))
            )
    
    fig_stability.update_layout(
        title=f"Temporal Stability (Absolute Rate of Change): {target_bag}",
        xaxis_title="Time",
        yaxis_title="Absolute Distance Change (m/sample)",
        height=500
    )
    figs["stability"] = fig_stability
    
    # 3. Scatter plots for inter-system comparison
    fig_scatter = make_subplots(
        rows=1, cols=3,
        subplot_titles=("FFT vs Sonar", "FFT vs DVL", "Sonar vs DVL")
    )
    
    scatter_pairs = [
        ("fft_distance_m", "sonar_distance_m", "red", 1, 1),
        ("fft_distance_m", "nav_distance_m", "green", 1, 2),
        ("sonar_distance_m", "nav_distance_m", "blue", 1, 3)
    ]
    
    for col1, col2, color, row, col in scatter_pairs:
        if col1 in sync_df.columns and col2 in sync_df.columns:
            valid = sync_df[[col1, col2]].dropna()
            if len(valid) > 0:
                fig_scatter.add_trace(
                    go.Scatter(x=valid[col1], y=valid[col2],
                              mode="markers", marker=dict(color=color, size=3),
                              showlegend=False),
                    row=row, col=col
                )
                # Add ideal y=x line
                min_val = min(valid[col1].min(), valid[col2].min())
                max_val = max(valid[col1].max(), valid[col2].max())
                fig_scatter.add_trace(
                    go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                              mode="lines", line=dict(color="black", dash="dash"),
                              showlegend=False),
                    row=row, col=col
                )
    
    fig_scatter.update_xaxes(title_text="Distance (m)")
    fig_scatter.update_yaxes(title_text="Distance (m)")
    fig_scatter.update_layout(height=400, title_text=f"Inter-System Agreement: {target_bag}")
    figs["scatter"] = fig_scatter
    
    return figs


def print_quality_metrics(sync_df: pd.DataFrame):
    """
    Print comprehensive quality metrics without ground truth.
    
    Args:
        sync_df: Synchronized DataFrame
    """
    print("\n=== QUALITY METRICS (NO GROUND TRUTH REQUIRED) ===\n")
    
    # Temporal stability
    print("1. TEMPORAL STABILITY (Lower is more stable)")
    print("   Measures consistency of measurements over time\n")
    stability = compute_temporal_stability_metrics(sync_df)
    for system, metrics in stability.items():
        print(f"  {system}:")
        print(f"    Distance change std: {metrics['distance_change_std']:.4f} m/sample")
        print(f"    Distance smoothness:  {metrics['distance_smoothness']:.2f} (higher = smoother)")
        print(f"    Pitch change std:     {metrics['pitch_change_std']:.3f} °/sample")
        print(f"    Pitch smoothness:     {metrics['pitch_smoothness']:.2f} (higher = smoother)")
    
    # Inter-system agreement
    print("\n2. INTER-SYSTEM AGREEMENT")
    print("   Measures how well systems agree with each other\n")
    agreement = compute_inter_system_agreement(sync_df)
    for pair, metrics in agreement.items():
        print(f"  {pair}:")
        if "distance_correlation" in metrics:
            print(f"    Distance correlation: {metrics['distance_correlation']:.3f} (1.0 = perfect)")
            print(f"    Distance MAE:         {metrics['distance_mae']:.3f} m")
            print(f"    Distance agreement:   {metrics['distance_agreement_score']:.3f} (higher = better)")
        if "pitch_correlation" in metrics:
            print(f"    Pitch correlation:    {metrics['pitch_correlation']:.3f} (1.0 = perfect)")
            print(f"    Pitch MAE:            {metrics['pitch_mae_deg']:.2f}°")
            print(f"    Pitch agreement:      {metrics['pitch_agreement_score']:.3f} (higher = better)")
    
    # Data completeness
    print("\n3. DATA COMPLETENESS")
    print("   Percentage of valid (non-NaN) measurements\n")
    systems = [
        ("FFT", "fft_distance_m", "fft_pitch_deg"),
        ("Sonar", "sonar_distance_m", "sonar_pitch_deg"),
        ("DVL", "nav_distance_m", "nav_pitch_deg")
    ]
    
    for name, dist_col, pitch_col in systems:
        if dist_col in sync_df.columns:
            dist_valid = sync_df[dist_col].notna().sum()
            pitch_valid = sync_df[pitch_col].notna().sum() if pitch_col in sync_df.columns else 0
            total = len(sync_df)
            print(f"  {name}:")
            print(f"    Distance: {dist_valid}/{total} ({100*dist_valid/total:.1f}%)")
            if pitch_col in sync_df.columns:
                print(f"    Pitch:    {pitch_valid}/{total} ({100*pitch_valid/total:.1f}%)")

def print_timeseries_analysis(sync_df: pd.DataFrame):
    """
    Print time series analysis metrics.
    
    Args:
        sync_df: Synchronized DataFrame
    """
    print("\n=== TIME SERIES ANALYSIS ===\n")
    
    print("4. MEASUREMENT FREQUENCY")
    print("   Average time between measurements\n")
    
    if "sync_timestamp" in sync_df.columns:
        time_diffs = sync_df["sync_timestamp"].diff().dt.total_seconds().dropna()
        if len(time_diffs) > 0:
            print(f"  Mean interval: {time_diffs.mean():.3f} seconds")
            print(f"  Std interval:  {time_diffs.std():.3f} seconds")
            print(f"  Median interval: {time_diffs.median():.3f} seconds")
            print(f"  Sampling rate: ~{1.0/time_diffs.mean():.2f} Hz")
    
    print("\n5. MEASUREMENT CONSISTENCY")
    print("   Coefficient of variation (CV = std/mean)\n")
    
    systems = [
        ("FFT", "fft_distance_m"),
        ("Sonar", "sonar_distance_m"),
        ("DVL", "nav_distance_m")
    ]
    
    for name, col in systems:
        if col in sync_df.columns:
            data = sync_df[col].dropna()
            if len(data) > 0 and data.mean() > 0:
                cv = data.std() / data.mean()
                print(f"  {name}: CV = {cv:.4f} ({'low' if cv < 0.1 else 'moderate' if cv < 0.2 else 'high'} variability)")


def create_timeseries_visualizations(sync_df: pd.DataFrame, target_bag: str) -> Dict[str, go.Figure]:
    """
    Create time series analysis visualizations.
    
    Args:
        sync_df: Synchronized DataFrame
        target_bag: Target bag identifier
        
    Returns:
        Dictionary of plotly figures
    """
    figs = {}
    
    # 1. Autocorrelation plots
    fig_acf = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Distance ACF (Full)", "Pitch ACF (Full)",
                       "Distance ACF (Zoomed)", "Pitch ACF (Zoomed)"),
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    systems = [
        ("FFT", "fft_distance_m", "fft_pitch_deg", "red"),
        ("Sonar", "sonar_distance_m", "sonar_pitch_deg", "blue"),
        ("DVL", "nav_distance_m", "nav_pitch_deg", "green")
    ]
    
    for name, dist_col, pitch_col, color in systems:
        if dist_col in sync_df.columns:
            valid_data = sync_df[dist_col].dropna()
            if len(valid_data) > 1:
                acf_dist = signal.correlate(valid_data, valid_data, mode="full")
                acf_dist = acf_dist[acf_dist.size // 2:] / acf_dist.max()
                lags_dist = np.arange(len(acf_dist))
                
                fig_acf.add_trace(
                    go.Scatter(x=lags_dist, y=acf_dist, mode="lines", name=f"{name} Dist",
                              line=dict(color=color, width=2)),
                    row=1, col=1
                )
                
                fig_acf.add_trace(
                    go.Scatter(x=lags_dist, y=acf_dist, mode="lines", name=f"{name} Dist (Zoom)",
                              line=dict(color=color, width=2), showlegend=False),
                    row=2, col=1
                )
        
        if pitch_col in sync_df.columns:
            valid_data = sync_df[pitch_col].dropna()
            if len(valid_data) > 1:
                acf_pitch = signal.correlate(valid_data, valid_data, mode="full")
                acf_pitch = acf_pitch[acf_pitch.size // 2:] / acf_pitch.max()
                lags_pitch = np.arange(len(acf_pitch))
                
                fig_acf.add_trace(
                    go.Scatter(x=lags_pitch, y=acf_pitch, mode="lines", name=f"{name} Pitch",
                              line=dict(color=color, width=2), showlegend=False),
                    row=1, col=2
                )
                
                fig_acf.add_trace(
                    go.Scatter(x=lags_pitch, y=acf_pitch, mode="lines", name=f"{name} Pitch (Zoom)",
                              line=dict(color=color, width=2), showlegend=False),
                    row=2, col=2
                )
    
    fig_acf.update_xaxes(title_text="Lag", row=1, col=1)
    fig_acf.update_xaxes(title_text="Lag", row=1, col=2)
    fig_acf.update_xaxes(title_text="Lag", range=[0, 50], row=2, col=1)
    fig_acf.update_xaxes(title_text="Lag", range=[0, 50], row=2, col=2)
    fig_acf.update_yaxes(title_text="Autocorrelation")
    fig_acf.update_layout(height=700, title_text=f"Autocorrelation Analysis: {target_bag}")
    figs["autocorrelation"] = fig_acf
    
    # 2. Rolling statistics (with uncertainty bands)
    fig_rolling = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Distance Rolling Mean ± Std", "Pitch Rolling Mean ± Std"),
        vertical_spacing=0.15
    )
    
    window = 20  # Rolling window size
    
    for name, dist_col, pitch_col, color in systems:
        if dist_col in sync_df.columns:
            # Distance rolling stats
            rolling_mean = sync_df[dist_col].rolling(window=window, center=True).mean()
            rolling_std = sync_df[dist_col].rolling(window=window, center=True).std()
            
            rgb_tuple = to_rgb(color)
            rgba_fill = f"rgba({int(rgb_tuple[0]*255)}, {int(rgb_tuple[1]*255)}, {int(rgb_tuple[2]*255)}, 0.2)"
            
            fig_rolling.add_trace(
                go.Scatter(x=sync_df["sync_timestamp"], y=rolling_mean,
                          mode="lines", name=f"{name}",
                          line=dict(color=color, width=2)),
                row=1, col=1
            )
            fig_rolling.add_trace(
                go.Scatter(x=sync_df["sync_timestamp"], y=rolling_mean + rolling_std,
                          mode="lines", line=dict(color=color, width=0.5, dash="dash"),
                          showlegend=False, hoverinfo="skip"),
                row=1, col=1
            )
            fig_rolling.add_trace(
                go.Scatter(x=sync_df["sync_timestamp"], y=rolling_mean - rolling_std,
                          mode="lines", line=dict(color=color, width=0.5, dash="dash"),
                          fill="tonexty", fillcolor=rgba_fill,
                          showlegend=False, hoverinfo="skip"),
                row=1, col=1
            )
        
        if pitch_col in sync_df.columns:
            # Pitch rolling stats
            rolling_mean = sync_df[pitch_col].rolling(window=window, center=True).mean()
            rolling_std = sync_df[pitch_col].rolling(window=window, center=True).std()
            
            rgb_tuple = to_rgb(color)
            rgba_fill = f"rgba({int(rgb_tuple[0]*255)}, {int(rgb_tuple[1]*255)}, {int(rgb_tuple[2]*255)}, 0.2)"
            
            fig_rolling.add_trace(
                go.Scatter(x=sync_df["sync_timestamp"], y=rolling_mean,
                          mode="lines", name=f"{name}",
                          line=dict(color=color, width=2), showlegend=False),
                row=2, col=1
            )
            fig_rolling.add_trace(
                go.Scatter(x=sync_df["sync_timestamp"], y=rolling_mean + rolling_std,
                          mode="lines", line=dict(color=color, width=0.5, dash="dash"),
                          showlegend=False, hoverinfo="skip"),
                row=2, col=1
            )
            fig_rolling.add_trace(
                go.Scatter(x=sync_df["sync_timestamp"], y=rolling_mean - rolling_std,
                          mode="lines", line=dict(color=color, width=0.5, dash="dash"),
                          fill="tonexty", fillcolor=rgba_fill,
                          showlegend=False, hoverinfo="skip"),
                row=2, col=1
            )
    
    fig_rolling.update_xaxes(title_text="Time")
    fig_rolling.update_yaxes(title_text="Distance (m)", row=1, col=1)
    fig_rolling.update_yaxes(title_text="Pitch (°)", row=2, col=1)
    fig_rolling.update_layout(height=700, title_text=f"Rolling Statistics (window={window}): {target_bag}")
    figs["rolling_stats"] = fig_rolling
    
    return figs

def export_raw_comparison_data(df_sonar_raw: pd.DataFrame, df_nav_raw: pd.DataFrame, 
                               df_fft_raw: pd.DataFrame, target_bag: str, 
                               output_dir: Path, tolerance_seconds: float = 0.5) -> Optional[Path]:
    """
    Generate and export raw (unsmoothed) synchronized comparison data to CSV.
    
    Args:
        df_sonar_raw: Raw sonar DataFrame
        df_nav_raw: Raw navigation DataFrame
        df_fft_raw: Raw FFT DataFrame
        target_bag: Target bag identifier for filename
        output_dir: Directory to save the CSV file
        tolerance_seconds: Time tolerance for synchronization
        
    Returns:
        Path to saved CSV file, or None if export failed
    """
    print("\n=== GENERATING RAW COMPARISON DATA ===\n")
    print("Creating unsmoothed synchronized dataset for export...")
    
    # Create synchronized dataset from raw data
    three_sync_df_raw, _, _ = prepare_three_system_comparison(
        target_bag,
        df_sonar_raw,
        df_nav_raw,
        df_fft_raw,
        tolerance_seconds=tolerance_seconds
    )
    
    if three_sync_df_raw is None or three_sync_df_raw.empty:
        print("✗ Could not generate raw comparison data")
        return None
    
    # Ensure XY columns exist
    three_sync_df_raw = ensure_xy_columns(three_sync_df_raw)
    
    # Select columns for export
    export_cols = [
        "sync_timestamp",
        # FFT data
        "fft_distance_m", "fft_pitch_deg", "fft_x", "fft_y",
        # Sonar data
        "sonar_distance_m", "sonar_pitch_deg", "sonar_x", "sonar_y",
        # DVL data
        "nav_distance_m", "nav_pitch_deg", "dvl_x", "dvl_y"
    ]
    
    # Filter to only existing columns
    export_cols = [c for c in export_cols if c in three_sync_df_raw.columns]
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / f"{target_bag}_raw_comparison.csv"
    three_sync_df_raw[export_cols].to_csv(output_path, index=False)
    
    # Report what was saved
    available_systems = []
    if any("fft" in c for c in export_cols):
        available_systems.append("FFT")
    if any("sonar" in c for c in export_cols):
        available_systems.append("Sonar")
    if any("nav" in c or "dvl" in c for c in export_cols):
        available_systems.append("DVL")
    
    print(f"✓ Saved raw comparison data: {output_path.name}")
    print(f"  Records: {len(three_sync_df_raw)}")
    print(f"  Columns: {len(export_cols)}")
    print(f"  Available systems: {', '.join(available_systems)}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB\n")
    
    return output_path
