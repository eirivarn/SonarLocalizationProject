# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
SOLAQUA Sonar Visualization Utilities

Pure functional API for sonar data visualization:
- Data I/O and frame extraction: utils.sonar_utils
- Display enhancement: utils.sonar_utils.enhance_intensity
- Cone rasterization: utils.sonar_utils.cone_raster_like_display_cell
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.io_utils import load_df
from utils.sonar_utils import (
    get_sonoptix_frame,
    apply_flips,
    enhance_intensity,
    cone_raster_like_display_cell,
)
from utils.config import SONAR_VIS_DEFAULTS, EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS

# ============================ FILE DISCOVERY ============================

def find_sonar_files(search_root: Optional[Union[str, Path]] = None) -> List[Path]:
    """
    Find candidate sonar files in the exports directory.
    If search_root is None, use the configured exports dir + 'by_bag'.
    """
    if search_root is None:
        search_root = Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    else:
        search_root = Path(search_root)

    if not search_root.exists():
        return []
    
    # Try specific pattern first
    candidates = sorted(list(search_root.glob("sensor_sonoptix_echo_image__*.csv")))
    
    # Fallback: broader search
    if not candidates:
        for p in search_root.rglob("*"):
            if p.suffix.lower() in {".csv", ".parquet"}:
                name = p.name.lower()
                if any(tok in name for tok in ["sonoptix", "echo"]):
                    candidates.append(p)
        candidates = sorted(candidates)
    
    return candidates

# ============================ DATA LOADING ============================

def load_sonar_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load sonar data from CSV or Parquet file."""
    df = load_df(Path(file_path))
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def get_frame_data(
    df: pd.DataFrame,
    frame_index: int,
    config: Optional[Dict] = None
) -> np.ndarray:
    """
    Extract a single frame and apply orientation flips.
    
    Args:
        df: Sonar data dataframe
        frame_index: Frame number to extract
        config: Visualization config (defaults to SONAR_VIS_DEFAULTS)
    """
    if config is None:
        config = SONAR_VIS_DEFAULTS.copy()
    
    M0 = get_sonoptix_frame(df, frame_index)
    if M0 is None:
        raise RuntimeError("Could not construct a Sonoptix frame from this file.")

    # Apply orientation
    M = M0.T if config["swap_hw"] else M0.copy()
    M = apply_flips(
        M,
        flip_range=config["flip_range"],
        flip_beams=config["flip_beams"],
    )
    return M


def enhance_data(data: np.ndarray, config: Optional[Dict] = None) -> np.ndarray:
    """
    Apply enhancement to sonar data.
    Returns data normalized to [0, 1] for display.
    """
    if config is None:
        config = SONAR_VIS_DEFAULTS.copy()
    
    return enhance_intensity(
        data,
        config["range_min_m"],
        config["range_max_m"],
        scale=config["enh_scale"],
        tvg=config["enh_tvg"],
        alpha_db_per_m=config["enh_alpha_db_per_m"],
        r0=config["enh_r0"],
        p_low=config["enh_p_low"],
        p_high=config["enh_p_high"],
        gamma=config["enh_gamma"],
        zero_aware=config["enh_zero_aware"],
        eps_log=config["enh_eps_log"],
    )

# ============================ MATPLOTLIB VISUALIZATIONS ============================

def plot_raw_and_enhanced(
    data: np.ndarray,
    frame_index: int,
    config: Optional[Dict] = None
) -> plt.Figure:
    """
    Plot raw and enhanced sonar data side by side.
    
    Args:
        data: Frame data
        frame_index: Frame number (for title)
        config: Visualization config
    """
    if config is None:
        config = SONAR_VIS_DEFAULTS.copy()
    
    data_enh = enhance_data(data, config)

    # Axes extents
    theta_min_deg = -0.5 * config["fov_deg"]
    theta_max_deg = 0.5 * config["fov_deg"]
    
    if config["swap_hw"]:
        extent_xy = (
            config["range_min_m"],
            config["range_max_m"],
            theta_min_deg,
            theta_max_deg,
        )
        xlab, ylab = "Range [m]", "Beam angle [deg]"
    else:
        extent_xy = (
            theta_min_deg,
            theta_max_deg,
            config["range_min_m"],
            config["range_max_m"],
        )
        xlab, ylab = "Beam angle [deg]", "Range [m]"

    fig, axes = plt.subplots(
        1, 2, figsize=config["figsize"], constrained_layout=True
    )

    # Raw
    vmin_raw, vmax_raw = float(np.nanmin(data)), float(np.nanmax(data))
    im0 = axes[0].imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent_xy,
        vmin=vmin_raw,
        vmax=vmax_raw,
        cmap=config["cmap_raw"],
    )
    axes[0].set_title(f"Raw (frame {frame_index})")
    axes[0].set_xlabel(xlab)
    axes[0].set_ylabel(ylab)
    if config["swap_hw"]:
        axes[0].set_xlim(config["range_min_m"], config["display_range_max_m"])
    else:
        axes[0].set_ylim(config["range_min_m"], config["display_range_max_m"])
    fig.colorbar(im0, ax=axes[0], label="Echo (raw units)")

    # Enhanced
    im1 = axes[1].imshow(
        data_enh,
        origin="lower",
        aspect="auto",
        extent=extent_xy,
        vmin=0,
        vmax=1,
        cmap=config["cmap_enh"],
    )
    axes[1].set_title(f"Enhanced (frame {frame_index})")
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(ylab)
    if config["swap_hw"]:
        axes[1].set_xlim(config["range_min_m"], config["display_range_max_m"])
    else:
        axes[1].set_ylim(config["range_min_m"], config["display_range_max_m"])
    fig.colorbar(im1, ax=axes[1], label="Echo (enhanced)")

    return fig


def plot_cone_view(
    data: np.ndarray,
    use_enhanced: bool = True,
    config: Optional[Dict] = None
) -> plt.Figure:
    """
    Plot sonar data as a geometric cone view in Cartesian coordinates.
    Uses the shared cone rasterization utility for consistency with videos.
    """
    if config is None:
        config = SONAR_VIS_DEFAULTS.copy()
    
    Z = enhance_data(data, config) if use_enhanced else data
    cmap_name = config["cmap_enh"] if use_enhanced else config["cmap_raw"]
    vmin, vmax = (0.0, 1.0) if use_enhanced else (
        float(np.nanmin(Z)),
        float(np.nanmax(Z)),
    )

    # Delegate cone rasterization
    cone, (x_min, x_max, y_min, y_max) = cone_raster_like_display_cell(
        Z,
        config["fov_deg"],
        config["range_min_m"],
        config["range_max_m"],
        config["display_range_max_m"],
        config["img_w"],
        config["img_h"],
    )

    # Configure colormap
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(config["bg_color"])

    # Plot
    fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
    im = ax.imshow(
        cone,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Starboard X [m] (+)")
    ax.set_ylabel("Forward Y [m]")
    ax.set_title("Sonar Cone — " + ("Enhanced" if use_enhanced else "Raw"))

    # Angle spokes
    half = 0.5 * config["fov_deg"]
    for a in np.linspace(-half, half, config["n_spokes"]):
        th = np.deg2rad(a + config["rotate_deg"])
        x_end = config["display_range_max_m"] * np.sin(th)
        y_end = config["display_range_max_m"] * np.cos(th)
        ax.plot([0, x_end], [0, y_end], lw=0.9, alpha=0.85, color="k")

    fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        shrink=0.9,
        label=("Echo (normalized)" if use_enhanced else "Echo (raw units)"),
    )
    return fig

# ============================ PLOTLY ANALYSIS VISUALIZATION ============================

def plot_distance_analysis(
    distance_results: pd.DataFrame,
    title: str = "Distance Analysis Over Time"
) -> Optional[go.Figure]:
    """Create interactive distance plot."""
    valid = distance_results[distance_results['detection_success']].copy()
    if len(valid) == 0:
        print("No valid data to plot")
        return None
    
    dist_col = 'distance_meters' if 'distance_meters' in valid.columns and valid['distance_meters'].notna().any() else 'distance_pixels'
    unit = "meters" if dist_col == 'distance_meters' else "pixels"
    distances = valid[dist_col].dropna()
    
    if len(distances) == 0:
        return None
    
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=('Distance vs Frame', 'Distance vs Time', 'Distribution', 'Trends'))
    
    fig.add_trace(go.Scatter(x=valid['frame_index'], y=distances, mode='lines+markers',
                  name='Distance', line=dict(color='blue')), row=1, col=1)
    
    x_time = valid.get('timestamp', valid['frame_index'])
    fig.add_trace(go.Scatter(x=x_time, y=distances, mode='lines+markers',
                  name='Over Time', line=dict(color='green')), row=1, col=2)
    
    fig.add_trace(go.Histogram(x=distances, nbinsx=30, name='Distribution',
                    marker_color='lightcoral', opacity=0.7), row=2, col=1)
    fig.add_vline(x=distances.mean(), line=dict(color='red', dash='dash'),
                 annotation_text=f'Mean: {distances.mean():.2f}', row=2, col=1)
    
    window_size = max(5, len(distances) // 20)
    smoothed = distances.rolling(window=window_size, center=True).mean()
    fig.add_trace(go.Scatter(x=valid['frame_index'], y=distances, mode='lines',
                  name='Raw', line=dict(color='lightcoral', width=1), opacity=0.5), row=2, col=2)
    fig.add_trace(go.Scatter(x=valid['frame_index'], y=smoothed, mode='lines',
                  name=f'Smoothed (n={window_size})', line=dict(color='darkred', width=3)), row=2, col=2)
    
    fig.update_layout(title_text=title, height=800, showlegend=True)
    fig.update_yaxes(title_text=f'Distance ({unit})', row=1, col=1)
    fig.update_yaxes(title_text=f'Distance ({unit})', row=1, col=2)
    
    return fig


def compare_sonar_vs_dvl(
    distance_results: pd.DataFrame,
    raw_data: Optional[Dict[str, pd.DataFrame]],
    sonar_coverage_m: float = 5.0,
    sonar_image_size: int = 700,
    bag_id: Optional[str] = None,
    save_plot: bool = True,
    plots_dir: Optional[Union[str, Path]] = None
) -> Tuple[Optional[go.Figure], Dict]:
    """
    Compare sonar and DVL distance measurements.
    
    Args:
        distance_results: DataFrame with sonar analysis results
        raw_data: Dictionary containing navigation data
        sonar_coverage_m: Sonar coverage in meters
        sonar_image_size: Image size in pixels
        bag_id: Bag identifier for saving plots (auto-detected from timestamp if None)
        save_plot: Whether to save the plot to disk
        plots_dir: Directory to save plots (uses /Volumes/LaCie/SOLAQUA/exports/plots if None)
    
    Returns:
        Tuple of (plotly figure, statistics dictionary)
    """
    
    if raw_data is None or 'navigation' not in raw_data or raw_data['navigation'] is None:
        return None, {'error': 'no_navigation_data'}
    
    nav = raw_data['navigation'].copy()
    nav['timestamp'] = pd.to_datetime(nav['timestamp'], errors='coerce')
    nav = nav.dropna(subset=['timestamp'])
    nav['relative_time'] = (nav['timestamp'] - nav['timestamp'].min()).dt.total_seconds()
    
    sonar = distance_results.copy()
    if 'distance_meters' not in sonar.columns or sonar['distance_meters'].isna().all():
        if 'distance_pixels' in sonar.columns:
            ppm = float(sonar_image_size) / float(sonar_coverage_m)
            sonar['distance_meters'] = sonar['distance_pixels'] / ppm
        else:
            return None, {'error': 'no_distance_data'}
    
    dvl_duration = float(max(1.0, nav['relative_time'].max() - nav['relative_time'].min()))
    if 'frame_index' not in sonar:
        sonar['frame_index'] = np.arange(len(sonar))
    N = max(1, len(sonar) - 1)
    sonar['synthetic_time'] = (sonar['frame_index'] / float(N)) * dvl_duration
    
    # Auto-detect bag_id from timestamp if not provided
    if bag_id is None and 'timestamp' in sonar.columns:
        try:
            first_ts = pd.to_datetime(sonar['timestamp'].iloc[0])
            bag_id = first_ts.strftime('%Y-%m-%d_%H-%M-%S')
        except:
            bag_id = 'unknown'
    elif bag_id is None:
        bag_id = 'unknown'
    
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=sonar['synthetic_time'], y=sonar['distance_meters'],
                  mode='lines', name='Sonar', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=nav['relative_time'], y=nav['NetDistance'],
                  mode='lines', name='DVL', line=dict(color='blue', width=3)))
    
    fig.update_layout(title=f"Sonar vs DVL Distance Comparison - {bag_id}", height=600)
    fig.update_xaxes(title_text="Time (seconds)")
    fig.update_yaxes(title_text="Distance (meters)")
    
    sonar_mean = float(np.nanmean(sonar['distance_meters']))
    dvl_mean = float(nav['NetDistance'].mean())
    
    stats = {
        'sonar_mean_m': sonar_mean,
        'dvl_mean_m': dvl_mean,
        'scale_ratio': sonar_mean / dvl_mean if dvl_mean else np.nan,
        'sonar_frames': len(sonar),
        'dvl_records': len(nav),
        'bag_id': bag_id,
    }
    
    print(f"\nComparison: Sonar={sonar_mean:.3f}m, DVL={dvl_mean:.3f}m, Ratio={stats['scale_ratio']:.3f}x")
    
    # Save plot if requested
    if save_plot and fig is not None:
        # Determine save directory
        if plots_dir is None:
            # Try default location first
            default_plots_dir = Path("/Volumes/LaCie/SOLAQUA/exports/plots")
            if default_plots_dir.parent.exists():
                plots_dir = default_plots_dir
            else:
                # Fallback: create 'plots' in current working directory
                plots_dir = Path.cwd() / "plots"
        else:
            plots_dir = Path(plots_dir)
        
        # Create directory if it doesn't exist
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as HTML (interactive) and PNG (static)
        html_path = plots_dir / f"sonar_vs_dvl_{bag_id}.html"
        png_path = plots_dir / f"sonar_vs_dvl_{bag_id}.png"
        
        try:
            fig.write_html(str(html_path))
            print(f"Saved plot: {html_path}")
        except Exception as e:
            print(f"Warning: Could not save HTML plot: {e}")
        
        try:
            fig.write_image(str(png_path), width=1200, height=600)
            print(f"Saved plot: {png_path}")
        except Exception as e:
            print(f"Warning: Could not save PNG plot (requires kaleido): {e}")
    
    return fig, stats