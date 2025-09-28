"""
SOLAQUA Sonar Visualization Utilities

This module contains functions for visualizing sonar data from Sonoptix and Ping360 systems.
It handles data loading, enhancement, and various visualization modes including:
- Raw and enhanced intensity displays
- Cartesian cone views with proper geometric mapping
- Polar displays for Ping360 data
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap


class SonarVisualizer:
    """
    Main class for sonar data visualization with configurable parameters.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the sonar visualizer with configuration parameters.
        
        Args:
            config: Dictionary with visualization parameters
        """
        # Default configuration
        self.config = {
            # Geometry / calibration
            'fov_deg': 120.0,
            'range_min_m': 0.0,
            'range_max_m': 30.0,
            'display_range_max_m': 10.0,
            
            # Orientation
            'swap_hw': False,
            'flip_beams': False,
            'flip_range': False,
            
            # Enhancement
            'enh_scale': 'db',
            'enh_tvg': 'amplitude',
            'enh_alpha_db_per_m': 0.0,
            'enh_eps_log': 1e-5,
            'enh_r0': 1e-6,
            'enh_p_low': 1.0,
            'enh_p_high': 99.5,
            'enh_gamma': 0.9,
            'enh_zero_aware': True,
            
            # Visualization
            'cmap_raw': 'viridis',
            'cmap_enh': 'viridis',
            'figsize': (12, 5.6),
            
            # Cone view
            'img_w': 900,
            'img_h': 700,
            'bg_color': 'black',
            'n_spokes': 5,
            'rotate_deg': 0.0
        }
        
        # Update with user config
        if config:
            self.config.update(config)
    
    def load_sonar_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load sonar data from CSV or Parquet file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with sonar data
        """
        from utils.sonar_utils import load_df
        
        path = Path(file_path)
        df = load_df(path)
        print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    
    def get_frame_data(self, df: pd.DataFrame, frame_index: int) -> np.ndarray:
        """
        Extract a single frame from the sonar dataframe.
        
        Args:
            df: Sonar dataframe
            frame_index: Index of frame to extract
            
        Returns:
            2D numpy array with sonar intensities
        """
        from utils.sonar_utils import get_sonoptix_frame, apply_flips
        
        M0 = get_sonoptix_frame(df, frame_index)
        if M0 is None:
            raise RuntimeError("Could not construct a Sonoptix frame from this file.")
        
        # Apply orientation
        M = M0.T if self.config['swap_hw'] else M0.copy()
        M = apply_flips(M, 
                       flip_range=self.config['flip_range'], 
                       flip_beams=self.config['flip_beams'])
        
        return M
    
    def enhance_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply enhancement to sonar data.
        
        Args:
            data: Raw sonar intensity data
            
        Returns:
            Enhanced data normalized to [0, 1]
        """
        from utils.sonar_utils import enhance_intensity
        
        return enhance_intensity(
            data,
            self.config['range_min_m'],
            self.config['range_max_m'],
            scale=self.config['enh_scale'],
            tvg=self.config['enh_tvg'],
            alpha_db_per_m=self.config['enh_alpha_db_per_m'],
            r0=self.config['enh_r0'],
            p_low=self.config['enh_p_low'],
            p_high=self.config['enh_p_high'],
            gamma=self.config['enh_gamma'],
            zero_aware=self.config['enh_zero_aware'],
            eps_log=self.config['enh_eps_log']
        )
    
    def plot_raw_and_enhanced(self, data: np.ndarray, frame_index: int) -> plt.Figure:
        """
        Plot raw and enhanced sonar data side by side.
        
        Args:
            data: Raw sonar data array
            frame_index: Frame index for labeling
            
        Returns:
            Matplotlib figure
        """
        # Enhance data
        data_enh = self.enhance_data(data)
        
        # Calculate extents
        H, W = data.shape
        theta_min_deg = -0.5 * self.config['fov_deg']
        theta_max_deg = 0.5 * self.config['fov_deg']
        
        if self.config['swap_hw']:
            extent_xy = (self.config['range_min_m'], self.config['range_max_m'], 
                        theta_min_deg, theta_max_deg)
            xlab, ylab = "Range [m]", "Beam angle [deg]"
        else:
            extent_xy = (theta_min_deg, theta_max_deg, 
                        self.config['range_min_m'], self.config['range_max_m'])
            xlab, ylab = "Beam angle [deg]", "Range [m]"
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=self.config['figsize'], constrained_layout=True)
        
        # Raw data
        vmin_raw, vmax_raw = float(np.nanmin(data)), float(np.nanmax(data))
        im0 = axes[0].imshow(data, origin="lower", aspect="auto", extent=extent_xy,
                            vmin=vmin_raw, vmax=vmax_raw, cmap=self.config['cmap_raw'])
        axes[0].set_title(f"Raw (frame {frame_index})")
        axes[0].set_xlabel(xlab)
        axes[0].set_ylabel(ylab)
        
        # Set display range
        if self.config['swap_hw']:
            axes[0].set_xlim(self.config['range_min_m'], self.config['display_range_max_m'])
        else:
            axes[0].set_ylim(self.config['range_min_m'], self.config['display_range_max_m'])
        
        fig.colorbar(im0, ax=axes[0], label="Echo (raw units)")
        
        # Enhanced data
        im1 = axes[1].imshow(data_enh, origin="lower", aspect="auto", extent=extent_xy,
                            vmin=0, vmax=1, cmap=self.config['cmap_enh'])
        axes[1].set_title(f"Enhanced (frame {frame_index})")
        axes[1].set_xlabel(xlab)
        axes[1].set_ylabel(ylab)
        
        if self.config['swap_hw']:
            axes[1].set_xlim(self.config['range_min_m'], self.config['display_range_max_m'])
        else:
            axes[1].set_ylim(self.config['range_min_m'], self.config['display_range_max_m'])
        
        fig.colorbar(im1, ax=axes[1], label="Echo (enhanced)")
        
        return fig
    
    def plot_cone_view(self, data: np.ndarray, use_enhanced: bool = True) -> plt.Figure:
        """
        Plot sonar data as a geometric cone view in Cartesian coordinates.
        
        Args:
            data: Raw sonar data array
            use_enhanced: Whether to use enhanced data
            
        Returns:
            Matplotlib figure
        """
        # Choose data and enhance if needed
        if use_enhanced:
            Z = self.enhance_data(data)
            vmin, vmax = 0.0, 1.0
            cmap_name = self.config['cmap_enh']
        else:
            Z = data
            vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
            cmap_name = self.config['cmap_raw']
        
        H, W = Z.shape
        
        # Calculate spatial bounds
        half = np.deg2rad(0.5 * self.config['fov_deg'])
        x_max = np.sin(half) * self.config['display_range_max_m']
        x_min = -x_max
        y_min = max(0.0, self.config['range_min_m'])
        y_max = self.config['display_range_max_m']
        
        # Create Cartesian grid
        x = np.linspace(x_min, x_max, self.config['img_w'])
        y = np.linspace(y_min, y_max, self.config['img_h'])
        Xg, Yg = np.meshgrid(x, y)
        
        # Convert to polar coordinates
        theta = np.arctan2(Xg, Yg)
        theta -= np.deg2rad(self.config['rotate_deg'])
        r = np.hypot(Xg, Yg)
        
        # Create mask for valid points
        mask = (
            (r >= self.config['range_min_m']) & (r <= y_max) &
            (theta >= -half) & (theta <= +half)
        )
        
        # Map to array indices
        rowf = (r - self.config['range_min_m']) / max(self.config['range_max_m'] - self.config['range_min_m'], 1e-12) * (H - 1)
        colf = (theta + half) / max(2*half, 1e-12) * (W - 1)
        
        rows = np.rint(np.clip(rowf, 0, H - 1)).astype(int)
        cols = np.rint(np.clip(colf, 0, W - 1)).astype(int)
        
        # Build the raster
        img = np.full((self.config['img_h'], self.config['img_w']), np.nan, dtype=float)
        flat = mask.ravel()
        img.ravel()[flat] = Z[rows.ravel()[flat], cols.ravel()[flat]]
        
        # Setup colormap
        cmap = plt.cm.get_cmap(cmap_name).copy()
        cmap.set_bad(self.config['bg_color'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(9.5, 8.0), constrained_layout=True)
        im = ax.imshow(img, origin="lower",
                      extent=(x_min, x_max, y_min, y_max),
                      aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        
        ax.set_xlabel("Starboard X [m] (+)")
        ax.set_ylabel("Forward Y [m]")
        ax.set_title("Sonar Cone — " + ("Enhanced" if use_enhanced else "Raw"))
        
        # Add angle spokes
        for a in np.linspace(-np.rad2deg(half), np.rad2deg(half), self.config['n_spokes']):
            th = np.deg2rad(a + self.config['rotate_deg'])
            x_end = y_max * np.sin(th)
            y_end = y_max * np.cos(th)
            ax.plot([0, x_end], [0, y_end], lw=0.9, alpha=0.85, color="k")
            ax.text(1.02*x_end, 1.02*y_end, f"{a:.0f}°", ha="center", va="center", fontsize=9)
        
        # Colorbar
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9,
                    label=("Echo (normalized)" if use_enhanced else "Echo (raw units)"))
        
        return fig
    
    def analyze_frame_statistics(self, df: pd.DataFrame, frame_index: int) -> Dict:
        """
        Analyze statistics for a specific frame.
        
        Args:
            df: Sonar dataframe
            frame_index: Frame index to analyze
            
        Returns:
            Dictionary with frame statistics
        """
        data = self.get_frame_data(df, frame_index)
        
        stats = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min_value': float(np.nanmin(data)),
            'max_value': float(np.nanmax(data)),
            'mean_value': float(np.nanmean(data)),
            'median_value': float(np.nanmedian(data)),
            'std_value': float(np.nanstd(data)),
            'nonzero_ratio': float((data > 0).mean()),
            'nan_ratio': float(np.isnan(data).mean()),
            'unique_values': len(np.unique(data[~np.isnan(data)]))
        }
        
        return stats


def find_sonar_files(search_root: Union[str, Path] = "exports/by_bag") -> List[Path]:
    """
    Find candidate sonar files in the exports directory.
    
    Args:
        search_root: Directory to search for sonar files
        
    Returns:
        List of paths to candidate sonar files
    """
    search_root = Path(search_root)
    candidates = []
    
    if search_root.exists():
        for p in search_root.rglob("*"):
            if p.suffix.lower() in {".csv", ".parquet"}:
                if any(tok in p.name.lower() for tok in ["sonoptix", "echo", "ping360", "ping"]):
                    candidates.append(p)
    
    return sorted(candidates)


def print_sonar_files(candidates: List[Path]) -> None:
    """
    Print the list of candidate sonar files.
    
    Args:
        candidates: List of file paths to print
    """
    for i, p in enumerate(candidates):
        print(f"[{i}] {p}")
    
    if not candidates:
        print("(no matching files found yet — export first)")


def quick_visualize(file_path: Union[str, Path], frame_index: int = 0, 
                   config: Optional[Dict] = None) -> Tuple[plt.Figure, plt.Figure]:
    """
    Quick visualization of a sonar frame with default settings.
    
    Args:
        file_path: Path to sonar data file
        frame_index: Frame index to visualize
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (raw/enhanced figure, cone figure)
    """
    viz = SonarVisualizer(config)
    df = viz.load_sonar_data(file_path)
    data = viz.get_frame_data(df, frame_index)
    
    fig1 = viz.plot_raw_and_enhanced(data, frame_index)
    fig2 = viz.plot_cone_view(data, use_enhanced=True)
    
    return fig1, fig2


def analyze_bag_directly(bag_path: Union[str, Path], topic: str, frame_index: int = 0) -> Dict:
    """
    Analyze sonar data directly from a bag file.
    
    Args:
        bag_path: Path to the bag file
        topic: Sonar topic name
        frame_index: Frame index to analyze
        
    Returns:
        Dictionary with analysis results
    """
    from rosbags.highlevel import AnyReader
    
    bag_path = Path(bag_path)
    
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"Topic {topic!r} not found in {bag_path}")
        
        con = conns[0]
        
        for i, (_, t_ns, raw) in enumerate(reader.messages(connections=[con])):
            if i == frame_index:
                msg = reader.deserialize(raw, con.msgtype)
                arr = np.array(msg.array_data.data, dtype=np.float32)
                
                return {
                    'frame_index': frame_index,
                    'dtype': str(arr.dtype),
                    'shape': arr.shape,
                    'min_max': (float(arr.min()), float(arr.max())),
                    'max_fractional_part': float(np.max(np.abs(np.modf(arr)[0]))),
                    'unique_count': len(np.unique(arr)),
                    'first_50_unique': np.unique(arr)[:50].tolist(),
                    'timestamp_ns': int(t_ns)
                }
        
        raise IndexError(f"Frame index {frame_index} not found")


def analyze_full_bag_statistics(bag_path: Union[str, Path], topic: str) -> Dict:
    """
    Analyze statistics across all frames in a bag.
    
    Args:
        bag_path: Path to the bag file  
        topic: Sonar topic name
        
    Returns:
        Dictionary with overall statistics
    """
    from collections import Counter
    from rosbags.highlevel import AnyReader
    
    bag_path = Path(bag_path)
    counts = Counter()
    nz_ratios = []
    frame_count = 0
    
    with AnyReader([bag_path]) as reader:
        con = [c for c in reader.connections if c.topic == topic][0]
        
        for _, _, raw in reader.messages(connections=[con]):
            msg = reader.deserialize(raw, con.msgtype)
            arr = np.array(msg.array_data.data, dtype=np.float32)
            counts.update(np.unique(arr).tolist())
            nz_ratios.append((arr > 0).mean())
            frame_count += 1
    
    return {
        'total_frames': frame_count,
        'unique_values_overall': len(counts),
        'median_nonzero_percent': float(100 * np.median(nz_ratios)),
        'mean_nonzero_percent': float(100 * np.mean(nz_ratios)),
        'std_nonzero_percent': float(100 * np.std(nz_ratios))
    }