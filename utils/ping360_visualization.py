# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
Ping360 Sonar Visualization Utilities

This module provides utilities for processing and visualizing Ping360 sonar data,
including data loading, preprocessing, TVG compensation, and visualization.
"""

import json
import pathlib
from typing import Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS

class Ping360Visualizer:
    """
    A class for processing and visualizing Ping360 sonar data.
    
    Handles loading CSV data, applying TVG compensation, gating near-field,
    and creating various visualizations of the sonar data.
    """
    
    def __init__(self):
        """Initialize the Ping360Visualizer with default parameters."""
        # Processing parameters
        self.gate_near_m = 0.6
        self.tvg_spreading_db = True
        self.tvg_alpha_db_per_m = 0.00
        self.db_eps = 1e-3
        self.angle_bin_deg = 360
        self.aggregation = "max"
        self.angle_offset_deg = 0.0
        self.clip_low_pct = 1.0
        self.clip_high_pct = 99.7
        
        # IMU/Yaw desmearing
        self.use_imu_yaw = False
        self.yaw_series_deg = None
        
        # Data storage
        self.df = None
        self.cfg = None
        self.stack = None
        self.stack_g = None
        self.r = None
        self.r_g = None
        self.angles_deg = None
        self.db_hat = None
        self.n_samples = None
        
        # Set matplotlib defaults
        plt.rcParams["figure.dpi"] = 110
        plt.rcParams["image.origin"] = "upper"
    
    def configure(self, **kwargs):
        """
        Configure processing parameters.
        
        Parameters:
        -----------
        gate_near_m : float, optional
            Near-field gating distance in meters (default: 0.6)
        tvg_spreading_db : bool, optional
            Apply 40*log10(r) spreading compensation (default: True)
        tvg_alpha_db_per_m : float, optional
            Absorption compensation in dB/m (default: 0.0)
        angle_offset_deg : float, optional
            Fixed mount offset rotation in degrees (default: 0.0)
        clip_low_pct : float, optional
            Low percentile for display clipping (default: 1.0)
        clip_high_pct : float, optional
            High percentile for display clipping (default: 99.7)
        use_imu_yaw : bool, optional
            Enable IMU yaw desmearing (default: False)
        yaw_series_deg : array-like, optional
            Yaw angles in degrees for desmearing
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"  Warning: Unknown parameter '{key}' ignored")
    
    def load_data(self, pings_csv: Union[str, pathlib.Path], 
                  config_csv: Optional[Union[str, pathlib.Path]] = None) -> pd.DataFrame:
        """
        Load Ping360 CSV data and parse JSON arrays.
        
        Parameters:
        -----------
        pings_csv : str or Path
            Path to the pings CSV file
        config_csv : str or Path, optional
            Path to the config CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded pings dataframe
        """
        pings_csv = pathlib.Path(pings_csv)
        assert pings_csv.exists(), f"Missing file: {pings_csv}"
        
        print(f"Loading Ping360 data from: {pings_csv.name}")
        
        self.df = pd.read_csv(pings_csv)
        
        if config_csv:
            config_csv = pathlib.Path(config_csv)
            self.cfg = pd.read_csv(config_csv) if config_csv.exists() else pd.DataFrame()
        else:
            self.cfg = pd.DataFrame()
        
        # Basic checks
        for col in ["data", "angle_deg"]:
            assert col in self.df.columns, f"CSV missing required column: {col}"
        
        # Parse JSON arrays from 'data'
        profiles = []
        lens = []
        
        print("Parsing JSON data arrays...")
        for s in self.df["data"].astype(str):
            try:
                arr = np.asarray(json.loads(s), dtype=np.float32)
            except Exception:
                arr = np.zeros(1, dtype=np.float32)
            profiles.append(arr)
            lens.append(arr.size)
        
        # Determine number_of_samples
        if "number_of_samples" in self.df.columns and self.df["number_of_samples"].notna().any():
            self.n_samples = int(self.df["number_of_samples"].mode().iloc[0])
        else:
            self.n_samples = int(max(lens))
        
        # Stack into uint8 (clip 0..255)
        self.stack = np.zeros((len(profiles), self.n_samples), dtype=np.uint8)
        for i, a in enumerate(profiles):
            a = a[:self.n_samples]
            a = np.clip(a, 0, 255).astype(np.uint8)
            self.stack[i, :a.size] = a
        
        print(f"Loaded {len(self.df)} pings, {self.n_samples} samples per ping")
        return self.df
    
    def compute_range_axis(self, default_c_ms: float = 1500.0) -> np.ndarray:
        """
        Compute range axis from sonar parameters.
        
        Parameters:
        -----------
        default_c_ms : float, optional
            Default sound speed in m/s (default: 1500.0)
            
        Returns:
        --------
        np.ndarray
            Range axis in meters
        """
        # Try sonar_range first
        if "sonar_range" in self.df.columns and self.df["sonar_range"].notna().any():
            R = float(self.df["sonar_range"].astype(float).mode().iloc[0])
            if R > 0:
                self.r = np.linspace(0.0, R, self.n_samples, dtype=np.float32)
                print(f"Using sonar_range: 0.0 → {R:.1f}m")
                return self.r
        
        # Try sample_period
        sp = self.df.get("sample_period", pd.Series([np.nan])).dropna().astype(float)
        if len(sp):
            sp_val = float(sp.mode().iloc[0])
            if 0 < sp_val <= 40000:
                # Ping360 convention: 25 ns ticks
                sample_dt = sp_val * 25e-9
            else:
                # treat as microseconds
                sample_dt = sp_val * 1e-6
            dr = (default_c_ms * sample_dt) / 2.0
            self.r = np.arange(self.n_samples, dtype=np.float32) * dr
            print(f"Computed from sample_period: 0.0 → {self.r[-1]:.1f}m")
            return self.r
        
        # Fallback: bin index
        self.r = np.arange(self.n_samples, dtype=np.float32)
        print("Using bin indices as range axis")
        return self.r
    
    def gate_near_field(self):
        """Apply near-field gating to remove TX ringdown artifacts."""
        if self.r is None:
            self.compute_range_axis()
        
        # Gate near range
        gate_bins = 0
        if self.r.size > 1 and np.isfinite(self.r).all():
            gate_bins = int(np.searchsorted(self.r, self.gate_near_m))
        
        self.stack_g = self.stack[:, gate_bins:]
        self.r_g = self.r[gate_bins:]
        
        print(f"Gated first {gate_bins} bins (~{self.gate_near_m}m)")
        print(f"New samples per ping: {self.stack_g.shape[1]}")
    
    def preprocess_data(self):
        """
        Apply comprehensive preprocessing including TVG, background removal, and angle processing.
        """
        if self.stack_g is None:
            self.gate_near_field()
        
        print("Applying preprocessing...")
        
        # 1) dB conversion
        db = 20.0 * np.log10(self.stack_g.astype(np.float32) / 255.0 + self.db_eps)
        
        # 2) TVG (spreading + absorption)
        tv = np.zeros_like(db)
        if self.r_g.size:
            if self.tvg_spreading_db:
                tv += 40.0 * np.log10(np.maximum(self.r_g, 0.5)[None, :])  # avoid -inf
            if self.tvg_alpha_db_per_m > 0.0:
                tv += 2.0 * self.tvg_alpha_db_per_m * self.r_g[None, :]
        
        db_tvg = db + tv
        
        # 3) Background removal per range bin
        self.db_hat = db_tvg - np.nanmedian(db_tvg, axis=0, keepdims=True)
        
        # 4) Angles (deg), optional desmear with yaw + fixed offset
        self.angles_deg = np.mod(self.df["angle_deg"].to_numpy(dtype=float) + self.angle_offset_deg, 360.0)
        
        if self.use_imu_yaw:
            if self.yaw_series_deg is None or len(self.yaw_series_deg) != len(self.angles_deg):
                raise ValueError("use_imu_yaw=True but yaw_series_deg missing / wrong length.")
            self.angles_deg = np.mod(self.angles_deg + self.yaw_series_deg, 360.0)
        
        # Diagnostics
        ang_span = (self.angles_deg.min(), self.angles_deg.max())
        ang_cov = np.unique(np.rint(self.angles_deg).astype(int) % 360).size
        print(f"📐 Angle span: {ang_span[0]:.1f}° → {ang_span[1]:.1f}°")
        print(f"   Coverage: ~{ang_cov} unique integer degrees")
        
        print(" Preprocessing complete")
    
    def plot_raw_stack(self, figsize: Tuple[float, float] = (9, 4)):
        """
        Plot raw and gated sonar stack as ping index vs range.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*2))
        
        # Raw stack
        im1 = ax1.imshow(self.stack, aspect="auto")
        ax1.set_title("Raw stack (ping index × range bin)")
        ax1.set_xlabel("range bin")
        ax1.set_ylabel("ping index")
        plt.colorbar(im1, ax=ax1)
        
        # Gated stack
        if self.stack_g is not None:
            im2 = ax2.imshow(self.stack_g, aspect="auto")
            ax2.set_title(f"Gated stack (ping index × range bin, gate={self.gate_near_m}m)")
            ax2.set_xlabel("range bin (after gate)")
            ax2.set_ylabel("ping index")
            plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    
    def verify_angle_consistency(self):
        """Verify consistency between angle_deg and angle_ping360 fields if present."""
        if "angle_ping360" in self.df.columns:
            err = (self.df["angle_deg"].astype(float) - self.df["angle_ping360"].astype(float) * 0.9).abs()
            print("Angle consistency (|angle_deg - 0.9*angle_ping360|) [deg]:")
            print(err.describe())
        else:
            print("ℹangle_ping360 not present; skipped consistency check.")
    
    def create_polar_image(self, use_processed: bool = True) -> np.ndarray:
        """
        Create a polar image by binning data into angle-range grid.
        
        Parameters:
        -----------
        use_processed : bool, optional
            Use processed data (db_hat) if True, raw gated data if False
            
        Returns:
        --------
        np.ndarray
            Polar image with shape (angle_bins, range_bins)
        """
        if self.angles_deg is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        # Choose data source
        if use_processed and self.db_hat is not None:
            data = self.db_hat
        else:
            data = self.stack_g.astype(float)
        
        # Create angle bins (0-360 degrees)
        angle_bins = np.linspace(0, 360, self.angle_bin_deg + 1)
        angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
        
        # Initialize polar image
        polar_image = np.full((self.angle_bin_deg, data.shape[1]), np.nan)
        
        # Bin the data by angle
        for i, angle in enumerate(self.angles_deg):
            # Find which angle bin this ping belongs to
            bin_idx = np.digitize(angle, angle_bins) - 1
            bin_idx = np.clip(bin_idx, 0, self.angle_bin_deg - 1)
            
            # Aggregate data in this bin (max, mean, etc.)
            if self.aggregation == "max":
                if np.isnan(polar_image[bin_idx, 0]):
                    polar_image[bin_idx, :] = data[i, :]
                else:
                    polar_image[bin_idx, :] = np.maximum(polar_image[bin_idx, :], data[i, :])
            elif self.aggregation == "mean":
                if np.isnan(polar_image[bin_idx, 0]):
                    polar_image[bin_idx, :] = data[i, :]
                else:
                    polar_image[bin_idx, :] = (polar_image[bin_idx, :] + data[i, :]) / 2
        
        return polar_image, angle_centers

    def plot_polar_radar(self, figsize: Tuple[float, float] = (10, 10), 
                        use_processed: bool = True, cmap: str = 'viridis') -> plt.Figure:
        """
        Create a circular radar-style polar plot of the sonar data.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        use_processed : bool, optional
            Use processed data (db_hat) if True, raw gated data if False
        cmap : str, optional
            Colormap name for the display
            
        Returns:
        --------
        matplotlib.Figure
            The created figure
        """
        polar_image, angle_centers = self.create_polar_image(use_processed)
        
        # Create polar plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Convert angles to radians and create meshgrid
        theta = np.deg2rad(angle_centers)
        r = self.r_g if self.r_g is not None else np.arange(polar_image.shape[1])
        
        # Create meshgrid for plotting
        R, T = np.meshgrid(r, theta)
        
        # Plot the data
        im = ax.pcolormesh(T, R, polar_image, cmap=cmap, shading='auto')
        
        # Customize the plot
        ax.set_theta_zero_location('N')  # North at top
        ax.set_theta_direction(-1)       # Clockwise
        ax.set_title('Ping360 Sonar - Polar View', pad=20, fontsize=14, fontweight='bold')
        ax.set_ylim(0, r.max())
        
        # Add range rings labels
        ax.set_ylabel('Range (m)', labelpad=40)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1)
        if use_processed:
            cbar.set_label('Echo Intensity (dB)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Echo Intensity (raw)', rotation=270, labelpad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Do not call plt.show() here; let the notebook handle display
        return fig

    def plot_cartesian_projection(self, figsize: Tuple[float, float] = (12, 10),
                                 use_processed: bool = True, cmap: str = 'viridis') -> plt.Figure:
        """
        Create a top-down Cartesian (X-Y) projection of the sonar data.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        use_processed : bool, optional
            Use processed data (db_hat) if True, raw gated data if False
        cmap : str, optional
            Colormap name for the display
            
        Returns:
        --------
        matplotlib.Figure
            The created figure
        """
        polar_image, angle_centers = self.create_polar_image(use_processed)
        
        # Create coordinate grids
        r = self.r_g if self.r_g is not None else np.arange(polar_image.shape[1])
        max_range = r.max()
        
        # Create Cartesian grid (reduced resolution for performance)
        x_bins = np.linspace(-max_range, max_range, 100)
        y_bins = np.linspace(-max_range, max_range, 100)
        X, Y = np.meshgrid(x_bins, y_bins)
        
        # Convert Cartesian to polar coordinates
        R_cart = np.sqrt(X**2 + Y**2)
        T_cart = np.rad2deg(np.arctan2(X, Y))  # Note: Y first for North-up
        T_cart = np.mod(T_cart, 360)  # Ensure 0-360 range
        
        # Initialize Cartesian image
        cart_image = np.full_like(X, np.nan)
        
        # Optimized interpolation using vectorized operations
        # Flatten grids for vectorized processing
        R_flat = R_cart.flatten()
        T_flat = T_cart.flatten()
        
        # Create mask for valid points (within range)
        valid_mask = (R_flat <= max_range) & (R_flat >= r.min())
        
        # Only process valid points
        R_valid = R_flat[valid_mask]
        T_valid = T_flat[valid_mask]
        
        # Find nearest angle bins (vectorized)
        angle_indices = np.searchsorted(angle_centers, T_valid)
        angle_indices = np.clip(angle_indices, 0, len(angle_centers) - 1)
        
        # Find nearest range bins (vectorized)  
        range_indices = np.searchsorted(r, R_valid)
        range_indices = np.clip(range_indices, 0, len(r) - 1)
        
        # Create flattened output
        cart_flat = np.full_like(R_flat, np.nan)
        
        # Copy intensity values for valid points
        for idx in range(len(R_valid)):
            angle_idx = angle_indices[idx]
            range_idx = range_indices[idx]
            if not np.isnan(polar_image[angle_idx, range_idx]):
                cart_flat[valid_mask.nonzero()[0][idx]] = polar_image[angle_idx, range_idx]
        
        # Reshape back to grid
        cart_image = cart_flat.reshape(X.shape)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the Cartesian projection
        im = ax.imshow(cart_image, extent=[-max_range, max_range, -max_range, max_range],
                      origin='lower', cmap=cmap, aspect='equal')
        
        # Add sonar position marker
        ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=3, label='Sonar Position')
        
        # Customize the plot
        ax.set_xlabel('Starboard Distance (m)', fontsize=12)
        ax.set_ylabel('Forward Distance (m)', fontsize=12) 
        ax.set_title('Ping360 Sonar - Top-Down View', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add range circles for reference
        for range_ring in [5, 10, 15, 20]:
            if range_ring <= max_range:
                circle = plt.Circle((0, 0), range_ring, fill=False, 
                                  color='white', alpha=0.5, linestyle='--')
                ax.add_patch(circle)
                ax.text(range_ring * 0.707, range_ring * 0.707, f'{range_ring}m',
                       color='white', fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if use_processed:
            cbar.set_label('Echo Intensity (dB)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Echo Intensity (raw)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        # Do not call plt.show() here; let the notebook handle display
        return fig

    def plot_range_angle_improved(self, figsize: Tuple[float, float] = (12, 8),
                                 use_processed: bool = True, cmap: str = 'viridis') -> plt.Figure:
        """
        Create an improved range-angle plot with proper angle binning.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        use_processed : bool, optional
            Use processed data (db_hat) if True, raw gated data if False
        cmap : str, optional
            Colormap name for the display
            
        Returns:
        --------
        matplotlib.Figure
            The created figure
        """
        polar_image, angle_centers = self.create_polar_image(use_processed)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create meshgrid for plotting
        r = self.r_g if self.r_g is not None else np.arange(polar_image.shape[1])
        A, R = np.meshgrid(angle_centers, r, indexing='ij')
        
        # Plot the data (transpose to get angle on x-axis, range on y-axis)
        im = ax.pcolormesh(A, R, polar_image, cmap=cmap, shading='auto')
        
        # Customize the plot
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Range (m)', fontsize=12)
        ax.set_title('Ping360 Sonar - Range vs Angle', fontsize=14, fontweight='bold')
        
        # Set angle limits based on actual data coverage
        if self.angles_deg is not None:
            ax.set_xlim(self.angles_deg.min() - 5, self.angles_deg.max() + 5)
        
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if use_processed:
            cbar.set_label('Echo Intensity (dB)', rotation=270, labelpad=20)
        else:
            cbar.set_label('Echo Intensity (raw)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        return fig

    def get_summary(self) -> dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
        --------
        dict
            Summary statistics
        """
        if self.df is None:
            return {"status": "No data loaded"}
        
        summary = {
            "num_pings": len(self.df),
            "samples_per_ping": self.n_samples,
            "range_span": f"{self.r[0]:.3f} → {self.r[-1]:.3f}" if self.r is not None else "Unknown",
            "angle_span": f"{self.angles_deg.min():.1f}° → {self.angles_deg.max():.1f}°" if self.angles_deg is not None else "Unknown",
            "processing_applied": self.db_hat is not None
        }
        
        return summary


def find_ping360_files(base_path: Union[str, pathlib.Path] = None) -> List[Tuple[pathlib.Path, Optional[pathlib.Path]]]:
    """
    Find Ping360 CSV files in the exports directory.
    
    Parameters:
    -----------
    base_path : str, optional
        Base path to search from (default: current directory)
        
    Returns:
    --------
    List[Tuple[Path, Optional[Path]]]
        List of (pings_csv, config_csv) tuples
    """
    from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    if base_path is None:
        exports_dir = pathlib.Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    else:
        base_path = pathlib.Path(base_path)
        # If caller provides a base path, treat it as the root and join the 'by_bag' subdir
        exports_dir = base_path / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')

    if not exports_dir.exists():
        return []
    
    ping_files = []
    ping_csvs = list(exports_dir.glob("sensor_ping360__*.csv"))
    
    for ping_csv in ping_csvs:
        # Try to find corresponding config file
        stem = ping_csv.stem.replace("sensor_ping360__", "sensor_ping360_config__")
        config_csv = exports_dir / f"{stem}.csv"
        
        if config_csv.exists():
            ping_files.append((ping_csv, config_csv))
        else:
            ping_files.append((ping_csv, None))
    
    return sorted(ping_files)


def print_ping360_files(base_path: Union[str, pathlib.Path] = "."):
    """
    Print available Ping360 files with indices for easy selection.
    
    Parameters:
    -----------
    base_path : str or Path, optional
        Base path to search from (default: current directory)
    """
    files = find_ping360_files(base_path)
    
    if not files:
        # Use configured path in the message for clarity
        print(f"No Ping360 files found in {pathlib.Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('by_bag','by_bag')}/")
        return
    
    print("Found Ping360 files:")
    for i, (ping_csv, config_csv) in enumerate(files):
        print(f"[{i}] {ping_csv.name}")
        if config_csv:
            print(f"    + {config_csv.name}")
    
    print(f"\n Found {len(files)} Ping360 datasets")
    print("Use the index number to select a file in the next cell")

