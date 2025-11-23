# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import cv2
import numpy as np
import pandas as pd

from utils.config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG, EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
from utils.image_enhancement import preprocess_edges
from utils.sonar_tracking import NetTracker
from utils.sonar_utils import load_cone_run_npz, to_uint8_gray
from utils.io_utils import get_available_npz_files

# ============================ DATA STRUCTURES ============================

@dataclass
class FrameAnalysisResult:
    """Analysis result for a single sonar frame with net detection data."""
    frame_idx: int
    timestamp: pd.Timestamp
    distance_pixels: Optional[float] = None
    angle_degrees: Optional[float] = None
    distance_meters: Optional[float] = None
    detection_success: bool = False
    contour_features: Dict = field(default_factory=dict)
    tracking_status: str = "NO_DETECTION"
    
    def to_dict(self) -> Dict:
        return {
            'frame_index': self.frame_idx,
            'timestamp': self.timestamp,
            'distance_pixels': self.distance_pixels,
            'angle_degrees': self.angle_degrees,
            'distance_meters': self.distance_meters,
            'detection_success': self.detection_success,
            'tracking_status': self.tracking_status,
            **self.contour_features,
        }


def analyze_npz_sequence(
    npz_file_index: int = 0,
    npz_dir: str | None = None,
    frame_start: int = 0,
    frame_count: int = 100,
    frame_step: int = 1,
    save_outputs: bool = False,
) -> pd.DataFrame:
    """
    Analyze a sequence of sonar frames for net detection and distance measurement.
    
    Processing Pipeline:
    1. Load cone-view sonar frames from NPZ file
    2. Convert to binary images (eliminates intensity variations)
    3. Apply edge enhancement using morphological operations
    4. Track net structures using NetTracker system
    5. Convert pixel distances to meters using spatial extent
    6. Return time series of measurements
    
    Args:
        npz_file_index: Index of NPZ file to process
        npz_dir: Directory containing NPZ files (None = use default)
        frame_start: First frame to process
        frame_count: Number of frames to process
        frame_step: Process every Nth frame
        save_outputs: Save results to CSV file
        
    Returns:
        DataFrame with columns: frame_index, timestamp, distance_pixels, 
        distance_meters, angle_degrees, detection_success, tracking_status, area
    """
    files = get_available_npz_files(npz_dir)
    npz_path = files[npz_file_index]
    
    cones, timestamps, extent, _ = load_cone_run_npz(npz_path)
    
    # Extract bag_id from filename - handle multiple possible suffixes
    # Filename formats:
    #   sensor_sonoptix_echo_image__YYYY-MM-DD_HH-MM-SS_video.npz
    #   sensor_sonoptix_echo_image__YYYY-MM-DD_HH-MM-SS_data_cones_video.npz
    bag_id = npz_path.stem.replace('sensor_sonoptix_echo_image__', '')
    
    # Remove all possible suffixes to get clean bag_id
    for suffix in ['_data_cones_video', '_video', '_data_cones', '_cones']:
        bag_id = bag_id.replace(suffix, '')
    
    frame_indices = list(range(
        max(0, frame_start),
        min(len(cones), frame_start + frame_count),
        max(1, frame_step)
    ))
    
    print(f"Analyzing {len(frame_indices)} frames from {npz_path.name}")
    print(f"Bag ID: {bag_id}")
    print(f"Using NetTracker system with binary processing and ellipse fitting")
    
    # Create NetTracker with combined configuration
    config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
    tracker = NetTracker(config)
    
    results = []
    
    for i, frame_idx in enumerate(frame_indices):
        frame_u8 = to_uint8_gray(cones[frame_idx])
        H, W = frame_u8.shape[:2]
        
        # 1. Binary conversion (eliminates signal strength dependency)
        binary = (frame_u8 > config['binary_threshold']).astype(np.uint8) * 255

        # 2. Edge enhancement using morphological operations
        try:
            _, edges = preprocess_edges(binary, config)
        except:
            edges = binary  # Fallback to binary if enhancement fails
        
        # 3. Track net using NetTracker system
        contour = tracker.find_and_update(edges, (H, W))
        distance_px, angle_deg = tracker.calculate_distance(W, H)
        
        # 4. Convert pixel distance to meters using spatial extent
        distance_m = None
        if distance_px is not None and extent is not None:
            # Meters per pixel in Y (range) direction
            px2m = (extent[3] - extent[2]) / H
            # Convert: y_min + pixels * scaling_factor  
            distance_m = extent[2] + distance_px * px2m
        
        # 5. Store comprehensive result
        results.append({
            'frame_index': frame_idx,
            'timestamp': pd.Timestamp(timestamps[frame_idx]),
            'distance_pixels': distance_px,
            'distance_meters': distance_m,
            'angle_degrees': angle_deg,
            'detection_success': (contour is not None),
            'tracking_status': tracker.get_status(),
            'area': float(cv2.contourArea(contour)) if contour is not None else 0.0
        })
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frame_indices)} | Status: {tracker.get_status()}")

    df = pd.DataFrame(results)
    
    # Store bag_id as metadata (accessible for plotting)
    df.attrs['bag_id'] = bag_id

    # Print analysis summary
    detection_rate = df['detection_success'].mean() * 100
    print(f"\nAnalysis Summary:")
    print(f"  Detection Rate: {detection_rate:.1f}%")
    print(f"  Valid Distance Measurements: {df['distance_meters'].notna().sum()}")
    if df['distance_meters'].notna().any():
        print(f"  Distance Range: {df['distance_meters'].min():.2f} - {df['distance_meters'].max():.2f} m")
    
    if save_outputs:
        output_root = Path(EXPORTS_DIR_DEFAULT)
        output_dir = output_root / EXPORTS_SUBDIRS.get('outputs', 'outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{bag_id}_analysis.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    return df