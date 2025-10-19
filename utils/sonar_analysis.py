# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import cv2

from utils.config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG
from utils.image_enhancement import preprocess_edges
from utils.sonar_tracking import NetTracker
from utils.sonar_utils import load_cone_run_npz, to_uint8_gray
from utils.io_utils import get_available_npz_files

# ============================ DATA STRUCTURES ============================

@dataclass
class FrameAnalysisResult:
    """Simplified result for CSV export."""
    frame_idx: int
    timestamp: pd.Timestamp
    distance_pixels: Optional[float] = None
    angle_degrees: Optional[float] = None
    distance_meters: Optional[float] = None
    detection_success: bool = False
    contour_features: Dict = field(default_factory=dict)
    tracking_status: str = "SIMPLE"
    
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
    """Clean analysis using NetTracker."""
    files = get_available_npz_files(npz_dir)
    npz_path = files[npz_file_index]
    
    cones, timestamps, extent, _ = load_cone_run_npz(npz_path)
    
    frame_indices = list(range(
        max(0, frame_start),
        min(len(cones), frame_start + frame_count),
        max(1, frame_step)
    ))
    
    print(f"Analyzing {len(frame_indices)} frames from {npz_path.name}")
    
    # Create tracker
    config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
    tracker = NetTracker(config)
    
    results = []
    
    for i, frame_idx in enumerate(frame_indices):
        frame_u8 = to_uint8_gray(cones[frame_idx])
        H, W = frame_u8.shape[:2]
        
        # Preprocess
        binary = (frame_u8 > config['binary_threshold']).astype(np.uint8) * 255
        
        try:
            from utils.image_enhancement import adaptive_linear_momentum_merge_fast, preprocess_edges
            momentum = adaptive_linear_momentum_merge_fast(binary, 
                angle_steps=config['adaptive_angle_steps'],
                base_radius=config['adaptive_base_radius'],
                max_elongation=config['adaptive_max_elongation'],
                momentum_boost=config['momentum_boost'],
                linearity_threshold=config['adaptive_linearity_threshold'],
                downscale_factor=config['downscale_factor'],
                top_k_bins=config['top_k_bins'],
                min_coverage_percent=config['min_coverage_percent'],
                gaussian_sigma=config['gaussian_sigma']
            )
            _, edges = preprocess_edges(momentum, config)
        except:
            edges = binary
        
        # Morphology
        if config.get('morph_close_kernel', 0) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config['morph_close_kernel'],) * 2)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        
        # Track
        contour = tracker.find_and_update(edges, (H, W))
        distance_px, angle_deg = tracker.calculate_distance(W, H)
        
        # Convert to meters
        distance_m = None
        if distance_px is not None and extent is not None:
            px2m = (extent[3] - extent[2]) / H
            distance_m = extent[2] + distance_px * px2m
        
        # CRITICAL FIX: Don't add 90° here - angle_deg is already the major axis angle
        # Just use the angle directly from the tracker
        
        results.append({
            'frame_index': frame_idx,
            'timestamp': pd.Timestamp(timestamps[frame_idx]),
            'distance_pixels': distance_px,
            'distance_meters': distance_m,
            'angle_degrees': angle_deg,  # Use directly - no modification needed
            'detection_success': (contour is not None),
            'tracking_status': tracker.get_status(),
            'area': float(cv2.contourArea(contour)) if contour is not None else 0.0
        })
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frame_indices)} | {tracker.get_status()}")
    
    df = pd.DataFrame(results)
    
    if save_outputs:
        from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
        output_dir = Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS['outputs']
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{npz_path.stem.replace('_video', '')}_analysis.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    return df