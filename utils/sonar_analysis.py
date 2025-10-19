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
from utils.sonar_tracking import (
    smooth_center_position, create_smooth_elliptical_aoi,
    split_contour_by_corridor
)
from utils.sonar_utils import load_cone_run_npz, to_uint8_gray
from utils.io_utils import get_available_npz_files

# ============================ DATA STRUCTURES ============================

@dataclass
class FrameAnalysisResult:
    """Lightweight container for frame analysis results."""
    frame_idx: int
    timestamp: pd.Timestamp
    distance_pixels: Optional[float] = None
    angle_degrees: Optional[float] = None
    distance_meters: Optional[float] = None
    detection_success: bool = False
    contour_features: Dict = field(default_factory=dict)
    tracking_status: str = "SIMPLE"
    best_contour: Optional[np.ndarray] = None
    stats: Dict = field(default_factory=dict)
    ellipse_mask: Optional[np.ndarray] = None
    corridor_mask: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dict (exclude numpy arrays for CSV)."""
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

@dataclass
class TrackingState:
    """Immutable container for persistent tracking state."""
    last_center: Optional[Tuple[float, float]] = None
    current_aoi: Optional[Dict] = None
    smoothed_center: Optional[Tuple[float, float]] = None
    previous_ellipse: Optional[Tuple] = None
    previous_distance_pixels: Optional[float] = None

# ============================ PURE ANALYSIS FUNCTIONS ============================

def compute_contour_features(contour: np.ndarray) -> Dict[str, float]:
    """Compute contour features (pure function)."""
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    ar = (max(w, h) / max(1, min(w, h))) if min(w, h) > 0 else 0.0

    if len(contour) >= 5:
        try:
            _, (minor, major), _ = cv2.fitEllipse(contour)
            ell = (major / minor) if minor > 0 else ar
        except Exception:
            ell = ar
    else:
        ell = ar

    return {
        'area': area,
        'aspect_ratio': ar,
        'ellipse_elongation': ell,
        'rect': (x, y, w, h),
        'centroid_x': float(x + w/2),
        'centroid_y': float(y + h/2),
    }


def contour_overlaps_aoi(
    contour: np.ndarray,
    aoi: Dict,
    min_overlap_percent: float = 0.7
) -> bool:
    """Check if contour overlaps with AOI (pure function)."""
    if aoi is None or 'mask' not in aoi:
        return False
    
    mask = aoi['mask']
    inside_count = 0
    
    for point in contour:
        x, y = point[0]
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[int(y), int(x)] > 0:
                inside_count += 1
    
    overlap_percent = inside_count / len(contour) if len(contour) > 0 else 0
    return overlap_percent >= min_overlap_percent


def find_best_contour(
    contours: List,
    img_config: Dict,
    state: TrackingState
) -> Tuple[Optional[np.ndarray], Optional[Dict], Dict]:
    """Find best contour from list (pure function)."""
    min_area = float(img_config.get('min_contour_area', 100))
    aoi_boost = float(img_config.get('aoi_boost_factor', 2.0))
    
    best, best_feat, best_score = None, None, 0.0
    total = 0

    for c in contours or []:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        total += 1
        
        feat = compute_contour_features(c)
        elongation = max(feat['aspect_ratio'], feat['ellipse_elongation'])
        base_score = area * elongation
        
        if state.current_aoi is not None:
            overlap_ok = contour_overlaps_aoi(c, state.current_aoi)
            if overlap_ok:
                base_score *= aoi_boost
        
        final_score = base_score
        if state.last_center is not None:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            distance = np.sqrt((cx - state.last_center[0])**2 + (cy - state.last_center[1])**2)
            distance_factor = max(0.01, 1.0 - distance / 50.0)
            final_score *= distance_factor
        
        if final_score > best_score:
            best, best_feat, best_score = c, feat, final_score

    stats = {'total_contours': total, 'best_score': best_score}
    return best, (best_feat or {}), stats


def calculate_distance_angle(
    contour: Optional[np.ndarray],
    state: TrackingState,
    image_width: int,
    image_height: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate distance and angle from contour (pure function)."""
    if contour is None or len(contour) < 5:
        return None, None
    
    try:
        (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
        
        if state.smoothed_center is not None:
            cx, cy = state.smoothed_center
        
        major_angle = angle if w >= h else angle + 90.0
        center_x = image_width / 2
        ang_r = np.radians(major_angle)
        cos_ang = np.cos(ang_r)
        
        if abs(cos_ang) > 1e-6:
            t = (center_x - cx) / cos_ang
            intersect_y = cy + t * np.sin(ang_r)
            distance = intersect_y
        else:
            distance = cy
        
        red_line_angle = (float(angle) + 90.0) % 360.0
        return float(distance), red_line_angle
    except Exception:
        return None, None


def update_tracking_state(
    features: Dict,
    best_contour: np.ndarray,
    image_shape: Tuple[int, int],
    state: TrackingState,
    tracking_config: Dict,
) -> TrackingState:
    """Update tracking state (returns new state, functional)."""
    H, W = image_shape
    new_center = (features['centroid_x'], features['centroid_y'])
    
    smoothing_alpha = tracking_config.get('center_smoothing_alpha', 0.3)
    smoothed_center = smooth_center_position(state.smoothed_center, new_center, smoothing_alpha)
    
    expansion_factor = tracking_config.get('ellipse_expansion_factor', 0.3)
    aoi_mask, ellipse_center, new_ellipse = create_smooth_elliptical_aoi(
        best_contour, expansion_factor, (H, W),
        state.previous_ellipse,
        tracking_config.get('center_smoothing_alpha', 0.3),
        tracking_config.get('ellipse_size_smoothing_alpha', 0.1),
        tracking_config.get('ellipse_orientation_smoothing_alpha', 0.1),
        tracking_config.get('ellipse_max_movement_pixels', 4.0)
    )
    
    new_aoi = {
        'mask': aoi_mask,
        'center': ellipse_center,
        'ellipse_params': new_ellipse,
        'smoothed_center': smoothed_center,
        'type': 'elliptical',
        'ellipse_mask': None,
        'corridor_mask': None,
    }
    
    if tracking_config.get('use_corridor_splitting', False):
        try:
            inside_c, corridor_c, other_c, ell_mask, corr_mask = split_contour_by_corridor(
                best_contour, new_ellipse, (H, W),
                band_k=tracking_config.get('corridor_band_k', 0.55),
                length_px=tracking_config.get('corridor_length_px', None),
                length_factor=tracking_config.get('corridor_length_factor', 1.25),
                widen=tracking_config.get('corridor_widen', 1.0),
                both_directions=tracking_config.get('corridor_both_directions', True)
            )
            new_aoi['ellipse_mask'] = ell_mask
            new_aoi['corridor_mask'] = corr_mask
        except Exception as e:
            print(f"Warning: Corridor splitting failed: {e}")
    
    return TrackingState(
        last_center=new_center,
        current_aoi=new_aoi,
        smoothed_center=smoothed_center,
        previous_ellipse=new_ellipse,
        previous_distance_pixels=state.previous_distance_pixels,
    )


def analyze_frame(
    frame_u8: np.ndarray,
    extent: Tuple[float, float, float, float],
    state: TrackingState,
    img_config: Dict,
    tracking_config: Dict,
) -> Tuple[FrameAnalysisResult, TrackingState]:
    """Analyze single frame (pure function, returns new state and result)."""
    H, W = frame_u8.shape[:2]
    
    raw_edges, edges_proc = preprocess_edges(frame_u8, img_config)
    
    mks = int(img_config.get('morph_close_kernel', 0))
    if mks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mks, mks))
        edges_proc = cv2.morphologyEx(edges_proc, cv2.MORPH_CLOSE, kernel)
    
    dil = int(img_config.get('edge_dilation_iterations', 0))
    if dil > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_proc = cv2.dilate(edges_proc, kernel2, iterations=dil)
    
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour, features, stats = find_best_contour(contours, img_config, state)
    
    new_state = state
    if best_contour is not None and features:
        new_state = update_tracking_state(features, best_contour, (H, W), state, tracking_config)
    
    distance_pixels, angle_degrees = calculate_distance_angle(best_contour, new_state, W, H)
    
    if distance_pixels is not None and state.previous_distance_pixels is not None:
        max_change = img_config.get('max_distance_change_pixels', 20)
        distance_change = abs(distance_pixels - state.previous_distance_pixels)
        if distance_change > max_change:
            direction = 1 if distance_pixels > state.previous_distance_pixels else -1
            distance_pixels = state.previous_distance_pixels + (direction * max_change)
    
    new_state.previous_distance_pixels = distance_pixels
    
    distance_meters = None
    if distance_pixels is not None and extent is not None:
        x_min, x_max, y_min, y_max = extent
        height_m = y_max - y_min
        px2m_y = height_m / H
        distance_meters = y_min + distance_pixels * px2m_y
    
    tracking_status = "TRACKED" if new_state.current_aoi else "SEARCHING"
    
    ellipse_mask = new_state.current_aoi.get('ellipse_mask') if new_state.current_aoi else None
    corridor_mask = new_state.current_aoi.get('corridor_mask') if new_state.current_aoi else None
    
    result = FrameAnalysisResult(
        frame_idx=0,
        timestamp=pd.Timestamp.now(),
        best_contour=best_contour,
        distance_pixels=distance_pixels,
        angle_degrees=angle_degrees,
        distance_meters=distance_meters,
        detection_success=best_contour is not None,
        contour_features=features,
        tracking_status=tracking_status,
        stats=stats,
        ellipse_mask=ellipse_mask,
        corridor_mask=corridor_mask,
    )
    
    return result, new_state

# ============================ ANALYSIS ENGINE ============================

def analyze_npz_sequence(
    npz_file_index: int = 0,
    frame_start: int = 0,
    frame_count: Optional[int] = None,
    frame_step: int = 1,
    npz_dir: Optional[str] = None,
    save_outputs: bool = False,
) -> pd.DataFrame:
    """Analyze distance over time from NPZ file."""
    print("=== DISTANCE ANALYSIS FROM NPZ ===")
    
    files = get_available_npz_files(npz_dir)
    if npz_file_index >= len(files):
        raise IndexError(f"NPZ file index {npz_file_index} not available")
    
    npz_file = files[npz_file_index]
    cones, timestamps, extent, _ = load_cone_run_npz(npz_file)
    T = len(cones)
    
    if frame_count is None:
        frame_count = T - frame_start
    actual = min(frame_count, max(0, (T - frame_start)) // max(1, frame_step))
    
    print(f"Processing {actual} frames from {frame_start} (step={frame_step})")
    
    results = []
    state = TrackingState()
    
    for i in range(actual):
        idx = frame_start + i * frame_step
        if idx >= T:
            break
        
        frame_u8 = to_uint8_gray(cones[idx])
        result, state = analyze_frame(
            frame_u8, extent, state,
            IMAGE_PROCESSING_CONFIG,
            TRACKING_CONFIG
        )
        result.frame_idx = idx
        result.timestamp = timestamps[idx] if idx < len(timestamps) else timestamps[0]
        
        results.append(result.to_dict())
    
    df = pd.DataFrame(results)
    print_analysis_summary(df)
    
    if save_outputs:
        from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
        outputs_dir = Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('outputs', 'outputs')
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = outputs_dir / f"{npz_file.stem}_analysis_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Analysis results saved to: {csv_path}")
    
    return df


def print_analysis_summary(df: pd.DataFrame) -> None:
    """Print analysis summary statistics."""
    total = len(df)
    successful = df['detection_success'].sum()
    success_rate = successful / total * 100 if total > 0 else 0
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total frames: {total}, Successful: {successful} ({success_rate:.1f}%)")
    
    if successful > 0:
        dist_col = 'distance_meters' if 'distance_meters' in df.columns and df['distance_meters'].notna().any() else 'distance_pixels'
        valid_distances = df.loc[df['detection_success'], dist_col].dropna()
        
        if len(valid_distances) > 0:
            unit = "meters" if dist_col == 'distance_meters' else "pixels"
            print(f"Distance ({unit}): mean={valid_distances.mean():.3f}, std={valid_distances.std():.3f}")
