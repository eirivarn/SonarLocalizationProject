# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import json
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.sonar_config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG
from utils.sonar_processing import preprocess_edges  # Use full version instead of simple
from utils.sonar_tracking import (
    smooth_center_position, create_smooth_elliptical_aoi,
    split_contour_by_corridor, build_aoi_corridor_mask
)

# ============================ CORE DATA STRUCTURES ============================

class FrameAnalysisResult:
    """Container for frame analysis results."""
    def __init__(self, frame_idx: int = 0, timestamp: pd.Timestamp = None, 
                 distance_pixels: Optional[float] = None, angle_degrees: Optional[float] = None,
                 distance_meters: Optional[float] = None, detection_success: bool = False,
                 contour_features: Optional[Dict] = None, tracking_status: str = "SIMPLE",
                 best_contour: Optional[np.ndarray] = None, stats: Optional[Dict] = None,
                 ellipse_mask: Optional[np.ndarray] = None, corridor_mask: Optional[np.ndarray] = None):  # ADD THESE
        self.frame_idx = frame_idx
        self.timestamp = timestamp or pd.Timestamp.now()
        self.distance_pixels = distance_pixels
        self.angle_degrees = angle_degrees
        self.distance_meters = distance_meters
        self.detection_success = detection_success
        self.contour_features = contour_features or {}
        self.tracking_status = tracking_status
        self.best_contour = best_contour
        self.stats = stats or {}
        self.ellipse_mask = ellipse_mask      # ADD THIS
        self.corridor_mask = corridor_mask    # ADD THIS

    def to_dict(self) -> Dict:
        result = {
            'frame_index': self.frame_idx,
            'timestamp': self.timestamp,
            'distance_pixels': self.distance_pixels,
            'angle_degrees': self.angle_degrees,
            'distance_meters': self.distance_meters,
            'detection_success': self.detection_success,
            'tracking_status': self.tracking_status,
            **self.contour_features
        }
        # Include masks if they exist (for video overlay)
        if self.ellipse_mask is not None:
            result['ellipse_mask'] = self.ellipse_mask
        if self.corridor_mask is not None:
            result['corridor_mask'] = self.corridor_mask
        return result

class SonarDataProcessor:
    """Core sonar data processor with tracking capabilities."""
    
    def __init__(self, img_config: Dict = None):
        self.img_config = img_config or IMAGE_PROCESSING_CONFIG
        self.reset_tracking()
        self.previous_distance_pixels = None
        self.previous_ellipse = None
        
    def reset_tracking(self):
        """Reset tracking state."""
        self.last_center = None
        self.current_aoi = None
        self.smoothed_center = None
        self.previous_ellipse = None
        
    def analyze_frame(self, frame_u8: np.ndarray, extent: Tuple[float,float,float,float] = None) -> FrameAnalysisResult:
        """Analyze frame and return results."""
        H, W = frame_u8.shape[:2]
        
        # Edge detection with FULL preprocessing (uses config settings)
        raw_edges, edges_proc = preprocess_edges(frame_u8, self.img_config)
        
        mks = int(self.img_config.get('morph_close_kernel', 0))
        if mks > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mks, mks))
            edges_proc = cv2.morphologyEx(edges_proc, cv2.MORPH_CLOSE, kernel)
        
        dil = int(self.img_config.get('edge_dilation_iterations', 0))
        if dil > 0:
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            edges_proc = cv2.dilate(edges_proc, kernel2, iterations=dil)
        
        contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Select best contour
        best_contour, features, stats = self._find_best_contour(contours)
        
        # Update tracking
        if best_contour is not None and features:
            self._update_tracking(features, best_contour, H, W)
        
        # Calculate distance and angle
        distance_pixels, angle_degrees = self._calculate_distance_angle(best_contour, W, H)
        
        # Apply distance smoothing
        if distance_pixels is not None and self.previous_distance_pixels is not None:
            max_change = self.img_config.get('max_distance_change_pixels', 20)
            distance_change = abs(distance_pixels - self.previous_distance_pixels)
            if distance_change > max_change:
                direction = 1 if distance_pixels > self.previous_distance_pixels else -1
                distance_pixels = self.previous_distance_pixels + (direction * max_change)
        
        if distance_pixels is not None:
            self.previous_distance_pixels = distance_pixels
        
        # Convert to meters
        distance_meters = None
        if distance_pixels is not None and extent is not None:
            x_min, x_max, y_min, y_max = extent
            height_m = y_max - y_min
            px2m_y = height_m / H
            distance_meters = y_min + distance_pixels * px2m_y
        
        status = "TRACKED" if self.current_aoi else "SEARCHING"
        
        # CRITICAL FIX: Extract masks from current_aoi
        ellipse_mask = None
        corridor_mask = None
        if self.current_aoi is not None:
            ellipse_mask = self.current_aoi.get('ellipse_mask')
            corridor_mask = self.current_aoi.get('corridor_mask')
        
        return FrameAnalysisResult(
            best_contour=best_contour,
            distance_pixels=distance_pixels,
            angle_degrees=angle_degrees,
            distance_meters=distance_meters,
            detection_success=best_contour is not None,
            contour_features=features,
            tracking_status=status,
            stats=stats,
            ellipse_mask=ellipse_mask,      # PASS THROUGH
            corridor_mask=corridor_mask      # PASS THROUGH
        )

    def _find_best_contour(self, contours) -> Tuple[Optional[np.ndarray], Optional[Dict], Dict]:
        """Find best contour using area and elongation."""
        min_area = float(self.img_config.get('min_contour_area', 100))
        aoi_boost = 2.0
        
        best, best_feat, best_score = None, None, 0.0
        total = 0

        for c in contours or []:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            total += 1
            
            feat = self._compute_contour_features(c)
            elongation = max(feat['aspect_ratio'], feat['ellipse_elongation'])
            base_score = area * elongation
            
            if self.current_aoi is not None:
                overlap_ok = self._contour_overlaps_aoi(c, min_overlap_percent=0.7)
                if overlap_ok:
                    base_score *= aoi_boost
            
            final_score = base_score
            if self.last_center is not None:
                cx, cy = feat['centroid_x'], feat['centroid_y']
                distance = np.sqrt((cx - self.last_center[0])**2 + (cy - self.last_center[1])**2)
                distance_factor = max(0.01, 1.0 - distance / 50.0)
                final_score *= distance_factor
            
            if final_score > best_score:
                best, best_feat, best_score = c, feat, final_score

        stats = {'total_contours': total, 'best_score': best_score}
        return best, (best_feat or {}), stats

    def _compute_contour_features(self, contour) -> Dict[str, float]:
        """Compute contour features for scoring."""
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        ar = (max(w, h) / max(1, min(w, h))) if min(w, h) > 0 else 0.0

        # Ellipse elongation
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

    def _update_tracking(self, features, best_contour, H, W):
        """Update tracking state with elliptical AOI."""
        new_center = (features['centroid_x'], features['centroid_y'])
        
        smoothing_alpha = TRACKING_CONFIG.get('center_smoothing_alpha', 0.3)
        self.smoothed_center = smooth_center_position(self.smoothed_center, new_center, smoothing_alpha)
        
        expansion_factor = TRACKING_CONFIG.get('ellipse_expansion_factor', 0.3)
        aoi_mask, ellipse_center, self.previous_ellipse = create_smooth_elliptical_aoi(
            best_contour, expansion_factor, (H, W), 
            self.previous_ellipse,
            TRACKING_CONFIG.get('center_smoothing_alpha', 0.3),
            TRACKING_CONFIG.get('ellipse_size_smoothing_alpha', 0.1),
            TRACKING_CONFIG.get('ellipse_orientation_smoothing_alpha', 0.1),
            TRACKING_CONFIG.get('ellipse_max_movement_pixels', 4.0)
        )
        
        self.current_aoi = {
            'mask': aoi_mask,
            'center': ellipse_center,
            'ellipse_params': self.previous_ellipse,
            'smoothed_center': self.smoothed_center,
            'type': 'elliptical'
        }
        self.last_center = new_center
        
        # CRITICAL FIX: Store masks for video overlay
        self.current_aoi['ellipse_mask'] = None
        self.current_aoi['corridor_mask'] = None
        
        # Optional: Add corridor splitting
        if TRACKING_CONFIG.get('use_corridor_splitting', False):
            try:
                inside_c, corridor_c, other_c, ell_mask, corr_mask = split_contour_by_corridor(
                    best_contour, self.previous_ellipse, (H, W),
                    band_k=TRACKING_CONFIG.get('corridor_band_k', 0.55),
                    length_px=TRACKING_CONFIG.get('corridor_length_px', None),
                    length_factor=TRACKING_CONFIG.get('corridor_length_factor', 1.25),
                    widen=TRACKING_CONFIG.get('corridor_widen', 1.0),
                    both_directions=TRACKING_CONFIG.get('corridor_both_directions', True)
                )
                # Store the masks for video overlay
                self.current_aoi['ellipse_mask'] = ell_mask
                self.current_aoi['corridor_mask'] = corr_mask
            except Exception as e:
                print(f"Warning: Corridor splitting failed: {e}")

    def _contour_overlaps_aoi(self, contour: np.ndarray, min_overlap_percent: float = 0.7) -> bool:
        """Check if contour overlaps with AOI."""
        if self.current_aoi is None or 'mask' not in self.current_aoi:
            return False
        
        mask = self.current_aoi['mask']
        inside_count = 0
        
        for point in contour:
            x, y = point[0]
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[int(y), int(x)] > 0:
                    inside_count += 1
        
        overlap_percent = inside_count / len(contour) if len(contour) > 0 else 0
        return overlap_percent >= min_overlap_percent

    def _calculate_distance_angle(self, contour, image_width: int, image_height: int) -> Tuple[Optional[float], Optional[float]]:
        """Calculate distance and angle from contour."""
        if contour is None or len(contour) < 5:
            return None, None
        
        try:
            (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
            
            # Use smoothed center if available
            if self.smoothed_center is not None:
                cx, cy = self.smoothed_center
            
            # Determine major axis orientation
            major_angle = angle if w >= h else angle + 90.0
            
            # Calculate intersection with center beam
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

# ============================ NPZ I/O ============================

def load_cone_run_npz(path: str | Path):
    """Load NPZ file with cones data."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "cones" not in keys or "extent" not in keys:
            raise KeyError(f"NPZ must contain 'cones' and 'extent'")
        
        cones = np.asarray(z["cones"], dtype=np.float32)
        extent = tuple(np.asarray(z["extent"], dtype=np.float64).tolist())

        # meta
        meta = {}
        if "meta_json" in keys:
            raw = z["meta_json"]
            meta = json.loads(raw.item() if getattr(raw, "ndim", 0) else raw.tolist())
        elif "meta" in keys:
            m = z["meta"]
            try:
                meta = m.item() if hasattr(m, "item") else m.tolist()
                if isinstance(meta, (bytes,str)): 
                    meta = json.loads(meta)
            except Exception:
                meta = {}

        ts = None
        if "ts_unix_ns" in keys:
            ts = pd.to_datetime(np.asarray(z["ts_unix_ns"], dtype=np.int64), utc=True)
        elif "ts" in keys:
            ts = pd.to_datetime(z["ts"], utc=True, errors='coerce')

        T = cones.shape[0]
        if ts is None or isinstance(ts, pd.Timestamp):
            ts = pd.DatetimeIndex([ts or pd.Timestamp.utcnow()]*T)
        else:
            ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
            if len(ts) != T:
                ts = pd.DatetimeIndex([ts[0]]*T) if len(ts)==1 else pd.to_datetime(range(T), unit="s", utc=True)

    return cones, ts, extent, meta

def get_available_npz_files(npz_dir: str | None = None) -> List[Path]:
    """Get list of available NPZ files."""
    from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    npz_dir = Path(npz_dir) if npz_dir is not None else Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('outputs', 'outputs')
    if not npz_dir.exists():
        return []
    return [f for f in npz_dir.glob("*_cones.npz") if not f.name.startswith('._')]

def to_uint8_gray(frame01: np.ndarray) -> np.ndarray:
    """Convert normalized frame to uint8 grayscale."""
    safe = np.nan_to_num(frame01, nan=0.0, posinf=1.0, neginf=0.0)
    safe = np.clip(safe, 0.0, 1.0)
    return (safe * 255.0).astype(np.uint8)

def get_pixel_to_meter_mapping(npz_file_path: Union[str, Path]) -> Dict[str, Any]:
    """Auto-detect pixel->meter mapping from NPZ file metadata."""
    npz_file_path = Path(npz_file_path)
    
    try:
        cones, ts, extent, meta = load_cone_run_npz(npz_file_path)
        T, H, W = cones.shape
        x_min, x_max, y_min, y_max = extent
        width_m = float(x_max - x_min)
        height_m = float(y_max - y_min)
        px2m_x = width_m / float(W)
        px2m_y = height_m / float(H)
        
        return {
            'pixels_to_meters_avg': 0.5 * (px2m_x + px2m_y),
            'px2m_x': px2m_x,
            'px2m_y': px2m_y,
            'image_shape': (H, W),
            'sonar_coverage_meters': max(width_m, height_m),
            'extent': extent,
            'source': 'npz_metadata',
            'success': True
        }
    except Exception as e:
        from utils.sonar_config import CONE_H_DEFAULT, CONE_W_DEFAULT, DISPLAY_RANGE_MAX_M_DEFAULT
        image_shape = (CONE_H_DEFAULT, CONE_W_DEFAULT)
        sonar_coverage_meters = DISPLAY_RANGE_MAX_M_DEFAULT * 2
        pixels_to_meters_avg = sonar_coverage_meters / max(image_shape)
        
        return {
            'pixels_to_meters_avg': pixels_to_meters_avg,
            'image_shape': image_shape,
            'source': 'config_defaults',
            'success': False,
            'error': str(e)
        }

# ============================ ANALYSIS ENGINE ============================

class DistanceAnalysisEngine:
    """Engine for distance analysis from NPZ files."""
    
    def __init__(self, processor: SonarDataProcessor = None):
        self.processor = processor or SonarDataProcessor()
        
    def analyze_npz_sequence(self, npz_file_index: int = 0, frame_start: int = 0,
                       frame_count: Optional[int] = None, frame_step: int = 1,
                       npz_dir: str = None, save_outputs: bool = False) -> pd.DataFrame:
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
        self.processor.reset_tracking()
        
        for i in range(actual):
            idx = frame_start + i * frame_step
            if idx >= T:
                break
                
            frame_u8 = to_uint8_gray(cones[idx])
            result = self.processor.analyze_frame(frame_u8, extent)
            result.frame_idx = idx
            result.timestamp = timestamps[idx] if idx < len(timestamps) else timestamps[0]
            
            results.append(result.to_dict())
        
        df = pd.DataFrame(results)
        self._print_analysis_summary(df)
        
        if save_outputs:
            from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
            outputs_dir = Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('outputs', 'outputs')
            outputs_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = outputs_dir / f"{npz_file.stem}_analysis_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Analysis results saved to: {csv_path}")
    
        return df
    
    def _print_analysis_summary(self, df: pd.DataFrame):
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

# ============================ VISUALIZATION ============================

class VisualizationEngine:
    """Engine for plotting distance analysis results."""
    
    @staticmethod
    def plot_distance_analysis(distance_results: pd.DataFrame, 
                             title: str = "Distance Analysis Over Time") -> Optional[go.Figure]:
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

class ComparisonEngine:
    """Engine for sonar vs DVL comparisons."""
    
    @staticmethod
    def compare_sonar_vs_dvl(distance_results: pd.DataFrame, raw_data: Dict[str, pd.DataFrame],
                           sonar_coverage_m: float = 5.0, sonar_image_size: int = 700) -> Tuple[Optional[go.Figure], Dict]:
        """Compare sonar and DVL distance measurements."""
        
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
        
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=sonar['synthetic_time'], y=sonar['distance_meters'],
                      mode='lines', name='Sonar', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=nav['relative_time'], y=nav['NetDistance'],
                      mode='lines', name='DVL', line=dict(color='blue', width=3)))
        
        fig.update_layout(title="Sonar vs DVL Distance Comparison", height=600)
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
        }
        
        print(f"\nComparison: Sonar={sonar_mean:.3f}m, DVL={dvl_mean:.3f}m, Ratio={stats['scale_ratio']:.3f}x")
        
        return fig, stats
