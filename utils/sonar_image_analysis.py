# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.sonar_utils import (
    load_df, get_sonoptix_frame,
    enhance_intensity, apply_flips, cone_raster_like_display_cell
)
from utils.sonar_config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG, VIDEO_CONFIG, ConeGridSpec

# ============================ CORE DATA STRUCTURES ============================

class FrameAnalysisResult:
    """CORE container for frame analysis results."""
    def __init__(self, frame_idx: int = 0, timestamp: pd.Timestamp = None, 
                 distance_pixels: Optional[float] = None, angle_degrees: Optional[float] = None,
                 distance_meters: Optional[float] = None, detection_success: bool = False,
                 contour_features: Optional[Dict] = None, tracking_status: str = "SIMPLE",
                 best_contour: Optional[np.ndarray] = None, stats: Optional[Dict] = None,
                 edges_processed: Optional[np.ndarray] = None):
        self.frame_idx = frame_idx
        self.timestamp = timestamp or pd.Timestamp.now()
        self.distance_pixels = distance_pixels
        self.angle_degrees = angle_degrees
        self.distance_meters = distance_meters
        self.detection_success = detection_success
        self.contour_features = contour_features or {}
        self.tracking_status = tracking_status
        # Core processing artifacts
        self.best_contour = best_contour
        self.stats = stats or {}
        self.edges_processed = edges_processed

    def to_dict(self) -> Dict:
        return {
            'frame_index': self.frame_idx,
            'timestamp': self.timestamp,
            'distance_pixels': self.distance_pixels,
            'angle_degrees': self.angle_degrees,
            'distance_meters': self.distance_meters,
            'detection_success': self.detection_success,
            'tracking_status': self.tracking_status,
            **self.contour_features
        }

class SonarDataProcessor:
    """Simplified sonar data processor - core functionality only."""
    
    def __init__(self, img_config: Dict = None):
        self.img_config = img_config or IMAGE_PROCESSING_CONFIG
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset tracking state."""
        self.last_center = None
        self.current_aoi = None
        
    def preprocess_frame(self, frame_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple preprocessing without object separation."""
        return preprocess_edges(frame_u8, self.img_config)
        
    def find_best_contour(self, contours):
        """Find the best contour using core selection logic."""
        return select_best_contour_core(
            contours, 
            self.last_center, 
            self.current_aoi, 
            self.img_config
        )
        
    def analyze_frame(self, frame_u8: np.ndarray, extent: Tuple[float,float,float,float] = None) -> FrameAnalysisResult:
        """Simplified frame analysis without complex object separation."""
        H, W = frame_u8.shape[:2]
        
        # STEP 1: Simple preprocessing
        _, edges_proc = self.preprocess_frame(frame_u8)
        
        # STEP 2: Find contours
        contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # STEP 3: Select best contour
        best_contour, features, stats = self.find_best_contour(contours)
        
        # Extract distance and angle
        distance_pixels, angle_degrees = _distance_angle_from_contour(best_contour, W, H)
        
        # Convert to meters if extent provided
        distance_meters = None
        if distance_pixels is not None and extent is not None:
            x_min, x_max, y_min, y_max = extent
            height_m = y_max - y_min
            px2m_y = height_m / H
            distance_meters = y_min + distance_pixels * px2m_y
        
        # Update simple tracking
        if best_contour is not None and features:
            self.last_center = (features['centroid_x'], features['centroid_y'])
            
            # Simple AOI: expand bounding rect
            x, y, w, h = features['rect']
            expansion = 25
            self.current_aoi = (
                max(0, x - expansion),
                max(0, y - expansion),
                min(W - max(0, x - expansion), w + 2*expansion),
                min(H - max(0, y - expansion), h + 2*expansion)
            )
        
        # Create simple tracking status
        status_parts = []
        if self.current_aoi:
            status_parts.append("TRACKED")
        else:
            status_parts.append("SEARCHING")
            
        tracking_status = "_".join(status_parts)
        
        # Create result
        return FrameAnalysisResult(
            best_contour=best_contour,
            distance_pixels=distance_pixels,
            angle_degrees=angle_degrees,
            distance_meters=distance_meters,
            detection_success=best_contour is not None,
            contour_features=features,
            tracking_status=tracking_status,
            stats=stats,
            edges_processed=edges_proc
        )

    def process_frame(self, frame_idx: int) -> Optional[FrameAnalysisResult]:
        """SIMPLIFIED process frame - user provides NPZ file index separately."""
        # NOTE: This method requires NPZ file to be loaded separately
        # Use DistanceAnalysisEngine.analyze_npz_sequence for full workflow
        try:
            # For now, just create a basic result - real implementation would need NPZ data
            return FrameAnalysisResult(
                frame_idx=frame_idx,
                timestamp=pd.Timestamp.now(), 
                distance_pixels=None,
                angle_degrees=None,
                distance_meters=None,
                detection_success=False,
                contour_features={},
                tracking_status="SIMPLE"
            )
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return None

# ============================ NPZ I/O ============================

def load_cone_run_npz(path: str | Path):
    """Robust loader: returns (cones[T,H,W] float32 ∈ [0,1], ts DatetimeIndex, extent tuple, meta dict)."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        if "cones" not in keys or "extent" not in keys:
            raise KeyError(f"NPZ must contain 'cones' and 'extent'. Keys: {sorted(keys)}")
        cones  = np.asarray(z["cones"], dtype=np.float32)
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
                if isinstance(meta, (bytes,str)): meta = json.loads(meta)
            except Exception:
                meta = {}

        # timestamps (many variants)
        ts = None
        if "ts_unix_ns" in keys:
            ts = pd.to_datetime(np.asarray(z["ts_unix_ns"], dtype=np.int64), utc=True)
        elif "ts" in keys:
            try:
                ts = pd.to_datetime(z["ts"], utc=True)
            except Exception:
                ts = pd.to_datetime(np.asarray(z["ts"], dtype="int64"), unit="s", utc=True)
        else:
            for k in ("ts_unix_ns","ts_ns","timestamps_ns","ts"):
                if isinstance(meta, dict) and (k in meta):
                    v = meta[k]
                    try:
                        ts = pd.to_datetime(np.asarray(v, dtype="int64"), utc=True) if "ns" in k \
                             else pd.to_datetime(v, utc=True)
                    except Exception:
                        ts = None
                    break

    # normalize ts to length T
    T = cones.shape[0]
    if ts is None:
        ts = pd.to_datetime(range(T), unit="s", utc=True)
    elif isinstance(ts, pd.Timestamp):
        ts = pd.DatetimeIndex([ts]*T)
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
        if len(ts) != T:
            ts = pd.DatetimeIndex([ts[0]]*T) if len(ts)==1 else pd.to_datetime(range(T), unit="s", utc=True)

    return cones, ts, extent, meta


def get_available_npz_files(npz_dir: str | None = None) -> List[Path]:
    from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    npz_dir = Path(npz_dir) if npz_dir is not None else Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('outputs', 'outputs')
    if not npz_dir.exists():
        return []
    return [f for f in npz_dir.glob("*_cones.npz") if not f.name.startswith('._')]

# ============================ Small utilities ============================

def to_uint8_gray(frame01: np.ndarray) -> np.ndarray:
    # Robust conversion: handle NaN/inf and out-of-range values before casting
    safe = np.nan_to_num(frame01, nan=0.0, posinf=1.0, neginf=0.0)
    safe = np.clip(safe, 0.0, 1.0)
    return (safe * 255.0).astype(np.uint8)

def apply_smoothing(series: pd.Series | np.ndarray,
                    window_size: int = 15,
                    polyorder: int = 3,
                    gaussian_size: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Return multiple smoothed variants and a recommended 'primary'."""
    x = pd.Series(series).astype(float)
    # moving average
    mavg = x.rolling(window=max(3, window_size), center=True, min_periods=1).mean().to_numpy()

    # savgol requires odd window <= len
    win = max(3, window_size)
    if win % 2 == 0: win += 1
    if len(x) > win:
        try:
            sv = savgol_filter(x.fillna(method='ffill').fillna(method='bfill').to_numpy(), win, polyorder)
        except Exception:
            sv = mavg
    else:
        sv = mavg

    # uniform filter as gaussian-ish
    gs = uniform_filter1d(x.fillna(method='ffill').fillna(method='bfill').to_numpy(),
                          size=max(1, gaussian_size or int(win*0.6)))

    return {'mavg': mavg, 'savgol': sv, 'gaussian': gs, 'primary': sv}

def rects_overlap(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax+aw < bx or bx+bw < ax or ay+ah < by or by+bh < ay)

# ============================ Core contour selection ============================

def select_best_contour_core(contours, last_center=None, aoi=None, cfg_img=IMAGE_PROCESSING_CONFIG) -> Tuple[Optional[np.ndarray], Optional[Dict], Dict]:
    """CORE contour selection: elongated + AOI tracking + simple scoring."""
    min_area = float(cfg_img.get('min_contour_area', 100))
    aoi_boost = 2.0  # Simple fixed boost for AOI
    
    best, best_feat, best_score = None, None, 0.0
    total = 0

    for c in contours or []:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        total += 1
        
        # Basic features
        feat = compute_contour_features(c)
        
        # CORE SCORING: area × elongation (aspect ratio or ellipse)
        elongation = max(feat['aspect_ratio'], feat['ellipse_elongation'])
        base_score = area * elongation
        
        # AOI boost: simple 2x boost if center is inside AOI
        if aoi is not None:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            ax, ay, aw, ah = aoi
            if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                base_score *= aoi_boost
        
        # Distance penalty if we have last position
        final_score = base_score
        if last_center is not None:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            distance = np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)
            # Penalty: max 50% reduction if very far (100+ pixels)
            distance_factor = max(0.5, 1.0 - distance / 200.0)
            final_score *= distance_factor
        
        if final_score > best_score:
            best, best_feat, best_score = c, feat, final_score

    stats = {'total_contours': total, 'best_score': best_score}
    return best, (best_feat or {}), stats

# ============================ Momentum vs Blur (shared) ============================

def directional_momentum_merge(frame, search_radius=3, momentum_threshold=0.2,
                               momentum_decay=0.8, momentum_boost=1.5):
    """
    Enhanced directional momentum merge that ignores signal strength considerations.
    Focuses purely on structural directional enhancement for better sonar processing.
    
    Key improvements:
    - Signal strength agnostic processing
    - Multiple scale enhancement  
    - Adaptive kernel sizing
    - Exponential boost curves to avoid saturation
    - No thresholding based on energy levels
    """
    
    # Convert to float32 for better precision during processing
    result = frame.astype(np.float32)
    
    # Multi-scale kernel generation based on search_radius
    kernel_scales = [1, 2, 3] if search_radius > 2 else [1, 2]
    
    all_responses = []
    
    for scale in kernel_scales:
        kernel_size = max(3, scale * 2 + 1)
        center = kernel_size // 2
        
        # Create stronger directional kernels
        h_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        v_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        d1_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        d2_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # Enhanced horizontal kernel with stronger center weighting
        for i in range(kernel_size):
            weight = 1.0 / (1.0 + abs(i - center) * 0.5)
            h_kernel[center, i] = weight
        h_kernel = h_kernel / np.sum(h_kernel)  # Normalize
        
        # Enhanced vertical kernel
        for i in range(kernel_size):
            weight = 1.0 / (1.0 + abs(i - center) * 0.5)
            v_kernel[i, center] = weight
        v_kernel = v_kernel / np.sum(v_kernel)  # Normalize
        
        # Enhanced diagonal kernels
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == j:  # Main diagonal
                    weight = 1.0 / (1.0 + abs(i - center) * 0.5)
                    d1_kernel[i, j] = weight
                if i + j == kernel_size - 1:  # Anti-diagonal
                    weight = 1.0 / (1.0 + abs(i - center) * 0.5)
                    d2_kernel[i, j] = weight
        
        d1_kernel = d1_kernel / np.sum(d1_kernel) if np.sum(d1_kernel) > 0 else d1_kernel
        d2_kernel = d2_kernel / np.sum(d2_kernel) if np.sum(d2_kernel) > 0 else d2_kernel
        
        # Apply directional filters
        h_response = cv2.filter2D(result, -1, h_kernel)
        v_response = cv2.filter2D(result, -1, v_kernel)
        d1_response = cv2.filter2D(result, -1, d1_kernel)
        d2_response = cv2.filter2D(result, -1, d2_kernel)
        
        # Take maximum response for this scale
        scale_response = np.maximum.reduce([h_response, v_response, d1_response, d2_response])
        all_responses.append(scale_response)
    
    # Combine multi-scale responses using weighted average
    if len(all_responses) == 1:
        enhanced = all_responses[0]
    else:
        # Weight larger scales more heavily for better long-range detection
        weights = np.array([1.0, 1.5, 2.0][:len(all_responses)])
        weights = weights / np.sum(weights)
        enhanced = np.zeros_like(all_responses[0])
        for i, response in enumerate(all_responses):
            enhanced += weights[i] * response
    
    # Apply exponential boost curve to avoid saturation
    # This replaces the old linear boost with signal strength considerations
    normalized_enhanced = enhanced / 255.0
    
    # Exponential boost function: more aggressive for weaker signals, 
    # but doesn't rely on signal strength thresholds
    exp_boost = momentum_boost * (1.0 - np.exp(-2.0 * normalized_enhanced))
    
    # Combine original and enhanced with exponential weighting
    final_result = result + exp_boost * enhanced
    
    # Soft clipping to allow stronger enhancement while avoiding hard saturation
    max_val = 255.0 * (1.0 + momentum_boost * 0.5)  # Allow temporary overflow
    soft_clipped = max_val * np.tanh(final_result / max_val)
    
    # Final normalization back to [0, 255]
    final_normalized = 255.0 * (soft_clipped / np.max(soft_clipped)) if np.max(soft_clipped) > 0 else soft_clipped
    
    return np.clip(final_normalized, 0, 255).astype(np.uint8)

def preprocess_edges(frame_u8: np.ndarray, cfg=IMAGE_PROCESSING_CONFIG) -> Tuple[np.ndarray, np.ndarray]:
    proc = directional_momentum_merge(
        frame_u8,
        search_radius=cfg.get('momentum_search_radius', 3),
        momentum_threshold=cfg.get('momentum_threshold', 0.2),
        momentum_decay=cfg.get('momentum_decay', 0.8),
        momentum_boost=cfg.get('momentum_boost', 1.5),
    )
    edges = cv2.Canny(proc, cfg.get('canny_low_threshold', 50), cfg.get('canny_high_threshold', 150))
    mks = int(cfg.get('morph_close_kernel', 0))
    dil = int(cfg.get('edge_dilation_iterations', 0))
    out = edges
    if mks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mks, mks))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if dil > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        out = cv2.dilate(out, kernel2, iterations=dil)
    return edges, out

# ============================ Contour features & scoring ============================

def compute_contour_features(contour) -> Dict[str, float]:
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    ar = (max(w, h) / max(1, min(w, h))) if min(w, h) > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    rect_area = float(w * h)
    extent = (area / rect_area) if rect_area > 0 else 0.0

    # ellipse elongation
    if len(contour) >= 5:
        try:
            _, (minor, major), _ = cv2.fitEllipse(contour)
            ell = (major / minor) if minor > 0 else ar
        except Exception:
            ell = ar
    else:
        ell = ar

    # straightness via line fit
    straight = 1.0
    if len(contour) >= 10:
        try:
            pts = contour.reshape(-1, 2).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            a, b, c = float(vy), float(-vx), float(vx * y0 - vy * x0)
            denom = np.sqrt(a*a + b*b) + 1e-12
            dists = np.abs(a*pts[:, 0] + b*pts[:, 1] + c) / denom
            avgd = float(np.mean(dists))
            maxd = max(w, h) * 0.1
            straight = max(0.1, 1.0 - (avgd / max(maxd, 1.0)))
        except Exception:
            straight = 0.5

    return {
        'area': area,
        'aspect_ratio': ar,
        'solidity': solidity,
        'extent': extent,
        'ellipse_elongation': ell,
        'straightness': straight,
        'rect': (x, y, w, h),
        'centroid_x': float(x + w/2),  # Center of bounding rect
        'centroid_y': float(y + h/2),
    }

# ============================ Distance and angle extraction ============================

def _distance_angle_from_contour(contour, image_width: int, image_height: int) -> Tuple[Optional[float], Optional[float]]:
    """Calculate net distance straight ahead of robot (center beam intersection)."""
    if contour is None or len(contour) < 5:
        return None, None
    try:
        (cx, cy), (minor_axis, major_axis), angle = cv2.fitEllipse(contour)

        # Calculate intersection of net's major axis with center beam (x = image_width/2)
        ang_r = np.radians(angle + 90.0)  # Major axis direction
        cos_ang = np.cos(ang_r)
        sin_ang = np.sin(ang_r)

        # Find intersection with vertical center line
        center_x = image_width / 2

        if abs(cos_ang) > 1e-6:  # Avoid division by zero
            # Solve: cx + t * cos_ang = center_x
            t = (center_x - cx) / cos_ang
            intersect_y = cy + t * sin_ang

            # Use the intersection point directly (blue dot position) - no bounds checking
            distance = intersect_y
        else:
            # Line is nearly vertical, use center y
            distance = cy

        red_line_angle = (float(angle) + 90.0) % 360.0
        return float(distance), red_line_angle
    except Exception:
        return None, None

def get_red_line_distance_and_angle(frame_u8: np.ndarray, prev_aoi: Optional[Tuple[int,int,int,int]] = None,
                                   ellipse_aoi: Optional[Tuple[float,float,float,float,float]] = None):
    """Return (distance_pixels, angle_deg) for the dominant elongated contour, or (None, None).
    
    Uses SAME contour selection logic as SonarDataProcessor to ensure consistency.
    Now supports elliptical AOI for more restrictive searching.
    """
    _, edges_proc = preprocess_edges(frame_u8, IMAGE_PROCESSING_CONFIG)
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Apply same AOI logic as unified processor
    aoi = None
    if prev_aoi is not None:
        ax, ay, aw, ah = prev_aoi
        exp = int(TRACKING_CONFIG.get('aoi_expansion_pixels', 10))
        H, W = frame_u8.shape[:2]
        aoi = (max(0, ax-exp), max(0, ay-exp),
               min(W - max(0, ax-exp), aw + 2*exp),
               min(H - max(0, ay-exp), ah + 2*exp))
    
    # Use CORE contour selection
    best, _, _ = select_best_contour_core(contours, None, None, IMAGE_PROCESSING_CONFIG)
    H, W = frame_u8.shape[:2]
    return _distance_angle_from_contour(best, W, H)

# ============================ Public: per-frame processing (video overlay) ============================

def process_frame_for_video(frame_u8: np.ndarray, prev_aoi: Optional[Tuple[int,int,int,int]] = None,
                           ellipse_aoi: Optional[Tuple[float,float,float,float,float]] = None):
    # edge pipeline
    edges_raw, edges_proc = preprocess_edges(frame_u8, IMAGE_PROCESSING_CONFIG)
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # AOI expansion
    aoi = None
    if prev_aoi is not None:
        ax, ay, aw, ah = prev_aoi
        exp = int(TRACKING_CONFIG.get('aoi_expansion_pixels', 10))
        H, W = frame_u8.shape[:2]
        aoi = (max(0, ax-exp), max(0, ay-exp),
               min(W - max(0, ax-exp), aw + 2*exp),
               min(H - max(0, ay-exp), ah + 2*exp))

    # choose contour with CORE selection
    best, feat, stats = select_best_contour_core(contours, None, None, IMAGE_PROCESSING_CONFIG)

    # draw
    out = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    if VIDEO_CONFIG.get('show_all_contours', True) and contours:
        cv2.drawContours(out, contours, -1, (255, 200, 100), 1)

    # draw AOI box (expanded rectangular)
    if aoi is not None:
        ex, ey, ew, eh = aoi
        cv2.rectangle(out, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)
        cv2.putText(out, 'AOI', (ex + 5, ey + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,255), 1)
    
    # draw elliptical AOI if provided
    if ellipse_aoi is not None:
        # Draw with expansion factor to show the actual AOI area (same as magenta ellipse but green)
        expansion_factor = 1.3
        (cx, cy), (minor_axis, major_axis), angle = ellipse_aoi
        expanded_ellipse = ((cx, cy), (minor_axis * expansion_factor, major_axis * expansion_factor), angle)
        cv2.ellipse(out, expanded_ellipse, (0, 255, 0), 2)
        cv2.putText(out, 'E-AOI', (int(cx - major_axis/4), int(cy - minor_axis/2 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,0), 1)

    next_aoi, status = prev_aoi, "LOST"
    if best is not None:
        cv2.drawContours(out, [best], -1, (0, 255, 0), 2)
        x, y, w, h = feat['rect']
        if VIDEO_CONFIG.get('show_bounding_box', True):
            cv2.rectangle(out, (x,y), (x+w, y+h), (0,0,255), 1)
        next_aoi = (x, y, w, h)
        if VIDEO_CONFIG.get('show_ellipse', True) and len(best) >= 5:
            try:
                # Fit ellipse on the detected contour
                ellipse = cv2.fitEllipse(best)
                (cx, cy), (minor, major), ang = ellipse
                
                # Draw the ellipse
                cv2.ellipse(out, ellipse, (255, 0, 255), 1)
                
                # 90°-rotated major-axis line (red)
                ang_r = np.radians(ang + 90.0)
                half = major * 0.5
                p1 = (int(cx + half*np.cos(ang_r)), int(cy + half*np.sin(ang_r)))
                p2 = (int(cx - half*np.cos(ang_r)), int(cy - half*np.sin(ang_r)))
                cv2.line(out, p1, p2, (0,0,255), 2)
                
                # Calculate and draw intersection point with center beam
                H_img, W_img = frame_u8.shape[:2]
                center_x = W_img / 2
                cos_ang = np.cos(ang_r)
                sin_ang = np.sin(ang_r)
                
                if abs(cos_ang) > 1e-6:  # Avoid division by zero
                    t = (center_x - cx) / cos_ang
                    intersect_y = cy + t * sin_ang
                    intersect_x = center_x
                else:  # Line is nearly vertical
                    intersect_x = cx
                    intersect_y = cy
                
                # Draw intersection point (blue circle)
                cv2.circle(out, (int(intersect_x), int(intersect_y)), 2, (255, 0, 0), -1)  # Filled blue circle
                cv2.circle(out, (int(intersect_x), int(intersect_y)), 5, (255, 0, 0), 2)  # Blue ring
                
            except Exception:
                pass
        # text & status
        cv2.putText(out, f'Area: {feat.get("area",0):.0f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (255,255,255), 2)
        cv2.putText(out, f'Score: {stats.get("best_score",0):.0f}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (255,255,255), 2)
        status = "TRACKED" if (aoi and rects_overlap(feat['rect'], aoi)) else "NEW"
    else:
        # dashed "searching" box if we had a previous AOI
        if prev_aoi is not None:
            ax, ay, aw, ah = prev_aoi
            dash, gap = 10, 5
            for X in range(ax, ax+aw, dash+gap):
                cv2.line(out, (X, ay), (min(X+dash, ax+aw), ay), (0, 100, 255), 2)
                cv2.line(out, (X, ay+ah), (min(X+dash, ax+aw), ay+ah), (0, 100, 255), 2)
            for Y in range(ay, ay+ah, dash+gap):
                cv2.line(out, (ax, Y), (ax, min(Y+dash, ay+ah)), (0, 100, 255), 2)
                cv2.line(out, (ax+aw, Y), (ax+aw, min(Y+dash, ay+ah)), (0, 100, 255), 2)
            cv2.putText(out, 'SEARCHING', (ax+5, ay+ah-10),
                        cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,100,255), 2)

    status_colors = {"TRACKED": (0,255,0), "NEW": (0,165,255), "LOST": (0,100,255)}
    cv2.putText(out, status, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], status_colors[status], 2)

    info_y = out.shape[0] - 40
    s = VIDEO_CONFIG['text_scale'] * 0.8
    cv2.putText(out, f'Total: {stats.get("total_contours",0)}', (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, s, (255,255,255), 1)
    cv2.putText(out, f'In AOI: {stats.get("aoi_contours",0)}', (10, info_y+15),
                cv2.FONT_HERSHEY_SIMPLEX, s, (255,255,255), 1)

    return out, next_aoi

# ============================ UNIFIED ANALYSIS ENGINE ============================

class DistanceAnalysisEngine:
    """Unified engine for distance analysis across different data sources."""
    
    def __init__(self, processor: SonarDataProcessor = None):
        self.processor = processor or SonarDataProcessor()
        
    def analyze_npz_sequence(self, npz_file_index: int = 0, frame_start: int = 0,
                           frame_count: Optional[int] = None, frame_step: int = 1,
                           npz_dir: str = None) -> pd.DataFrame:
        """Analyze distance over time from NPZ file."""
        print("=== DISTANCE ANALYSIS FROM NPZ ===")
        
        files = get_available_npz_files(npz_dir)
        if npz_file_index >= len(files):
            raise IndexError(f"NPZ file index {npz_file_index} not available")
            
        npz_file = files[npz_file_index]
        print(f"Analyzing: {npz_file}")
        
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
            
            if (i + 1) % 50 == 0:
                success_rate = sum(1 for r in results if r['detection_success']) / len(results) * 100
                print(f"  Processed {i+1}/{actual} frames (Success rate: {success_rate:.1f}%)")
        
        df = pd.DataFrame(results)
        self._print_analysis_summary(df)
        return df
    
    def analyze_sonar_csv(self, target_bag: str, exports_folder: Path,
                         frame_start: int = 0, frame_count: Optional[int] = None,
                         frame_step: int = 1, **sonar_params) -> pd.DataFrame:
        """Analyze distance from sonar CSV data with full processing pipeline."""
        print("=== DISTANCE ANALYSIS FROM SONAR CSV ===")
        print(f"Target Bag: {target_bag}")
        
        # Set default sonar parameters
        params = {
            'FOV_DEG': 120.0, 'RANGE_MIN_M': 0.5, 'RANGE_MAX_M': 10.0,
            'DISPLAY_RANGE_MAX_M': 10.0, 'FLIP_BEAMS': True, 'FLIP_RANGE': False,
            'USE_ENHANCED': True, 'ENH_SCALE': "db", 'ENH_TVG': "amplitude",
            'ENH_ALPHA_DB_PER_M': 0.0, 'ENH_R0': 1e-2, 'ENH_P_LOW': 1.0,
            'ENH_P_HIGH': 99.5, 'ENH_GAMMA': 0.9, 'ENH_ZERO_AWARE': True,
            'ENH_EPS_LOG': 1e-6, 'CONE_W': 900, 'CONE_H': 700
        }
        params.update(sonar_params)
        
        # Load sonar data
        from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
        sonar_csv_file = (Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('by_bag','by_bag') / 
                         f"sensor_sonoptix_echo_image__{target_bag}_video.csv")
        
        if not sonar_csv_file.exists():
            raise FileNotFoundError(f"Sonar CSV not found: {sonar_csv_file}")
            
        df = load_df(sonar_csv_file)
        if "ts_utc" not in df.columns:
            if "t" not in df.columns:
                raise ValueError("Missing timestamp column")
            df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
        
        T = len(df)
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
                
            try:
                # Process sonar data
                cone_frame, extent = self._process_sonar_frame(df, idx, params)
                result = self.processor.analyze_frame(cone_frame, extent)
                result.frame_idx = idx
                result.timestamp = df.loc[idx, "ts_utc"]
                
                results.append(result.to_dict())
                
                if (i + 1) % 50 == 0:
                    success_rate = sum(1 for r in results if r['detection_success']) / len(results) * 100
                    print(f"  Processed {i+1}/{actual} frames (Success rate: {success_rate:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue
        
        df_result = pd.DataFrame(results)
        self._print_analysis_summary(df_result)
        return df_result
    
    def _process_sonar_frame(self, df: pd.DataFrame, idx: int, params: Dict) -> Tuple[np.ndarray, Tuple]:
        """Process single sonar frame using provided parameters."""
        M0 = get_sonoptix_frame(df, idx)
        if M0 is None:
            raise ValueError(f"Could not get sonar frame at index {idx}")
            
        M = apply_flips(M0, flip_range=params['FLIP_RANGE'], flip_beams=params['FLIP_BEAMS'])
        
        if params['USE_ENHANCED']:
            Z = enhance_intensity(
                M, params['RANGE_MIN_M'], params['RANGE_MAX_M'], 
                scale=params['ENH_SCALE'], tvg=params['ENH_TVG'],
                alpha_db_per_m=params['ENH_ALPHA_DB_PER_M'], r0=params['ENH_R0'],
                p_low=params['ENH_P_LOW'], p_high=params['ENH_P_HIGH'], 
                gamma=params['ENH_GAMMA'], zero_aware=params['ENH_ZERO_AWARE'], 
                eps_log=params['ENH_EPS_LOG']
            )
        else:
            Z = M
            
        # Create cone visualization
        cone, extent = cone_raster_like_display_cell(
            Z, params['FOV_DEG'], params['RANGE_MIN_M'], params['RANGE_MAX_M'], 
            params['DISPLAY_RANGE_MAX_M'], params['CONE_W'], params['CONE_H']
        )
        cone = np.flipud(cone)  # Match video generation
        
        # Convert to uint8
        cone_normalized = np.ma.masked_invalid(cone)
        cone_u8 = (cone_normalized * 255).astype(np.uint8)
        
        return cone_u8, extent
    
    def _print_analysis_summary(self, df: pd.DataFrame):
        """Print analysis summary statistics."""
        total = len(df)
        successful = df['detection_success'].sum()
        success_rate = successful / total * 100 if total > 0 else 0
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Total frames processed: {total}")
        print(f"Successful detections: {successful} ({success_rate:.1f}%)")
        
        if successful > 0:
            # Use meters if available, otherwise pixels
            dist_col = 'distance_meters' if 'distance_meters' in df.columns and df['distance_meters'].notna().any() else 'distance_pixels'
            valid_distances = df.loc[df['detection_success'], dist_col].dropna()
            
            if len(valid_distances) > 0:
                unit = "meters" if dist_col == 'distance_meters' else "pixels"
                print(f"Distance statistics ({unit}):")
                print(f"  - Mean: {valid_distances.mean():.3f}")
                print(f"  - Std:  {valid_distances.std():.3f}")
                print(f"  - Min:  {valid_distances.min():.3f}")
                print(f"  - Max:  {valid_distances.max():.3f}")
                print(f"  - Range: {(valid_distances.max()-valid_distances.min()):.3f}")

# ============================ UNIFIED VISUALIZATION & COMPARISON ============================

class VisualizationEngine:
    """Unified engine for plotting and visualization."""
    
    @staticmethod
    def plot_distance_analysis(distance_results: pd.DataFrame, 
                             title: str = "Distance Analysis Over Time",
                             use_plotly: bool = True) -> Optional[object]:
        """Unified distance plotting with optional interactive visualization."""
        valid = distance_results[distance_results['detection_success']].copy()
        if len(valid) == 0:
            print("No valid data to plot")
            return None
            
        # Determine which distance column to use
        if 'distance_meters' in valid.columns and valid['distance_meters'].notna().any():
            dist_col, unit = 'distance_meters', 'meters'
        else:
            dist_col, unit = 'distance_pixels', 'pixels'
            
        distances = valid[dist_col].dropna()
        if len(distances) == 0:
            print("No valid distance data to plot")
            return None
            
        # Try interactive plotting
        if use_plotly:
            try:
                return VisualizationEngine._plot_interactive_distance(valid, dist_col, unit, title)
            except ImportError:
                print("Error: Plotly not available. Please install plotly: pip install plotly")
                return None
        else:
            print("Error: Only Plotly visualization is supported. Set use_plotly=True")
            return None
    
    @staticmethod
    def _plot_interactive_distance(valid: pd.DataFrame, dist_col: str, unit: str, title: str):
        """Create interactive plotly visualization."""
        
        
        distances = valid[dist_col]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Distance vs Frame Index',
                f'Distance vs Time',
                f'Distance Distribution',
                f'Distance Trends'
            )
        )
        
        # Frame index plot
        fig.add_trace(
            go.Scatter(x=valid['frame_index'], y=distances, mode='lines+markers',
                      name='Distance', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Time plot (use frame index as proxy if no time column)
        x_time = valid.get('timestamp', valid['frame_index'])
        fig.add_trace(
            go.Scatter(x=x_time, y=distances, mode='lines+markers',
                      name='Distance over Time', line=dict(color='green')),
            row=1, col=2
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=distances, nbinsx=30, name='Distribution',
                        marker_color='lightcoral', opacity=0.7),
            row=2, col=1
        )
        
        # Add mean/median lines
        fig.add_vline(x=distances.mean(), line=dict(color='red', dash='dash'),
                     annotation_text=f'Mean: {distances.mean():.2f}{unit}', row=2, col=1)
        
        # Trends with smoothing
        window_size = max(5, len(distances) // 20)
        smoothed = distances.rolling(window=window_size, center=True).mean()
        
        fig.add_trace(
            go.Scatter(x=valid['frame_index'], y=distances, mode='lines',
                      name='Raw', line=dict(color='lightcoral', width=1), opacity=0.5),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=valid['frame_index'], y=smoothed, mode='lines',
                      name=f'Smoothed (n={window_size})', line=dict(color='darkred', width=3)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(title_text=title, height=800, showlegend=True)
        fig.update_xaxes(title_text='Frame Index', row=1, col=1)
        fig.update_xaxes(title_text='Time', row=1, col=2)
        fig.update_xaxes(title_text=f'Distance ({unit})', row=2, col=1)
        fig.update_xaxes(title_text='Frame Index', row=2, col=2)
        fig.update_yaxes(title_text=f'Distance ({unit})', row=1, col=1)
        fig.update_yaxes(title_text=f'Distance ({unit})', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_yaxes(title_text=f'Distance ({unit})', row=2, col=2)
        
        return fig

class ComparisonEngine:
    """Unified engine for sonar vs DVL comparisons."""
    
    @staticmethod
    def compare_sonar_vs_dvl(distance_results: pd.DataFrame, raw_data: Dict[str, pd.DataFrame],
                           sonar_coverage_m: float = 5.0, sonar_image_size: int = 700,
                           window_size: int = 15, use_plotly: bool = True) -> Tuple[object, Dict]:
        """Unified comparison between sonar and DVL data."""
        
        # Validate inputs
        if raw_data is None or 'navigation' not in raw_data or raw_data['navigation'] is None:
            return None, {'error': 'no_navigation_data'}
        if distance_results is None:
            return None, {'error': 'no_distance_results'}
        
        # Prepare navigation data
        nav = raw_data['navigation'].copy()
        nav['timestamp'] = pd.to_datetime(nav['timestamp'], errors='coerce')
        nav = nav.dropna(subset=['timestamp'])
        nav['relative_time'] = (nav['timestamp'] - nav['timestamp'].min()).dt.total_seconds()
        
        # Prepare sonar data
        sonar = distance_results.copy()
        
        # Ensure distance_meters column exists
        if 'distance_meters' not in sonar.columns or sonar['distance_meters'].isna().all():
            if 'distance_pixels' in sonar.columns:
                # Convert pixels to meters
                ppm = float(sonar_image_size) / float(sonar_coverage_m)
                sonar['distance_meters'] = sonar['distance_pixels'] / ppm
            else:
                return None, {'error': 'no_distance_data'}
        
        # Create time alignment for sonar data
        dvl_duration = float(max(1.0, nav['relative_time'].max() - nav['relative_time'].min()))
        if 'frame_index' not in sonar:
            sonar['frame_index'] = np.arange(len(sonar))
        N = max(1, len(sonar) - 1)
        sonar['synthetic_time'] = (sonar['frame_index'] / float(N)) * dvl_duration
        
        # Apply smoothing to sonar data
        sonar_smooth = apply_smoothing(sonar['distance_meters'], window_size)
        for key, values in sonar_smooth.items():
            sonar[f'distance_meters_{key}'] = values
        
        # Create visualization - only Plotly supported
        if use_plotly:
            try:
                fig = ComparisonEngine._create_interactive_comparison(sonar, nav)
            except ImportError:
                print("Error: Plotly not available. Please install plotly: pip install plotly")
                return None, {'error': 'plotly_not_available'}
        else:
            print("Error: Only Plotly visualization is supported. Set use_plotly=True")
            return None, {'error': 'only_plotly_supported'}
        
        # Calculate statistics
        sonar_mean = float(np.nanmean(sonar['distance_meters']))
        dvl_mean = float(nav['NetDistance'].mean())
        
        stats = {
            'sonar_mean_m': sonar_mean,
            'dvl_mean_m': dvl_mean,
            'scale_ratio': sonar_mean / dvl_mean if dvl_mean else np.nan,
            'sonar_duration_s': float(sonar['synthetic_time'].max()),
            'dvl_duration_s': dvl_duration,
            'sonar_frames': len(sonar),
            'dvl_records': len(nav)
        }
        
        # Print summary
        print("\n📊 SONAR vs DVL COMPARISON STATISTICS:")
        print("="*50)
        print(f"Sonar mean distance: {sonar_mean:.3f} m")
        print(f"DVL mean distance:   {dvl_mean:.3f} m")
        print(f"Scale ratio (Sonar/DVL): {stats['scale_ratio']:.3f}x")
        print(f"Sonar duration: {stats['sonar_duration_s']:.1f}s ({stats['sonar_frames']} frames)")
        print(f"DVL duration:   {stats['dvl_duration_s']:.1f}s ({stats['dvl_records']} records)")
        
        return fig, stats
    
    @staticmethod
    def _create_interactive_comparison(sonar: pd.DataFrame, nav: pd.DataFrame):
        """Create interactive plotly comparison."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Check if pitch data is available
        has_pitch = 'NetPitch' in nav.columns and nav['NetPitch'].notna().any()
        
        if has_pitch:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Distance Comparison", "Pitch Comparison")
            )
        else:
            fig = make_subplots()
        
        # Distance traces
        fig.add_trace(
            go.Scatter(x=sonar['synthetic_time'], y=sonar['distance_meters'],
                      mode='lines', name='Sonar Raw', line=dict(color='rgba(255,0,0,0.3)')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=sonar['synthetic_time'], y=sonar['distance_meters_primary'],
                      mode='lines', name='Sonar Smoothed', line=dict(color='red', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=nav['relative_time'], y=nav['NetDistance'],
                      mode='lines', name='DVL Distance', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Pitch traces if available
        if has_pitch:
            if 'angle_degrees' in sonar.columns:
                fig.add_trace(
                    go.Scatter(x=sonar['synthetic_time'], y=sonar['angle_degrees'],
                              mode='lines', name='Sonar Angle', line=dict(color='orange', width=3)),
                    row=2, col=1
                )
            fig.add_trace(
                go.Scatter(x=nav['relative_time'], y=np.degrees(nav['NetPitch']),
                          mode='lines', name='DVL Pitch', line=dict(color='green', width=3)),
                row=2, col=1
            )
        
        # Update layout
        title = "Interactive Distance & Pitch Comparison" if has_pitch else "Interactive Distance Comparison"
        fig.update_layout(title=f"🔄 {title}: Sonar vs DVL", hovermode='x unified',
                         height=800 if has_pitch else 600)
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", showgrid=True)
        fig.update_yaxes(title_text="Distance (meters)", showgrid=True)
        if has_pitch:
            fig.update_yaxes(title_text="Pitch (degrees)", row=2, col=1, showgrid=True)
        
        return fig

# ============================ UTILITY FUNCTIONS ============================

def get_red_line_distance_and_angle(frame_u8: np.ndarray, prev_aoi: Optional[Tuple[int,int,int,int]] = None):
    """Return (distance_pixels, angle_deg) for the dominant elongated contour, or (None, None).
    
    Uses SAME contour selection logic as process_frame_for_video to ensure consistency.
    """
    _, edges_proc = preprocess_edges(frame_u8, IMAGE_PROCESSING_CONFIG)
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Apply same AOI logic as video processing
    aoi = None
    if prev_aoi is not None:
        ax, ay, aw, ah = prev_aoi
        exp = int(TRACKING_CONFIG.get('aoi_expansion_pixels', 10))
        H, W = frame_u8.shape[:2]
        aoi = (max(0, ax-exp), max(0, ay-exp),
               min(W - max(0, ax-exp), aw + 2*exp),
               min(H - max(0, ay-exp), ah + 2*exp))
    
    # Use CORE contour selection
    best, _, _ = select_best_contour_core(contours, None, aoi, IMAGE_PROCESSING_CONFIG)
    H, W = frame_u8.shape[:2]
    return _distance_angle_from_contour(best, W, H)

# ============================ VIDEO PROCESSING FUNCTIONS ============================

def process_frame_for_video(frame_u8: np.ndarray, prev_aoi: Optional[Tuple[int,int,int,int]] = None):
    # edge pipeline
    edges_raw, edges_proc = preprocess_edges(frame_u8, IMAGE_PROCESSING_CONFIG)
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # AOI expansion
    aoi = None
    if prev_aoi is not None:
        ax, ay, aw, ah = prev_aoi
        exp = int(TRACKING_CONFIG.get('aoi_expansion_pixels', 10))
        H, W = frame_u8.shape[:2]
        aoi = (max(0, ax-exp), max(0, ay-exp),
               min(W - max(0, ax-exp), aw + 2*exp),
               min(H - max(0, ay-exp), ah + 2*exp))

    # choose contour - CORE selection
    best, feat, stats = select_best_contour_core(contours, None, aoi, IMAGE_PROCESSING_CONFIG)

    # draw
    out = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    if VIDEO_CONFIG.get('show_all_contours', True) and contours:
        cv2.drawContours(out, contours, -1, (255, 200, 100), 1)

    # draw AOI box (expanded)
    if aoi is not None:
        ex, ey, ew, eh = aoi
        cv2.rectangle(out, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)
        cv2.putText(out, 'AOI', (ex + 5, ey + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,255), 1)

    next_aoi, status = prev_aoi, "LOST"
    if best is not None:
        cv2.drawContours(out, [best], -1, (0, 255, 0), 2)
        x, y, w, h = feat['rect']
        if VIDEO_CONFIG.get('show_bounding_box', True):
            cv2.rectangle(out, (x,y), (x+w, y+h), (0,0,255), 1)
        next_aoi = (x, y, w, h)
        if VIDEO_CONFIG.get('show_ellipse', True) and len(best) >= 5:
            try:
                # Fit ellipse on the detected contour
                ellipse = cv2.fitEllipse(best)
                (cx, cy), (minor, major), ang = ellipse
                
                # Draw the ellipse
                cv2.ellipse(out, ellipse, (255, 0, 255), 1)
                
                # 90°-rotated major-axis line (red)
                ang_r = np.radians(ang + 90.0)
                half = major * 0.5
                p1 = (int(cx + half*np.cos(ang_r)), int(cy + half*np.sin(ang_r)))
                p2 = (int(cx - half*np.cos(ang_r)), int(cy - half*np.sin(ang_r)))
                cv2.line(out, p1, p2, (0,0,255), 2)
                
                # Calculate and draw intersection point with center beam
                H_img, W_img = frame_u8.shape[:2]
                center_x = W_img / 2
                cos_ang = np.cos(ang_r)
                sin_ang = np.sin(ang_r)
                
                if abs(cos_ang) > 1e-6:  # Avoid division by zero
                    t = (center_x - cx) / cos_ang
                    intersect_y = cy + t * sin_ang
                    intersect_x = center_x
                else:  # Line is nearly vertical
                    intersect_x = cx
                    intersect_y = cy
                
                # Draw intersection point (blue circle)
                cv2.circle(out, (int(intersect_x), int(intersect_y)), 2, (255, 0, 0), -1)  # Filled blue circle
                cv2.circle(out, (int(intersect_x), int(intersect_y)), 5, (255, 0, 0), 2)  # Blue ring
                
            except Exception:
                pass
        # text & status
        cv2.putText(out, f'Area: {feat.get("area",0):.0f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (255,255,255), 2)
        cv2.putText(out, f'Score: {stats.get("best_score",0):.0f}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (255,255,255), 2)
        status = "TRACKED" if (aoi and rects_overlap(feat['rect'], aoi)) else "NEW"
    else:
        # dashed "searching" box if we had a previous AOI
        if prev_aoi is not None:
            ax, ay, aw, ah = prev_aoi
            dash, gap = 10, 5
            for X in range(ax, ax+aw, dash+gap):
                cv2.line(out, (X, ay), (min(X+dash, ax+aw), ay), (0, 100, 255), 2)
                cv2.line(out, (X, ay+ah), (min(X+dash, ax+aw), ay+ah), (0, 100, 255), 2)
            for Y in range(ay, ay+ah, dash+gap):
                cv2.line(out, (ax, Y), (ax, min(Y+dash, ay+ah)), (0, 100, 255), 2)
                cv2.line(out, (ax+aw, Y), (ax+aw, min(Y+dash, ay+ah)), (0, 100, 255), 2)
            cv2.putText(out, 'SEARCHING', (ax+5, ay+ah-10),
                        cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,100,255), 2)

    status_colors = {"TRACKED": (0,255,0), "NEW": (0,165,255), "LOST": (0,100,255)}
    cv2.putText(out, status, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], status_colors[status], 2)

    info_y = out.shape[0] - 40
    s = VIDEO_CONFIG['text_scale'] * 0.8
    cv2.putText(out, f'Total: {stats.get("total_contours",0)}', (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, s, (255,255,255), 1)
    cv2.putText(out, f'In AOI: {stats.get("aoi_contours",0)}', (10, info_y+15),
                cv2.FONT_HERSHEY_SIMPLEX, s, (255,255,255), 1)

    return out, next_aoi

# ============================ Frame export & video ============================

def pick_and_save_frame(npz_file_index: int = 0, frame_position: str | float = 'middle',
                        output_path: str = "sample_frame.png", npz_dir: str | None = None) -> Dict:
    files = get_available_npz_files(npz_dir)
    if not files: raise FileNotFoundError(f"No NPZ files found in {npz_dir}")
    if npz_file_index >= len(files): raise IndexError(f"NPZ index {npz_file_index} out of range 0..{len(files)-1}")

    cones, timestamps, extent, meta = load_cone_run_npz(files[npz_file_index])
    T = len(cones)
    if frame_position == 'start': idx = 0
    elif frame_position == 'end': idx = T-1
    elif frame_position == 'middle': idx = T//2
    else:
        idx = int(frame_position * T) if isinstance(frame_position, float) and frame_position <= 1.0 else int(frame_position)
        idx = max(0, min(idx, T-1))

    u8 = to_uint8_gray(cones[idx])
    op = Path(output_path)
    cv2.imwrite(str(op), u8)
    print(f"Frame saved to: {op}")
    print(f"Source: {files[npz_file_index].name}, Frame {idx}/{T-1}")
    print(f"Timestamp: {timestamps[idx].strftime('%H:%M:%S')}")
    print(f"Shape: {u8.shape}")
    return {
        'saved_path': str(op), 'npz_file': files[npz_file_index].name, 'frame_index': idx,
        'total_frames': T, 'timestamp': timestamps[idx], 'shape': u8.shape, 'extent': extent
    }

def load_saved_frame(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return img

def create_contour_detection_video(npz_file_index=0, frame_start=0, frame_count=100,
                                   frame_step=5, output_path='contour_detection_video.mp4'):
    """Create video with enhanced error handling and distance overlay."""
    print("=== CONTOUR DETECTION VIDEO CREATION ===")
    print(f"Creating video with {frame_count} frames, stepping by {frame_step}...")
    
    files = get_available_npz_files()
    if npz_file_index >= len(files):
        print(f"Error: NPZ file index {npz_file_index} not available")
        return None

    cones, _, _, _ = load_cone_run_npz(files[npz_file_index])
    T = len(cones)
    actual = int(min(frame_count, max(0, (T - frame_start)) // max(1, frame_step)))
    if actual <= 0:
        print("Error: Not enough frames to process")
        return None

    first = to_uint8_gray(cones[frame_start])
    H, W = first.shape
    outp = Path(output_path)
    
    # Ensure output directory exists
    try:
        outp.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Try primary mp4 writer
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_CONFIG['fps'], (W, H))
    if not vw.isOpened():
        # Fallback: try AVI with XVID
        fallback_path = outp.with_suffix('.avi')
        vw = cv2.VideoWriter(str(fallback_path), cv2.VideoWriter_fourcc(*'XVID'), VIDEO_CONFIG['fps'], (W, H))
        if vw.isOpened():
            print(f"Warning: mp4 writer failed, falling back to AVI: {fallback_path}")
            output_path = str(fallback_path)
        else:
            print("Error: Could not open video writer (mp4v and XVID both failed).")
            return None

    print("Processing frames...")
    aoi = None
    processor = SonarDataProcessor()  # Use new unified processor
    tracked = new = lost = 0

    for i in range(actual):
        idx = frame_start + i * frame_step
        frame_u8 = to_uint8_gray(cones[idx])
        vis, next_aoi = process_frame_for_video(frame_u8, aoi)
        
        # Compute detected net distance using unified processor
        try:
            dist_px, ang_deg = get_red_line_distance_and_angle(frame_u8, aoi)
            if dist_px is not None:
                dist_text = f"Distance: {dist_px:.1f}px"
            else:
                dist_text = "Distance: N/A"
        except Exception:
            dist_text = "Distance: N/A"
        
        # Add distance overlay
        try:
            text_x = max(10, W - 320)
            cv2.putText(vis, dist_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        except Exception:
            pass
            
        # Update tracking stats
        if next_aoi is not None:
            if aoi is None:
                new += 1
            elif aoi == next_aoi:
                lost += 1
            else:
                tracked += 1
        else:
            lost += 1
        aoi = next_aoi
        
        # Add frame counter
        cv2.putText(vis, f'Frame: {idx}', (W - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        vw.write(vis)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{actual} frames")

    vw.release()
    
    print(f"\n=== VIDEO CREATION COMPLETE ===")
    print(f"Video saved to: {output_path}")
    print(f"Video specs: {W}x{H}, {VIDEO_CONFIG['fps']} fps, {actual} frames")
    print(f"Tracking stats:")
    total_det = tracked + new
    print(f"  - Tracked frames: {tracked}")
    print(f"  - New detections: {new}")
    print(f"  - Lost/searching frames: {lost}")
    if total_det > 0:
        print(f"  - Detection success rate: {total_det/actual*100:.1f}%")
        print(f"  - Tracking continuity:   {tracked/max(1,total_det)*100:.1f}%")
    
    return output_path

def create_enhanced_contour_detection_video_with_processor(npz_file_index=0, frame_start=0, frame_count=100,
                                                          frame_step=5, output_path='enhanced_video.mp4',
                                                          processor: SonarDataProcessor = None):
    """Create video using the simplified SonarDataProcessor."""
    print("=== ENHANCED VIDEO CREATION (Simplified) ===")
    print(f"Creating video with simplified processor...")
    print(f"Frames: {frame_count}, step: {frame_step}")
    
    if processor is None:
        processor = SonarDataProcessor()
        
    files = get_available_npz_files()
    if npz_file_index >= len(files):
        print(f"Error: NPZ file index {npz_file_index} not available")
        return None

    cones, timestamps, extent, _ = load_cone_run_npz(files[npz_file_index])
    T = len(cones)
    actual = int(min(frame_count, max(0, (T - frame_start)) // max(1, frame_step)))
    if actual <= 0:
        print("Error: Not enough frames to process")
        return None

    first = to_uint8_gray(cones[frame_start])
    H, W = first.shape
    outp = Path(output_path)
    
    # Ensure output directory exists
    try:
        outp.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Try primary mp4 writer
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_CONFIG['fps'], (W, H))
    if not vw.isOpened():
        # Fallback: try AVI with XVID
        fallback_path = outp.with_suffix('.avi')
        vw = cv2.VideoWriter(str(fallback_path), cv2.VideoWriter_fourcc(*'XVID'), VIDEO_CONFIG['fps'], (W, H))
        if vw.isOpened():
            print(f"⚠️ Fallback: Using {fallback_path} (AVI format)")
            outp = fallback_path
        else:
            print("❌ Error: Could not initialize video writer")
            return None

    # Reset processor tracking
    processor.reset_tracking()
    
    # Tracking statistics
    tracked, new, lost = 0, 0, 0
    
    print(f"✅ Processing {actual} frames with simplified processor...")
    
    for i in range(actual):
        idx = frame_start + i * frame_step
        frame_u8 = to_uint8_gray(cones[idx])
        
        # Use unified processor for analysis
        result = processor.analyze_frame(frame_u8, extent)
        
        # Create visualization
        vis = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        
        # Find contours for drawing all contours
        _, edges_proc = processor.preprocess_frame(frame_u8)
        contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all contours
        if VIDEO_CONFIG.get('show_all_contours', True) and contours:
            cv2.drawContours(vis, contours, -1, (255, 200, 100), 1)
        
        # Draw rectangular AOI (yellow)
        if processor.current_aoi is not None:
            ax, ay, aw, ah = processor.current_aoi
            cv2.rectangle(vis, (ax, ay), (ax+aw, ay+ah), (0, 255, 255), 2)
            cv2.putText(vis, 'AOI', (ax + 5, ay + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,255), 1)
        
        # Draw best contour and features
        if result.detection_success and result.best_contour is not None:
            best_contour = result.best_contour
            
            # Draw best contour (green)
            cv2.drawContours(vis, [best_contour], -1, (0, 255, 0), 2)
            
            # Draw bounding box
            if VIDEO_CONFIG.get('show_bounding_box', True) and result.contour_features:
                x, y, w, h = result.contour_features['rect']
                cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 1)
            
            # Draw ellipse and red line
            if VIDEO_CONFIG.get('show_ellipse', True) and len(best_contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(best_contour)
                    (cx, cy), (minor, major), ang = ellipse
                    
                    # Draw the ellipse (magenta)
                    cv2.ellipse(vis, ellipse, (255, 0, 255), 1)
                    
                    # 90°-rotated major-axis line (red)
                    ang_r = np.radians(ang + 90.0)
                    half = major * 0.5
                    p1 = (int(cx + half*np.cos(ang_r)), int(cy + half*np.sin(ang_r)))
                    p2 = (int(cx - half*np.cos(ang_r)), int(cy - half*np.sin(ang_r)))
                    cv2.line(vis, p1, p2, (0,0,255), 2)
                    
                    # Blue dot at intersection with center beam
                    if result.distance_pixels is not None:
                        center_x = W // 2
                        dot_y = int(result.distance_pixels)
                        cv2.circle(vis, (center_x, dot_y), 4, (255, 0, 0), -1)
                        
                        # Distance text
                        if result.distance_meters is not None:
                            dist_text = f"Dist: {result.distance_meters:.2f}m"
                        else:
                            dist_text = f"Dist: {result.distance_pixels:.1f}px"
                        cv2.putText(vis, dist_text, (center_x + 10, dot_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (255, 0, 0), 1)
                except:
                    pass
        
        # Update tracking statistics
        status = result.tracking_status
        if "TRACKED" in status:
            tracked += 1
        elif "NEW" in status:
            new += 1
        else:
            lost += 1
        
        # Add processing info (simplified)
        frame_info = f'Frame: {idx} | {status} | Simplified Processing'
        cv2.putText(vis, frame_info, (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        vw.write(vis)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{actual} frames")

    vw.release()
    
    print(f"\n=== ENHANCED VIDEO CREATION COMPLETE ===")
    print(f"Video saved to: {output_path}")
    print(f"Video specs: {W}x{H}, {VIDEO_CONFIG['fps']} fps, {actual} frames")
    print(f"🟢 SIMPLIFIED TRACKING STATS:")
    total_det = tracked + new
    print(f"  - Total detected frames: {total_det}")
    print(f"  - Lost/searching frames: {lost}")
    if total_det > 0:
        print(f"  - Detection success rate: {total_det/actual*100:.1f}%")
    
    return output_path

def load_saved_frame(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return img

