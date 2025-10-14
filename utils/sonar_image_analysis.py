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
        self.previous_distance_pixels = None
        self.previous_center = None
        self.previous_angle_degrees = None
        # Add ellipse state for temporal smoothing
        self.previous_ellipse = None  # Store previous ellipse parameters
        
    def reset_tracking(self):
        """Reset tracking state."""
        self.last_center = None
        self.current_aoi = None
        self.smoothed_center = None
        self.previous_ellipse = None
        
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
        
        # Update tracking with elliptical AOI FIRST
        if best_contour is not None and features:
            # Update center tracking
            new_center = (features['centroid_x'], features['centroid_y'])
            
            # Import TRACKING_CONFIG for elliptical AOI parameters
            from utils.sonar_config import TRACKING_CONFIG
            
            # Smooth center position
            if TRACKING_CONFIG.get('use_elliptical_aoi', True):
                smoothing_alpha = TRACKING_CONFIG.get('center_smoothing_alpha', 0.3)
                self.smoothed_center = smooth_center_position(self.smoothed_center, new_center, smoothing_alpha)
                
                # Create elliptical AOI with temporal smoothing
                expansion_factor = TRACKING_CONFIG.get('ellipse_expansion_factor', 0.3)
                ellipse_smoothing_alpha = TRACKING_CONFIG.get('ellipse_smoothing_alpha', 0.2)
                max_movement = TRACKING_CONFIG.get('ellipse_max_movement_pixels', 4.0)
                
                aoi_mask, ellipse_center, self.previous_ellipse = create_smooth_elliptical_aoi(
                    best_contour, expansion_factor, (H, W), 
                    self.previous_ellipse, ellipse_smoothing_alpha, max_movement
                )
                
                # Store AOI as mask and ellipse center for visualization
                self.current_aoi = {
                    'mask': aoi_mask,
                    'center': ellipse_center,
                    'smoothed_center': self.smoothed_center,
                    'type': 'elliptical'
                }
            else:
                # Fallback to rectangular AOI
                x, y, w, h = features['rect']
                expansion = 25
                self.current_aoi = {
                    'rect': (
                        max(0, x - expansion),
                        max(0, y - expansion),
                        min(W - max(0, x - expansion), w + 2*expansion),
                        min(H - max(0, y - expansion), h + 2*expansion)
                    ),
                    'type': 'rectangular'
                }
            
            # Update last_center for backward compatibility
            self.last_center = new_center
        
        # Extract distance and angle using smoothed center for stable tracking
        if best_contour is not None and self.smoothed_center is not None:
            # Use smoothed center for stable distance/angle measurements
            distance_pixels, angle_degrees = _distance_angle_from_smoothed_center(
                best_contour, self.smoothed_center, W, H
            )
        else:
            # Fallback to raw contour calculation
            distance_pixels, angle_degrees = _distance_angle_from_contour(best_contour, W, H)
        
        # Apply distance change threshold to prevent sudden jumps
        if distance_pixels is not None and self.previous_distance_pixels is not None:
            max_change = self.img_config.get('max_distance_change_pixels', 20)
            distance_change = abs(distance_pixels - self.previous_distance_pixels)
            
            if distance_change > max_change:
                # Large distance change detected - clamp to maximum allowed change
                direction = 1 if distance_pixels > self.previous_distance_pixels else -1
                distance_pixels = self.previous_distance_pixels + (direction * max_change)
        
        # Update previous distance for next frame
        if distance_pixels is not None:
            self.previous_distance_pixels = distance_pixels
        
        # Convert to meters if extent provided
        distance_meters = None
        if distance_pixels is not None and extent is not None:
            x_min, x_max, y_min, y_max = extent
            height_m = y_max - y_min
            px2m_y = height_m / H
            distance_meters = y_min + distance_pixels * px2m_y
        
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

def create_smooth_elliptical_aoi(contour: np.ndarray, expansion_factor: float, image_shape: Tuple[int, int], 
                                previous_ellipse: Optional[Tuple] = None, smoothing_alpha: float = 0.2,
                                max_movement_pixels: float = 4.0) -> Tuple[np.ndarray, Tuple[float, float], Tuple]:
    """Create an elliptical AOI with temporal smoothing.
    
    Args:
        contour: Input contour
        expansion_factor: Factor to expand the ellipse (e.g., 0.3 = 30% larger)
        image_shape: (height, width) of the image
        previous_ellipse: Previous ellipse parameters ((center), (axes), angle) or None
        smoothing_alpha: Smoothing factor for ellipse parameters (0.0 = no smoothing, 1.0 = use previous only)
        max_movement_pixels: Maximum pixels center can move per frame
        
    Returns:
        Tuple of (ellipse_mask, (center_x, center_y), ellipse_params)
    """
    H, W = image_shape
    
    # Fit ellipse to current contour
    if len(contour) >= 5:  # Need at least 5 points to fit ellipse
        current_ellipse = cv2.fitEllipse(contour)
        (curr_center_x, curr_center_y), (curr_width, curr_height), curr_angle = current_ellipse
    else:
        # Fallback to circular AOI if we can't fit ellipse
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            curr_center_x = int(moments['m10'] / moments['m00'])
            curr_center_y = int(moments['m01'] / moments['m00'])
        else:
            curr_center_x, curr_center_y = W//2, H//2
        
        # Estimate size from contour area
        area = cv2.contourArea(contour)
        radius = max(10, np.sqrt(area) * 2)  # Conservative estimate
        curr_width = curr_height = radius * 2
        curr_angle = 0
    
    # Apply temporal smoothing if we have previous ellipse
    if previous_ellipse is not None:
        (prev_center, prev_axes, prev_angle) = previous_ellipse
        prev_center_x, prev_center_y = prev_center
        prev_width, prev_height = prev_axes
        
        # Smooth center position with movement limit
        center_dx = curr_center_x - prev_center_x
        center_dy = curr_center_y - prev_center_y
        center_distance = np.sqrt(center_dx**2 + center_dy**2)
        
        if center_distance > max_movement_pixels:
            # Limit movement to maximum allowed
            scale = max_movement_pixels / center_distance
            center_dx *= scale
            center_dy *= scale
        
        smoothed_center_x = prev_center_x + smoothing_alpha * center_dx
        smoothed_center_y = prev_center_y + smoothing_alpha * center_dy
        
        # Smooth ellipse parameters
        smoothed_width = prev_width + smoothing_alpha * (curr_width - prev_width)
        smoothed_height = prev_height + smoothing_alpha * (curr_height - prev_height)
        
        # Smooth angle (handle angle wraparound)
        angle_diff = curr_angle - prev_angle
        # Normalize angle difference to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        smoothed_angle = prev_angle + smoothing_alpha * angle_diff
        
        center_x, center_y = smoothed_center_x, smoothed_center_y
        width, height = smoothed_width, smoothed_height
        angle = smoothed_angle
    else:
        # First frame - use current ellipse as-is
        center_x, center_y = curr_center_x, curr_center_y
        width, height = curr_width, curr_height
        angle = curr_angle
    
    # Expand the ellipse
    expanded_width = width * (1 + expansion_factor)
    expanded_height = height * (1 + expansion_factor)
    expanded_ellipse = ((center_x, center_y), (expanded_width, expanded_height), angle)
    
    # Create mask
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, expanded_ellipse, 255, -1)
    
    # Return ellipse parameters for next frame
    ellipse_params = ((center_x, center_y), (width, height), angle)
    
    return mask, (center_x, center_y), ellipse_params

def create_elliptical_aoi(contour: np.ndarray, expansion_factor: float, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Create an elliptical AOI from a contour.
    
    Args:
        contour: Input contour
        expansion_factor: Factor to expand the ellipse (e.g., 0.3 = 30% larger)
        image_shape: (height, width) of the image
        
    Returns:
        Tuple of (ellipse_mask, (center_x, center_y))
    """
    H, W = image_shape
    
    # Fit ellipse to contour
    if len(contour) >= 5:  # Need at least 5 points to fit ellipse
        ellipse = cv2.fitEllipse(contour)
        (center_x, center_y), (width, height), angle = ellipse
        
        # Expand the ellipse
        expanded_width = width * (1 + expansion_factor)
        expanded_height = height * (1 + expansion_factor)
        expanded_ellipse = ((center_x, center_y), (expanded_width, expanded_height), angle)
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, expanded_ellipse, 255, -1)
        
        return mask, (center_x, center_y)
    else:
        # Fallback to circular AOI if we can't fit ellipse
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        else:
            center_x, center_y = W//2, H//2
        
        # Create circular mask with expansion
        radius = int(cv2.contourArea(contour) ** 0.5 * (1 + expansion_factor))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        return mask, (center_x, center_y)

def point_in_aoi(point_x: float, point_y: float, aoi) -> bool:
    """Check if a point is inside the AOI (Area of Interest).
    
    Args:
        point_x, point_y: Point coordinates
        aoi: AOI definition (can be rectangular tuple or elliptical dict)
        
    Returns:
        True if point is inside AOI
    """
    if aoi is None:
        return False
    
    if isinstance(aoi, dict):
        aoi_type = aoi.get('type', 'rectangular')
        
        if aoi_type == 'elliptical' and 'mask' in aoi:
            # Check if point is inside elliptical mask
            mask = aoi['mask']
            if 0 <= point_y < mask.shape[0] and 0 <= point_x < mask.shape[1]:
                return mask[int(point_y), int(point_x)] > 0
            return False
        elif aoi_type == 'rectangular' and 'rect' in aoi:
            # Check if point is inside rectangular AOI
            ax, ay, aw, ah = aoi['rect']
            return ax <= point_x <= ax + aw and ay <= point_y <= ay + ah
    else:
        # Legacy rectangular AOI (tuple format)
        ax, ay, aw, ah = aoi
        return ax <= point_x <= ax + aw and ay <= point_y <= ay + ah
    
    return False

def smooth_center_position(current_center: Tuple[float, float], 
                          new_center: Tuple[float, float], 
                          smoothing_alpha: float) -> Tuple[float, float]:
    """Smooth the center position using exponential moving average.
    
    Args:
        current_center: Current smoothed center position
        new_center: New detected center position
        smoothing_alpha: Smoothing factor (0.0 = no change, 1.0 = instant jump)
        
    Returns:
        New smoothed center position
    """
    if current_center is None:
        return new_center
    
    curr_x, curr_y = current_center
    new_x, new_y = new_center
    
    smoothed_x = curr_x + smoothing_alpha * (new_x - curr_x)
    smoothed_y = curr_y + smoothing_alpha * (new_y - curr_y)
    
    return (smoothed_x, smoothed_y)

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
            if point_in_aoi(cx, cy, aoi):
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

# ============================ FAST adaptive linear merging ============================

# Global kernel cache for optimization
_KERNEL_CACHE = {}
_CACHE_MAX_SIZE = 200  # Limit cache size to prevent memory issues

def _get_cache_key(kernel_type, *params):
    """Generate cache key for kernel caching."""
    return (kernel_type,) + tuple(float(p) for p in params)

def clear_kernel_cache():
    """Clear the kernel cache to free memory."""
    global _KERNEL_CACHE
    _KERNEL_CACHE.clear()

def _cache_kernel(cache_key, kernel):
    """Add kernel to cache with size management."""
    global _KERNEL_CACHE
    if len(_KERNEL_CACHE) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_KERNEL_CACHE))
        del _KERNEL_CACHE[oldest_key]
    _KERNEL_CACHE[cache_key] = kernel

def adaptive_linear_momentum_merge_fast(
    frame: np.ndarray,
    angle_steps: int = 36,
    base_radius: int = 3,
    max_elongation: float = 3.0,
    momentum_boost: float = 0.8,
    linearity_threshold: float = 0.15,
    downscale_factor: int = 2,
    top_k_bins: int = 8,
    min_coverage_percent: float = 0.5,
    gaussian_sigma: float = 1.0
) -> np.ndarray:
    """
    ADVANCED OPTIMIZED version using structure tensors and sophisticated filtering:
    1. Structure tensor-based orientation detection (replaces oriented gradient bank)
    2. Top-K bin selection for processing only most significant angles
    3. Separable Gaussian blur instead of circular kernels  
    4. ROI-based convolution processing
    5. Aggressive bin filtering by coverage and linearity
    6. Quantized angle management throughout pipeline
    
    Args:
        frame: Input grayscale image (0-255)
        angle_steps: Number of angle bins for quantization
        base_radius: Base kernel radius for enhancement
        max_elongation: Maximum elongation factor for elliptical kernels
        momentum_boost: Enhancement strength multiplier
        linearity_threshold: Minimum linearity for processing
        downscale_factor: Factor for downsampling during analysis
        top_k_bins: Maximum number of angle bins to process
        min_coverage_percent: Minimum pixel coverage (%) for bin processing
        gaussian_sigma: Sigma for Gaussian blur enhancement
    
    Returns:
        Enhanced grayscale image (0-255)
    """
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    result = frame.astype(np.float32)
    h, w = result.shape
    
    # ADVANCED OPTIMIZATION 1: Early exit for low contrast images
    frame_std = np.std(result)
    if frame_std < 5.0:
        # Apply separable Gaussian blur for mild enhancement
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.3 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Downsampled dimensions for structure tensor analysis
    h_small = max(h // downscale_factor, 32)
    w_small = max(w // downscale_factor, 32)
    frame_small = cv2.resize(result, (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    # ADVANCED OPTIMIZATION 2: Structure tensor-based orientation detection
    # First compute gradients for structure tensor
    grad_x = cv2.Sobel(frame_small, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame_small, cv2.CV_64F, 0, 1, ksize=3)
    
    # Use existing structure tensor function for superior orientation and linearity detection
    orientations, linearity_map_small = compute_structure_tensor_field_fast(
        grad_x, grad_y, 
        sigma=1.5  # Sigma for Gaussian smoothing in structure tensor
    )
    
    # ADVANCED OPTIMIZATION 3: Quantize orientations to integer angle bins
    # Convert orientations (-π/2 to π/2) to angle bins (0 to angle_steps-1)
    orientations_normalized = (orientations + np.pi/2) / np.pi  # Normalize to [0, 1]
    direction_bin_map_small = np.round(orientations_normalized * (angle_steps - 1)).astype(np.int32)
    direction_bin_map_small = np.clip(direction_bin_map_small, 0, angle_steps - 1)
    
    # Normalize linearity map
    max_linearity = np.max(linearity_map_small)
    if max_linearity > 0:
        linearity_map_small = linearity_map_small / max_linearity
    else:
        # No linearity detected - apply Gaussian blur enhancement
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # ADVANCED OPTIMIZATION 4: Upsample maps to full resolution
    linearity_map = cv2.resize(linearity_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
    direction_bin_map = cv2.resize(direction_bin_map_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    
    # Create binary mask for linear regions
    linear_mask = linearity_map > linearity_threshold
    
    if np.sum(linear_mask) == 0:
        # No linear patterns - apply Gaussian blur enhancement
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # ADVANCED OPTIMIZATION 5: Aggressive bin filtering and Top-K selection
    unique_bins, bin_counts = np.unique(direction_bin_map[linear_mask], return_counts=True)
    
    # Calculate coverage percentages and total linearity per bin
    total_linear_pixels = np.sum(linear_mask)
    bin_coverage_percent = (bin_counts / total_linear_pixels) * 100.0
    
    # Calculate total linearity strength per bin
    bin_linearity_totals = []
    for bin_idx in unique_bins:
        bin_mask = (direction_bin_map == bin_idx) & linear_mask
        total_linearity = np.sum(linearity_map[bin_mask])
        bin_linearity_totals.append(total_linearity)
    bin_linearity_totals = np.array(bin_linearity_totals)
    
    # Filter bins by minimum coverage
    coverage_filter = bin_coverage_percent >= min_coverage_percent
    filtered_bins = unique_bins[coverage_filter]
    filtered_linearity = bin_linearity_totals[coverage_filter]
    
    if len(filtered_bins) == 0:
        # No bins meet coverage criteria - apply Gaussian blur
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # ADVANCED OPTIMIZATION 6: Select top-K bins by linearity strength
    if len(filtered_bins) > top_k_bins:
        top_k_indices = np.argsort(filtered_linearity)[-top_k_bins:]
        significant_bins = filtered_bins[top_k_indices]
    else:
        significant_bins = filtered_bins
    
    # Base enhancement with separable Gaussian blur (faster than circular convolution)
    enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
    
    # ADVANCED OPTIMIZATION 7: ROI-based convolution processing
    for angle_bin in significant_bins:
        # Convert quantized bin back to angle for kernel creation
        angle_degrees = float(angle_bin * 180.0 / angle_steps)
        
        # Create binary mask for this bin
        bin_mask = (direction_bin_map == angle_bin) & linear_mask
        
        if np.sum(bin_mask) == 0:
            continue
            
        # ADVANCED OPTIMIZATION 8: ROI bounding box calculation
        # Find bounding box of the mask to limit convolution area
        rows, cols = np.where(bin_mask)
        if len(rows) == 0:
            continue
            
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        
        # Expand ROI by kernel radius to account for convolution border effects
        kernel_margin = base_radius + 2
        roi_row_start = max(0, row_min - kernel_margin)
        roi_row_end = min(h, row_max + kernel_margin + 1)
        roi_col_start = max(0, col_min - kernel_margin)
        roi_col_end = min(w, col_max + kernel_margin + 1)
        
        # Extract ROI for processing
        roi_frame = result[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_mask = bin_mask[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_linearity = linearity_map[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        
        if roi_frame.size == 0:
            continue
        
        # Calculate average elongation for this bin within ROI
        avg_elongation = 1 + (max_elongation - 1) * np.mean(roi_linearity[roi_mask])
        
        # ADVANCED OPTIMIZATION 9: Cached elliptical kernel with ROI convolution
        cache_key = _get_cache_key('ellipse', base_radius, avg_elongation, angle_degrees)
        if cache_key not in _KERNEL_CACHE:
            _cache_kernel(cache_key, create_elliptical_kernel_fast(base_radius, avg_elongation, angle_degrees))
        ellipse_kernel = _KERNEL_CACHE[cache_key]
        
        # Apply elliptical convolution only to ROI
        roi_enhanced = cv2.filter2D(roi_frame, -1, ellipse_kernel)
        
        # ADVANCED OPTIMIZATION 10: Masked blending within ROI
        blend_weights = roi_linearity * roi_mask.astype(np.float32)
        
        # Update enhanced image only within ROI
        roi_current = enhanced[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_blended = roi_current * (1.0 - blend_weights) + roi_enhanced * blend_weights
        enhanced[roi_row_start:roi_row_end, roi_col_start:roi_col_end] = roi_blended
    
    # Final enhancement combination with adaptive clipping
    final_result = result + momentum_boost * enhanced
    
    # ADVANCED OPTIMIZATION 11: Adaptive soft clipping based on image statistics
    clip_upper = 255.0 * (1.0 + momentum_boost * 0.2)
    final_result = np.clip(final_result, 0.0, clip_upper)
    
    # Smooth saturation using tanh for natural look
    final_result = 255.0 * np.tanh(final_result / 255.0)
    
    return np.clip(final_result, 0.0, 255.0).astype(np.uint8)


# ============================ ENHANCED LINE DIRECTION DETECTION ============================

def compute_structure_tensor_field_fast(grad_x: np.ndarray, grad_y: np.ndarray, 
                                       sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast vectorized structure tensor computation for entire image.
    
    Args:
        grad_x, grad_y: Gradient components
        sigma: Gaussian smoothing parameter
        
    Returns:
        orientation_map: Dominant orientation at each pixel (0-180°)
        coherency_map: Coherency (linearity measure) at each pixel (0-1)
    """
    # Structure tensor components
    Jxx = grad_x * grad_x
    Jyy = grad_y * grad_y  
    Jxy = grad_x * grad_y
    
    # Apply Gaussian smoothing (vectorized)
    kernel_size = max(3, int(4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel_gauss = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = kernel_gauss @ kernel_gauss.T
    
    Jxx_smooth = cv2.filter2D(Jxx, -1, kernel_2d)
    Jyy_smooth = cv2.filter2D(Jyy, -1, kernel_2d)
    Jxy_smooth = cv2.filter2D(Jxy, -1, kernel_2d)
    
    # Vectorized eigenvalue analysis
    trace = Jxx_smooth + Jyy_smooth
    det = Jxx_smooth * Jyy_smooth - Jxy_smooth * Jxy_smooth
    
    # Compute orientation (vectorized)
    orientation_map = np.zeros_like(Jxx_smooth)
    coherency_map = np.zeros_like(Jxx_smooth)
    
    # Mask for valid regions (non-zero gradients)
    valid_mask = np.abs(Jxy_smooth) > 1e-6
    
    # Vectorized orientation computation
    orientation_map[valid_mask] = 0.5 * np.arctan2(2 * Jxy_smooth[valid_mask], 
                                                   Jxx_smooth[valid_mask] - Jyy_smooth[valid_mask])
    orientation_map = (orientation_map * 180 / np.pi + 180) % 180
    
    # Handle near-zero cases
    horizontal_mask = (~valid_mask) & (Jxx_smooth > Jyy_smooth)
    vertical_mask = (~valid_mask) & (Jxx_smooth <= Jyy_smooth)
    orientation_map[horizontal_mask] = 0    # Horizontal
    orientation_map[vertical_mask] = 90     # Vertical
    
    # Vectorized coherency computation (safely)
    valid_coherency_mask = (trace > 1e-6) & (det >= 0)
    coherency_map[valid_coherency_mask] = ((trace[valid_coherency_mask] - 
                                          2 * np.sqrt(det[valid_coherency_mask])) / 
                                         trace[valid_coherency_mask])
    coherency_map = np.clip(coherency_map, 0, 1)
    
    return orientation_map, coherency_map


def create_oriented_gradient_kernel_fast(angle_degrees, size):
    """Fast simplified gradient kernel creation."""
    if size % 2 == 0:
        size += 1
    
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    
    # Simplified gradient using Sobel-like pattern
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Create simple 3-point gradient along the direction
    if size >= 3:
        # Use only center line for speed
        for i in range(size):
            offset = i - center
            if abs(offset) <= 1:
                kernel[center, i] = offset * cos_a
        
        for i in range(size):
            offset = i - center  
            if abs(offset) <= 1:
                kernel[i, center] += offset * sin_a
    
    # Ensure zero sum
    kernel = kernel - np.mean(kernel)
    return kernel

def create_circular_kernel_fast(radius):
    """Fast circular kernel with simplified weights."""
    size = 2 * radius + 1
    center = radius
    y, x = np.ogrid[:size, :size]
    
    # Distance from center
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Simple step function instead of smooth falloff
    kernel = (dist <= radius).astype(np.float32)
    
    # Normalize
    return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

def create_elliptical_kernel_fast(base_radius, elongation_factor, angle_degrees):
    """Fast elliptical kernel with simplified computation."""
    # Simplified ellipse - just stretch circular kernel
    elongated_radius = int(base_radius * elongation_factor)
    size = 2 * elongated_radius + 1
    center = size // 2
    
    # Create basic ellipse
    y, x = np.ogrid[:size, :size]
    y_c, x_c = y - center, x - center
    
    # Rotate coordinates
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    x_rot = x_c * cos_a + y_c * sin_a
    y_rot = -x_c * sin_a + y_c * cos_a
    
    # Ellipse equation (simplified)
    ellipse_dist = (x_rot / elongated_radius) ** 2 + (y_rot / base_radius) ** 2
    kernel = (ellipse_dist <= 1.0).astype(np.float32)
    
    # Normalize
    return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

def preprocess_edges(frame_u8: np.ndarray, cfg=IMAGE_PROCESSING_CONFIG) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess sonar frame for edge detection using binary conversion followed by enhancement.
    
    SIGNAL-STRENGTH INDEPENDENT APPROACH:
    1. Convert to binary frame immediately (removes signal strength dependency)
    2. Apply structural enhancement on binary data
    3. Direct edge detection without additional Canny step
    
    Args:
        frame_u8: Input sonar frame (uint8)
        cfg: Configuration dictionary with processing parameters
        
    Returns:
        Tuple of (raw_edges, processed_edges) - both are binary edge maps
    """
    
    # STEP 1: Convert to binary frame immediately - removes signal strength dependency
    binary_threshold = cfg.get('binary_threshold', 128)  # Default threshold
    binary_frame = (frame_u8 > binary_threshold).astype(np.uint8) * 255
    
    # STEP 2: Apply structural enhancement on binary data using custom adaptive linear merging
    enhanced_binary = adaptive_linear_momentum_merge_fast(
        binary_frame,
        base_radius=cfg.get('adaptive_base_radius', 2),
        max_elongation=cfg.get('adaptive_max_elongation', 4),
        linearity_threshold=cfg.get('adaptive_linearity_threshold', 0.3),
        momentum_boost=cfg.get('momentum_boost', 1.5),
        angle_steps=cfg.get('adaptive_angle_steps', 9)
    )
    
    # STEP 3: Extract edges from enhanced binary frame
    # For binary data, we can use simple edge detection or morphological operations
    
    # Raw edges: simple gradient-based edge detection on binary frame
    kernel_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    raw_edges = cv2.filter2D(binary_frame, cv2.CV_32F, kernel_edge)
    raw_edges = np.clip(raw_edges, 0, 255).astype(np.uint8)
    raw_edges = (raw_edges > 0).astype(np.uint8) * 255
    
    # Enhanced edges: edge detection on enhanced binary frame  
    enhanced_edges = cv2.filter2D(enhanced_binary, cv2.CV_32F, kernel_edge)
    enhanced_edges = np.clip(enhanced_edges, 0, 255).astype(np.uint8)
    enhanced_edges = (enhanced_edges > 0).astype(np.uint8) * 255
    
    # Post-process edges with morphological operations
    mks = int(cfg.get('morph_close_kernel', 0))
    dil = int(cfg.get('edge_dilation_iterations', 0))
    out = enhanced_edges
    
    if mks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mks, mks))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    
    if dil > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        out = cv2.dilate(out, kernel2, iterations=dil)
    
    return raw_edges, out

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
        (cx, cy), (w, h), angle = cv2.fitEllipse(contour)

        # Determine the major axis orientation correctly
        if w >= h:
            major_angle = angle
        else:
            major_angle = angle + 90.0

        # Calculate intersection of net's major axis with center beam (x = image_width/2)
        ang_r = np.radians(major_angle)  # Major axis direction
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

def _distance_angle_from_smoothed_center(contour, smoothed_center: Tuple[float, float], 
                                       image_width: int, image_height: int) -> Tuple[Optional[float], Optional[float]]:
    """Calculate net distance using smoothed center position for stable tracking."""
    if contour is None or len(contour) < 5 or smoothed_center is None:
        return None, None
    
    try:
        # Get ellipse angle from contour, but use smoothed center for position
        (_, _), (w, h), angle = cv2.fitEllipse(contour)
        
        # Determine the major axis orientation correctly
        if w >= h:
            major_angle = angle
        else:
            major_angle = angle + 90.0
        
        # Use smoothed center position
        cx, cy = smoothed_center
        
        # Calculate intersection of net's major axis with center beam (x = image_width/2)
        ang_r = np.radians(major_angle)  # Major axis direction
        cos_ang = np.cos(ang_r)
        sin_ang = np.sin(ang_r)
        
        # Find intersection with vertical center line
        center_x = image_width / 2
        
        if abs(cos_ang) > 1e-6:  # Avoid division by zero
            # Solve: cx + t * cos_ang = center_x
            t = (center_x - cx) / cos_ang
            intersect_y = cy + t * sin_ang
            
            # Use the intersection point directly (blue dot position)
            distance = intersect_y
        else:
            # Line is nearly vertical, use smoothed center y
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
        print("\nSONAR vs DVL COMPARISON STATISTICS:")
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
        fig.update_layout(title=f"{title}: Sonar vs DVL", hovermode='x unified',
                         height=800 if has_pitch else 600)
        
        # Update axes
        fig.update_xaxes(title_text="Time (seconds)", showgrid=True)
        fig.update_yaxes(title_text="Distance (meters)", showgrid=True)
        if has_pitch:
            fig.update_yaxes(title_text="Pitch (degrees)", row=2, col=1, showgrid=True)
        
        return fig

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
            print(f"Fallback: Using {fallback_path} (AVI format)")
            outp = fallback_path
        else:
            print("Error: Could not initialize video writer")
            return None

    # Reset processor tracking
    processor.reset_tracking()
    
    # Tracking statistics
    tracked, new, lost = 0, 0, 0
    
    print(f"Processing {actual} frames with simplified processor...")
    
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
        
        # Draw AOI (yellow)
        if processor.current_aoi is not None:
            if isinstance(processor.current_aoi, dict):
                aoi_type = processor.current_aoi.get('type', 'rectangular')
                
                if aoi_type == 'elliptical':
                    # Draw elliptical AOI mask outline
                    aoi_mask = processor.current_aoi['mask']
                    aoi_contours, _ = cv2.findContours(aoi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if aoi_contours:
                        cv2.drawContours(vis, aoi_contours, -1, (0, 255, 255), 2)
                    
                    # Draw ellipse center (current detection)
                    ellipse_center = processor.current_aoi['center']
                    cv2.circle(vis, (int(ellipse_center[0]), int(ellipse_center[1])), 3, (0, 255, 255), -1)
                    
                    # Draw smoothed center (tracking center) - red dot
                    if 'smoothed_center' in processor.current_aoi and processor.current_aoi['smoothed_center']:
                        smoothed = processor.current_aoi['smoothed_center']
                        cv2.circle(vis, (int(smoothed[0]), int(smoothed[1])), 5, (0, 0, 255), -1)
                        cv2.putText(vis, 'TRACK', (int(smoothed[0]) + 8, int(smoothed[1]) - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale']*0.7, (0,0,255), 1)
                    
                    cv2.putText(vis, 'AOI-E', (int(ellipse_center[0]) + 8, int(ellipse_center[1]) + 8),
                               cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,255), 1)
                
                elif aoi_type == 'rectangular' and 'rect' in processor.current_aoi:
                    # Draw rectangular AOI
                    ax, ay, aw, ah = processor.current_aoi['rect']
                    cv2.rectangle(vis, (ax, ay), (ax+aw, ay+ah), (0, 255, 255), 2)
                    cv2.putText(vis, 'AOI-R', (ax + 5, ay + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,255), 1)
            else:
                # Legacy rectangular AOI (backward compatibility)
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
    print(f"SIMPLIFIED TRACKING STATS:")
    total_det = tracked + new
    print(f"  - Total detected frames: {total_det}")
    print(f"  - Lost/searching frames: {lost}")
    if total_det > 0:
        print(f"  - Detection success rate: {total_det/actual*100:.1f}%")
    
    return output_path
