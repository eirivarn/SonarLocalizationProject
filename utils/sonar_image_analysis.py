# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

from utils.sonar_utils import (
    load_df, get_sonoptix_frame,
    enhance_intensity, apply_flips, cone_raster_like_display_cell
)
from utils.sonar_config import IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG, VIDEO_CONFIG, ConeGridSpec

# ============================ ELLIPSE MANIPULATION UTILITIES ============================

def modify_ellipse_aspect_ratio(ellipse_params, aspect_ratio: float = 1.0):
    """
    Modify the aspect ratio of an ellipse to make it thinner/longer or wider/shorter.
    
    Args:
        ellipse_params: OpenCV ellipse parameters ((cx, cy), (minor_axis, major_axis), angle)
        aspect_ratio: Desired major_axis / minor_axis ratio
                     >1.0 = thinner, longer ellipse 
                     <1.0 = wider, shorter ellipse
                     1.0 = no change
    
    Returns:
        Modified ellipse parameters with adjusted axes
    """
    if ellipse_params is None or aspect_ratio == 1.0:
        return ellipse_params
        
    (cx, cy), (minor_axis, major_axis), angle = ellipse_params
    
    # Calculate the area to preserve (optional - you can comment this out if you want size changes)
    original_area = np.pi * (minor_axis / 2) * (major_axis / 2)
    
    # Apply aspect ratio modification
    # If aspect_ratio > 1.0, we want major_axis / minor_axis = aspect_ratio
    if major_axis >= minor_axis:
        # Normal case: major is already larger
        new_minor = minor_axis / np.sqrt(aspect_ratio)
        new_major = major_axis * np.sqrt(aspect_ratio)
    else:
        # Edge case: minor is larger, so swap and apply
        new_major = minor_axis * np.sqrt(aspect_ratio)
        new_minor = major_axis / np.sqrt(aspect_ratio)
    
    # Ensure we maintain reasonable sizes (prevent too small ellipses)
    new_minor = max(new_minor, 5.0)  # Minimum 5 pixels
    new_major = max(new_major, 5.0)  # Minimum 5 pixels
    
    return ((cx, cy), (new_minor, new_major), angle)

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
    """Advanced processor: Object separation + momentum merging to prevent merging."""
    
    def __init__(self, img_config: Dict = None):
        self.img_config = img_config or IMAGE_PROCESSING_CONFIG
        self.last_center = None  # Track last center position
        self.current_aoi = None  # Track rectangular AOI around last detection
        self.exclusion_zones = []  # Track other large contours to prevent merging
        self.object_ownership_map = None  # Track which pixels belong to which objects
        self.frame_shape = None  # Remember frame dimensions
        self.last_valid_distance = None  # Track last valid distance for filtering
        
        # ELLIPSE TRACKING STATE for smooth movement
        self.last_ellipse_center = None  # (cx, cy) for smooth interpolation
        self.last_ellipse_params = None  # Full ellipse parameters for consistency
        
    def reset_tracking(self):
        """Reset all tracking state."""
        self.last_center = None
        self.current_aoi = None
        self.exclusion_zones = []
        self.object_ownership_map = None
        self.frame_shape = None
        self.last_valid_distance = None
        self.last_ellipse_center = None
        self.last_ellipse_params = None
        
    def preprocess_frame(self, frame_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocessing with object separation to prevent merging."""
        # Apply momentum-based merging first (this helps connect separated net parts)
        if self.img_config.get('use_momentum_merging', True):
            momentum_enhanced = directional_momentum_merge(
                frame_u8, 
                search_radius=self.img_config.get('momentum_search_radius', 1),
                momentum_threshold=self.img_config.get('momentum_threshold', 0.1),
                momentum_decay=self.img_config.get('momentum_decay', 0.9),
                momentum_boost=self.img_config.get('momentum_boost', 10.0)
            )
        else:
            momentum_enhanced = frame_u8
        
        # Use object separation preprocessing
        return preprocess_edges_with_object_separation(momentum_enhanced, self.object_ownership_map, self.img_config)
        
    def update_object_ownership(self, frame_u8: np.ndarray):
        """Update object ownership tracking to prevent merging"""
        H, W = frame_u8.shape[:2]
        self.frame_shape = (H, W)
        
        # First pass: find all contours without any filtering
        _, raw_edges = preprocess_edges(frame_u8, self.img_config)
        raw_contours, _ = cv2.findContours(raw_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create object masks
        object_masks = create_object_masks(raw_contours, (H, W))
        
        # Update ownership tracking
        self.object_ownership_map = track_object_ownership(
            object_masks, 
            self.object_ownership_map, 
            (H, W)
        )
        
        return object_masks
        
    def find_best_contour(self, contours):
        """Find the best contour - object separation already handled in preprocessing."""
        return select_best_contour_core(
            contours, 
            self.last_center, 
            self.current_aoi, 
            self.img_config
        )
        
    def analyze_frame(self, frame_u8: np.ndarray, extent: Tuple[float,float,float,float] = None) -> FrameAnalysisResult:
        """Frame analysis with optional object ownership tracking to prevent merging."""
        H, W = frame_u8.shape[:2]
        
        # STEP 1: Conditionally update object ownership based on config toggle
        if self.img_config.get('use_pixel_ownership', True):
            # Use pixel ownership tracking to prevent merging
            object_masks = self.update_object_ownership(frame_u8)
            # STEP 2: Preprocess with object separation (masks out other objects)
            _, edges_proc = self.preprocess_frame(frame_u8)
        else:
            # Skip pixel ownership tracking for faster processing
            # STEP 2: Simple preprocessing without object separation
            _, edges_proc = preprocess_edges(frame_u8, self.img_config)
        
        # STEP 3: Find contours on separated edges (net should be separated now)
        contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # STEP 4: Select best contour (should be clean net contour)
        best_contour, features, stats = self.find_best_contour(contours)
        
        # STEP 5: Update exclusion zones for visualization
        self.exclusion_zones = update_exclusion_zones(
            contours, 
            self.exclusion_zones, 
            best_contour, 
            self.img_config
        )
        
        # Extract distance and angle
        distance_pixels, angle_degrees = _distance_angle_from_contour(best_contour, W, H)
        
        # DISTANCE VALIDATION: Filter out invalid distances (negative or outside image bounds)
        if self.img_config.get('use_distance_validation', True):
            distance_pixels = self._validate_distance(distance_pixels, H)
        
        # Convert to meters if extent provided
        distance_meters = None
        if distance_pixels is not None and extent is not None:
            x_min, x_max, y_min, y_max = extent
            height_m = y_max - y_min
            px2m_y = height_m / H
            distance_meters = y_min + distance_pixels * px2m_y
        
        # Update SMOOTH ellipse tracking (prevents jumping/flickering)
        if best_contour is not None and features and len(best_contour) >= 5:
            # Get current ellipse parameters
            try:
                current_ellipse = cv2.fitEllipse(best_contour)
                current_center = current_ellipse[0]  # (cx, cy)
                
                # Apply smooth tracking with movement limiting
                smooth_center = self._smooth_ellipse_movement(current_center, W, H)
                
                # Apply aspect ratio modification to the ellipse
                aspect_ratio = TRACKING_CONFIG.get('ellipse_aspect_ratio', 1.0)
                modified_ellipse = modify_ellipse_aspect_ratio(current_ellipse, aspect_ratio)
                
                # Update the ellipse center to use the smoothed center
                if modified_ellipse is not None:
                    _, (minor_axis, major_axis), angle = modified_ellipse
                    modified_ellipse = (smooth_center, (minor_axis, major_axis), angle)
                
                # Update tracking state with smoothed center and modified ellipse
                self.last_center = smooth_center
                self.last_ellipse_center = smooth_center
                self.last_ellipse_params = modified_ellipse
                
                # Create AOI around smoothed ellipse center
                expansion = TRACKING_CONFIG.get('aoi_expansion_pixels', 25)
                x, y = int(smooth_center[0]), int(smooth_center[1])
                self.current_aoi = (
                    max(0, x - expansion),
                    max(0, y - expansion),
                    min(W, 2*expansion),  # width
                    min(H, 2*expansion)   # height
                )
            except:
                # Fallback to simple centroid tracking if ellipse fitting fails
                self.last_center = (features['centroid_x'], features['centroid_y'])
                x, y, w, h = features['rect']
                expansion = 25
                self.current_aoi = (
                    max(0, x - expansion),
                    max(0, y - expansion),
                    min(W - max(0, x - expansion), w + 2*expansion),
                    min(H - max(0, y - expansion), h + 2*expansion)
                )
        
        # Create tracking status with smooth ellipse information
        status_parts = []
        if self.current_aoi:
            if self.last_ellipse_center is not None:
                status_parts.append("SMOOTH_ELLIPSE_AOI")
            else:
                status_parts.append("SEPARATED_AOI")
        else:
            status_parts.append("SEPARATED_SEARCH")
            
        if len(self.exclusion_zones) > 0:
            status_parts.append(f"EX{len(self.exclusion_zones)}")
            
        # Count separated objects
        if self.object_ownership_map is not None:
            unique_objects = len(np.unique(self.object_ownership_map)) - 1  # -1 for background
            if unique_objects > 1:
                status_parts.append(f"OBJ{unique_objects}")
        
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

    def _validate_distance(self, distance_pixels: Optional[float], image_height: int) -> Optional[float]:
        """
        Validate distance measurement and use last valid value if current is invalid.
        
        Args:
            distance_pixels: Raw distance measurement in pixels
            image_height: Height of the image in pixels
            
        Returns:
            Validated distance or None if no valid distance available
        """
        # Check if current distance is valid
        if distance_pixels is not None:
            # Distance must be non-negative and within image bounds
            if 0 <= distance_pixels <= image_height:
                # Valid distance - store it and return
                self.last_valid_distance = distance_pixels
                return distance_pixels
            else:
                # Invalid distance - use last valid distance if available
                if self.last_valid_distance is not None:
                    return self.last_valid_distance
                else:
                    return None
        else:
            # No distance detected - use last valid distance if available
            if self.last_valid_distance is not None:
                return self.last_valid_distance
            else:
                return None

    def _smooth_ellipse_movement(self, current_center: tuple, image_width: int, image_height: int) -> tuple:
        """
        Apply smooth ellipse movement with speed limiting to prevent jumping/flickering.
        
        Args:
            current_center: (cx, cy) of current detected ellipse
            image_width: Width of image for bounds checking
            image_height: Height of image for bounds checking
            
        Returns:
            Smoothed ellipse center (cx, cy)
        """
        from utils.sonar_config import TRACKING_CONFIG
        
        # Get smoothing parameters
        smoothing_alpha = TRACKING_CONFIG.get('ellipse_smoothing_alpha', 0.2)
        max_movement = TRACKING_CONFIG.get('ellipse_max_movement_pixels', 4.0)
        
        # If no previous ellipse, use current as-is
        if self.last_ellipse_center is None:
            return current_center
        
        # Calculate movement distance
        last_cx, last_cy = self.last_ellipse_center
        curr_cx, curr_cy = current_center
        
        dx = curr_cx - last_cx
        dy = curr_cy - last_cy
        movement_distance = (dx*dx + dy*dy) ** 0.5  # Using power instead of np.sqrt
        
        # Limit movement speed to prevent jumping
        if movement_distance > max_movement:
            # Scale down movement to maximum allowed
            scale = max_movement / movement_distance
            dx *= scale
            dy *= scale
            
        # Apply exponential smoothing
        smooth_cx = last_cx + smoothing_alpha * dx
        smooth_cy = last_cy + smoothing_alpha * dy
        
        # Ensure smoothed center stays within image bounds
        smooth_cx = max(0, min(image_width - 1, smooth_cx))
        smooth_cy = max(0, min(image_height - 1, smooth_cy))
        
        return (smooth_cx, smooth_cy)

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


def list_npz_files(npz_dir: str | None = None) -> None:
    npz_files = get_available_npz_files(npz_dir)
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        return

    print(f"Available NPZ files in {npz_dir}:")
    for i, npz_path in enumerate(npz_files):
        try:
            cones, timestamps, _, _ = load_cone_run_npz(npz_path)
            print(f"  {i}: {npz_path.name}")
            print(f"     {cones.shape[0]} frames, {timestamps[0].strftime('%H:%M:%S')} to {timestamps[-1].strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"  {i}: {npz_path.name} - Error: {e}")

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

def contour_rect(contour) -> Tuple[int,int,int,int]:
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

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

# ============================ Object Separation and Masking Functions ============================

def create_object_masks(contours, frame_shape):
    """Create individual masks for each contour to track object ownership"""
    H, W = frame_shape
    object_masks = []
    
    for i, contour in enumerate(contours):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        object_masks.append({
            'mask': mask,
            'contour': contour,
            'area': cv2.contourArea(contour),
            'id': i
        })
    
    return object_masks

def track_object_ownership(current_masks, previous_ownership_map, frame_shape):
    """Track which pixels belong to which objects across frames"""
    from utils.sonar_config import EXCLUSION_CONFIG
    min_area = EXCLUSION_CONFIG.get('min_secondary_area', 50)
    
    H, W = frame_shape
    
    # Initialize ownership map if needed
    if previous_ownership_map is None:
        ownership_map = np.zeros((H, W), dtype=np.int32)  # 0 = unassigned
    else:
        ownership_map = previous_ownership_map.copy()
    
    # Sort masks by area (largest first - likely the net)
    sorted_masks = sorted(current_masks, key=lambda x: x['area'], reverse=True)
    
    # Assign object IDs (1 = net, 2+ = other objects)
    for mask_idx, mask_data in enumerate(sorted_masks):
        if mask_data['area'] < min_area:
            continue
            
        object_id = mask_idx + 1  # Start from 1 (0 is unassigned)
        mask = mask_data['mask']
        
        # For the largest object (likely net), be more permissive
        if mask_idx == 0:  # Largest object (net)
            # Only claim pixels that are currently unassigned or were previously net
            net_pixels = (mask > 0) & ((ownership_map == 0) | (ownership_map == 1))
            ownership_map[net_pixels] = 1
        else:  # Smaller objects (fish, debris)
            # Claim all pixels for this object (more aggressive)
            object_pixels = mask > 0
            ownership_map[object_pixels] = object_id
    
    return ownership_map

def create_net_protection_mask(ownership_map, net_id=1):
    """Create a mask that protects the net from merging with other objects"""
    # Create mask where only net pixels (ID=1) and unassigned pixels (ID=0) are allowed
    net_protection_mask = ((ownership_map == net_id) | (ownership_map == 0)).astype(np.uint8) * 255
    return net_protection_mask

def preprocess_edges_with_object_separation(frame_u8, ownership_map, cfg=IMAGE_PROCESSING_CONFIG):
    """Preprocess edges but mask out pixels belonging to other objects"""
    # Standard preprocessing
    proc = prepare_input_gray(frame_u8, cfg)
    edges = cv2.Canny(proc, cfg.get('canny_low_threshold', 50), cfg.get('canny_high_threshold', 150))
    
    # Morphological operations
    mks = int(cfg.get('morph_close_kernel', 0))
    dil = int(cfg.get('edge_dilation_iterations', 0))
    out = edges
    if mks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mks, mks))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if dil > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        out = cv2.dilate(out, kernel2, iterations=dil)
    
    # CRITICAL: Apply object separation mask
    if ownership_map is not None:
        net_protection_mask = create_net_protection_mask(ownership_map)
        # Only keep edges that are in net-allowed areas
        out = cv2.bitwise_and(out, net_protection_mask)
    
    return edges, out

# ============================ Original Spatial Exclusion Memory Functions ============================

def update_exclusion_memory_map(memory_map, contours, selected_contour, frame_shape, decay_rate=0.9):
    """Update spatial memory map of excluded areas"""
    from utils.sonar_config import EXCLUSION_CONFIG
    min_area = EXCLUSION_CONFIG.get('min_secondary_area', 50)
    memory_radius = EXCLUSION_CONFIG.get('exclusion_radius', 10)
    
    H, W = frame_shape
    
    # Initialize memory map if needed
    if memory_map is None:
        memory_map = np.zeros((H, W), dtype=np.float32)
    else:
        # Decay existing memory
        memory_map *= decay_rate
    
    # Add exclusion areas for other contours (not the selected net contour)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Skip if this is the selected net contour
        if selected_contour is not None and np.array_equal(contour, selected_contour):
            continue
        
        # Create mask for this contour with expanded area
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Expand the exclusion area around the contour
        if memory_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (memory_radius*2, memory_radius*2))
            mask = cv2.dilate(mask, kernel)
        
        # Add to memory map (areas where other contours were present)
        memory_map[mask > 0] = np.maximum(memory_map[mask > 0], 1.0)
    
    return memory_map

def filter_contour_by_exclusion_memory(contour, memory_map, threshold=0.3):
    """Filter contour points that are in excluded memory areas"""
    if memory_map is None or contour is None or len(contour) == 0:
        return contour
    
    # Check which contour points are in excluded areas
    filtered_points = []
    
    for point in contour:
        x, y = point[0]
        if 0 <= y < memory_map.shape[0] and 0 <= x < memory_map.shape[1]:
            # If this point has low exclusion memory, keep it
            if memory_map[y, x] < threshold:
                filtered_points.append(point)
        else:
            # Keep points outside frame bounds
            filtered_points.append(point)
    
    # Return filtered contour, but ensure we don't remove too many points
    if len(filtered_points) < len(contour) * 0.3:  # Keep at least 30% of points
        return contour  # Return original if too much was filtered
    
    return np.array(filtered_points, dtype=contour.dtype) if filtered_points else contour

def select_best_contour_with_spatial_memory(contours, last_center=None, aoi=None, exclusion_zones=None, memory_map=None, cfg_img=IMAGE_PROCESSING_CONFIG):
    """Enhanced contour selection that uses spatial memory to prevent merging"""
    min_area = float(cfg_img.get('min_contour_area', 100))
    aoi_boost = 2.0
    
    from utils.sonar_config import EXCLUSION_CONFIG
    exclusion_penalty = 0.5  # Stronger penalty for spatial memory
    exclusion_radius = EXCLUSION_CONFIG.get('exclusion_radius', 10)
    memory_threshold = 0.3  # Threshold for exclusion memory
    
    best, best_feat, best_score = None, None, 0.0
    total = 0

    for c in contours or []:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        total += 1
        
        # SPATIAL MEMORY FILTERING: Remove points that were near other contours
        filtered_contour = filter_contour_by_exclusion_memory(c, memory_map, memory_threshold)
        
        # Recalculate area after filtering
        filtered_area = cv2.contourArea(filtered_contour)
        if filtered_area < min_area:
            continue  # Skip if too much was filtered out
        
        # Calculate features on filtered contour
        feat = compute_contour_features(filtered_contour)
        
        # CORE SCORING: area × elongation (using filtered contour)
        elongation = max(feat['aspect_ratio'], feat['ellipse_elongation'])
        base_score = filtered_area * elongation
        
        # MEMORY PENALTY: Penalize contours that overlap with exclusion memory
        memory_penalty = 1.0
        if memory_map is not None:
            # Check how much of the contour overlaps with excluded areas
            mask = np.zeros(memory_map.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [filtered_contour], 255)
            
            # Calculate overlap with exclusion memory
            overlap_area = np.sum((mask > 0) & (memory_map > memory_threshold))
            contour_area_pixels = np.sum(mask > 0)
            
            if contour_area_pixels > 0:
                overlap_ratio = overlap_area / contour_area_pixels
                memory_penalty = 1.0 - (overlap_ratio * exclusion_penalty)
        
        # AOI boost
        if aoi is not None:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            ax, ay, aw, ah = aoi
            if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                base_score *= aoi_boost
        
        # Traditional exclusion zone penalty (for immediate zones)
        zone_penalty = 1.0
        if exclusion_zones:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            for zone in exclusion_zones:
                zone_dist = np.sqrt((cx - zone['center'][0])**2 + (cy - zone['center'][1])**2)
                if zone_dist < exclusion_radius:
                    proximity_factor = 1.0 - (zone_dist / exclusion_radius)
                    penalty = 0.3 * zone['confidence'] * proximity_factor
                    zone_penalty *= (1.0 - penalty)
        
        # Distance penalty if we have last position
        distance_penalty = 1.0
        if last_center is not None:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            distance = np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)
            distance_penalty = max(0.5, 1.0 - distance / 200.0)
        
        # FINAL SCORE with spatial memory consideration
        final_score = base_score * memory_penalty * zone_penalty * distance_penalty
        
        if final_score > best_score:
            best, best_feat, best_score = filtered_contour, feat, final_score

    stats = {
        'total_contours': total,
        'best_score': best_score,
        'exclusion_zones_count': len(exclusion_zones) if exclusion_zones else 0,
        'memory_active': memory_map is not None
    }
    return best, (best_feat or {}), stats

# ============================ Original Exclusion Zone Functions ============================

def update_exclusion_zones(contours, current_exclusion_zones, selected_contour, cfg=None):
    """Update exclusion zones by tracking other large contours"""
    if cfg is None:
        cfg = IMAGE_PROCESSING_CONFIG
    
    from utils.sonar_config import EXCLUSION_CONFIG
    
    # Check if exclusions are enabled
    if not EXCLUSION_CONFIG.get('enable_exclusions', True):
        return []
    
    min_area = EXCLUSION_CONFIG.get('min_secondary_area', 50)
    max_zones = EXCLUSION_CONFIG.get('max_exclusion_zones', 5)
    zone_decay_frames = EXCLUSION_CONFIG.get('zone_decay_frames', 3)
    zone_decay = 1.0 - (1.0 / max(1, zone_decay_frames))  # Convert frames to decay factor
    
    # Decay existing zones
    decayed_zones = []
    for zone in current_exclusion_zones:
        zone['confidence'] *= zone_decay
        if zone['confidence'] > 0.1:  # Keep zones above threshold
            decayed_zones.append(zone)
    
    # Find new large contours (excluding the selected one)
    new_zones = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Skip if this is the selected contour
        if selected_contour is not None and np.array_equal(contour, selected_contour):
            continue
            
        # Get centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Check if this matches an existing zone (update instead of create new)
        updated_existing = False
        for zone in decayed_zones:
            zone_dist = np.sqrt((cx - zone['center'][0])**2 + (cy - zone['center'][1])**2)
            if zone_dist < 30:  # Close to existing zone
                zone['center'] = (cx, cy)
                zone['area'] = area
                zone['confidence'] = min(1.0, zone['confidence'] + 0.3)
                updated_existing = True
                break
                
        if not updated_existing:
            new_zones.append({
                'center': (cx, cy),
                'area': area,
                'confidence': 0.5,
                'contour': contour
            })
    
    # Combine and limit zones
    all_zones = decayed_zones + new_zones
    all_zones.sort(key=lambda z: z['confidence'], reverse=True)
    return all_zones[:max_zones]

def select_best_contour_with_exclusion(contours, last_center=None, aoi=None, exclusion_zones=None, cfg_img=IMAGE_PROCESSING_CONFIG):
    """Enhanced contour selection that avoids exclusion zones"""
    min_area = float(cfg_img.get('min_contour_area', 100))
    aoi_boost = 2.0
    
    from utils.sonar_config import EXCLUSION_CONFIG
    
    # Check if exclusions are enabled
    if not EXCLUSION_CONFIG.get('enable_exclusions', True) or not exclusion_zones:
        exclusion_zones = []
    
    exclusion_penalty = 0.3  # Fixed penalty for exclusion zones
    exclusion_radius = EXCLUSION_CONFIG.get('exclusion_radius', 10)
    
    best, best_feat, best_score = None, None, 0.0
    total = 0

    for c in contours or []:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        total += 1
        
        # Basic features
        feat = compute_contour_features(c)
        
        # CORE SCORING: area × elongation
        elongation = max(feat['aspect_ratio'], feat['ellipse_elongation'])
        base_score = area * elongation
        
        # AOI boost
        if aoi is not None:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            ax, ay, aw, ah = aoi
            if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                base_score *= aoi_boost
        
        # Exclusion zone penalty
        if exclusion_zones:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            for zone in exclusion_zones:
                zone_dist = np.sqrt((cx - zone['center'][0])**2 + (cy - zone['center'][1])**2)
                if zone_dist < exclusion_radius:
                    # Apply penalty based on zone confidence and proximity
                    proximity_factor = 1.0 - (zone_dist / exclusion_radius)
                    penalty = exclusion_penalty * zone['confidence'] * proximity_factor
                    base_score *= (1.0 - penalty)
        
        # Distance penalty if we have last position
        final_score = base_score
        if last_center is not None:
            cx, cy = feat['centroid_x'], feat['centroid_y']
            distance = np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)
            distance_factor = max(0.5, 1.0 - distance / 200.0)
            final_score *= distance_factor
        
        if final_score > best_score:
            best, best_feat, best_score = c, feat, final_score

    stats = {
        'total_contours': total, 
        'best_score': best_score,
        'exclusion_zones_count': len(exclusion_zones) if exclusion_zones else 0
    }
    return best, (best_feat or {}), stats

# ============================ Momentum vs Blur (shared) ============================

def directional_momentum_merge(frame, search_radius=3, momentum_threshold=0.2,
                               momentum_decay=0.8, momentum_boost=1.5):
    if search_radius <= 2:
        return fast_directional_enhance(frame, momentum_threshold, momentum_boost)

    result = frame.astype(np.float32)
    grad_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
    energy_map = np.sqrt(grad_x**2 + grad_y**2) / 255.0
    if np.max(energy_map) < momentum_threshold:
        return frame

    kernel_size = 5
    center = 2
    h_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32); h_kernel[center, :] = [0.1, 0.2, 0.4, 0.2, 0.1]
    v_kernel = np.zeros_like(h_kernel); v_kernel[:, center] = [0.1, 0.2, 0.4, 0.2, 0.1]
    d1_kernel = np.zeros_like(h_kernel); d2_kernel = np.zeros_like(h_kernel)
    for i in range(kernel_size):
        d1_kernel[i, i] = 0.2
        d2_kernel[i, kernel_size-1-i] = 0.2
    d1_kernel[center, center] = 0.4; d2_kernel[center, center] = 0.4

    responses = [
        cv2.filter2D(result, -1, h_kernel),
        cv2.filter2D(result, -1, v_kernel),
        cv2.filter2D(result, -1, d1_kernel),
        cv2.filter2D(result, -1, d2_kernel),
    ]
    enhanced = np.maximum.reduce(responses)
    boost_factor = 1.0 + momentum_boost * np.clip(energy_map, 0, 1)
    result = enhanced * boost_factor
    return np.clip(result, 0, 255).astype(np.uint8)

def fast_directional_enhance(frame, threshold=0.2, boost=1.5):
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    h_enhanced = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, h_kernel)
    v_enhanced = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, v_kernel)
    enhanced = np.maximum(h_enhanced, v_enhanced)
    grad = cv2.Laplacian(frame, cv2.CV_32F)
    grad_norm = np.abs(grad) / 255.0
    boost_mask = grad_norm > threshold
    result = frame.astype(np.float32)
    result[boost_mask] += boost * enhanced[boost_mask]
    return np.clip(result, 0, 255).astype(np.uint8)

def prepare_input_gray(frame_u8: np.ndarray, cfg=IMAGE_PROCESSING_CONFIG) -> np.ndarray:
    """Prepare input using momentum merging for sharp, well-defined objects"""
    return directional_momentum_merge(
        frame_u8,
        search_radius=cfg.get('momentum_search_radius', 3),
        momentum_threshold=cfg.get('momentum_threshold', 0.2),
        momentum_decay=cfg.get('momentum_decay', 0.8),
        momentum_boost=cfg.get('momentum_boost', 1.5),
    )

def preprocess_edges(frame_u8: np.ndarray, cfg=IMAGE_PROCESSING_CONFIG) -> Tuple[np.ndarray, np.ndarray]:
    proc = prepare_input_gray(frame_u8, cfg)
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
    x, y, w, h = contour_rect(contour)
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

# ============================ LEGACY FUNCTION ADAPTERS ============================

def analyze_red_line_distance_over_time(npz_file_index: int = 0, frame_start: int = 0,
                                       frame_count: Optional[int] = None, frame_step: int = 1,
                                       save_error_frames: bool = False, error_frames_dir: str = None) -> pd.DataFrame:
    """Legacy adapter for NPZ analysis."""
    engine = DistanceAnalysisEngine()
    return engine.analyze_npz_sequence(npz_file_index, frame_start, frame_count, frame_step)

def analyze_red_line_distance_from_sonar_csv(TARGET_BAG: str, EXPORTS_FOLDER: Path,
                                           frame_start: int = 0, frame_count: Optional[int] = None,
                                           frame_step: int = 1, **kwargs) -> pd.DataFrame:
    """Legacy adapter for sonar CSV analysis."""
    engine = DistanceAnalysisEngine()
    return engine.analyze_sonar_csv(TARGET_BAG, EXPORTS_FOLDER, frame_start, frame_count, frame_step, **kwargs)

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
            
        # Try interactive plotting first if requested
        if use_plotly:
            try:
                return VisualizationEngine._plot_interactive_distance(valid, dist_col, unit, title)
            except ImportError:
                print("Plotly not available, falling back to matplotlib")
        
        return VisualizationEngine._plot_matplotlib_distance(valid, dist_col, unit, title)
    
    @staticmethod
    def _plot_interactive_distance(valid: pd.DataFrame, dist_col: str, unit: str, title: str):
        """Create interactive plotly visualization."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
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
        
        fig.show()
        return fig
    
    @staticmethod
    def _plot_matplotlib_distance(valid: pd.DataFrame, dist_col: str, unit: str, title: str):
        """Create matplotlib visualization."""
        distances = valid[dist_col]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Frame index plot
        axes[0,0].plot(valid['frame_index'], distances, 'b-', alpha=0.7)
        axes[0,0].scatter(valid['frame_index'], distances, c='red', s=20, alpha=0.6)
        axes[0,0].set_title('Distance vs Frame Index')
        axes[0,0].set_xlabel('Frame')
        axes[0,0].set_ylabel(f'Distance ({unit})')
        axes[0,0].grid(True, alpha=0.3)
        
        # Time plot
        x_time = valid.get('timestamp', valid['frame_index'])
        axes[0,1].plot(x_time, distances, 'g-', alpha=0.7)
        axes[0,1].scatter(x_time, distances, c='red', s=20, alpha=0.6)
        axes[0,1].set_title('Distance vs Time')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel(f'Distance ({unit})')
        axes[0,1].grid(True, alpha=0.3)
        
        # Histogram
        axes[1,0].hist(distances, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1,0].axvline(distances.mean(), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {distances.mean():.2f}{unit}')
        axes[1,0].axvline(distances.median(), color='green', linestyle='--', linewidth=2,
                         label=f'Median: {distances.median():.2f}{unit}')
        axes[1,0].set_title('Distance Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Trends
        window_size = max(5, len(distances) // 20)
        smoothed = distances.rolling(window=window_size, center=True).mean()
        std = distances.rolling(window=window_size, center=True).std()
        
        axes[1,1].plot(valid['frame_index'], distances, 'lightcoral', alpha=0.5, label='Raw data')
        axes[1,1].plot(valid['frame_index'], smoothed, 'darkred', linewidth=2,
                      label=f'Smoothed (n={window_size})')
        axes[1,1].fill_between(valid['frame_index'], smoothed - std, smoothed + std,
                              alpha=0.3, color='red', label='±1 Std Dev')
        axes[1,1].set_title('Distance Trends')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {distances.mean():.2f}{unit}\nStd: {distances.std():.2f}{unit}\nRange: {distances.min():.2f}-{distances.max():.2f}{unit}'
        axes[0,0].text(0.02, 0.98, stats_text, transform=axes[0,0].transAxes, fontsize=9,
                      va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
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
        
        # Create visualization
        if use_plotly:
            try:
                fig = ComparisonEngine._create_interactive_comparison(sonar, nav)
            except ImportError:
                print("Plotly not available, falling back to matplotlib")
                fig = ComparisonEngine._create_matplotlib_comparison(sonar, nav, window_size)
        else:
            fig = ComparisonEngine._create_matplotlib_comparison(sonar, nav, window_size)
        
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
        
        fig.show()
        return fig
    
    @staticmethod
    def _create_matplotlib_comparison(sonar: pd.DataFrame, nav: pd.DataFrame, window_size: int):
        """Create matplotlib comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle('🔄 SONAR vs DVL DISTANCE COMPARISON', fontsize=16, fontweight='bold')
        
        # Distance over time
        ax1 = axes[0, 0]
        ax1.plot(sonar['synthetic_time'], sonar['distance_meters'], 'r-', linewidth=1, alpha=0.3, label='Sonar Raw')
        ax1.plot(sonar['synthetic_time'], sonar['distance_meters_primary'], 'r-', linewidth=2, alpha=0.8, label='Sonar Smoothed')
        ax1.plot(nav['relative_time'], nav['NetDistance'], 'b-', linewidth=2, alpha=0.8, label='DVL NetDistance')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Distance (m)')
        ax1.set_title('Distance Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Smoothing comparison detail
        ax2 = axes[0, 1]
        sl = slice(100, min(200, len(sonar)))
        ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters'].iloc[sl], 'gray', linewidth=1, alpha=0.7, label='Raw')
        ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters_mavg'].iloc[sl], 'orange', linewidth=1.5, alpha=0.8, label='Moving Avg')
        ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters_savgol'].iloc[sl], 'red', linewidth=2, alpha=0.9, label='Savitzky-Golay')
        ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters_gaussian'].iloc[sl], 'purple', linewidth=1.5, alpha=0.8, label='Gaussian')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Distance (m)')
        ax2.set_title('Smoothing Methods (Detail)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Distribution comparison
        ax3 = axes[0, 2]
        ax3.hist(sonar['distance_meters'], bins=30, alpha=0.7, color='red', label=f'Sonar (n={len(sonar)})', density=True)
        ax3.hist(nav['NetDistance'], bins=30, alpha=0.7, color='blue', label=f'DVL (n={len(nav)})', density=True)
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Statistics table
        ax4 = axes[1, 0]
        stats_df = pd.DataFrame({
            'Measurement': ['Sonar', 'DVL'],
            'Count': [len(sonar), len(nav)],
            'Mean (m)': [np.nanmean(sonar['distance_meters']), nav['NetDistance'].mean()],
            'Std (m)': [np.nanstd(sonar['distance_meters']), nav['NetDistance'].std()],
            'Min (m)': [np.nanmin(sonar['distance_meters']), nav['NetDistance'].min()],
            'Max (m)': [np.nanmax(sonar['distance_meters']), nav['NetDistance'].max()],
        })
        ax4.axis('off')
        ax4.text(0.05, 0.95, stats_df.round(3).to_string(index=False),
                transform=ax4.transAxes, fontfamily='monospace', fontsize=10, va='top')
        ax4.set_title('Statistical Comparison')
        
        # Bar comparison
        ax5 = axes[1, 1]
        x = np.arange(len(stats_df))
        w = 0.35
        ax5.bar(x - w/2, stats_df['Mean (m)'], w, label='Mean', alpha=0.8, color=['red','blue'])
        ax5.bar(x + w/2, stats_df['Std (m)'], w, label='Std', alpha=0.8, color=['lightcoral','lightblue'])
        ax5.set_xticks(x)
        ax5.set_xticklabels(stats_df['Measurement'])
        ax5.set_title('Mean ± Std')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Noise reduction comparison
        ax6 = axes[1, 2]
        noise_levels = [
            np.nanstd(sonar['distance_meters']),
            np.nanstd(sonar['distance_meters_mavg']),
            np.nanstd(sonar['distance_meters_savgol']),
            np.nanstd(sonar['distance_meters_gaussian']),
        ]
        bars = ax6.bar(['Raw', 'Moving Avg', 'Savitzky-Golay', 'Gaussian'], noise_levels, alpha=0.7)
        ax6.set_ylabel('Std (m)')
        ax6.set_title('Noise Reduction by Smoothing')
        ax6.grid(True, alpha=0.3)
        for b, v in zip(bars, noise_levels):
            ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.3f}m', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        return fig

# ============================ LEGACY PLOTTING FUNCTION ADAPTERS ============================

def plot_time_based_analysis(distance_results: pd.DataFrame, pixels_to_meters_avg: float = 0.01,
                            estimated_fps: float = 15):
    """Legacy adapter for time-based plotting."""
    return VisualizationEngine.plot_distance_analysis(distance_results, "Time-Based Distance Analysis", use_plotly=False)

def plot_real_world_distance_analysis(distance_results: pd.DataFrame, image_shape=(700, 900),
                                     sonar_coverage_meters=5.0):
    """Legacy adapter for real-world distance plotting."""
    return VisualizationEngine.plot_distance_analysis(distance_results, "Real-World Distance Analysis")

def compare_sonar_vs_dvl(distance_results: pd.DataFrame, raw_data: Dict[str, pd.DataFrame],
                        distance_measurements: Dict[str, pd.DataFrame] = None,
                        sonar_coverage_m: float = 5.0, sonar_image_size: int = 700,
                        window_size: int = 15):
    """Unified comparison function."""
    return ComparisonEngine.compare_sonar_vs_dvl(distance_results, raw_data, sonar_coverage_m, 
                                                sonar_image_size, window_size, use_plotly=False)

def interactive_distance_comparison(distance_results: pd.DataFrame, raw_data: Dict[str, pd.DataFrame],
                                  distance_measurements: Dict[str, pd.DataFrame] = None,
                                  sonar_coverage_m: float = 5.0, sonar_image_size: int = 700):
    """Interactive comparison using plotly."""
    return ComparisonEngine.compare_sonar_vs_dvl(distance_results, raw_data, sonar_coverage_m, 
                                                sonar_image_size, use_plotly=True)

def detailed_sonar_dvl_comparison(distance_results, raw_data, sonar_coverage_m: float = 5.0,
                                 sonar_image_size: int = 700, window_size: int = 15):
    """Legacy API preserved."""
    return ComparisonEngine.compare_sonar_vs_dvl(distance_results, raw_data, sonar_coverage_m, 
                                                sonar_image_size, window_size, use_plotly=False)

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

# ============================ DIAGNOSTIC & UTILITY FUNCTIONS ============================

def basic_image_processing_pipeline(img_u8: np.ndarray, show=True, figsize=(15, 10)) -> Dict[str, np.ndarray]:
    """Sharp image processing pipeline with momentum merging"""
    momentum_enhanced = directional_momentum_merge(img_u8)
    edges = cv2.Canny(img_u8, 50, 150)
    edges_momentum = cv2.Canny(momentum_enhanced, 50, 150)
    _, thresh = cv2.threshold(img_u8, 127, 255, cv2.THRESH_BINARY)
    diff = cv2.absdiff(edges, edges_momentum)

    res = {'original': img_u8, 'momentum_enhanced': momentum_enhanced, 'edges': edges, 'edges_momentum': edges_momentum, 'thresh': thresh, 'diff': diff}
    if show:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes[0,0].imshow(img_u8, cmap='gray');       axes[0,0].set_title('Original');        axes[0,0].axis('off')
        axes[0,1].imshow(momentum_enhanced, cmap='gray');      axes[0,1].set_title('Momentum Enhanced');   axes[0,1].axis('off')
        axes[0,2].imshow(thresh, cmap='gray');       axes[0,2].set_title('Binary Threshold');axes[0,2].axis('off')
        axes[1,0].imshow(edges, cmap='gray');        axes[1,0].set_title('Canny Edges');     axes[1,0].axis('off')
        axes[1,1].imshow(edges_momentum, cmap='gray');axes[1,1].set_title('Canny + Momentum');    axes[1,1].axis('off')
        axes[1,2].imshow(diff, cmap='gray');         axes[1,2].set_title('Edge Difference'); axes[1,2].axis('off')
        plt.tight_layout(); plt.show()
    return res

def visualize_processing_steps(frame_index=50, npz_file_index=0, figsize=(15, 5)):
    files = get_available_npz_files()
    if npz_file_index >= len(files):
        print(f"Error: NPZ file index {npz_file_index} not available"); return
    cones, _, _, _ = load_cone_run_npz(files[npz_file_index])
    if frame_index >= len(cones):
        print(f"Error: Frame {frame_index} not available (max: {len(cones)-1})"); return

    u8 = to_uint8_gray(cones[frame_index])
    proc = prepare_input_gray(u8, IMAGE_PROCESSING_CONFIG)
    edges_raw = cv2.Canny(proc, IMAGE_PROCESSING_CONFIG['canny_low_threshold'], IMAGE_PROCESSING_CONFIG['canny_high_threshold'])
    _, edges_proc = preprocess_edges(u8, IMAGE_PROCESSING_CONFIG)
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = IMAGE_PROCESSING_CONFIG['min_contour_area']
    filt = [c for c in contours if cv2.contourArea(c) >= min_area]

    cont_vis = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if filt: cv2.drawContours(cont_vis, filt, -1, (0,255,0), 2)
    print(f"Found {len(filt)} contours (area >= {min_area})" if filt else f"No contours found with area >= {min_area}")

    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].imshow(u8, cmap='gray');               ax[0].set_title(f'Original Frame {frame_index}'); ax[0].axis('off')
    ax[1].imshow(proc, cmap='gray');             ax[1].set_title('Momentum Merge (Sharp)'); ax[1].axis('off')
    ax[2].imshow(edges_raw, cmap='gray');        ax[2].set_title(f'Canny {IMAGE_PROCESSING_CONFIG["canny_low_threshold"]},{IMAGE_PROCESSING_CONFIG["canny_high_threshold"]}'); ax[2].axis('off')
    ax[3].imshow(edges_proc, cmap='gray');       ax[3].set_title(f'Close:{IMAGE_PROCESSING_CONFIG["morph_close_kernel"]} Dilate:{IMAGE_PROCESSING_CONFIG["edge_dilation_iterations"]}'); ax[3].axis('off')
    ax[4].imshow(cv2.cvtColor(cont_vis, cv2.COLOR_BGR2RGB)); ax[4].set_title(f'Contours ≥{min_area}'); ax[4].axis('off')
    plt.tight_layout(); plt.show()

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
    op = Path(output_path); cv2.imwrite(str(op), u8)
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
    print("=== CONTOUR DETECTION VIDEO CREATION ===")
    print(f"Creating video with {frame_count} frames, stepping by {frame_step}...")
    print(f"  Image Processing: MOMENTUM MERGING (radius={IMAGE_PROCESSING_CONFIG['momentum_search_radius']}, "
          f"threshold={IMAGE_PROCESSING_CONFIG['momentum_threshold']}, decay={IMAGE_PROCESSING_CONFIG['momentum_decay']}), "
          f"canny=({IMAGE_PROCESSING_CONFIG['canny_low_threshold']}, {IMAGE_PROCESSING_CONFIG['canny_high_threshold']}), "
          f"min_area={IMAGE_PROCESSING_CONFIG['min_contour_area']})")
    print(f"  Tracking: boost={TRACKING_CONFIG['aoi_boost_factor']}x, expansion={TRACKING_CONFIG['aoi_expansion_pixels']}px")
    print(f"  Video: fps={VIDEO_CONFIG['fps']}, show_contours={VIDEO_CONFIG['show_all_contours']}, show_ellipse={VIDEO_CONFIG['show_ellipse']}")

    files = get_available_npz_files()
    if npz_file_index >= len(files):
        print(f"Error: NPZ file index {npz_file_index} not available"); return None

    cones, _, _, _ = load_cone_run_npz(files[npz_file_index])
    T = len(cones)
    actual = int(min(frame_count, max(0, (T - frame_start)) // max(1, frame_step)))
    if actual <= 0:
        print("Error: Not enough frames to process"); return None

    first = to_uint8_gray(cones[frame_start]); H, W = first.shape
    outp = Path(output_path)
    # Ensure output directory exists
    try:
        outp.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Try primary mp4 writer
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_CONFIG['fps'], (W, H))
    if not vw.isOpened():
        # Fallback: try AVI with XVID (more widely supported in some OpenCV builds)
        fallback_path = outp.with_suffix('.avi')
        vw = cv2.VideoWriter(str(fallback_path), cv2.VideoWriter_fourcc(*'XVID'), VIDEO_CONFIG['fps'], (W, H))
        if vw.isOpened():
            print(f"Warning: mp4 writer failed, falling back to AVI: {fallback_path}")
            output_path = str(fallback_path)
        else:
            print("Error: Could not open video writer (mp4v and XVID both failed).")
            print("Possible causes: output directory missing, or your OpenCV build lacks the needed codecs (mp4/ffmpeg).")
            print(f"Tried paths: {outp} and {fallback_path}")
            print("Try installing OpenCV with FFmpeg support, or use an .avi output by specifying output_path with .avi extension.")
            return None

    print("Processing frames...")
    aoi = None
    tracked = new = lost = 0

    for i in range(actual):
        idx = frame_start + i * frame_step
        frame_u8 = to_uint8_gray(cones[idx])
        vis, next_aoi = process_frame_for_video(frame_u8, aoi)
        # Compute detected net distance (pixels) and overlay on the frame
        # Use SAME AOI context to ensure consistency with blue circle
        try:
            dist_px, ang_deg = get_red_line_distance_and_angle(frame_u8, aoi)
            if dist_px is not None:
                dist_text = f"Detected net distance: {dist_px:.1f}px"
            else:
                dist_text = "Detected net distance: N/A"
        except Exception:
            dist_text = "Detected net distance: N/A"
        # Put the distance text near the existing frame counter text (right side)
        try:
            text_x = max(10, W - 320)
            # Place the distance a bit further down (y=50) to avoid overlapping the top-right frame counter
            cv2.putText(vis, dist_text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        except Exception:
            # If overlay fails for any reason, continue without breaking the video creation
            pass
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
                                                          frame_step=5, output_path='elliptical_aoi_video.mp4',
                                                          processor: SonarDataProcessor = None):
    """Create video using the unified SonarDataProcessor with elliptical AOI support."""
    print("=== ENHANCED ELLIPTICAL AOI VIDEO CREATION ===")
    print(f"Creating video with elliptical AOI tracking...")
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
    tracked, new, lost, ellipse_tracked = 0, 0, 0, 0
    
    print(f"✅ Processing {actual} frames with elliptical AOI...")
    
    for i in range(actual):
        idx = frame_start + i * frame_step
        frame_u8 = to_uint8_gray(cones[idx])
        
        # Use unified processor for analysis
        result = processor.analyze_frame(frame_u8, extent)
        
        # Create visualization with elliptical AOI
        vis = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        
        # Find contours for drawing all contours
        _, edges_proc = processor.preprocess_frame(frame_u8)
        contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all contours
        if VIDEO_CONFIG.get('show_all_contours', True) and contours:
            cv2.drawContours(vis, contours, -1, (255, 200, 100), 1)
        
        # Draw elliptical AOI (yellow) if ellipse tracking is active
        if processor.last_ellipse_params is not None:
            # Draw the smoothed ellipse AOI
            ellipse_aoi = processor.last_ellipse_params
            cv2.ellipse(vis, ellipse_aoi, (0, 255, 255), 2)  # Yellow ellipse
            
            # Add AOI label at ellipse center
            cx, cy = ellipse_aoi[0]
            cv2.putText(vis, 'ELLIPSE AOI', (int(cx) + 10, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,255), 1)
        
        # Fallback: Draw rectangular AOI (yellow) if no ellipse tracking
        elif processor.current_aoi is not None:
            ax, ay, aw, ah = processor.current_aoi
            cv2.rectangle(vis, (ax, ay), (ax+aw, ay+ah), (0, 255, 255), 2)
            cv2.putText(vis, 'RECT AOI', (ax + 5, ay + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0,255,255), 1)
        
        # Draw object ownership map (colored overlay) - CRITICAL for showing separation
        if processor.object_ownership_map is not None:
            ownership_map = processor.object_ownership_map
            # Create colored overlay for different objects
            ownership_overlay = np.zeros_like(vis)
            
            # Assign different colors to different objects
            unique_objects = np.unique(ownership_map)
            colors = [
                [0, 0, 0],        # Background (ID=0)
                [0, 255, 0],      # Net (ID=1) - Green
                [0, 0, 255],      # Object 2 - Red  
                [255, 0, 0],      # Object 3 - Blue
                [0, 255, 255],    # Object 4 - Yellow
                [255, 0, 255],    # Object 5 - Magenta
                [255, 255, 0],    # Object 6 - Cyan
            ]
            
            for obj_id in unique_objects:
                if obj_id > 0 and obj_id < len(colors):
                    mask = ownership_map == obj_id
                    ownership_overlay[mask] = colors[obj_id]
            
            # Blend with original image (lighter overlay)
            alpha = 0.2  # Light transparency to see the separation
            vis = cv2.addWeighted(vis, 1-alpha, ownership_overlay, alpha, 0)
        
        # Draw exclusion zones (orange circles with labels)
        if processor.exclusion_zones:
            for i, zone in enumerate(processor.exclusion_zones):
                cx, cy = zone['center']
                radius = int(zone.get('area', 100) ** 0.5 * 0.5)  # Scale radius based on area
                confidence = zone['confidence']
                
                # Draw circle with opacity based on confidence
                color = (0, 165, 255)  # Orange in BGR
                thickness = max(1, int(confidence * 3))
                cv2.circle(vis, (cx, cy), radius, color, thickness)
                
                # Draw center dot
                cv2.circle(vis, (cx, cy), 3, color, -1)
                
                # Add label
                label = f'EX{i} ({confidence:.1f})'
                cv2.putText(vis, label, (cx + radius + 5, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
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
            if "ELLIPSE" in status:
                ellipse_tracked += 1
        elif "NEW" in status:
            new += 1
        else:
            lost += 1
        
        # Add frame info and status - enhanced for object separation
        exclusion_count = len(processor.exclusion_zones)
        
        frame_info = f'Frame: {idx} | {status}'
        cv2.putText(vis, frame_info, (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Add object separation info (CRITICAL feature visualization)
        separation_info_parts = []
        if exclusion_count > 0:
            separation_info_parts.append(f'Zones: {exclusion_count}')
        
        if processor.object_ownership_map is not None:
            unique_objects = len(np.unique(processor.object_ownership_map)) - 1  # -1 for background
            if unique_objects > 1:
                separation_info_parts.append(f'Objects: {unique_objects}')
        
        if separation_info_parts:
            separation_info = ' | '.join(separation_info_parts) + ' (separated at pixel level)'
            cv2.putText(vis, separation_info, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,100), 1)
        
        vw.write(vis)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{actual} frames | Ellipse tracking: {ellipse_tracked}")

    vw.release()
    
    print(f"\n=== ENHANCED VIDEO CREATION COMPLETE ===")
    print(f"Video saved to: {output_path}")
    print(f"Video specs: {W}x{H}, {VIDEO_CONFIG['fps']} fps, {actual} frames")
    print(f"🟢 ELLIPTICAL AOI TRACKING STATS:")
    total_det = tracked + new
    print(f"  - Total detected frames: {total_det}")
    print(f"  - Ellipse-guided frames: {ellipse_tracked}")
    print(f"  - Ellipse effectiveness: {ellipse_tracked/max(1,total_det)*100:.1f}%")
    print(f"  - Lost/searching frames: {lost}")
    if total_det > 0:
        print(f"  - Detection success rate: {total_det/actual*100:.1f}%")
    
    return output_path

# --- Backwards compatibility shim for legacy callers ---
def detailed_sonar_dvl_comparison(
    distance_results,
    raw_data,
    sonar_coverage_m: float = 5.0,
    sonar_image_size: int = 700,
    window_size: int = 15,
):
    """
    Legacy API preserved. Delegates to compare_sonar_vs_dvl and returns (fig, stats).
    """
    return compare_sonar_vs_dvl(
        distance_results=distance_results,
        raw_data=raw_data,
        distance_measurements=None,   # legacy version didn't accept this
        sonar_coverage_m=sonar_coverage_m,
        sonar_image_size=sonar_image_size,
        window_size=window_size,
    )
