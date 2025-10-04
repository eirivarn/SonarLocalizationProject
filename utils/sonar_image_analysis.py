# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
from utils.sonar_config import IMAGE_PROCESSING_CONFIG, ELONGATION_CONFIG, TRACKING_CONFIG, VIDEO_CONFIG, ConeGridSpec

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

def elapsed_seconds_from_timestamps(stamps: pd.Series | pd.DatetimeIndex | np.ndarray,
                                    estimated_fps: float,
                                    count: int) -> np.ndarray:
    """Return elapsed seconds from timestamps if valid, else an index/fps ramp."""
    try:
        t = pd.to_datetime(stamps)
        if t.isnull().all():
            raise ValueError
        return (t - t[0]).total_seconds()
    except Exception:
        return np.arange(count) / float(max(estimated_fps, 1e-6))

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
    if cfg.get('use_momentum_merging', True):
        return directional_momentum_merge(
            frame_u8,
            search_radius=cfg.get('momentum_search_radius', 3),
            momentum_threshold=cfg.get('momentum_threshold', 0.2),
            momentum_decay=cfg.get('momentum_decay', 0.8),
            momentum_boost=cfg.get('momentum_boost', 1.5),
        )
    else:
        k = cfg.get('blur_kernel_size', (15, 15))
        return cv2.GaussianBlur(frame_u8, k, 0)

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
    }

def score_contour(feat: Dict[str, float], w=ELONGATION_CONFIG) -> float:
    comp = (
        feat['aspect_ratio'] * w['aspect_ratio_weight'] +
        feat['ellipse_elongation'] * w['ellipse_elongation_weight'] +
        (1 - feat['solidity']) * w['solidity_weight'] +
        feat['extent'] * w['extent_weight'] +
        min(feat['aspect_ratio'] / 10.0, 0.5) * w['perimeter_weight']
    )
    comp *= (0.5 + 1.5 * feat['straightness'])  # straightness boost
    return float(feat['area'] * comp)

def select_best_contour(contours, cfg_img=IMAGE_PROCESSING_CONFIG, cfg_track=TRACKING_CONFIG,
                        aoi: Optional[Tuple[int,int,int,int]] = None) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    min_area = float(cfg_img.get('min_contour_area', 100))
    boost = float(cfg_track.get('aoi_boost_factor', 1.0))
    best, best_feat, best_score = None, None, 0.0
    aoi_count, total = 0, 0

    for c in contours or []:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        total += 1
        feat = compute_contour_features(c)
        s = score_contour(feat)
        if aoi is not None and rects_overlap(feat['rect'], aoi):
            s *= boost
            aoi_count += 1
        if s > best_score:
            best, best_feat, best_score = c, feat, s

    stats = {'total_contours': total, 'aoi_contours': aoi_count, 'best_score': best_score}
    return best, (best_feat or {}), stats

# ============================ Public: distance+angle extraction ============================

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
    
    # Use same contour selection as video
    best, _, _ = select_best_contour(contours, IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG, aoi)
    H, W = frame_u8.shape[:2]
    return _distance_angle_from_contour(best, W, H)

# ============================ Public: per-frame processing (video overlay) ============================

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

    # choose contour
    best, feat, stats = select_best_contour(contours, IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG, aoi)

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

def analyze_red_line_distance_over_time(npz_file_index: int = 0,
                                        frame_start: int = 0,
                                        frame_count: Optional[int] = None,
                                        frame_step: int = 1,
                                        save_error_frames: bool = False,
                                        error_frames_dir: str | None = None) -> pd.DataFrame:
    """Analyze red line distance over time - uses SAME calculation as video display.

    If error_frames_dir is None the directory will be created under
    EXPORTS_DIR_DEFAULT / EXPORTS_SUBDIRS['outputs'] / 'error_frames'.
    """
    print("=== RED LINE DISTANCE ANALYSIS OVER TIME ===")
    files = get_available_npz_files()
    if npz_file_index >= len(files):
        print(f"Error: NPZ file index {npz_file_index} not available")
        return None
    npz_file = files[npz_file_index]
    print(f"Analyzing: {npz_file}")

    cones, ts, extent, _ = load_cone_run_npz(npz_file)
    x_min, x_max, y_min, y_max = extent
    height_m = y_max - y_min
    H = cones.shape[1]  # assuming cones[T,H,W]
    px2m_y = height_m / H
    T = len(cones)
    if frame_count is None:
        frame_count = T - frame_start
    actual = int(min(frame_count, max(0, (T - frame_start)) // max(1, frame_step)))
    print(f"Total frames available: {T}")
    print(f"Analyzing frames {frame_start} to {frame_start + actual * frame_step} (step={frame_step})")

    # Set up error frame saving
    if save_error_frames:
        from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
        if error_frames_dir is not None:
            error_dir = Path(error_frames_dir)
        else:
            error_dir = Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('outputs', 'outputs') / 'error_frames'
        error_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving error frames to: {error_dir}")

    out = {'frame_index': [], 'timestamp': [], 'distance_pixels': [], 'angle_degrees': [], 'detection_success': [], 'distance_meters': []}
    success = 0
    aoi = None  # Track AOI across frames like video does

    for i in range(actual):
        idx = frame_start + i * frame_step
        if idx >= T: break
        frame_u8 = to_uint8_gray(cones[idx])
        # Use the SAME function as video display - with AOI tracking
        distance_pixels, angle = get_red_line_distance_and_angle(frame_u8, aoi)
        
        # Update AOI for next frame (same logic as video)
        if distance_pixels is not None:
            _, edges_proc = preprocess_edges(frame_u8, IMAGE_PROCESSING_CONFIG)
            contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Apply same AOI expansion as video
            expanded_aoi = None
            if aoi is not None:
                ax, ay, aw, ah = aoi
                exp = int(TRACKING_CONFIG.get('aoi_expansion_pixels', 10))
                H, W = frame_u8.shape[:2]
                expanded_aoi = (max(0, ax-exp), max(0, ay-exp),
                               min(W - max(0, ax-exp), aw + 2*exp),
                               min(H - max(0, ay-exp), ah + 2*exp))
            
            best, feat, _ = select_best_contour(contours, IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG, expanded_aoi)
            if best is not None:
                x, y, w, h = feat['rect']
                aoi = (x, y, w, h)  # Update AOI for next frame
        
        out['frame_index'].append(idx)
        out['timestamp'].append(ts[idx] if idx < len(ts) else idx)
        out['distance_pixels'].append(distance_pixels)
        out['angle_degrees'].append(angle)
        ok = distance_pixels is not None
        out['detection_success'].append(ok)
        out['distance_meters'].append(y_min + distance_pixels * px2m_y if ok else None)
        if ok: success += 1
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{actual} frames (Success rate: {success/(i+1)*100:.1f}%)")

    df = pd.DataFrame(out)
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total frames processed: {len(df)}")
    print(f"Successful detections: {success} ({(success/len(df)*100 if len(df) else 0):.1f}%)")

    if success:
        vals = df.loc[df['detection_success'], 'distance_pixels']
        print("Distance statistics (pixels):")
        print(f"  - Mean: {vals.mean():.2f}")
        print(f"  - Std:  {vals.std():.2f}")
        print(f"  - Min:  {vals.min():.2f}")
        print(f"  - Max:  {vals.max():.2f}")
        print(f"  - Range:{(vals.max()-vals.min()):.2f}")
    return df


def analyze_red_line_distance_from_sonar_csv(TARGET_BAG: str,
                                             EXPORTS_FOLDER: Path,
                                             frame_start: int = 0,
                                             frame_count: Optional[int] = None,
                                             frame_step: int = 1,
                                             # --- sonar processing params (same as video generation) ---
                                             FOV_DEG: float = 120.0,
                                             RANGE_MIN_M: float = 0.5,
                                             RANGE_MAX_M: float = 10.0,
                                             DISPLAY_RANGE_MAX_M: float = 10.0,
                                             FLIP_BEAMS: bool = True,
                                             FLIP_RANGE: bool = False,
                                             USE_ENHANCED: bool = True,
                                             ENH_SCALE: str = "db",
                                             ENH_TVG: str = "amplitude",
                                             ENH_ALPHA_DB_PER_M: float = 0.0,
                                             ENH_R0: float = 1e-2,
                                             ENH_P_LOW: float = 1.0,
                                             ENH_P_HIGH: float = 99.5,
                                             ENH_GAMMA: float = 0.9,
                                             ENH_ZERO_AWARE: bool = True,
                                             ENH_EPS_LOG: float = 1e-6,
                                             CONE_W: int = 900,
                                             CONE_H: int = 700) -> pd.DataFrame:
    """
    Analyze red line distance using the same sonar processing pipeline as video generation.
    This ensures consistency between video visualization and analysis.
    """
    print("=== RED LINE DISTANCE ANALYSIS FROM SONAR CSV ===")
    print(f"🎯 Target Bag: {TARGET_BAG}")
    print(f"   Cone Size: {CONE_W}x{CONE_H}")
    print(f"   Range: {RANGE_MIN_M}-{DISPLAY_RANGE_MAX_M}m | FOV: {FOV_DEG}°")

    from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    sonar_csv_file = (Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('by_bag','by_bag')) / f"sensor_sonoptix_echo_image__{TARGET_BAG}_video.csv" if EXPORTS_DIR_DEFAULT else EXPORTS_FOLDER / "by_bag" / f"sensor_sonoptix_echo_image__{TARGET_BAG}_video.csv"
    if not sonar_csv_file.exists():
        print(f"❌ ERROR: Sonar CSV not found: {sonar_csv_file}")
        return None

    from utils.sonar_utils import load_df, get_sonoptix_frame, apply_flips
    from utils.sonar_config import ENHANCE_DEFAULTS

    print(f"   Loading sonar data: {sonar_csv_file.name}")
    df = load_df(sonar_csv_file)
    if "ts_utc" not in df.columns:
        if "t" not in df.columns:
            print("❌ ERROR: Missing timestamp column")
            return None
        df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")

    T = len(df)
    if frame_count is None:
        frame_count = T - frame_start
    actual = int(min(frame_count, max(0, (T - frame_start)) // max(1, frame_step)))
    print(f"Total frames available: {T}")
    print(f"Analyzing frames {frame_start} to {frame_start + actual * frame_step} (step={frame_step})")

    out = {'frame_index': [], 'timestamp': [], 'distance_pixels': [], 'angle_degrees': [], 'detection_success': []}
    success = 0

    for i in range(actual):
        idx = frame_start + i * frame_step
        if idx >= T: break

        try:
            # Process sonar data exactly like video generation (up to cone creation)
            M0 = get_sonoptix_frame(df, idx)
            if M0 is None:
                continue

            M = apply_flips(M0, flip_range=FLIP_RANGE, flip_beams=FLIP_BEAMS)

            if USE_ENHANCED:
                from utils.sonar_utils import enhance_intensity
                Z = enhance_intensity(
                    M, RANGE_MIN_M, RANGE_MAX_M, scale=ENH_SCALE, tvg=ENH_TVG,
                    alpha_db_per_m=ENH_ALPHA_DB_PER_M, r0=ENH_R0,
                    p_low=ENH_P_LOW, p_high=ENH_P_HIGH, gamma=ENH_GAMMA,
                    zero_aware=ENH_ZERO_AWARE, eps_log=ENH_EPS_LOG)
            else:
                Z = M

            # Create cone visualization (same as video generation)
            cone, extent = cone_raster_like_display_cell(
                Z, FOV_DEG, RANGE_MIN_M, RANGE_MAX_M, DISPLAY_RANGE_MAX_M, CONE_W, CONE_H
            )
            cone = np.flipud(cone)  # Same vertical flip as video generation

            # Compute pixel-to-meter conversion from extent (same as notebook)
            x_min, x_max, y_min, y_max = extent
            width_m = x_max - x_min
            height_m = y_max - y_min
            H, W = cone.shape
            px2m_x = width_m / W
            px2m_y = height_m / H
            pixels_to_meters = 0.5 * (px2m_x + px2m_y)

            # Convert to grayscale for analysis (this is our "raw input")
            cone_normalized = np.ma.masked_invalid(cone)
            cone_u8 = (cone_normalized * 255).astype(np.uint8)

            # Apply line detection algorithm
            distance, angle = get_red_line_distance_and_angle(cone_u8)

            out['frame_index'].append(idx)
            out['timestamp'].append(df.loc[idx, "ts_utc"])
            out['distance_pixels'].append(distance)
            out['angle_degrees'].append(angle)
            ok = distance is not None
            out['detection_success'].append(ok)
            out['distance_meters'].append(y_min + distance * px2m_y if ok else None)
            if ok: success += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{actual} frames (Success rate: {success/(i+1)*100:.1f}%)")

        except Exception as e:
            print(f"❌ Error processing frame {idx}: {e}")
            continue

    df_result = pd.DataFrame(out)
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total frames processed: {len(df_result)}")
    print(f"Successful detections: {success} ({(success/len(df_result)*100 if len(df_result) else 0):.1f}%)")

    if success:
        vals = df_result.loc[df_result['detection_success'], 'distance_meters']
        print("Distance statistics (meters):")
        print(f"  - Mean: {vals.mean():.2f}")
        print(f"  - Std:  {vals.std():.2f}")
        print(f"  - Min:  {vals.min():.2f}")
        print(f"  - Max:  {vals.max():.2f}")
        print(f"  - Range:{(vals.max()-vals.min()):.2f}")
    return df_result

# ============================ Public: plotting helpers ============================

def plot_time_based_analysis(distance_results: pd.DataFrame,
                             pixels_to_meters_avg: float = 0.01,
                             estimated_fps: float = 15):
    valid = distance_results[distance_results['detection_success']].copy()
    if 'distance_meters' in valid.columns:
        valid['distance_meters'] = valid['distance_meters']
    else:
        valid['distance_meters'] = valid['distance_pixels'] * pixels_to_meters_avg
    t = elapsed_seconds_from_timestamps(valid.get('timestamp', None), estimated_fps, len(valid))
    win = max(5, len(valid) // 20)
    mv = valid['distance_meters'].rolling(window=win, center=True).mean()
    sd = valid['distance_meters'].rolling(window=win, center=True).std()

    plt.figure(figsize=(14, 7))
    plt.plot(t, valid['distance_meters'], 'lightblue', alpha=0.5, label='Raw data')
    plt.plot(t, mv, 'darkblue', linewidth=2, label=f'Moving Avg (n={win})')
    plt.fill_between(t, mv - sd, mv + sd, alpha=0.3, color='blue', label='±1 Std Dev')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('Distance (meters)')
    plt.title('Red Line Distance Over Time (Time-Based Analysis)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def plot_real_world_distance_analysis(distance_results: pd.DataFrame,
                                      image_shape=(700, 900),
                                      sonar_coverage_meters=5.0):
    valid = distance_results[distance_results['detection_success']].copy()
    if 'distance_meters' in valid.columns:
        dist = valid['distance_meters']
    else:
        H, W = image_shape
        px2m = (sonar_coverage_meters / W + sonar_coverage_meters / H) / 2.0
        dist = valid['distance_pixels'] * px2m
        valid['distance_meters'] = dist
    win = max(5, len(valid)//20)
    mv = dist.rolling(window=win, center=True).mean()
    sd = dist.rolling(window=win, center=True).std()

    # Create interactive plots using plotly
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Red Line Distance vs Frame Index',
                'Red Line Distance vs Time',
                'Red Line Distance Distribution',
                'Distance Trends and Variability'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )

        # Top-left: Distance vs Frame Index
        fig.add_trace(
            go.Scatter(
                x=valid['frame_index'],
                y=dist,
                mode='lines+markers',
                name='Distance',
                line=dict(color='blue', width=2),
                marker=dict(color='red', size=6),
                showlegend=False
            ),
            row=1, col=1
        )

        # Top-right: Distance vs Time (using frame index as proxy for time)
        fig.add_trace(
            go.Scatter(
                x=valid['frame_index'],
                y=dist,
                mode='lines+markers',
                name='Distance over Time',
                line=dict(color='green', width=2),
                marker=dict(color='red', size=6),
                showlegend=False
            ),
            row=1, col=2
        )

        # Bottom-left: Histogram
        fig.add_trace(
            go.Histogram(
                x=dist,
                nbinsx=max(5, min(30, len(dist)//5)),
                name='Distance Distribution',
                marker_color='lightcoral',
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )

        # Add mean and median lines to histogram
        fig.add_vline(
            x=dist.mean(),
            line=dict(color='red', dash='dash', width=2),
            annotation_text=f'Mean: {dist.mean():.2f}m',
            row=2, col=1
        )
        fig.add_vline(
            x=dist.median(),
            line=dict(color='green', dash='dash', width=2),
            annotation_text=f'Median: {dist.median():.2f}m',
            row=2, col=1
        )

        # Bottom-right: Trends and variability
        fig.add_trace(
            go.Scatter(
                x=valid['frame_index'],
                y=dist,
                mode='lines',
                name='Raw data',
                line=dict(color='lightcoral', width=1),
                opacity=0.5
            ),
            row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=valid['frame_index'],
                y=mv,
                mode='lines',
                name=f'Moving Avg (window={win})',
                line=dict(color='darkred', width=3)
            ),
            row=2, col=2
        )

        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=valid['frame_index'].tolist() + valid['frame_index'].tolist()[::-1],
                y=(mv + sd).tolist() + (mv - sd).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text='Red Line Distance Analysis Over Time (Real-World Distances)',
            title_font_size=16,
            title_font_weight='bold',
            height=800,
            showlegend=True
        )

        # Update axis labels
        fig.update_xaxes(title_text='Frame Index', row=1, col=1)
        fig.update_xaxes(title_text='Frame Index', row=1, col=2)
        fig.update_xaxes(title_text='Distance (m)', row=2, col=1)
        fig.update_xaxes(title_text='Frame Index', row=2, col=2)

        fig.update_yaxes(title_text='Distance (m)', row=1, col=1)
        fig.update_yaxes(title_text='Distance (m)', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        fig.update_yaxes(title_text='Distance (m)', row=2, col=2)

        # Add statistics annotation
        stats_text = f'Mean: {dist.mean():.2f}m<br>Std: {dist.std():.2f}m<br>Range: {dist.min():.2f}-{dist.max():.2f}m'
        fig.add_annotation(
            text=stats_text,
            xref='paper', yref='paper',
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor='wheat',
            opacity=0.8
        )

        fig.show()

    except ImportError:
        print("⚠️ Plotly not available, falling back to matplotlib plots")
        # Fallback to original matplotlib plots
        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Red Line Distance Analysis Over Time (Real-World Distances)', fontsize=16, fontweight='bold')

        ax[0,0].plot(valid['frame_index'], dist, 'b-', alpha=0.7, linewidth=1)
        ax[0,0].scatter(valid['frame_index'], dist, c='red', s=20, alpha=0.6)
        ax[0,0].set_title('Red Line Distance vs Frame Index'); ax[0,0].set_xlabel('Frame'); ax[0,0].set_ylabel('Distance (m)')
        ax[0,0].grid(True, alpha=0.3)
        ax[0,0].text(0.02, 0.98, f'Mean: {dist.mean():.2f}m\nStd: {dist.std():.2f}m\nRange: {dist.min():.2f}-{dist.max():.2f}m',
                     transform=ax[0,0].transAxes, fontsize=9, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax[0,1].plot(valid['frame_index'], dist, 'g-', alpha=0.7, linewidth=1)
        ax[0,1].scatter(valid['frame_index'], dist, c='red', s=20, alpha=0.6)
        ax[0,1].set_title('Red Line Distance vs Time'); ax[0,1].set_xlabel('Frame'); ax[0,1].set_ylabel('Distance (m)')
        ax[0,1].grid(True, alpha=0.3)

        nbins = max(5, min(30, len(dist)//5))
        ax[1,0].hist(dist, bins=nbins, alpha=0.7, color='lightcoral', edgecolor='black')
        ax[1,0].axvline(dist.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {dist.mean():.2f}m')
        ax[1,0].axvline(dist.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {dist.median():.2f}m')
        ax[1,0].set_title('Red Line Distance Distribution'); ax[1,0].legend(); ax[1,0].grid(True, alpha=0.3)

        ax[1,1].plot(valid['frame_index'], dist, 'lightcoral', alpha=0.5, label='Raw data')
        ax[1,1].plot(valid['frame_index'], mv, 'darkred', linewidth=2, label=f'Moving Avg (window={win})')
        ax[1,1].fill_between(valid['frame_index'], mv - sd, mv + sd, alpha=0.3, color='red', label='±1 Std Dev')
        ax[1,1].set_title('Distance Trends and Variability'); ax[1,1].legend(); ax[1,1].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

# ============================ Public: SONAR vs DVL (merged/DRY) ============================

def compare_sonar_vs_dvl(distance_results: pd.DataFrame,
                         raw_data: Dict[str, pd.DataFrame],
                         distance_measurements: Dict[str, pd.DataFrame] | None = None,
                         sonar_coverage_m: float = 5.0,
                         sonar_image_size: int = 700,
                         window_size: int = 15):
    """
    Unified, DRY version of the comparison that also reproduces the detailed views.
    - Uses shared smoothing and time alignment.
    """
    if raw_data is None or 'navigation' not in raw_data or raw_data['navigation'] is None:
        print("❌ No DVL navigation data available.")
        return None, {'error': 'no_navigation_data'}
    if distance_results is None:
        print("❌ Sonar distance_results not found.")
        return None, {'error': 'no_distance_results'}

    nav = raw_data['navigation'].copy()
    nav['timestamp'] = pd.to_datetime(nav['timestamp'], errors='coerce')
    nav = nav.dropna(subset=['timestamp'])
    nav['relative_time'] = (nav['timestamp'] - nav['timestamp'].min()).dt.total_seconds()

    # Prefer explicit sonar measurement input if provided, else convert from distance_results
    if distance_measurements and isinstance(distance_measurements.get('sonar'), pd.DataFrame):
        sonar = distance_measurements['sonar'].copy()
        if 'frame_index' not in sonar and 'frame_idx' in sonar:
            sonar = sonar.rename(columns={'frame_idx': 'frame_index'})
        if 'distance_pixels' not in sonar and 'distance' in sonar:
            sonar = sonar.rename(columns={'distance': 'distance_pixels'})
    else:
        sonar = distance_results.copy()

    if 'distance_meters' in sonar.columns:
        sonar['distance_meters_raw'] = sonar['distance_meters']
    else:
        ppm = float(sonar_image_size) / float(sonar_coverage_m)
        sonar['distance_meters'] = sonar['distance_pixels'] / ppm
        sonar['distance_meters_raw'] = sonar['distance_meters']

    # stretch sonar time to nav span
    dvl_duration = float(max(1.0, (nav['relative_time'].max() - nav['relative_time'].min())))
    N = max(1, len(sonar) - 1)
    if 'frame_index' not in sonar:
        sonar['frame_index'] = np.arange(len(sonar))
    sonar['synthetic_time'] = (sonar['frame_index'] / float(N)) * dvl_duration

    # plots
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle('🔄 SONAR vs DVL DISTANCE COMPARISON (WITH SMOOTHING)', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.plot(sonar['synthetic_time'], sonar['distance_meters_raw'], 'r-', linewidth=1, alpha=0.3, label='Sonar Raw')
    ax1.plot(sonar['synthetic_time'], sonar['distance_meters'], 'r-', linewidth=2, alpha=0.8, label='Sonar Smoothed')
    ax1.plot(nav['relative_time'], nav['NetDistance'], 'b-', linewidth=2, alpha=0.8, label='DVL NetDistance')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Distance (m)'); ax1.set_title('Distance Over Time'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    sl = slice(100, min(200, len(sonar)))
    ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters_raw'].iloc[sl], 'gray', linewidth=1, alpha=0.7, label='Raw')
    ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters_mavg'].iloc[sl], 'orange', linewidth=1.5, alpha=0.8, label='Moving Avg')
    ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters_savgol'].iloc[sl], 'red', linewidth=2, alpha=0.9, label='Savitzky-Golay')
    ax2.plot(sonar['synthetic_time'].iloc[sl], sonar['distance_meters_gaussian'].iloc[sl], 'purple', linewidth=1.5, alpha=0.8, label='Gaussian')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Distance (m)'); ax2.set_title('Smoothing Methods (Detail)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    ax3.hist(sonar['distance_meters'], bins=30, alpha=0.7, color='red', label=f'Sonar (n={len(sonar)})', density=True)
    ax3.hist(nav['NetDistance'], bins=30, alpha=0.7, color='blue', label=f'DVL (n={len(nav)})', density=True)
    ax3.set_xlabel('Distance (m)'); ax3.set_ylabel('Density'); ax3.set_title('Distribution Comparison'); ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    stats_df = pd.DataFrame({
        'Measurement': ['Sonar Red Line', 'DVL Navigation'],
        'Count': [len(sonar), len(nav)],
        'Mean (m)': [float(np.nanmean(sonar['distance_meters'])), float(nav['NetDistance'].mean())],
        'Std (m)': [float(np.nanstd(sonar['distance_meters'])), float(nav['NetDistance'].std())],
        'Min (m)': [float(np.nanmin(sonar['distance_meters'])), float(nav['NetDistance'].min())],
        'Max (m)': [float(np.nanmax(sonar['distance_meters'])), float(nav['NetDistance'].max())],
    })
    stats_df['CV (%)'] = stats_df['Std (m)'] / stats_df['Mean (m)'] * 100
    ax4.axis('off')
    ax4.text(0.05, 0.95, stats_df.round(3).to_string(index=False),
             transform=ax4.transAxes, fontfamily='monospace', fontsize=10, va='top')
    ax4.set_title('Statistical Comparison')

    ax5 = axes[1, 1]
    x = np.arange(len(stats_df)); w = 0.35
    ax5.bar(x - w/2, stats_df['Mean (m)'], w, label='Mean', alpha=0.8, color=['red','blue'])
    ax5.bar(x + w/2, stats_df['Std (m)'], w, label='Std', alpha=0.8, color=['lightcoral','lightblue'])
    ax5.set_xticks(x); ax5.set_xticklabels(stats_df['Measurement']); ax5.set_title('Mean ± Std'); ax5.legend(); ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    noise_levels = [
        float(np.nanstd(sonar['distance_meters_raw'])),
        float(np.nanstd(sonar['distance_meters_mavg'])),
        float(np.nanstd(sonar['distance_meters_savgol'])),
        float(np.nanstd(sonar['distance_meters_gaussian'])),
    ]
    bars = ax6.bar(['Raw', 'Moving Avg', 'Savitzky-Golay', 'Gaussian'], noise_levels, alpha=0.7)
    ax6.set_ylabel('Std (m)'); ax6.set_title('Noise Reduction by Smoothing'); ax6.grid(True, alpha=0.3)
    for b, v in zip(bars, noise_levels):
        ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f'{v:.3f}m', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(); plt.show()

    sonar_mean = float(np.nanmean(sonar['distance_meters']))
    dvl_mean = float(nav['NetDistance'].mean())
    ratio = sonar_mean / dvl_mean if dvl_mean else np.nan
    sonar_span = float(np.nanmax(sonar['synthetic_time'])) if len(sonar) else 0.0
    dvl_span = float(nav['relative_time'].max()) if len(nav) else 0.0

    print("\n📊 DETAILED COMPARISON STATISTICS:")
    print("="*50)
    print(stats_df.round(3).to_string(index=False))
    print("\n🔍 SCALE ANALYSIS:")
    print(f"   Sonar mean: {sonar_mean:.3f} m")
    print(f"   DVL mean:   {dvl_mean:.3f} m")
    print(f"   Scale ratio (Sonar/DVL): {ratio:.3f}x")
    print("\n⏱️ TIME ANALYSIS:")
    print(f"   Sonar duration (stretched): {sonar_span:.1f}s ({len(sonar)} frames)")
    print(f"   DVL duration:               {dvl_span:.1f}s ({len(nav)} records)")
    print(f"   ✅ Temporal alignment: Both now span ~{dvl_span:.1f}s")

    return fig, {
        'sonar_mean_m': sonar_mean,
        'dvl_mean_m': dvl_mean,
        'scale_ratio': ratio,
        'sonar_duration_stretched_s': sonar_span,
        'dvl_duration_s': dvl_span,
    }

# ============================ Interactive Plotly Version ============================

def interactive_distance_comparison(distance_results: pd.DataFrame,
                                   raw_data: Dict[str, pd.DataFrame],
                                   distance_measurements: Dict[str, pd.DataFrame] | None = None,
                                   sonar_coverage_m: float = 5.0,
                                   sonar_image_size: int = 700):
    """
    Interactive plotly version showing distance over time with raw/smoothed sonar and DVL data.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("❌ Plotly not available. Install with: pip install plotly")
        return None, {'error': 'plotly_not_available'}

    if raw_data is None or 'navigation' not in raw_data or raw_data['navigation'] is None:
        return None, {'error': 'no_navigation_data'}
    if distance_results is None:
        return None, {'error': 'no_distance_results'}

    nav = raw_data['navigation'].copy()
    nav['timestamp'] = pd.to_datetime(nav['timestamp'], errors='coerce')
    nav = nav.dropna(subset=['timestamp'])
    nav['relative_time'] = (nav['timestamp'] - nav['timestamp'].min()).dt.total_seconds()

    # Prepare sonar data
    if distance_measurements and isinstance(distance_measurements.get('sonar'), pd.DataFrame):
        sonar = distance_measurements['sonar'].copy()
        if 'frame_index' not in sonar and 'frame_idx' in sonar:
            sonar = sonar.rename(columns={'frame_idx': 'frame_index'})
        if 'distance_pixels' not in sonar and 'distance' in sonar:
            sonar = sonar.rename(columns={'distance': 'distance_pixels'})
    else:
        sonar = distance_results.copy()

    if 'distance_meters' in sonar.columns:
        sonar['distance_meters_raw'] = sonar['distance_meters']
    else:
        ppm = float(sonar_image_size) / float(sonar_coverage_m)
        sonar['distance_meters'] = sonar['distance_pixels'] / ppm
        sonar['distance_meters_raw'] = sonar['distance_meters']

    # Apply smoothing to sonar data
    from scipy.signal import savgol_filter
    window_size = min(15, len(sonar) // 2 * 2 + 1)  # Ensure odd window size
    if len(sonar) > window_size:
        sonar['distance_meters_smoothed'] = savgol_filter(sonar['distance_meters'], window_size, 3)
    else:
        sonar['distance_meters_smoothed'] = sonar['distance_meters']

    # Stretch sonar time to match DVL duration
    dvl_duration = float(max(1.0, (nav['relative_time'].max() - nav['relative_time'].min())))
    N = max(1, len(sonar) - 1)
    if 'frame_index' not in sonar:
        sonar['frame_index'] = np.arange(len(sonar))
    sonar['time_seconds'] = (sonar['frame_index'] / float(N)) * dvl_duration

    # Check if pitch data is available
    has_pitch = 'NetPitch' in nav.columns and nav['NetPitch'].notna().any()

    # Create interactive plot with subplots
    if has_pitch:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Distance Comparison", "Pitch Comparison"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
    else:
        fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Add distance traces
    fig.add_trace(
        go.Scatter(
            x=sonar['time_seconds'],
            y=sonar['distance_meters_raw'],
            mode='lines',
            name='Sonar Raw Distance',
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            hovertemplate='Time: %{x:.1f}s<br>Distance: %{y:.3f}m<extra>Sonar Raw</extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sonar['time_seconds'],
            y=sonar['distance_meters_smoothed'],
            mode='lines',
            name='Sonar Smoothed Distance',
            line=dict(color='rgba(255, 0, 0, 1)', width=3),
            hovertemplate='Time: %{x:.1f}s<br>Distance: %{y:.3f}m<extra>Sonar Smoothed</extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=nav['relative_time'],
            y=nav['NetDistance'],
            mode='lines',
            name='DVL Distance',
            line=dict(color='rgba(0, 0, 255, 1)', width=3),
            hovertemplate='Time: %{x:.1f}s<br>Distance: %{y:.3f}m<extra>DVL Distance</extra>'
        ),
        row=1, col=1
    )

    # Add pitch traces if available
    if has_pitch:
        # Sonar pitch (from angle_degrees, convert to similar scale as DVL)
        sonar_pitch = sonar.get('angle_degrees', pd.Series([0] * len(sonar)))
        
        fig.add_trace(
            go.Scatter(
                x=sonar['time_seconds'],
                y=sonar_pitch,
                mode='lines',
                name='Sonar Pitch (from contour)',
                line=dict(color='rgba(255, 165, 0, 1)', width=3),
                hovertemplate='Time: %{x:.1f}s<br>Pitch: %{y:.1f}°<extra>Sonar Pitch</extra>'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=nav['relative_time'],
                y=np.degrees(nav['NetPitch']),  # Convert radians to degrees
                mode='lines',
                name='DVL Pitch',
                line=dict(color='rgba(0, 128, 0, 1)', width=3),
                hovertemplate='Time: %{x:.1f}s<br>Pitch: %{y:.1f}°<extra>DVL Pitch</extra>'
            ),
            row=2, col=1
        )

    # Update layout
    if has_pitch:
        fig.update_layout(
            title="🔄 Interactive Distance & Pitch Comparison: Sonar vs DVL",
            hovermode='x unified',
            height=800
        )
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Distance (meters)", row=1, col=1)
        fig.update_yaxes(title_text="Pitch (degrees)", row=2, col=1)
    else:
        fig.update_layout(
            title="🔄 Interactive Distance Comparison: Sonar vs DVL",
            xaxis_title="Time (seconds)",
            yaxis_title="Distance (meters)",
            hovermode='x unified'
        )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Calculate basic stats
    sonar_mean = float(np.nanmean(sonar['distance_meters_smoothed']))
    dvl_mean = float(nav['NetDistance'].mean())

    return fig, {
        'sonar_mean_m': sonar_mean,
        'dvl_mean_m': dvl_mean,
        'sonar_frames': len(sonar),
        'dvl_records': len(nav),
        'time_span_s': dvl_duration
    }

# ============================ Visualization helpers (kept but DRY inside) ============================

def basic_image_processing_pipeline(img_u8: np.ndarray, show=True, figsize=(15, 10)) -> Dict[str, np.ndarray]:
    blurred = cv2.GaussianBlur(img_u8, (15, 15), 0)
    edges = cv2.Canny(img_u8, 50, 150)
    edges_blurred = cv2.Canny(blurred, 50, 150)
    _, thresh = cv2.threshold(img_u8, 127, 255, cv2.THRESH_BINARY)
    diff = cv2.absdiff(edges, edges_blurred)

    res = {'original': img_u8, 'blurred': blurred, 'edges': edges, 'edges_blurred': edges_blurred, 'thresh': thresh, 'diff': diff}
    if show:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes[0,0].imshow(img_u8, cmap='gray');       axes[0,0].set_title('Original');        axes[0,0].axis('off')
        axes[0,1].imshow(blurred, cmap='gray');      axes[0,1].set_title('Gaussian Blur');   axes[0,1].axis('off')
        axes[0,2].imshow(thresh, cmap='gray');       axes[0,2].set_title('Binary Threshold');axes[0,2].axis('off')
        axes[1,0].imshow(edges, cmap='gray');        axes[1,0].set_title('Canny Edges');     axes[1,0].axis('off')
        axes[1,1].imshow(edges_blurred, cmap='gray');axes[1,1].set_title('Canny + Blur');    axes[1,1].axis('off')
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
    ax[1].imshow(proc, cmap='gray');             ax[1].set_title('Momentum Merge' if IMAGE_PROCESSING_CONFIG['use_momentum_merging'] else f'Gaussian {IMAGE_PROCESSING_CONFIG["blur_kernel_size"]}'); ax[1].axis('off')
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
    if IMAGE_PROCESSING_CONFIG.get('use_momentum_merging', True):
        print(f"  Image Processing: MOMENTUM MERGING (radius={IMAGE_PROCESSING_CONFIG['momentum_search_radius']}, "
              f"threshold={IMAGE_PROCESSING_CONFIG['momentum_threshold']}, decay={IMAGE_PROCESSING_CONFIG['momentum_decay']}), "
              f"canny=({IMAGE_PROCESSING_CONFIG['canny_low_threshold']}, {IMAGE_PROCESSING_CONFIG['canny_high_threshold']}), "
              f"min_area={IMAGE_PROCESSING_CONFIG['min_contour_area']}")
    else:
        print(f"  Image Processing: blur={IMAGE_PROCESSING_CONFIG['blur_kernel_size']}, "
              f"canny=({IMAGE_PROCESSING_CONFIG['canny_low_threshold']}, {IMAGE_PROCESSING_CONFIG['canny_high_threshold']}), "
              f"min_area={IMAGE_PROCESSING_CONFIG['min_contour_area']}")
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
