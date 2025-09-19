# classical_segmentation_utils.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json, math

import numpy as np
import pandas as pd
import cv2

# ============================ I/O (robust) ============================

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
            try: ts = pd.to_datetime(z["ts"], utc=True)
            except Exception: ts = pd.to_datetime(np.asarray(z["ts"], dtype="int64"), unit="s", utc=True)
        else:
            # try in meta
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

# ============================ helpers ============================

def gaussian_blur01(img01: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Blur [0,1] image with Gaussian. ksize must be odd."""
    if ksize % 2 == 0: ksize += 1
    x = np.nan_to_num(img01, nan=0.0)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return cv2.GaussianBlur(x, (ksize, ksize), sigmaX=float(sigma), borderType=cv2.BORDER_REPLICATE)

def segment_percentile(
    img01: np.ndarray,
    p: float = 99.5,
    min_area_px: int = 20,
    open_close: int = 1
) -> np.ndarray:
    """
    Simple segmentation by percentile threshold on positives, then morphology.
    Returns binary mask (H,W).
    """
    X = np.nan_to_num(img01, nan=0.0).astype(np.float32)
    pos = X[X > 0]
    thr = float(np.percentile(pos, p)) if pos.size else 1.0
    mask = (X >= thr)

    m = (mask.astype(np.uint8) * 255)
    if open_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for _ in range(open_close):
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # area filter
    n, labels, stats, _ = cv2.connectedComponentsWithStats((m>0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_area_px)
    return keep[labels]

def extent_px_to_m(extent: Tuple[float,float,float,float], H: int, W: int, i_row: float, j_col: float):
    x_min, x_max, y_min, y_max = extent
    x = x_min + (x_max-x_min) * (j_col/max(W-1,1))
    y = y_min + (y_max-y_min) * (i_row/max(H-1,1))
    return x, y

# ============================ straight-sides scoring ============================

def _approx_edges(mask: np.ndarray, eps_ratio: float = 0.03):
    """
    Get dominant straight edges via contour polygon approximation.
    Returns list of line segments: [(x0,y0,x1,y1,length,theta_rad), ...]
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segs = []
    for c in cnts:
        if len(c) < 5: 
            continue
        peri = cv2.arcLength(c, True)
        eps  = eps_ratio * peri
        poly = cv2.approxPolyDP(c, eps, True)   # Nx1x2
        poly = poly.reshape(-1, 2)
        if len(poly) < 3:
            continue
        # edges between consecutive vertices (closed polygon)
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i+1) % len(poly)]
            dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
            L = float(np.hypot(dx, dy))
            if L < 1.0: 
                continue
            theta = math.atan2(dy, dx)  # radians
            segs.append((p0[0], p0[1], p1[0], p1[1], L, theta))
    # sort by length desc
    segs.sort(key=lambda s: -s[4])
    return segs

def _nearly_parallel(theta1: float, theta2: float, tol_deg: float = 12.0) -> bool:
    """Are two angles nearly parallel (mod π) within tolerance?"""
    d = abs((theta1 - theta2 + math.pi/2) % math.pi - math.pi/2)  # distance on circle modulo pi
    return math.degrees(d) <= tol_deg

def _opposite_sides(seg1, seg2, cx, cy) -> bool:
    """
    Check that two line segments lie on opposite sides of the component centroid.
    We use average normal direction to decide sign separation.
    """
    x0,y0,x1,y1,_,th1 = seg1
    u1 = np.array([math.cos(th1), math.sin(th1)], dtype=float)  # direction
    n1 = np.array([-u1[1], u1[0]], dtype=float)                 # normal
    # segment midpoints
    m1 = np.array([(x0+x1)/2.0, (y0+y1)/2.0])
    x0,y0,x1,y1,_,th2 = seg2
    u2 = np.array([math.cos(th2), math.sin(th2)], dtype=float)
    n2 = np.array([-u2[1], u2[0]], dtype=float)
    m2 = np.array([(x0+x1)/2.0, (y0+y1)/2.0])

    # use average normal (handles parallel case)
    n = (n1 + n2); 
    if np.linalg.norm(n) < 1e-6: n = n1
    n = n / max(np.linalg.norm(n), 1e-6)

    s1 = np.dot(m1 - np.array([cx, cy]), n)
    s2 = np.dot(m2 - np.array([cx, cy]), n)
    return s1 * s2 < 0  # different signs → opposite sides of centroid

def component_has_two_straight_sides(
    mask: np.ndarray,
    *,
    min_edge_len_px: int = 40,
    tol_parallel_deg: float = 12.0,
) -> Tuple[bool, Optional[Tuple]]:
    """
    Decide if a component has two long, near-parallel, opposite edges.
    Returns (ok, (seg1, seg2, cx, cy)) where seg = (x0,y0,x1,y1,L,theta).
    """
    ys, xs = np.where(mask)
    if xs.size == 0:
        return False, None
    cx, cy = xs.mean(), ys.mean()  # centroid in px (x,y)
    segs = _approx_edges(mask, eps_ratio=0.03)
    # take top ~8 longest segments and try pairs
    K = min(8, len(segs))
    best = None
    for i in range(K):
        if segs[i][4] < min_edge_len_px: 
            continue
        for j in range(i+1, K):
            if segs[j][4] < min_edge_len_px: 
                continue
            if not _nearly_parallel(segs[i][5], segs[j][5], tol_parallel_deg):
                continue
            if not _opposite_sides(segs[i], segs[j], cx, cy):
                continue
            # found a valid pair
            best = (segs[i], segs[j], cx, cy)
            return True, best
    return False, None

# ============================ overlay & export ============================

def _cmap_rgb(name="viridis", N=256):
    import matplotlib.cm as cm
    lut = (cm.get_cmap(name, N)(np.linspace(0,1,N))[:, :3] * 255 + 0.5).astype(np.uint8)
    return lut

def gray01_to_rgb(img01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(img01, nan=0.0)
    x = np.clip(x, 0.0, 1.0)
    idx = (x * (len(lut)-1)).astype(np.int32)
    return lut[idx]

def draw_component_edges(rgb: np.ndarray, seg1, seg2, color=(255,80,80)):
    for seg in (seg1, seg2):
        x0,y0,x1,y1,_,_ = seg
        cv2.line(rgb, (int(x0),int(y0)), (int(x1),int(y1)), color, 2, cv2.LINE_AA)

def analyze_run_straight_sides(
    npz_path: str | Path,
    *,
    blur_ksize=5,
    blur_sigma=1.2,
    thr_percentile=99.5,
    min_area_px=50,
    open_close=1,
    edge_min_len_px=40,
    tol_parallel_deg=12.0,
    save_csv: Optional[str | Path] = None,
    save_mp4: Optional[str | Path] = None,
    fps: float = 15.0,
    cmap: str = "viridis",
    alpha_mask: float = 0.25,
    progress: bool = True
) -> pd.DataFrame:
    """
    End-to-end classical pipeline:
      blur → percentile threshold → morphology → components → keep those with two straight, near-parallel sides.
    Writes CSV + optional overlay MP4. Returns DataFrame of matches.
    """
    cones, ts, extent, meta = load_cone_run_npz(npz_path)
    T, H, W = cones.shape

    lut = _cmap_rgb(cmap)
    vw = None
    if save_mp4:
        Path(save_mp4).parent.mkdir(parents=True, exist_ok=True)
        rgb0 = gray01_to_rgb(cones[0], lut)
        vw = cv2.VideoWriter(str(save_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows: List[Dict] = []

    for t in range(T):
        Z = cones[t]
        # 1) blur
        B = gaussian_blur01(Z, ksize=blur_ksize, sigma=blur_sigma)

        # 2) segment
        mask = segment_percentile(B, p=thr_percentile, min_area_px=min_area_px, open_close=open_close)

        # 3) connected components
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        found_pairs = []

        for cid in range(1, n):
            comp = (labels == cid)
            ok, info = component_has_two_straight_sides(
                comp, min_edge_len_px=edge_min_len_px, tol_parallel_deg=tol_parallel_deg
            )
            if not ok:
                continue
            seg1, seg2, cx, cy = info
            # summarize in meters
            # centroid (meters)
            xm, ym = extent_px_to_m(extent, H, W, cy, cx)
            # bbox from contours (px)
            ys, xs = np.where(comp)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            area = int(comp.sum())
            rows.append(dict(
                t_idx=t, ts=str(ts[t]),
                cx_px=float(cx), cy_px=float(cy), x_m=float(xm), y_m=float(ym),
                x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1),
                area_px=area,
                edge1=(float(seg1[0]),float(seg1[1]),float(seg1[2]),float(seg1[3])),
                edge2=(float(seg2[0]),float(seg2[1]),float(seg2[2]),float(seg2[3])),
                edge_len1=float(seg1[4]), edge_len2=float(seg2[4]),
                edge_theta1=float(seg1[5]), edge_theta2=float(seg2[5]),
                method="classical_straight_sides"
            ))
            found_pairs.append((seg1, seg2))

        # 4) overlay frame for video
        if vw is not None:
            rgb = gray01_to_rgb(Z, lut)
            # translucent component mask
            if mask.any():
                overlay = np.zeros_like(rgb)
                overlay[mask] = (255, 140, 0)
                cv2.addWeighted(overlay, float(alpha_mask), rgb, 1.0-float(alpha_mask), 0.0, dst=rgb)
            # draw the two straight edges for each matched component
            for (seg1, seg2) in found_pairs:
                draw_component_edges(rgb, seg1, seg2, color=(0,255,0))
            # timestamp text
            cv2.putText(rgb, f"{ts[t]}  t={t}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            vw.write(rgb[:, :, ::-1])

        if progress and (t % 100 == 0):
            print(f"[analyze] t={t}/{T}  comps={n-1}  matched={len(found_pairs)}")

    if vw is not None:
        vw.release()
        if progress: print(f"[analyze] wrote video → {save_mp4}")

    df = pd.DataFrame(rows)
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        if progress: print(f"[analyze] wrote csv → {save_csv}")

    return df


# === Largest-blob analysis (no straight-line check) ============================

def _largest_component(mask: np.ndarray):
    """
    Return (label_id, area, labels) where label_id is the largest nonzero component.
    If none, returns (0, 0, labels) and labels is zeros.
    """
    if mask.dtype != np.uint8:
        m8 = (mask.astype(np.uint8) & 1)
    else:
        m8 = mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return 0, 0, np.zeros_like(m8, dtype=np.int32)
    # stats[1:, CC_STAT_AREA] are the component areas; pick the largest
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = int(stats[idx, cv2.CC_STAT_AREA])
    return int(idx), area, labels

def analyze_run_largest_blob(
    npz_path: str | Path,
    *,
    blur_ksize=5,
    blur_sigma=1.2,
    thr_percentile=99.5,
    min_area_px=50,
    open_close=1,
    save_csv: Optional[str | Path] = None,
    save_mp4: Optional[str | Path] = None,
    fps: float = 15.0,
    cmap: str = "viridis",
    alpha_mask: float = 0.30,
    progress: bool = True
) -> pd.DataFrame:
    """
    Classical pipeline:
      blur → percentile threshold → morphology → keep ONLY the largest blob per frame.
    Writes CSV + optional overlay MP4. Returns DataFrame with one row per frame (if any blob found).
    """
    cones, ts, extent, meta = load_cone_run_npz(npz_path)
    T, H, W = cones.shape

    lut = _cmap_rgb(cmap)
    vw = None
    if save_mp4:
        Path(save_mp4).parent.mkdir(parents=True, exist_ok=True)
        rgb0 = gray01_to_rgb(cones[0], lut)
        vw = cv2.VideoWriter(str(save_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows: List[Dict] = []

    for t in range(T):
        Z = cones[t]

        # 1) blur
        B = gaussian_blur01(Z, ksize=blur_ksize, sigma=blur_sigma)

        # 2) segment
        mask = segment_percentile(B, p=thr_percentile, min_area_px=min_area_px, open_close=open_close)

        # 3) largest component
        lab_id, area, labels = _largest_component(mask.astype(np.uint8))
        found = (lab_id > 0 and area > 0)
        if found:
            comp = (labels == lab_id)
            ys, xs = np.where(comp)
            # centroid (px)
            cy = float(ys.mean()); cx = float(xs.mean())
            # bbox (px)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            # centroid in meters
            x_m, y_m = extent_px_to_m(extent, H, W, cy, cx)

            rows.append(dict(
                t_idx=t, ts=str(ts[t]),
                cx_px=cx, cy_px=cy, x_m=x_m, y_m=y_m,
                x0=x0, y0=y0, x1=x1, y1=y1,
                area_px=int(area),
                method="classical_largest_blob"
            ))

        # 4) overlay frame for video
        if vw is not None:
            rgb = gray01_to_rgb(Z, lut)
            if found:
                overlay = np.zeros_like(rgb)
                overlay[labels == lab_id] = (255, 140, 0)
                cv2.addWeighted(overlay, float(alpha_mask), rgb, 1.0 - float(alpha_mask), 0.0, dst=rgb)
                # draw bbox
                cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)
            # timestamp text
            cv2.putText(rgb, f"{ts[t]}  t={t}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            vw.write(rgb[:, :, ::-1])

        if progress and (t % 100 == 0):
            print(f"[largest_blob] t={t}/{T}  area={area if found else 0}")

    if vw is not None:
        vw.release()
        if progress: print(f"[largest_blob] wrote video → {save_mp4}")

    df = pd.DataFrame(rows)
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        if progress: print(f"[largest_blob] wrote csv → {save_csv}")

    return df
