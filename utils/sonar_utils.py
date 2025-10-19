# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
Core Sonar Utilities

Frame extraction, intensity enhancement, and cone operations.
This module focuses on core sonar data processing.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# Import from our new modular structure
from utils.io_utils import (
    load_df, parse_json_cell, load_png_rgb, ensure_ts_col,
    read_video_index, normalize_ts_utc, nearest_video_index
)

# --------------------------- Cone rasterization ------------------------------

@dataclass
class ConeGridSpec:
    """Specification for cone grid dimensions."""
    img_w: int = 900
    img_h: int = 700


def rasterize_cone(
    Z: np.ndarray,
    *,
    fov_deg: float,
    rmin: float,
    rmax: float,
    y_zoom: Optional[float] = None,
    grid: ConeGridSpec = ConeGridSpec(),
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Rasterize a polar fan (rows=range, cols=beams) into Cartesian (x,y) with forward +Y,
    starboard +X, returning (cone_image, extent=(x_min,x_max,y_min,y_max)).
    """
    H, W = map(int, Z.shape)
    half = np.deg2rad(0.5 * float(fov_deg))
    y_min = max(0.0, float(rmin))
    y_max = float(y_zoom if y_zoom is not None else rmax)
    x_max = np.sin(half) * y_max
    x_min = -x_max

    x = np.linspace(x_min, x_max, grid.img_w)
    y = np.linspace(y_min, y_max, grid.img_h)
    Xg, Yg = np.meshgrid(x, y)

    theta = np.arctan2(Xg, Yg)   # angle from +Y axis
    r = np.hypot(Xg, Yg)

    mask = (r >= rmin) & (r <= y_max) & (theta >= -half) & (theta <= +half)

    # map (r,theta) -> (row, col)
    rowf = (r - rmin) / max((rmax - rmin), 1e-12) * (H - 1)
    colf = (theta + half) / max((2 * half), 1e-12) * (W - 1)
    rows = np.rint(np.clip(rowf, 0, H - 1)).astype(np.int32)
    cols = np.rint(np.clip(colf, 0, W - 1)).astype(np.int32)

    cone = np.full((grid.img_h, grid.img_w), np.nan, dtype=float)
    mflat = mask.ravel()
    cone.ravel()[mflat] = Z[rows.ravel()[mflat], cols.ravel()[mflat]]
    extent = (x_min, x_max, y_min, y_max)
    return cone, extent

# --------------------------- Frame extraction --------------------------------

def infer_hw(labels: Optional[Iterable[str]], sizes: Optional[Iterable[int]], data_len: int) -> Tuple[Optional[int], Optional[int]]:
    """Infer [H,W] from dim labels/sizes or brute-force pairs whose product equals len(data)."""
    labels = [str(l or "").lower() for l in (labels or [])]
    sizes = [int(s) for s in (sizes or [])]
    try:
        h_idx = max(i for i, l in enumerate(labels) if any(k in l for k in ("height", "rows", "beams")))
        w_idx = max(i for i, l in enumerate(labels) if any(k in l for k in ("width", "cols", "bins", "range", "samples")))
        H, W = sizes[h_idx], sizes[w_idx]
        if H > 0 and W > 0 and H * W == data_len:
            return H, W
    except Exception:
        pass

    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            H, W = sizes[i], sizes[j]
            if H > 0 and W > 0 and H * W == data_len:
                return H, W
    return None, None


def get_sonoptix_frame(dfs: pd.DataFrame, idx: int) -> Optional[np.ndarray]:
    """
    Extract a single sonar image (rows=range, cols=beams) from a Sonoptix-style row.
    Tries ['image'] first, then ['data'] with ['dim_*'] / ['rows','cols'] hints.
    """
    if "image" in dfs.columns:
        img = parse_json_cell(dfs.loc[idx, "image"])
        if isinstance(img, list) and img and isinstance(img[0], list):
            return np.asarray(img, dtype=float)

    if "data" in dfs.columns:
        data = parse_json_cell(dfs.loc[idx, "data"])
        labels = parse_json_cell(dfs.loc[idx, "dim_labels"]) if "dim_labels" in dfs.columns else None
        sizes = parse_json_cell(dfs.loc[idx, "dim_sizes"]) if "dim_sizes" in dfs.columns else None
        if isinstance(data, list):
            if isinstance(sizes, list):
                H, W = infer_hw(labels, sizes, len(data))
                if H and W:
                    return np.asarray(data, dtype=float).reshape(H, W)
            H = int(dfs.loc[idx, "rows"]) if "rows" in dfs.columns and pd.notna(dfs.loc[idx, "rows"]) else None
            W = int(dfs.loc[idx, "cols"]) if "cols" in dfs.columns and pd.notna(dfs.loc[idx, "cols"]) else None
            if H and W and H * W == len(data):
                return np.asarray(data, dtype=float).reshape(H, W)
    return None


def apply_flips(M: np.ndarray, *, flip_range: bool = False, flip_beams: bool = False) -> np.ndarray:
    """Optionally flip range (rows) or beams (cols)."""
    if flip_range:
        M = M[::-1, :]
    if flip_beams:
        M = M[:, ::-1]
    return M


# --------------------------- Processing / scaling ----------------------------

def enhance_intensity(
    M: np.ndarray,
    range_min: float,
    range_max: float,
    *,
    scale: str = "db",          # "db" or "linear"
    tvg: Optional[str] = "amplitude",  # "amplitude", "power", or None
    alpha_db_per_m: float = 0.0,
    r0: float = 1e-3,
    p_low: float = 1.0,
    p_high: float = 99.5,
    gamma: float = 0.6,
    zero_aware: bool = True,
    eps_log: float = 1e-3,
) -> np.ndarray:
    """
    Display-oriented intensity enhancement: log scaling, time-varying gain, absorption,
    percentile stretch, and gamma. Returns normalized [0,1] array.
    """
    M = np.asarray(M, dtype=float)
    X = 20.0 * np.log10(np.maximum(M, eps_log)) if scale == "db" else M.copy()

    # Range compensation (TVG)
    n_rows = M.shape[0]
    r_edges = np.linspace(range_min, range_max, n_rows + 1)
    r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
    R = np.broadcast_to(r_cent[:, None], M.shape)

    if tvg == "amplitude":
        X = X + 20.0 * np.log10(np.maximum(R, r0))
    elif tvg == "power":
        X = X + 40.0 * np.log10(np.maximum(R, r0))

    if alpha_db_per_m > 0:
        X = X + 2.0 * alpha_db_per_m * R

    # Contrast stretching
    mask = np.isfinite(X)
    if zero_aware:
        mask &= (M > 0)
    vals = X[mask]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = np.percentile(vals, p_low)
        vmax = np.percentile(vals, p_high)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6

    Y = np.clip((X - vmin) / (vmax - vmin), 0, 1)
    if gamma != 1.0:
        Y = np.power(Y, 1.0 / gamma)
    return Y


# --------------------------- Cone operations ---------------------------------

# ConeGridSpec is imported from utils.rendering and re-exported via __all__

def iter_cone_frames(
    sonar_df: pd.DataFrame,
    *,
    fov_deg: float,
    rmin: float,
    rmax: float,
    y_zoom: float | None = None,
    grid: ConeGridSpec = ConeGridSpec(img_w=900, img_h=700),
    flip_range: bool = False,
    flip_beams: bool = False,
    enhanced: bool = True,
    enhance_kwargs: Dict = None,
    frame_range: tuple[int, int] | None = None,
):
    """
    Generator yielding (cone_image[H,W] float32 with NaNs, ts_utc[pandas.Timestamp]) for each row.
    Skips rows where a sonar frame cannot be constructed.
    """
    sonar_df = ensure_ts_col(sonar_df)
    start = 0 if frame_range is None else max(0, int(frame_range[0]))
    stop  = len(sonar_df) if frame_range is None else min(len(sonar_df), int(frame_range[1]))
    enhance_kwargs = enhance_kwargs or {}

    for i in range(start, stop):
        M = get_sonoptix_frame(sonar_df, i)
        if M is None:
            continue
        M = apply_flips(M, flip_range=flip_range, flip_beams=flip_beams)
        Z = enhance_intensity(M, rmin, rmax, **enhance_kwargs) if enhanced else M
        cone, extent = rasterize_cone(
            Z, fov_deg=fov_deg, rmin=rmin, rmax=rmax, y_zoom=y_zoom, grid=grid
        )
        ts = sonar_df.loc[i, "ts_utc"]
        yield cone.astype(np.float32), pd.to_datetime(ts, utc=True), extent


def save_cone_run_npz(
    sonar_df: pd.DataFrame,
    out_path: Path,
    *,
    fov_deg: float,
    rmin: float,
    rmax: float,
    y_zoom: float | None = None,
    grid: ConeGridSpec = ConeGridSpec(img_w=900, img_h=700),
    flip_range: bool = False,
    flip_beams: bool = False,
    enhanced: bool = True,
    enhance_kwargs: Dict = None,
    frame_range: tuple[int, int] | None = None,
    attrs: Dict | None = None,
    progress: bool = True,
) -> Dict:
    """
    Build a (T,H,W) stack of cone images and write a single compressed .npz file with:
      - cones: float32 array shape (T,H,W) with NaNs as background
      - ts_unix_ns: int64 timestamps in UTC (ns since epoch)
      - extent: float64 (x_min, x_max, y_min, y_max)
      - meta_json: UTF-8 JSON with parameters/attrs

    Returns dict with shapes and counts.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cones: List[np.ndarray] = []
    ts_ns: List[int] = []
    extent_ref = None

    for k, (cone, ts, extent) in enumerate(
        iter_cone_frames(
            sonar_df,
            fov_deg=fov_deg, rmin=rmin, rmax=rmax, y_zoom=y_zoom, grid=grid,
            flip_range=flip_range, flip_beams=flip_beams,
            enhanced=enhanced, enhance_kwargs=enhance_kwargs,
            frame_range=frame_range,
        )
    ):
        if extent_ref is None:
            extent_ref = np.asarray(extent, dtype=np.float64)
        cones.append(cone)
        ts_ns.append(int(pd.Timestamp(ts).value))  # nanoseconds since epoch
        if progress and (k % 200 == 0):
            print(f"  accumulated {k} cone frames ...")

    if not cones:
        raise RuntimeError("No cone frames could be produced from sonar_df.")

    # Stack: (T,H,W)
    cones_stack = np.stack(cones, axis=0).astype(np.float32)
    ts_unix_ns = np.asarray(ts_ns, dtype=np.int64)

    meta = dict(
        format="uuv_sonar_cone_npz/v1",
        fov_deg=float(fov_deg),
        rmin=float(rmin),
        rmax=float(rmax),
        y_zoom=(None if y_zoom is None else float(y_zoom)),
        grid=dict(img_w=int(grid.img_w), img_h=int(grid.img_h)),
        flips=dict(range=bool(flip_range), beams=bool(flip_beams)),
        enhanced=bool(enhanced),
        enhance_kwargs=(enhance_kwargs or {}),
        display_range_max_m=(None if y_zoom is None else float(y_zoom)),
        extent=(None if extent_ref is None else [float(x) for x in extent_ref]),
    )
    if attrs:
        meta["attrs"] = attrs

    np.savez_compressed(
        out_path,
        cones=cones_stack,
        ts_unix_ns=ts_unix_ns,
        extent=extent_ref,
        meta_json=json.dumps(meta, ensure_ascii=False)
    )

    if progress:
        T, H, W = cones_stack.shape
        print(f"[save_cone_run_npz] wrote {out_path} (cones={T}x{H}x{W})")

    return {
        "out_path": out_path,
        "num_frames": int(cones_stack.shape[0]),
        "shape": tuple(cones_stack.shape),
        "extent": tuple(float(x) for x in extent_ref),
    }


def load_cone_run_npz(path: str | Path):
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        keys = set(npz.files)

        cones = np.asarray(npz["cones"], dtype=np.float32)
        extent = tuple(np.asarray(npz["extent"], dtype=np.float64).tolist())

        # read meta (optional)
        meta = {}
        if "meta_json" in keys:
            raw_meta = npz["meta_json"]
            meta = json.loads(raw_meta.item() if getattr(raw_meta, "ndim", 0) else raw_meta.tolist())
        elif "meta" in keys:
            m = npz["meta"]
            try:
                meta = m.item() if hasattr(m, "item") else m.tolist()
                if isinstance(meta, (bytes, str)):
                    meta = json.loads(meta)
            except Exception:
                meta = {}

        # timestamps from various places
        ts = None
        if "ts_unix_ns" in keys:
            ts_ns = np.asarray(npz["ts_unix_ns"], dtype=np.int64)
            ts = pd.to_datetime(ts_ns, utc=True)
        elif "ts" in keys:
            ts_raw = npz["ts"]
            try:
                ts = pd.to_datetime(ts_raw, utc=True)
            except Exception:
                ts = pd.to_datetime(np.asarray(ts_raw, dtype="int64"), unit="s", utc=True)
        else:
            for k in ("ts_unix_ns", "ts_ns", "timestamps_ns", "ts"):
                if isinstance(meta, dict) and k in meta:
                    cand = meta[k]
                    try:
                        if "ns" in k:
                            ts = pd.to_datetime(np.asarray(cand, dtype="int64"), utc=True)
                        else:
                            ts = pd.to_datetime(cand, utc=True)
                    except Exception:
                        ts = None
                    break

        # --- normalize ts ---
        T = cones.shape[0]
        if ts is None:
            ts = pd.to_datetime(np.arange(T), unit="s", utc=True)  # dummy timeline
        elif isinstance(ts, pd.Timestamp):
            # scalar → repeat to match frames
            ts = pd.DatetimeIndex([ts] * T)
        else:
            ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
            if len(ts) != T:
                if len(ts) == 1:
                    ts = pd.DatetimeIndex(np.repeat(ts.values, T))
                else:
                    raise ValueError(f"Timestamp length ({len(ts)}) != frames ({T})")

    return cones, ts, extent, meta

def cone_raster_like_display_cell(Z: np.ndarray,
                                  fov_deg: float, r_min: float, r_max: float,
                                  y_max: float, img_w: int, img_h: int):
    """Rasterize sonar data into a cone display matching matplotlib cell format."""
    H, W = Z.shape
    half = np.deg2rad(0.5 * fov_deg)
    y_min = max(0.0, float(r_min))
    x_max =  np.sin(half) * y_max
    x_min = -x_max

    x = np.linspace(x_min, x_max, img_w)
    y = np.linspace(y_min, y_max, img_h)
    Xg, Yg = np.meshgrid(x, y)

    theta = np.arctan2(Xg, Yg)  # from +Y axis, starboard positive
    r     = np.hypot(Xg, Yg)

    mask = (r >= r_min) & (r <= y_max) & (theta >= -half) & (theta <= +half)

    rowf = (r - r_min) / max((r_max - r_min), 1e-12) * (H - 1)
    colf = (theta + half) / max((2*half),     1e-12) * (W - 1)

    rows = np.rint(np.clip(rowf, 0, H - 1)).astype(np.int32)
    cols = np.rint(np.clip(colf, 0, W - 1)).astype(np.int32)

    cone = np.full((img_h, img_w), np.nan, dtype=float)
    flat = mask.ravel()
    cone.ravel()[flat] = Z[rows.ravel()[flat], cols.ravel()[flat]]
    return cone, (x_min, x_max, y_min, y_max)


def to_uint8_gray(frame01: np.ndarray) -> np.ndarray:
    """Convert normalized frame to uint8 grayscale."""
    safe = np.nan_to_num(frame01, nan=0.0, posinf=1.0, neginf=0.0)
    safe = np.clip(safe, 0.0, 1.0)
    return (safe * 255.0).astype(np.uint8)


def get_pixel_to_meter_mapping(npz_file_path: Path) -> Dict[str, any]:
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
        from utils.config import CONE_H_DEFAULT, CONE_W_DEFAULT, DISPLAY_RANGE_MAX_M_DEFAULT
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


# --------------------------- Public API --------------------------------------

__all__ = [
    # Cone rasterization
    "ConeGridSpec",
    "rasterize_cone",
    # Frame extraction
    "infer_hw",
    "get_sonoptix_frame",
    "apply_flips",
    # Intensity processing
    "enhance_intensity",
    # Cone operations
    "iter_cone_frames",
    "save_cone_run_npz",
    "load_cone_run_npz",
    "cone_raster_like_display_cell",
    "to_uint8_gray",
    "get_pixel_to_meter_mapping",
    # Re-exported from io_utils for backward compatibility
    "load_df",
    "parse_json_cell",
    "load_png_rgb",
    "ensure_ts_col",
    "read_video_index",
    "normalize_ts_utc",
    "nearest_video_index",
]
