# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

# uuv_sonar_viz.py
"""
Utilities for loading Sonoptix-style multibeam sonar frames, matching video frames by
timestamp, enhancing/rasterizing sonar intensity, and producing ready-made plots.

Usage (in a notebook):
    import uuv_sonar_viz as usv

    dfs = usv.load_df(SONAR_FILE)
    dfv = usv.read_video_index(VIDEO_SEQ_DIR)

    # pick sonar frame
    M = usv.get_sonoptix_frame(dfs, frame_index)
    M = usv.apply_flips(M, flip_range=False, flip_beams=False)

    # enhance for display
    Z_enh = usv.enhance_intensity(M, range_min, range_max)

    # match video frame
    ts_target = usv.ensure_ts_col(dfs).loc[frame_index, "ts_utc"]
    j_best, dt_best = usv.nearest_video_index(dfv, ts_target, tolerance=pd.Timedelta("75ms"))
    img_rgb = usv.load_png_rgb(VIDEO_SEQ_DIR / dfv.loc[j_best, "file"])

    # quick plots
    usv.plot_video_and_sonar(img_rgb, Z_enh, fov_deg=120, rmin=0, rmax=30, y_zoom=10)
    usv.plot_video_and_cone(img_rgb, Z_enh, fov_deg=120, rmin=0, rmax=30, y_zoom=10)
"""

from __future__ import annotations
from matplotlib.backends.backend_agg import FigureCanvasAgg

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import json
from typing import Dict, List


# --------------------------- I/O + parsing helpers ---------------------------

def load_df(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet into a DataFrame."""
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def parse_json_cell(v):
    """Parse a JSON-encoded cell if it's a string, else pass through."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v


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


def load_png_rgb(path: Path) -> np.ndarray:
    """Load a PNG (or any OpenCV-readable image) as RGB numpy array."""
    bgr = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ensure_ts_col(dfs: pd.DataFrame, *, src_col: str = "t") -> pd.DataFrame:
    """
    Ensure a tz-aware UTC timestamp column 'ts_utc' exists.
    If missing, derive from 't' seconds epoch.
    """
    dfs = dfs.copy()
    if "ts_utc" not in dfs.columns:
        if src_col not in dfs.columns:
            raise RuntimeError("Sonar DataFrame missing both 'ts_utc' and epoch seconds column.")
        dfs["ts_utc"] = pd.to_datetime(dfs[src_col], unit="s", utc=True)
    else:
        dfs["ts_utc"] = pd.to_datetime(dfs["ts_utc"], utc=True, errors="coerce")
    return dfs


def read_video_index(seq_dir: Path) -> pd.DataFrame:
    """
    Return DataFrame with at least ['file','ts_utc'] from a frames folder.
    Uses index.csv if present; otherwise attempts to parse timestamps from filenames
    like YYYYmmdd_HHMMSS_micro+ZZZZ.png.
    """
    seq_dir = Path(seq_dir)
    idx_path = seq_dir / "index.csv"
    if idx_path.exists():
        dfv = pd.read_csv(idx_path)
        if "ts_utc" in dfv.columns:
            dfv["ts_utc"] = pd.to_datetime(dfv["ts_utc"], utc=True)
        elif "t_use" in dfv.columns:  # seconds epoch
            dfv["ts_utc"] = pd.to_datetime(dfv["t_use"], unit="s", utc=True)
        else:
            raise RuntimeError("index.csv missing ts_utc/t_use columns.")
        if "file" not in dfv.columns:
            dfv["file"] = [p.name for p in seq_dir.glob("*.png")]
        return dfv[["file", "ts_utc"]].dropna().sort_values("ts_utc").reset_index(drop=True)

    # Fallback: parse from filenames
    rows = []
    rx = re.compile(r"(\d{8})_(\d{6})_(\d{6})([+\-]\d{4})")
    for p in sorted(seq_dir.glob("*.png")):
        m = rx.search(p.stem)
        if not m:
            continue
        ymd, hms, micro, tz = m.groups()
        dt = datetime.strptime(ymd + hms + micro + tz, "%Y%m%d%H%M%S%f%z")
        rows.append({"file": p.name, "ts_utc": dt.astimezone(timezone.utc)})
    if not rows:
        raise FileNotFoundError("No index.csv and could not parse timestamps from filenames.")
    return pd.DataFrame(rows).sort_values("ts_utc").reset_index(drop=True)


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


# --------------------------- Timestamp matching ------------------------------

def normalize_ts_utc(ts) -> pd.Timestamp:
    """Ensure a pandas Timestamp is tz-aware UTC."""
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def nearest_video_index(
    dfv: pd.DataFrame,
    ts_target,
    *,
    tolerance: Optional[pd.Timedelta] = None
) -> Tuple[int, Optional[pd.Timedelta]]:
    """
    Find nearest frame index in dfv (expects 'ts_utc' tz-aware). Returns (idx, |dt|).
    If tolerance is given and nearest exceeds it, returns the index but you may treat
    it as a soft failure by checking the returned dt.
    """
    dfv = dfv.copy()
    dfv["ts_utc"] = pd.to_datetime(dfv["ts_utc"], utc=True, errors="coerce")
    dfv = dfv.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    if dfv.empty:
        raise RuntimeError("Video index has no valid timestamps.")

    ts_target = normalize_ts_utc(ts_target)
    idx_near = pd.Index(dfv["ts_utc"]).get_indexer([ts_target], method="nearest")[0]
    if idx_near < 0:
        raise RuntimeError("Could not find a nearest video frame (indexer returned -1).")
    dt_best = abs(dfv.loc[idx_near, "ts_utc"] - ts_target)
    if tolerance is not None and dt_best > tolerance:
        # not raising—caller can decide; we still return the nearest + dt.
        pass
    return int(idx_near), pd.Timedelta(dt_best)


# --------------------------- Cone rasterization ------------------------------

@dataclass
class ConeGridSpec:
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


# --------------------------- Plotting helpers --------------------------------

def plot_video_and_sonar(
    img_rgb: np.ndarray,
    Z: np.ndarray,
    *,
    fov_deg: float,
    rmin: float,
    rmax: float,
    y_zoom: Optional[float] = None,
    cmap: str = "viridis",
    enhanced: bool = True,
    ts_video_oslo: Optional[pd.Timestamp] = None,
    ts_sonar_oslo: Optional[pd.Timestamp] = None,
    frame_index: Optional[int] = None,
    vmin_raw: Optional[float] = None,
    vmax_raw: Optional[float] = None,
    figsize: Tuple[float, float] = (13.2, 5.8),
) -> None:
    """
    Side-by-side: video frame and sonar (polar image with beam angle on X, range on Y).
    Z should be in [0,1] if enhanced=True; otherwise provide vmin_raw/vmax_raw.
    """
    theta_min_deg = -0.5 * float(fov_deg)
    theta_max_deg = +0.5 * float(fov_deg)
    extent_xy = (theta_min_deg, theta_max_deg, rmin, rmax)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Left: video
    axes[0].imshow(img_rgb)
    title_video = "Video"
    if ts_video_oslo is not None:
        try:
            title_video += f" @ {pd.to_datetime(ts_video_oslo).tz_convert('Europe/Oslo'):%Y-%m-%d %H:%M:%S.%f %Z}"
        except Exception:
            title_video += f" @ {ts_video_oslo}"
    axes[0].set_title(title_video)
    axes[0].axis("off")

    # Right: sonar
    if enhanced:
        im = axes[1].imshow(Z, origin="lower", aspect="auto", extent=extent_xy, vmin=0, vmax=1, cmap=cmap)
        cb_label = "Echo (norm.)"
        mode_name = "Enhanced"
    else:
        vmin = float(np.nanmin(Z)) if vmin_raw is None else vmin_raw
        vmax = float(np.nanmax(Z)) if vmax_raw is None else vmax_raw
        im = axes[1].imshow(Z, origin="lower", aspect="auto", extent=extent_xy, vmin=vmin, vmax=vmax, cmap=cmap)
        cb_label = "Echo (raw)"
        mode_name = "Raw"

    title_sonar = f"Sonar — {mode_name}"
    if frame_index is not None:
        title_sonar += f" (frame {frame_index})"
    if ts_sonar_oslo is not None:
        try:
            title_sonar += f" @ {pd.to_datetime(ts_sonar_oslo).tz_convert('Europe/Oslo'):%H:%M:%S.%f %Z}"
        except Exception:
            title_sonar += f" @ {ts_sonar_oslo}"
    axes[1].set_title(title_sonar)
    axes[1].set_xlabel("Beam angle [deg]")
    axes[1].set_ylabel("Range [m]")
    axes[1].set_ylim(rmin, y_zoom if y_zoom is not None else rmax)
    fig.colorbar(im, ax=axes[1], label=cb_label)
    plt.show()


def plot_video_and_cone(
    img_rgb: np.ndarray,
    Z: np.ndarray,
    *,
    fov_deg: float,
    rmin: float,
    rmax: float,
    y_zoom: Optional[float] = None,
    cmap: str = "viridis",
    enhanced: bool = True,
    frame_index: Optional[int] = None,
    ts_video_oslo: Optional[pd.Timestamp] = None,
    ts_sonar_oslo: Optional[pd.Timestamp] = None,
    grid: ConeGridSpec = ConeGridSpec(),
    n_spokes: int = 5,
    figsize: Tuple[float, float] = (13.6, 6.8),
) -> None:
    """
    Video + Cartesian rasterized sonar cone. Z should be [0,1] if enhanced=True.
    """
    cone, extent = rasterize_cone(Z, fov_deg=fov_deg, rmin=rmin, rmax=rmax, y_zoom=y_zoom, grid=grid)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Left: video
    axes[0].imshow(img_rgb)
    title_video = "Video"
    if ts_video_oslo is not None:
        try:
            title_video += f" @ {pd.to_datetime(ts_video_oslo).tz_convert('Europe/Oslo'):%Y-%m-%d %H:%M:%S.%f %Z}"
        except Exception:
            title_video += f" @ {ts_video_oslo}"
    axes[0].set_title(title_video)
    axes[0].axis("off")

    # Right: cone
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    vmin, vmax = (0.0, 1.0) if enhanced else (float(np.nanmin(Z)), float(np.nanmax(Z)))

    im = axes[1].imshow(cone, origin="lower", extent=extent, aspect="equal", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    # Spokes
    half = 0.5 * fov_deg
    y_max = extent[3]
    for a in np.linspace(-half, half, n_spokes):
        th = np.deg2rad(a)
        axes[1].plot([0, y_max * np.sin(th)], [0, y_max * np.cos(th)], color="k", lw=0.9, alpha=0.85)
        axes[1].text(1.02 * y_max * np.sin(th), 1.02 * y_max * np.cos(th), f"{a:.0f}°", ha="center", va="center", fontsize=9)

    title_cone = "Sonar Cone"
    if enhanced:
        title_cone += " — Enhanced"
    if frame_index is not None:
        title_cone += f" (frame {frame_index})"
    if ts_sonar_oslo is not None:
        try:
            title_cone += f" @ {pd.to_datetime(ts_sonar_oslo).tz_convert('Europe/Oslo'):%H:%M:%S.%f %Z}"
        except Exception:
            title_cone += f" @ {ts_sonar_oslo}"
    axes[1].set_title(title_cone)

    axes[1].set_xlabel("Starboard X [m] (+)")
    axes[1].set_ylabel("Forward Y [m]")

    fig.colorbar(im, ax=axes[1], pad=0.02, shrink=0.9, label=("Echo (normalized)" if enhanced else "Echo (raw units)"))
    plt.show()


# --------------------------- Video writing ------------------------------------
from matplotlib.backends.backend_agg import FigureCanvasAgg

def _render_array_with_matplotlib(fig, axes) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[..., :3].copy()
    plt.close(fig)
    return rgb

def render_cone_side_by_side_frame(
    img_rgb: np.ndarray,
    Z: np.ndarray,
    *,
    fov_deg: float,
    rmin: float,
    rmax: float,
    y_zoom: float,
    cmap: str = "viridis",
    enhanced: bool = True,
    ts_video_oslo=None,
    ts_sonar_oslo=None,
    frame_index: int | None = None,
    figsize=(12.8, 7.2),
    n_spokes: int = 5,
    grid: ConeGridSpec = ConeGridSpec(img_w=900, img_h=700),
) -> np.ndarray:
    """Render one frame (left: video, right: sonar cone) to an RGB array."""
    cone, extent = rasterize_cone(
        Z, fov_deg=fov_deg, rmin=rmin, rmax=rmax, y_zoom=y_zoom, grid=grid
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Left: video
    axes[0].imshow(img_rgb)
    title_video = "Video"
    if ts_video_oslo is not None:
        try:
            title_video += f" @ {pd.to_datetime(ts_video_oslo).tz_convert('Europe/Oslo'):%Y-%m-%d %H:%M:%S.%f %Z}"
        except Exception:
            title_video += f" @ {ts_video_oslo}"
    axes[0].set_title(title_video)
    axes[0].axis("off")

    # Right: cone
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    vmin, vmax = (0.0, 1.0) if enhanced else (float(np.nanmin(Z)), float(np.nanmax(Z)))
    im = axes[1].imshow(cone, origin="lower", extent=extent, aspect="equal",
                        cmap=cmap_obj, vmin=vmin, vmax=vmax)

    # Spokes
    half = 0.5 * fov_deg
    y_max = extent[3]
    for a in np.linspace(-half, half, n_spokes):
        th = np.deg2rad(a)
        axes[1].plot([0, y_max*np.sin(th)], [0, y_max*np.cos(th)],
                     color="k", lw=0.9, alpha=0.85)
        axes[1].text(1.02*y_max*np.sin(th), 1.02*y_max*np.cos(th), f"{a:.0f}°",
                     ha="center", va="center", fontsize=9)

    title_cone = "Sonar Cone"
    if enhanced:
        title_cone += " — Enhanced"
    if frame_index is not None:
        title_cone += f" (frame {frame_index})"
    if ts_sonar_oslo is not None:
        try:
            title_cone += f" @ {pd.to_datetime(ts_sonar_oslo).tz_convert('Europe/Oslo'):%H:%M:%S.%f %Z}"
        except Exception:
            title_cone += f" @ {ts_sonar_oslo}"
    axes[1].set_title(title_cone)
    axes[1].set_xlabel("Starboard X [m] (+)")
    axes[1].set_ylabel("Forward Y [m]")

    fig.colorbar(im, ax=axes[1], pad=0.02, shrink=0.9,
                 label=("Echo (normalized)" if enhanced else "Echo (raw units)"))

    return _render_array_with_matplotlib(fig, axes)


def save_alignment_video_cone(
    sonar_df: pd.DataFrame,
    video_idx: pd.DataFrame,
    video_dir: Path,
    *,
    out_path: Path,
    fov_deg: float,
    rmin: float,
    rmax: float,
    y_zoom: float,
    time_tolerance: pd.Timedelta = pd.Timedelta("75ms"),
    flip_range: bool = False,
    flip_beams: bool = False,
    enhanced: bool = True,
    cmap: str = "viridis",
    fps: float = 15.0,
    frame_range: tuple[int, int] | None = None,  # e.g. (0, 500)
    figsize=(12.8, 7.2),
    n_spokes: int = 5,
    grid: ConeGridSpec = ConeGridSpec(img_w=900, img_h=700),
    progress: bool = True,
) -> dict:
    """
    Create an MP4 of (video + sonar cone) for each sonar timestamp (nearest-matched).
    Skips sonar frames with no video match inside the tolerance.
    """
    video_dir = Path(video_dir)
    out_path = Path(out_path)

    sonar_df = ensure_ts_col(sonar_df)
    video_idx = video_idx.copy()
    video_idx["ts_utc"] = pd.to_datetime(video_idx["ts_utc"], utc=True, errors="coerce")
    video_idx = video_idx.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    if video_idx.empty:
        raise RuntimeError("Video index has no valid timestamps.")

    # helper to build one output frame
    def build_frame(i):
        M = get_sonoptix_frame(sonar_df, i)
        if M is None:
            return None
        M = apply_flips(M, flip_range=flip_range, flip_beams=flip_beams)
        Z = enhance_intensity(M, rmin, rmax) if enhanced else M

        ts_target = sonar_df.loc[i, "ts_utc"]
        try:
            j_best, dt = nearest_video_index(video_idx, ts_target, tolerance=time_tolerance)
        except Exception:
            return None
        if dt is None or dt > time_tolerance:
            return None

        img_rgb = load_png_rgb(video_dir / video_idx.loc[j_best, "file"])
        frame = render_cone_side_by_side_frame(
            img_rgb, Z,
            fov_deg=fov_deg, rmin=rmin, rmax=rmax, y_zoom=y_zoom,
            cmap=cmap, enhanced=enhanced,
            ts_video_oslo=video_idx.loc[j_best, "ts_utc"],
            ts_sonar_oslo=ts_target, frame_index=i,
            figsize=figsize, n_spokes=n_spokes, grid=grid,
        )
        return frame

    # Find size via first renderable match
    start_i = 0 if frame_range is None else max(0, int(frame_range[0]))
    end_i   = len(sonar_df) if frame_range is None else min(len(sonar_df), int(frame_range[1]))

    first = None
    for i in range(start_i, end_i):
        first = build_frame(i)
        if first is not None:
            break
    if first is None:
        raise RuntimeError("No frames matched within tolerance; nothing to write.")

    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    written = 0
    skipped = 0
    if progress:
        print(f"[save_alignment_video_cone] Writing {out_path} at {fps:.1f} fps, size={W}x{H} ...")

    writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
    written += 1

    for j in range(i + 1, end_i):
        frame = build_frame(j)
        if frame is None:
            skipped += 1
            continue
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        written += 1
        if progress and (written % 50 == 0):
            print(f"  wrote {written} frames (+{skipped} skipped)")

    writer.release()
    if progress:
        print(f"[done] wrote={written}, skipped={skipped}, output='{out_path}'")
    return {"written": written, "skipped": skipped, "out_path": out_path}

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


def box_blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Apply a simple box blur for display smoothing (requires scipy)."""
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(img, size=k, mode="nearest")
    except ImportError:
        print("Warning: scipy not available for box_blur, returning original image")
        return img


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

# --------------------------- Public API --------------------------------------

__all__ = [
    # I/O & parsing
    "load_df",
    "parse_json_cell",
    "infer_hw",
    "get_sonoptix_frame",
    "apply_flips",
    "load_png_rgb",
    "ensure_ts_col",
    "read_video_index",
    # processing
    "enhance_intensity",
    "box_blur",
    "cone_raster_like_display_cell",
    # matching
    "nearest_video_index",
    # cone
    "ConeGridSpec",
    "rasterize_cone",
    # plotting
    "plot_video_and_sonar",
    "plot_video_and_cone",
    # video writing
    "render_cone_side_by_side_frame", 
    "save_alignment_video_cone",
    # cone run saving/loading
    "iter_cone_frames", 
    "save_cone_run_npz", 
    "load_cone_run_npz",
    # utility functions
    "load_png_bgr",
    "to_local_tz",
    "ts_for_filename",
    "put_text_overlay"
]


# Additional utility functions for video generation
def load_png_bgr(path) -> np.ndarray:
    """Load PNG image as BGR array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_local_tz(ts, tz_name="Europe/Oslo"):
    """Convert timestamp to local timezone."""
    try: 
        return ts.tz_convert(tz_name)
    except Exception: 
        return ts


def ts_for_filename(ts, tz_name="Europe/Oslo"):
    """Format timestamp for filename."""
    if pd.isna(ts): 
        ts = pd.Timestamp.utcnow().tz_localize("UTC")
    return to_local_tz(ts, tz_name).strftime("%Y%m%d_%H%M%S_%f%z")


def put_text_overlay(bgr, text, y, x=10, scale=0.55):
    """Add text overlay with white text and black outline."""
    cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 1, cv2.LINE_AA)
