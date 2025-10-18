# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
Rendering and Visualization Utilities

Functions for rasterizing sonar cones, creating matplotlib plots, and generating videos.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Import utilities from other modules
from utils.io_utils import load_png_rgb, ensure_ts_col, nearest_video_index
# Note: sonar_utils imports are done locally to avoid circular imports


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
        axes[1].text(1.02 * y_max * np.sin(th), 1.02 * y_max * np.cos(th), f"{a:.0f}", ha="center", va="center", fontsize=9)

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
        axes[1].text(1.02*y_max*np.sin(th), 1.02*y_max*np.cos(th), f"{a:.0f}",
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
    # Local import to avoid circular dependency
    from utils.sonar_utils import get_sonoptix_frame, apply_flips, enhance_intensity
    
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


# --------------------------- Public API --------------------------------------

__all__ = [
    "ConeGridSpec",
    "rasterize_cone",
    "plot_video_and_sonar",
    "plot_video_and_cone",
    "render_cone_side_by_side_frame",
    "save_alignment_video_cone",
]
