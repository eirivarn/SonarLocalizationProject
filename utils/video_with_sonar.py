import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from pathlib import Path
from datetime import datetime

from utils.sonar_utils import (
    load_df, parse_json_cell, infer_hw, get_sonoptix_frame,
    enhance_intensity, read_video_index, apply_flips, cone_raster_like_display_cell
)

def load_png_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_local(ts, tz_name="Europe/Oslo"):
    try:
        return ts.tz_convert(tz_name)
    except Exception:
        return ts

def ts_for_name(ts, tz_name="Europe/Oslo"):
    if pd.isna(ts):
        ts = pd.Timestamp.utcnow().tz_localize("UTC")
    return to_local(ts, tz_name).strftime("%Y%m%d_%H%M%S_%f%z")

def put_text(bgr, s, y, x=10, scale=0.55):
    cv2.putText(bgr, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(bgr, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),   1, cv2.LINE_AA)

def export_video_with_sonar_display(
    SONAR_FILE,
    VIDEO_SEQ_DIR,
    OUT_DIR,
    START_IDX=0,
    END_IDX=None,
    STRIDE=1,
    TARGET_FPS=None,
    TIME_TOLERANCE=pd.Timedelta("100ms"),
    FOV_DEG=120.0,
    RANGE_MIN_M=0.0,
    RANGE_MAX_M=30.0,
    DISPLAY_RANGE_MAX_M=10.0,
    FLIP_BEAMS=True,
    FLIP_RANGE=False,
    USE_ENHANCED=True,
    ENH_SCALE="db",
    ENH_TVG="amplitude",
    ENH_ALPHA_DB_PER_M=0.0,
    ENH_R0=1e-2,
    ENH_P_LOW=1.0,
    ENH_P_HIGH=99.5,
    ENH_GAMMA=0.9,
    ENH_ZERO_AWARE=True,
    ENH_EPS_LOG=1e-6,
    CMAP_RAW="viridis",
    CMAP_ENH="viridis",
    CONE_W=900,
    CONE_H=700,
    VIDEO_HEIGHT=700,
    PAD_BETWEEN=8,
    FONT_SCALE=0.55,
    CONE_FLIP_VERTICAL=True
):
    df = load_df(SONAR_FILE)
    if "ts_utc" not in df.columns:
        if "t" not in df.columns:
            raise RuntimeError("Sonar CSV missing 't' column for timestamps.")
        df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
    N = len(df)
    i0 = int(max(0, START_IDX))
    i1 = int(N if END_IDX is None else min(N, END_IDX))
    idxs = list(range(i0, i1, max(1, STRIDE)))
    if not idxs:
        raise RuntimeError("No frames selected; adjust START/END/STRIDE.")
    if TARGET_FPS is None:
        ts = pd.to_datetime(df.loc[idxs, "ts_utc"], utc=True, errors="coerce").dropna().sort_values().reset_index(drop=True)
        dt_s = ts.diff().dt.total_seconds().to_numpy()[1:]
        dt_s = dt_s[(dt_s > 1e-6) & (dt_s < 5.0)]
        fps = float(np.clip(1.0/np.median(dt_s), 1.0, 60.0)) if dt_s.size else 12.0
    else:
        fps = float(TARGET_FPS)
    dfv = read_video_index(VIDEO_SEQ_DIR)
    dfv = dfv.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    video_idx = pd.Index(dfv["ts_utc"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first_ts = pd.to_datetime(df.loc[idxs[0], "ts_utc"], utc=True, errors="coerce")
    out_name = ts_for_name(first_ts, "Europe/Oslo") + "_sonar_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    cmap_name = (CMAP_ENH if USE_ENHANCED else CMAP_RAW)
    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_bad((0,0,0,1))
    frames_written = 0
    half = np.deg2rad(0.5 * FOV_DEG)
    x_max_m = np.sin(half) * DISPLAY_RANGE_MAX_M
    def x_px(xm):
        return int(round((xm + x_max_m) / (2*x_max_m) * (CONE_W-1)))
    def y_px_nominal(ym):
        return int(round((ym - 0.0) / (DISPLAY_RANGE_MAX_M - 0.0 + 1e-12) * (CONE_H-1)))
    def y_px(ym):
        yp = y_px_nominal(ym)
        if CONE_FLIP_VERTICAL:
            yp = (CONE_H - 1) - yp
        return yp
    for k, i in enumerate(idxs):
        M0 = get_sonoptix_frame(df, i)
        if M0 is None:
            continue
        M = apply_flips(M0, flip_range=FLIP_RANGE, flip_beams=FLIP_BEAMS)
        H, W = M.shape
        Z = enhance_intensity(M, RANGE_MIN_M, RANGE_MAX_M,
                             scale=ENH_SCALE, tvg=ENH_TVG, alpha_db_per_m=ENH_ALPHA_DB_PER_M,
                             r0=ENH_R0, p_low=ENH_P_LOW, p_high=ENH_P_HIGH,
                             gamma=ENH_GAMMA, zero_aware=ENH_ZERO_AWARE, eps_log=ENH_EPS_LOG) if USE_ENHANCED else M
        cone, (x_min, x_max, y_min, y_max) = cone_raster_like_display_cell(
            Z, FOV_DEG, RANGE_MIN_M, RANGE_MAX_M, DISPLAY_RANGE_MAX_M, CONE_W, CONE_H
        )
        cone_rgb = (cmap(np.ma.masked_invalid(cone))[:, :, :3] * 255).astype(np.uint8)
        cone_bgr = cv2.cvtColor(cone_rgb, cv2.COLOR_RGB2BGR)
        if CONE_FLIP_VERTICAL:
            cone_bgr = cv2.flip(cone_bgr, 0)
        ts_target = pd.to_datetime(df.loc[i, "ts_utc"], utc=True, errors="coerce")
        idx_near = video_idx.get_indexer([ts_target], method="nearest")[0]
        ts_cam = dfv.loc[idx_near, "ts_utc"]
        dt = abs(ts_cam - ts_target)
        cam_file = VIDEO_SEQ_DIR / dfv.loc[idx_near, "file"]
        cam_bgr = load_png_bgr(cam_file)
        vh, vw0 = cam_bgr.shape[:2]
        scale = VIDEO_HEIGHT / vh
        cam_resized = cv2.resize(cam_bgr, (int(round(vw0*scale)), VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)
        pad = np.zeros((CONE_H, PAD_BETWEEN, 3), dtype=np.uint8)
        composite = np.hstack([cam_resized, pad, cone_bgr])
        if writer is None:
            out_size = (composite.shape[1], composite.shape[0])
            writer = cv2.VideoWriter(str(OUT_DIR / out_name), fourcc, fps, out_size, True)
            if not writer.isOpened():
                raise RuntimeError(f"Could not open writer: {OUT_DIR/out_name}")
        ts_cam_loc   = to_local(ts_cam,   "Europe/Oslo")
        ts_sonar_loc = to_local(ts_target,"Europe/Oslo")
        put_text(composite, f"VIDEO  @ {ts_cam_loc:%Y-%m-%d %H:%M:%S.%f %Z}", 24)
        put_text(composite, f"SONAR  @ {ts_sonar_loc:%Y-%m-%d %H:%M:%S.%f %Z}   Δt={dt.total_seconds():.3f}s", 48)
        put_text(composite, f"FOV={FOV_DEG:.0f}°, range={RANGE_MIN_M:.0f}-{DISPLAY_RANGE_MAX_M:.0f} m  ({'enhanced' if USE_ENHANCED else 'raw'})", 72)
        x_off = cam_resized.shape[1] + PAD_BETWEEN
        N_SPOKES = 5
        for a in np.linspace(-np.rad2deg(half), np.rad2deg(half), N_SPOKES):
            th = np.deg2rad(a)
            x_end = x_px(DISPLAY_RANGE_MAX_M * np.sin(th))
            y_end = y_px(DISPLAY_RANGE_MAX_M * np.cos(th))
            apex_x = x_off + x_px(0.0)
            apex_y = y_px(0.0)
            cv2.line(composite, (apex_x, apex_y), (x_off + x_end, y_end), (0,0,0), 2, cv2.LINE_AA)
            cv2.line(composite, (apex_x, apex_y), (x_off + x_end, y_end), (255,255,255), 1, cv2.LINE_AA)
        writer.write(composite)
        frames_written += 1
    if writer is not None:
        writer.release()
    print(f"Wrote {frames_written} frames to {OUT_DIR/out_name} @ {fps:.2f} FPS")
