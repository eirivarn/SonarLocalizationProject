import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from utils.sonar_utils import (
    load_df, get_sonoptix_frame,
    enhance_intensity, read_video_index, apply_flips, cone_raster_like_display_cell
)
from utils.sonar_config import (
    SONAR_VIS_DEFAULTS, ENHANCE_DEFAULTS,
    CONE_W_DEFAULT, CONE_H_DEFAULT, CONE_FLIP_VERTICAL_DEFAULT,
    CMAP_NAME_DEFAULT, DISPLAY_RANGE_MAX_M_DEFAULT, FOV_DEG_DEFAULT,
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

def export_optimized_sonar_video(
    TARGET_BAG: str,
    EXPORTS_FOLDER: Path,
    START_IDX: int = 0,
    END_IDX: int | None = 600,
    STRIDE: int = 1,
    # --- camera (optional) ---
    VIDEO_SEQ_DIR: Path | None = None,   # if provided, we add the camera panel
    VIDEO_HEIGHT: int = 700,
    PAD_BETWEEN: int = 8,
    FONT_SCALE: float = 0.55,
    # --- sonar / display params (optimized defaults) ---
    FOV_DEG: float = FOV_DEG_DEFAULT,
    RANGE_MIN_M: float = SONAR_VIS_DEFAULTS["range_min_m"],
    RANGE_MAX_M: float = SONAR_VIS_DEFAULTS["range_max_m"],
    DISPLAY_RANGE_MAX_M: float = DISPLAY_RANGE_MAX_M_DEFAULT,
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
    CONE_W: int = CONE_W_DEFAULT,
    CONE_H: int = CONE_H_DEFAULT,
    CONE_FLIP_VERTICAL: bool = CONE_FLIP_VERTICAL_DEFAULT,
    CMAP_NAME: str = CMAP_NAME_DEFAULT,
    # --- net overlay (optional) ---
    INCLUDE_NET: bool = True,
    NET_DISTANCE_TOLERANCE: float = 0.5,  # seconds
    NET_PITCH_TOLERANCE: float = 0.3,     # seconds
    # --- sonar analysis overlay (optional) ---
    SONAR_DISTANCE_RESULTS: pd.DataFrame | None = None,  # DataFrame with sonar analysis results
):
    """
    Optimized sonar + (optional) net-line overlay + (optional) sonar analysis overlay.
    If VIDEO_SEQ_DIR is given, the output includes the actual camera frame (side-by-side).
    If SONAR_DISTANCE_RESULTS is provided, displays both DVL and sonar analysis distances.

    Output is saved under EXPORTS_FOLDER / 'videos' with an 'optimized_sync' name.
    """
    import time
    import matplotlib.cm as cm

    # expected helpers in this module:
    #  - load_df, get_sonoptix_frame, apply_flips, enhance_intensity
    #  - cone_raster_like_display_cell, read_video_index
    #  - load_png_bgr, to_local, put_text, ts_for_name

    print("🛠️ OPTIMIZED SONAR VIDEO")
    print("=" * 70)
    print(f"🎯 Target Bag: {TARGET_BAG}")
    print(f"   Cone Size: {CONE_W}x{CONE_H}")
    print(f"   Range: {RANGE_MIN_M}-{DISPLAY_RANGE_MAX_M}m | FOV: {FOV_DEG}°")
    print(f"   🎥 Camera: {'enabled' if VIDEO_SEQ_DIR is not None else 'disabled'}")
    print(f"   🕸  Net-line: {'enabled' if INCLUDE_NET else 'disabled'}"
          + (f" (dist tol={NET_DISTANCE_TOLERANCE}s, pitch tol={NET_PITCH_TOLERANCE}s)" if INCLUDE_NET else ""))
    print(f"   📊 Sonar Analysis: {'enabled' if SONAR_DISTANCE_RESULTS is not None else 'disabled'}")

    # --- Load sonar timestamps/frames ---
    sonar_csv_file = Path(EXPORTS_FOLDER) / "by_bag" / f"sensor_sonoptix_echo_image__{TARGET_BAG}_video.csv"
    if not sonar_csv_file.exists():
        print(f"❌ ERROR: Sonar CSV not found: {sonar_csv_file}")
        return

    print(f"   Loading sonar data: {sonar_csv_file.name}")
    t0 = time.time()
    df = load_df(sonar_csv_file)
    if "ts_utc" not in df.columns:
        if "t" not in df.columns:
            print("❌ ERROR: Missing timestamp column")
            return
        df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
    print(f"   ✅ Loaded {len(df)} sonar frames in {time.time()-t0:.2f}s")

    # --- Optional: load navigation (NetDistance/NetPitch) ---
    nav_complete = None
    if INCLUDE_NET:
        nav_file = Path(EXPORTS_FOLDER) / "by_bag" / f"navigation_plane_approximation__{TARGET_BAG}_data.csv"
        if nav_file.exists():
            t0 = time.time()
            nav_complete = pd.read_csv(nav_file)
            nav_complete["timestamp"] = pd.to_datetime(nav_complete["ts_utc"])
            nav_complete = nav_complete.sort_values("timestamp")
            print(f"   ✅ Loaded {len(nav_complete)} navigation records in {time.time()-t0:.2f}s")
            avail = [c for c in ["NetDistance", "NetPitch", "timestamp"] if c in nav_complete.columns]
            print(f"      Available: {avail}")
        else:
            print("   ⚠️ Navigation file not found; net-line overlay disabled")
            INCLUDE_NET = False

    # --- Optional: load camera index ---
    dfv = None
    video_idx = None
    if VIDEO_SEQ_DIR is not None:
        try:
            dfv = read_video_index(VIDEO_SEQ_DIR)
            dfv = dfv.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
            video_idx = pd.Index(dfv["ts_utc"])
            print(f"   ✅ Loaded {len(dfv)} camera index entries")
        except Exception as e:
            print(f"   ❌ Camera index load failed: {e}")
            VIDEO_SEQ_DIR = None

    # --- Frame subset and natural FPS ---
    N = len(df)
    i0 = max(0, START_IDX)
    i1 = min(N, END_IDX if END_IDX is not None else N)
    frame_indices = list(range(i0, i1, max(1, STRIDE)))
    print(f"   Frames: {i0}..{i1-1} (step {STRIDE}) => {len(frame_indices)}")

    ts_sample = pd.to_datetime(df.loc[frame_indices[:50], "ts_utc"], utc=True, errors="coerce").dropna().sort_values()
    dt_s = ts_sample.diff().dt.total_seconds().to_numpy()[1:]
    dt_s = dt_s[(dt_s > 1e-6) & (dt_s < 5.0)]
    natural_fps = float(np.clip(1.0/np.median(dt_s), 1.0, 60.0)) if dt_s.size else 15.0
    print(f"   Natural FPS: {natural_fps:.1f}")

    # --- Output writer (size set after first composed frame) ---
    out_dir = Path(EXPORTS_FOLDER) / "videos"
    out_dir.mkdir(exist_ok=True)
    first_ts = pd.to_datetime(df.loc[frame_indices[0], "ts_utc"], utc=True, errors="coerce")
    out_name = f"{TARGET_BAG}_optimized_sync_{'withcam_' if VIDEO_SEQ_DIR is not None else ''}{'withsonar_' if SONAR_DISTANCE_RESULTS is not None else ''}{'nonet_' if not INCLUDE_NET else ''}{ts_for_name(first_ts)}.mp4"
    out_path = out_dir / out_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    meta_extent = None

    # --- Color map and geometry helpers ---
    cmap = cm.get_cmap(CMAP_NAME).copy()
    cmap.set_bad((0,0,0,1))
    half_fov = np.deg2rad(FOV_DEG / 2)
    x_max_m = np.sin(half_fov) * DISPLAY_RANGE_MAX_M

    def x_px(xm: float) -> int:
        return int(round((xm + x_max_m) / (2 * x_max_m) * (CONE_W - 1)))

    def y_px(ym: float) -> int:
        # top = max range, bottom = 0; flip if desired
        y = int(round((ym - RANGE_MIN_M) / (DISPLAY_RANGE_MAX_M - RANGE_MIN_M) * (CONE_H - 1)))
        return (CONE_H - 1) - y  # Flipped for display

    grid_blue = (255, 150, 50)
    vehicle_center = (CONE_W // 2, CONE_H - 1)

    # --- frame creation (optimized logic) ---
    def make_cone_frame(frame_idx: int) -> np.ndarray | None:
        try:
            # sonar frame
            M0 = get_sonoptix_frame(df, frame_idx)
            if M0 is None:
                return None
            M = apply_flips(M0, flip_range=FLIP_RANGE, flip_beams=FLIP_BEAMS)
            Z = (enhance_intensity(
                    M, RANGE_MIN_M, RANGE_MAX_M, scale=ENH_SCALE, tvg=ENH_TVG,
                    alpha_db_per_m=ENH_ALPHA_DB_PER_M, r0=ENH_R0,
                    p_low=ENH_P_LOW, p_high=ENH_P_HIGH, gamma=ENH_GAMMA,
                    zero_aware=ENH_ZERO_AWARE, eps_log=ENH_EPS_LOG)
                 if USE_ENHANCED else M)

            cone, _ext = cone_raster_like_display_cell(
                Z, FOV_DEG, RANGE_MIN_M, RANGE_MAX_M, DISPLAY_RANGE_MAX_M, CONE_W, CONE_H
            )
            # Optimized path used vertical flip; keep it:
            cone = np.flipud(cone)
            # remember extent for metadata (use latest)
            try:
                meta_extent = tuple(map(float, _ext))
            except Exception:
                meta_extent = None

            cone_rgb = (cmap(np.ma.masked_invalid(cone))[:, :, :3] * 255).astype(np.uint8)
            cone_bgr = cv2.cvtColor(cone_rgb, cv2.COLOR_RGB2BGR)

            # --- optional net-line overlay (optimized sync) ---
            status_text = None
            line_color = (128, 128, 128)
            if INCLUDE_NET or SONAR_DISTANCE_RESULTS is not None:
                net_angle_deg = 0.0
                net_distance = None
                sonar_distance_m = None
                sync_status = "NO_DATA"

                ts_target = pd.to_datetime(df.loc[frame_idx, "ts_utc"], utc=True, errors="coerce")

                # --- DVL Navigation Data ---
                if INCLUDE_NET:
                    if nav_complete is not None and len(nav_complete) > 0:
                        diffs = abs(nav_complete["timestamp"] - ts_target)
                        idx = diffs.idxmin()
                        min_dt = diffs.iloc[idx]
                        rec = nav_complete.loc[idx]

                        if min_dt <= pd.Timedelta(f"{NET_DISTANCE_TOLERANCE}s") and "NetDistance" in rec and pd.notna(rec["NetDistance"]):
                            net_distance = float(rec["NetDistance"])
                            sync_status = "DISTANCE_OK"
                            if min_dt <= pd.Timedelta(f"{NET_PITCH_TOLERANCE}s") and "NetPitch" in rec and pd.notna(rec["NetPitch"]):
                                # NetPitch sign convention: invert sign to match display coordinate system
                                net_angle_deg = -float(np.degrees(rec["NetPitch"]))
                                sync_status = "FULL_SYNC"

                # --- Sonar Analysis Data ---
                if SONAR_DISTANCE_RESULTS is not None and len(SONAR_DISTANCE_RESULTS) > 0:
                    # Convert pixel distance to meters using the extent
                    try:
                        # Calculate pixel-to-meter conversion
                        x_min, x_max, y_min, y_max = meta_extent if meta_extent is not None else (-DISPLAY_RANGE_MAX_M, DISPLAY_RANGE_MAX_M, RANGE_MIN_M, DISPLAY_RANGE_MAX_M)
                        width_m = float(x_max - x_min)
                        height_m = float(y_max - y_min)
                        px2m_x = width_m / float(CONE_W)
                        px2m_y = height_m / float(CONE_H)
                        pixels_to_meters_avg = 0.5 * (px2m_x + px2m_y)

                        # Find closest sonar analysis measurement
                        sonar_diffs = abs(SONAR_DISTANCE_RESULTS["timestamp"] - ts_target)
                        sonar_idx = sonar_diffs.idxmin()
                        min_sonar_dt = sonar_diffs.iloc[sonar_idx]
                        sonar_rec = SONAR_DISTANCE_RESULTS.loc[sonar_idx]

                        if min_sonar_dt <= pd.Timedelta(f"{NET_DISTANCE_TOLERANCE}s") and sonar_rec.get("detection_success", False):
                            sonar_distance_px = sonar_rec["distance_pixels"]
                            sonar_distance_m = sonar_rec.get("distance_meters", sonar_distance_px * pixels_to_meters_avg)
                            if sync_status == "NO_DATA":
                                sync_status = "SONAR_ONLY"
                            elif sync_status in ["DISTANCE_OK", "FULL_SYNC"]:
                                sync_status = "BOTH_SYNC"
                    except Exception as e:
                        print(f"Warning: Could not sync sonar analysis data: {e}")

                # --- Draw overlays ---
                if sync_status in ["DISTANCE_OK", "FULL_SYNC", "SONAR_ONLY", "BOTH_SYNC"]:
                    net_half_width = 2.0

                    # Draw DVL net line (if available)
                    if net_distance is not None and net_distance <= DISPLAY_RANGE_MAX_M:
                        if sync_status == "FULL_SYNC":
                            dvl_line_color = (0, 255, 255)  # Yellow
                            dvl_center_color = (0, 0, 255)  # Red
                            dvl_label = f"DVL: {net_distance:.2f}m @ {net_angle_deg:.1f}°"
                        else:
                            dvl_line_color = (0, 165, 255)  # Orange
                            dvl_center_color = (0, 100, 255)  # Dark orange
                            dvl_label = f"DVL: {net_distance:.2f}m @ 0.0°"

                        net_angle_rad = np.radians(net_angle_deg) if sync_status == "FULL_SYNC" else 0.0

                        x1, y1 = -net_half_width, net_distance
                        x2, y2 = +net_half_width, net_distance

                        cos_a, sin_a = np.cos(net_angle_rad), np.sin(net_angle_rad)
                        rx1 = x1 * cos_a - (y1 - net_distance) * sin_a
                        ry1 = x1 * sin_a + (y1 - net_distance) * cos_a + net_distance
                        rx2 = x2 * cos_a - (y2 - net_distance) * sin_a
                        ry2 = x2 * sin_a + (y2 - net_distance) * cos_a + net_distance

                        px1, py1 = x_px(rx1), y_px(ry1)
                        px2, py2 = x_px(rx2), y_px(ry2)

                        cv2.line(cone_bgr, (px1, py1), (px2, py2), dvl_line_color, 3)
                        for (px, py) in [(px1, py1), (px2, py2)]:
                            cv2.circle(cone_bgr, (px, py), 4, (255, 255, 255), -1)
                            cv2.circle(cone_bgr, (px, py), 4, (0, 0, 0), 1)

                        center_px, center_py = x_px(0), y_px(net_distance)
                        cv2.circle(cone_bgr, (center_px, center_py), 3, dvl_center_color, -1)
                        cv2.circle(cone_bgr, (center_px, center_py), 3, (255, 255, 255), 1)

                    # Draw Sonar analysis line (if available)
                    if sonar_distance_m is not None and sonar_distance_m <= DISPLAY_RANGE_MAX_M:
                        sonar_line_color = (255, 0, 255)  # Magenta
                        sonar_center_color = (128, 0, 128)  # Dark magenta
                        sonar_label = f"SONAR: {sonar_distance_m:.2f}m"

                        # Sonar analysis is always horizontal (no angle info)
                        sx1, sy1 = -net_half_width, sonar_distance_m
                        sx2, sy2 = +net_half_width, sonar_distance_m

                        spx1, spy1 = x_px(sx1), y_px(sy1)
                        spx2, spy2 = x_px(sx2), y_px(sy2)

                        cv2.line(cone_bgr, (spx1, spy1), (spx2, spy2), sonar_line_color, 3)
                        for (px, py) in [(spx1, spy1), (spx2, spy2)]:
                            cv2.circle(cone_bgr, (px, py), 4, (255, 255, 255), -1)
                            cv2.circle(cone_bgr, (px, py), 4, (0, 0, 0), 1)

                        center_px, center_py = x_px(0), y_px(sonar_distance_m)
                        cv2.circle(cone_bgr, (center_px, center_py), 3, sonar_center_color, -1)
                        cv2.circle(cone_bgr, (center_px, center_py), 3, (255, 255, 255), 1)

                    # Create status text
                    status_lines = []
                    if net_distance is not None:
                        status_lines.append(dvl_label)
                    if sonar_distance_m is not None:
                        status_lines.append(sonar_label)
                    if status_lines:
                        status_text = " | ".join(status_lines)

                else:
                    if INCLUDE_NET or SONAR_DISTANCE_RESULTS is not None:
                        status_text = f"NET: NO SYNC DATA (tol: {NET_DISTANCE_TOLERANCE}s)"
                        line_color = (128, 128, 128)

            # grid rings and bearings
            for r_m in [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]:
                if r_m <= DISPLAY_RANGE_MAX_M:
                    ring_y = y_px(r_m)
                    radius_px = abs(vehicle_center[1] - ring_y)
                    cv2.circle(cone_bgr, vehicle_center, radius_px, grid_blue, 1)
                    if r_m in [2.0, 5.0, 10.0]:
                        label_pos = (vehicle_center[0] + 15, ring_y + 5)
                        cv2.putText(cone_bgr, f"{r_m:.0f}m", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_blue, 1)

            for bearing_deg in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
                if abs(bearing_deg) <= FOV_DEG / 2:
                    bearing_rad = np.radians(bearing_deg)
                    x_end = np.sin(bearing_rad) * DISPLAY_RANGE_MAX_M
                    y_end = np.cos(bearing_rad) * DISPLAY_RANGE_MAX_M
                    px_end, py_end = x_px(x_end), y_px(y_end)
                    cv2.line(cone_bgr, vehicle_center, (px_end, py_end), grid_blue, 1)
                    if bearing_deg % 30 == 0:
                        label_pos = (x_px(x_end * 0.85), y_px(y_end * 0.85))
                        cv2.putText(cone_bgr, f"{bearing_deg:+d}°", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_blue, 1)

            # annotate status if net considered
            if (INCLUDE_NET or SONAR_DISTANCE_RESULTS is not None) and status_text:
                cv2.putText(cone_bgr, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(cone_bgr, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # footer info
            frame_info = f"Frame {frame_idx}/{len(frame_indices)} | {TARGET_BAG} | Opt.Sync"
            cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            return cone_bgr

        except Exception as e:
            print(f"❌ Frame {frame_idx} error: {e}")
            return None

    # --- generate video ---
    frames_written = 0
    for k, frame_idx in enumerate(frame_indices):
        cone_frame = make_cone_frame(frame_idx)
        if cone_frame is None:
            continue

        # compose (camera optional)
        if VIDEO_SEQ_DIR is not None and dfv is not None and video_idx is not None:
            ts_target = pd.to_datetime(df.loc[frame_idx, "ts_utc"], utc=True, errors="coerce")
            idx_near = video_idx.get_indexer([ts_target], method="nearest")[0]
            ts_cam = dfv.loc[idx_near, "ts_utc"]
            dt = abs(ts_cam - ts_target)
            cam_file = VIDEO_SEQ_DIR / dfv.loc[idx_near, "file"]

            cam_bgr = load_png_bgr(cam_file)
            vh, vw0 = cam_bgr.shape[:2]
            scale = VIDEO_HEIGHT / vh
            cam_resized = cv2.resize(cam_bgr, (int(round(vw0 * scale)), VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)

            pad = np.zeros((CONE_H, PAD_BETWEEN, 3), dtype=np.uint8)
            composite = np.hstack([cam_resized, pad, cone_frame])

            ts_cam_loc   = to_local(ts_cam,   "Europe/Oslo")
            ts_sonar_loc = to_local(ts_target,"Europe/Oslo")
            put_text(composite, f"VIDEO  @ {ts_cam_loc:%Y-%m-%d %H:%M:%S.%f %Z}", 24, scale=FONT_SCALE)
            put_text(composite, f"SONAR  @ {ts_sonar_loc:%Y-%m-%d %H:%M:%S.%f %Z}   Δt={dt.total_seconds():.3f}s", 48, scale=FONT_SCALE)

        else:
            # sonar-only output canvas
            composite = cone_frame

        # init writer on first valid frame
        if writer is None:
            out_size = (composite.shape[1], composite.shape[0])
            writer = cv2.VideoWriter(str(out_path), fourcc, natural_fps, out_size, True)
            if not writer.isOpened():
                raise RuntimeError(f"Could not open writer: {out_path}")

        writer.write(composite)
        frames_written += 1

    if writer is not None:
        writer.release()
    # Write metadata beside the exported video for reproducible pixel<->meter mapping
    try:
        meta = {
            "format": "solaqua.optimized_sync.video.meta.v1",
            "target_bag": str(TARGET_BAG),
            "created_at_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "frames_written": int(frames_written),
            "natural_fps": float(natural_fps),
            "fov_deg": float(FOV_DEG),
            "range_min_m": float(RANGE_MIN_M),
            "range_max_m": float(RANGE_MAX_M),
            "display_range_max_m": float(DISPLAY_RANGE_MAX_M),
            "cone_w": int(CONE_W),
            "cone_h": int(CONE_H),
            "cone_flip_vertical": bool(CONE_FLIP_VERTICAL),
            "cmap": str(CMAP_NAME),
            "include_net": bool(INCLUDE_NET),
            "include_sonar_analysis": SONAR_DISTANCE_RESULTS is not None,
            "flip_range": bool(FLIP_RANGE),
            "flip_beams": bool(FLIP_BEAMS),
        }
        if meta_extent is not None:
            meta["extent"] = list(meta_extent)
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"\n🎉 DONE! Wrote {frames_written} frames to {out_path} @ {natural_fps:.2f} FPS")
        print(f"Metadata saved to: {meta_path}")
    except Exception as e:
        print(f"Warning: could not write metadata: {e}")
