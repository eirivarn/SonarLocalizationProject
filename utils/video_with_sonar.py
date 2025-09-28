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

def export_optimized_sonar_video(
    TARGET_BAG,
    EXPORTS_FOLDER,
    START_IDX=0,
    END_IDX=600,
    STRIDE=1,
):
    """
    Optimized sonar video generation moved from notebook 10.

    This function encapsulates the optimized, frequency-aware synchronization
    and net overlay generation that was implemented interactively in the
    notebook. The code is kept identical in logic to the notebook; it is
    wrapped here so it can be imported and reused programmatically.

    Args:
        TARGET_BAG: bag identifier string
        EXPORTS_FOLDER: path to exports folder
        START_IDX: start frame index
        END_IDX: end frame index (inclusive upper bound-like behavior)
        STRIDE: frame stride
    """
    # The following block is a near-verbatim move of the notebook's core
    # optimized sonar video generation logic. Variable names and flow
    # are preserved to avoid changing the algorithm.

    import cv2
    import numpy as np
    import time
    from datetime import datetime
    from pathlib import Path
    import matplotlib.cm as cm
    import pandas as pd

    # Import fast utility functions from this package
    try:
        from utils.sonar_utils import (
            load_df, get_sonoptix_frame, apply_flips, enhance_intensity,
            cone_raster_like_display_cell
        )
    except Exception:
        # Fall back to top-level imports if relative import fails
        from sonar_utils import (
            load_df, get_sonoptix_frame, apply_flips, enhance_intensity,
            cone_raster_like_display_cell
        )

    print("🛠️ OPTIMIZED SONAR VIDEO (FREQUENCY-AWARE SYNCHRONIZATION + NET)")
    print("=" * 70)

    # OPTIMIZED CONFIGURATION BASED ON FREQUENCY ANALYSIS
    RANGE_MIN_M = 0.0
    RANGE_MAX_M = 30.0
    DISPLAY_RANGE_MAX_M = 4.0
    FOV_DEG = 120.0
    FLIP_BEAMS = True
    FLIP_RANGE = False

    # FREQUENCY-OPTIMIZED SYNCHRONIZATION SETTINGS
    NET_DISTANCE_TOLERANCE = 0.5
    NET_PITCH_TOLERANCE = 0.3
    GENERAL_SENSOR_TOLERANCE = 1.0

    # Enhanced settings
    USE_ENHANCED = True
    ENH_SCALE = "db"
    ENH_TVG = "amplitude"
    ENH_ALPHA_DB_PER_M = 0.0
    ENH_R0 = 1e-2
    ENH_P_LOW = 1.0
    ENH_P_HIGH = 99.5
    ENH_GAMMA = 0.9
    ENH_ZERO_AWARE = True
    ENH_EPS_LOG = 1e-6

    # Video settings
    CONE_W, CONE_H = 900, 700
    VIDEO_WIDTH = CONE_W
    VIDEO_HEIGHT = CONE_H
    START_IDX = START_IDX
    END_IDX = END_IDX
    STRIDE = STRIDE

    print(f"🎯 OPTIMIZED CONFIGURATION:")
    print(f"   Target Bag: {TARGET_BAG}")
    print(f"   Cone Size: {CONE_W}x{CONE_H}")
    print(f"   Range: {RANGE_MIN_M}-{DISPLAY_RANGE_MAX_M}m")
    print(f"   FOV: {FOV_DEG}°")
    print(f"   🔧 OPTIMIZED TOLERANCES (based on frequency analysis):")
    print(f"      • Net Distance: {NET_DISTANCE_TOLERANCE}s (was 1.0s)")
    print(f"      • Net Pitch: {NET_PITCH_TOLERANCE}s (was 5.0s)")
    print(f"      • Other Sensors: {GENERAL_SENSOR_TOLERANCE}s")

    # STEP 1: EFFICIENT DATA LOADING WITH NETPITCH INCLUDED
    print("\n📡 LOADING DATA WITH OPTIMIZED APPROACH...")

    sonar_csv_file = Path(EXPORTS_FOLDER) / "by_bag" / f"sensor_sonoptix_echo_image__{TARGET_BAG}_video.csv"
    if not sonar_csv_file.exists():
        print(f"❌ ERROR: Sonar CSV not found: {sonar_csv_file}")
        return
    else:
        print(f"   Loading sonar data: {sonar_csv_file.name}")
        load_start = time.time()
        df = load_df(sonar_csv_file)
        if "ts_utc" not in df.columns:
            if "t" not in df.columns:
                print("❌ ERROR: Missing timestamp column")
                return
            else:
                df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
        load_time = time.time() - load_start
        print(f"   ✅ Loaded {len(df)} sonar frames in {load_time:.2f}s")

        # OPTIMIZED NAVIGATION DATA LOADING (includes NetPitch from start)
        print(f"   Loading COMPLETE navigation data (including NetPitch)...")
        nav_file_complete = Path(EXPORTS_FOLDER) / "by_bag" / f"navigation_plane_approximation__{TARGET_BAG}_data.csv"

        nav_complete = None
        if nav_file_complete.exists():
            nav_load_start = time.time()
            nav_complete = pd.read_csv(nav_file_complete)
            nav_complete['timestamp'] = pd.to_datetime(nav_complete['ts_utc'])
            nav_complete = nav_complete.sort_values('timestamp')
            nav_load_time = time.time() - nav_load_start
            print(f"   ✅ Loaded {len(nav_complete)} navigation records in {nav_load_time:.2f}s")
            print(f"      Available columns: {[col for col in nav_complete.columns if col in ['NetDistance', 'NetPitch', 'timestamp']]}")
            if 'NetPitch' in nav_complete.columns:
                valid_pitch_count = nav_complete['NetPitch'].dropna().shape[0]
                print(f"      ✅ NetPitch: {valid_pitch_count} valid records")
            else:
                print(f"      ❌ NetPitch column missing!")
            if 'NetDistance' in nav_complete.columns:
                valid_dist_count = nav_complete['NetDistance'].dropna().shape[0]
                print(f"      ✅ NetDistance: {valid_dist_count} valid records")

        # Frame selection
        N = len(df)
        i0 = max(0, START_IDX)
        i1 = min(N, END_IDX if END_IDX else N)
        frame_indices = list(range(i0, i1, STRIDE))

        print(f"   Processing frames {i0}-{i1-1} (total: {len(frame_indices)})")

        # Calculate natural FPS
        ts = pd.to_datetime(df.loc[frame_indices[:50], "ts_utc"], utc=True, errors="coerce").dropna().sort_values()
        dt_s = ts.diff().dt.total_seconds().to_numpy()[1:]
        dt_s = dt_s[(dt_s > 1e-6) & (dt_s < 5.0)]
        natural_fps = float(np.clip(1.0/np.median(dt_s), 1.0, 60.0)) if dt_s.size else 15.0

        print(f"   Natural FPS: {natural_fps:.1f}")

        # STEP 2: SETUP OUTPUT
        output_dir = Path(EXPORTS_FOLDER) / "videos"
        output_dir.mkdir(exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{TARGET_BAG}_optimized_sync_{timestamp_str}.mp4"
        output_path = output_dir / output_filename

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, natural_fps, (VIDEO_WIDTH, VIDEO_HEIGHT))

        print(f"   Output: {output_filename}")

        cmap_name = "viridis"
        cmap = cm.get_cmap(cmap_name).copy()
        cmap.set_bad((0,0,0,1))

        half_fov = np.deg2rad(FOV_DEG / 2)
        x_max_m = np.sin(half_fov) * DISPLAY_RANGE_MAX_M

        def x_px(xm):
            return int(round((xm + x_max_m) / (2 * x_max_m) * (CONE_W - 1)))

        def y_px(ym):
            return int(round((DISPLAY_RANGE_MAX_M - ym) / DISPLAY_RANGE_MAX_M * (CONE_H - 1)))

        # STEP 3: OPTIMIZED FRAME PROCESSING FUNCTION
        def create_optimized_frame(frame_idx):
            try:
                M0 = get_sonoptix_frame(df, frame_idx)
                if M0 is None:
                    return None

                M = apply_flips(M0, flip_range=FLIP_RANGE, flip_beams=FLIP_BEAMS)

                if USE_ENHANCED:
                    Z = enhance_intensity(M, RANGE_MIN_M, RANGE_MAX_M,
                                        scale=ENH_SCALE, tvg=ENH_TVG, alpha_db_per_m=ENH_ALPHA_DB_PER_M,
                                        r0=ENH_R0, p_low=ENH_P_LOW, p_high=ENH_P_HIGH,
                                        gamma=ENH_GAMMA, zero_aware=ENH_ZERO_AWARE, eps_log=ENH_EPS_LOG)
                else:
                    Z = M

                cone, (x_min, x_max, y_min, y_max) = cone_raster_like_display_cell(
                    Z, FOV_DEG, RANGE_MIN_M, RANGE_MAX_M, DISPLAY_RANGE_MAX_M, CONE_W, CONE_H,
                )

                cone = np.flipud(cone)

                cone_rgb = (cmap(np.ma.masked_invalid(cone))[:, :, :3] * 255).astype(np.uint8)
                cone_bgr = cv2.cvtColor(cone_rgb, cv2.COLOR_RGB2BGR)

                net_angle_deg = 0.0
                net_distance = None
                sync_status = "NO_DATA"

                ts_target = pd.to_datetime(df.loc[frame_idx, "ts_utc"], utc=True, errors="coerce")

                if nav_complete is not None and len(nav_complete) > 0:
                    nav_time_diffs = abs(nav_complete['timestamp'] - ts_target)
                    closest_nav_idx = nav_time_diffs.idxmin()
                    min_time_diff = nav_time_diffs.iloc[closest_nav_idx]

                    closest_nav_record = nav_complete.loc[closest_nav_idx]

                    if min_time_diff <= pd.Timedelta(f'{NET_DISTANCE_TOLERANCE}s'):
                        if 'NetDistance' in closest_nav_record and pd.notna(closest_nav_record['NetDistance']):
                            net_distance = closest_nav_record['NetDistance']
                            sync_status = "DISTANCE_OK"
                            if min_time_diff <= pd.Timedelta(f'{NET_PITCH_TOLERANCE}s'):
                                if 'NetPitch' in closest_nav_record and pd.notna(closest_nav_record['NetPitch']):
                                    net_angle_rad = closest_nav_record['NetPitch']
                                    net_angle_deg = np.degrees(net_angle_rad)
                                    sync_status = "FULL_SYNC"

                    if frame_idx < 3:
                        print(f"      Frame {frame_idx}: Δt={min_time_diff.total_seconds():.3f}s, Status={sync_status}")

                vehicle_center = (CONE_W // 2, CONE_H - 1)
                grid_blue = (255, 150, 50)

                for r_m in [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]:
                    if r_m <= DISPLAY_RANGE_MAX_M:
                        ring_y = y_px(r_m)
                        radius_px = abs(vehicle_center[1] - ring_y)
                        cv2.circle(cone_bgr, vehicle_center, radius_px, grid_blue, 1)
                        if r_m in [2.0, 5.0, 10.0]:
                            label_pos = (vehicle_center[0] + 15, ring_y + 5)
                            cv2.putText(cone_bgr, f"{r_m:.0f}m", label_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_blue, 1)

                for bearing_deg in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
                    if abs(bearing_deg) <= FOV_DEG / 2:
                        bearing_rad = np.radians(bearing_deg)
                        x_end = np.sin(bearing_rad) * DISPLAY_RANGE_MAX_M
                        y_end = np.cos(bearing_rad) * DISPLAY_RANGE_MAX_M
                        px_end, py_end = x_px(x_end), y_px(y_end)
                        cv2.line(cone_bgr, vehicle_center, (px_end, py_end), grid_blue, 1)
                        if bearing_deg % 30 == 0:
                            label_pos = (x_px(x_end * 0.85), y_px(y_end * 0.85))
                            cv2.putText(cone_bgr, f"{bearing_deg:+d}°", label_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_blue, 1)

                if sync_status in ["DISTANCE_OK", "FULL_SYNC"] and net_distance is not None and net_distance <= DISPLAY_RANGE_MAX_M:
                    net_half_width = 2.0

                    if sync_status == "FULL_SYNC":
                        net_angle_rad = np.radians(net_angle_deg)
                        line_color = (0, 255, 255)
                        center_color = (0, 0, 255)
                        status_text = f"NET: {net_distance:.2f}m @ {net_angle_deg:.1f}° (SYNCED)"
                    else:
                        net_angle_rad = 0.0
                        net_angle_deg = 0.0
                        line_color = (0, 165, 255)
                        center_color = (0, 100, 255)
                        status_text = f"NET: {net_distance:.2f}m @ 0.0° (DIST-ONLY)"

                    x1, y1 = -net_half_width, net_distance
                    x2, y2 = net_half_width, net_distance

                    cos_a, sin_a = np.cos(net_angle_rad), np.sin(net_angle_rad)
                    rx1 = x1 * cos_a - (y1 - net_distance) * sin_a
                    ry1 = x1 * sin_a + (y1 - net_distance) * cos_a + net_distance
                    rx2 = x2 * cos_a - (y2 - net_distance) * sin_a
                    ry2 = x2 * sin_a + (y2 - net_distance) * cos_a + net_distance

                    px1, py1 = x_px(rx1), y_px(ry1)
                    px2, py2 = x_px(rx2), y_px(ry2)

                    cv2.line(cone_bgr, (px1, py1), (px2, py2), line_color, 5)
                    cv2.circle(cone_bgr, (px1, py1), 6, (255, 255, 255), -1)
                    cv2.circle(cone_bgr, (px2, py2), 6, (255, 255, 255), -1)
                    cv2.circle(cone_bgr, (px1, py1), 6, (0, 0, 0), 2)
                    cv2.circle(cone_bgr, (px2, py2), 6, (0, 0, 0), 2)

                    center_px, center_py = x_px(0), y_px(net_distance)
                    cv2.circle(cone_bgr, (center_px, center_py), 5, center_color, -1)
                    cv2.circle(cone_bgr, (center_px, center_py), 5, (255, 255, 255), 2)
                else:
                    status_text = f"NET: NO SYNC DATA (tolerance: {NET_DISTANCE_TOLERANCE}s)"
                    line_color = (128, 128, 128)

                cv2.putText(cone_bgr, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(cone_bgr, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           line_color, 2, cv2.LINE_AA)

                frame_info = f"Frame {frame_idx}/{len(frame_indices)} | {TARGET_BAG} | Opt.Sync"
                cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 0), 1, cv2.LINE_AA)

                return cone_bgr

            except Exception as e:
                print(f"❌ Frame {frame_idx} error: {e}")
                return None

        # STEP 4: GENERATE VIDEO WITH PERFORMANCE TRACKING
        print("\n🎬 GENERATING OPTIMIZED VIDEO...")
        print("-" * 50)

        frames_written = 0
        processing_times = []
        sync_stats = {"FULL_SYNC": 0, "DISTANCE_OK": 0, "NO_DATA": 0}
        start_total = time.time()

        for k, frame_idx in enumerate(frame_indices):
            frame_start = time.time()
            frame = create_optimized_frame(frame_idx)
            if frame is not None:
                video_writer.write(frame)
                frames_written += 1
                frame_time = time.time() - frame_start
                processing_times.append(frame_time)
                if frames_written % 25 == 0:
                    progress = frames_written / len(frame_indices) * 100
                    avg_time = np.mean(processing_times[-25:])
                    fps_processing = 1.0 / avg_time if avg_time > 0 else 0
                    eta_seconds = (len(frame_indices) - frames_written) * avg_time
                    eta_minutes = eta_seconds / 60
                    print(f"   {progress:5.1f}% | Frame {frames_written:3d}/{len(frame_indices)} | "
                          f"{avg_time:.3f}s/frame | {fps_processing:.1f} proc.FPS | ETA: {eta_minutes:.1f}min")

        # STEP 5: FINALIZE AND REPORT
        video_writer.release()
        cv2.destroyAllWindows()
        total_time = time.time() - start_total

        print("\n" + "=" * 70)
        print("🎉 OPTIMIZED SONAR VIDEO COMPLETE!")
        print(f"   📁 Output: {output_path}")
        print(f"   🎞 Frames: {frames_written}/{len(frame_indices)}")
        print(f"   ⏱  Total time: {total_time/60:.1f} minutes")

        if processing_times:
            avg_frame_time = np.mean(processing_times)
            print("   📊 Performance:")
            print(f"      • Avg per frame: {avg_frame_time:.3f}s")
            print(f"      • Processing speed: {frames_written/total_time:.1f} FPS")
            theoretical_realtime = 1.0 / natural_fps
            if avg_frame_time > 0:
                speed_factor = theoretical_realtime / avg_frame_time
                if speed_factor >= 1:
                    print(f"      • {speed_factor:.1f}x faster than real-time! 🚀")
                else:
                    print(f"      • {1/speed_factor:.1f}x slower than real-time")

        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            video_duration = frames_written / natural_fps
            print(f"   📦 File size: {file_size_mb:.1f} MB")
            print(f"   ⏱ Duration: {video_duration:.1f}s")
            print("\n🎯 OPTIMIZATION FEATURES:")
            print(f"   ✅ Frequency-optimized sync tolerances ({NET_DISTANCE_TOLERANCE}s dist, {NET_PITCH_TOLERANCE}s pitch)")
            print("   ✅ Single-pass navigation data loading (includes NetPitch)")
            print("   ✅ Eliminated per-frame file I/O")
            print("   ✅ Status-based net line styling (yellow=full sync, orange=dist-only)")
            print("   ✅ Real-time synchronization quality indicators")
            print("\n✅ SUCCESS! Optimized video saved to:")
            print(f"   {output_path}")
        else:
            print("\n❌ ERROR: Video file not created!")
