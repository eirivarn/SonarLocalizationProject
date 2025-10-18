# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from utils.sonar_utils import (
    load_df, get_sonoptix_frame,
    enhance_intensity, read_video_index, apply_flips, cone_raster_like_display_cell,
    get_video_overlay_info
)
from utils.sonar_config import (
    SONAR_VIS_DEFAULTS, ENHANCE_DEFAULTS,
    CONE_W_DEFAULT, CONE_H_DEFAULT, CONE_FLIP_VERTICAL_DEFAULT,
    CMAP_NAME_DEFAULT, DISPLAY_RANGE_MAX_M_DEFAULT, FOV_DEG_DEFAULT,
    EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS,
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

def prepare_three_system_data(
    target_bag: str,
    exports_folder: Path,
    net_analysis_results: pd.DataFrame,
    raw_data: dict,
    fft_csv_path: Path | None = None,
    use_notebook08_sync: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Prepare synchronized data for three-system video generation.
    
    Returns:
        tuple: (net_analysis_sync, fft_net_data_sync)
    """
    print("🔄 PREPARING THREE-SYSTEM SYNCHRONIZED DATA")
    print("=" * 60)
    
    # STEP 1: Load FFT data using notebook 08's method
    fft_net_data = None
    if fft_csv_path and fft_csv_path.exists():
        try:
            # Try to use notebook 08's FFT loading methods
            from utils.relative_fft_analysis import load_relative_fft_data, convert_fft_to_xy_coordinates
            
            print("✓ Using notebook 08's FFT loading methods...")
            fft_net_data = load_relative_fft_data(fft_csv_path)
            fft_net_data = convert_fft_to_xy_coordinates(fft_net_data)
            print(f"✓ Loaded FFT data via notebook 08 method: {len(fft_net_data)} records")
            
        except ImportError:
            print("✗ Notebook 08 FFT utilities not available - using fallback method")
            # Fallback to manual method
            fft_net_data = pd.read_csv(fft_csv_path)
            fft_net_data = fft_net_data.rename(columns={'time': 'timestamp', 'distance': 'distance_cm', 'pitch': 'pitch_rad'})
            
            # Convert units
            if 'distance_cm' in fft_net_data.columns:
                fft_net_data['distance_m'] = fft_net_data['distance_cm'] / 100.0
            if 'pitch_rad' in fft_net_data.columns:
                fft_net_data['pitch_deg'] = np.degrees(fft_net_data['pitch_rad'])
            
            # Handle timestamps - convert Unix timestamps to timezone-naive datetime
            fft_timestamp_values = pd.to_numeric(fft_net_data['timestamp'], errors='coerce')
            if fft_timestamp_values.min() > 1e9:
                fft_net_data['timestamp'] = pd.to_datetime(fft_timestamp_values, unit='s', utc=True).dt.tz_localize(None)
                print(f"✓ Converted Unix timestamps to timezone-naive")
        except Exception as e:
            print(f"✗ FFT loading error: {e}")
            fft_net_data = None
    elif fft_csv_path:
        print(f"✗ FFT file not found: {fft_csv_path.name}")
    
    # STEP 2: Use notebook 08's synchronization approach if available
    if fft_net_data is not None and use_notebook08_sync:
        try:
            # Import notebook 08's synchronization utilities
            from utils.net_relative_utils import run_complete_three_system_analysis
            
            print("✓ Using notebook 08's three-system synchronization...")
            
            # Run the same synchronization that works in notebook 08  
            sync_df, stats, _, _, _, _ = run_complete_three_system_analysis(
                df_sonar=net_analysis_results,
                df_nav=raw_data['navigation'],
                df_fft=fft_net_data,
                target_bag=target_bag,
                exports_folder=exports_folder
            )
            
            print(f"✓ Three-system synchronization completed: {len(sync_df)} synchronized records")
            
            # Extract the synchronized data for video generation
            net_analysis_sync = sync_df[['timestamp', 'sonar_distance_m', 'sonar_angle_deg', 'detection_success']].copy()
            net_analysis_sync = net_analysis_sync.rename(columns={
                'sonar_distance_m': 'distance_meters',
                'sonar_angle_deg': 'angle_degrees'
            })
            net_analysis_sync['distance_pixels'] = net_analysis_sync['distance_meters'] / 0.01  # Approximate
            net_analysis_sync['tracking_status'] = 'SYNCHRONIZED'
            
            # Use synchronized FFT data
            fft_net_data_sync = sync_df[['timestamp', 'fft_distance_m', 'fft_pitch_deg']].copy()
            fft_net_data_sync = fft_net_data_sync.rename(columns={
                'fft_distance_m': 'distance_m',
                'fft_pitch_deg': 'pitch_deg'
            })
            
            print(f"✓ Prepared synchronized datasets for video generation")
            return net_analysis_sync, fft_net_data_sync
            
        except ImportError:
            print("✗ Could not import notebook 08 synchronization utilities")
            print("Falling back to individual synchronization...")
            
        except Exception as e:
            print(f"✗ Notebook 08 synchronization failed: {e}")
            print("Falling back to individual synchronization...")
    
    # STEP 3: Fallback synchronization - ensure timezone-naive timestamps
    print("✓ Using fallback synchronization method")
    
    net_analysis_sync = net_analysis_results.copy()
    if 'timestamp' in net_analysis_sync.columns:
        net_analysis_sync['timestamp'] = pd.to_datetime(net_analysis_sync['timestamp']).dt.tz_localize(None)
    
    fft_net_data_sync = None
    if fft_net_data is not None:
        fft_net_data_sync = fft_net_data.copy()
        if 'timestamp' in fft_net_data_sync.columns:
            if hasattr(fft_net_data_sync['timestamp'].iloc[0], 'tz') and fft_net_data_sync['timestamp'].dt.tz is not None:
                fft_net_data_sync['timestamp'] = fft_net_data_sync['timestamp'].dt.tz_localize(None)
    
    print(f"✓ Fallback synchronization completed")
    return net_analysis_sync, fft_net_data_sync

def export_optimized_sonar_video(
    TARGET_BAG: str,
    EXPORTS_FOLDER: Path | None,
    START_IDX: int = 0,
    END_IDX: int | None = 600,
    STRIDE: int = 1,
    # --- camera (automated or manual) ---
    VIDEO_SEQ_DIR: Path | None = None,   # if provided, overrides auto-detection
    AUTO_DETECT_VIDEO: bool = True,      # if True, automatically find video files based on bag name
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
    SONAR_RESULTS: pd.DataFrame | None = None,  # DataFrame with sonar analysis results
    # --- FFT net position overlay (optional) ---
    FFT_NET_DATA: pd.DataFrame | None = None,  # DataFrame with FFT net position data
    # --- NEW: DVL data and sync parameters ---
    DVL_NAV_DATA: pd.DataFrame | None = None,  # DataFrame with DVL navigation data
    SYNC_WINDOW_SECONDS: float = 0.1,  # Time window for synchronization
):
    """
    Optimized sonar + (optional) net-line overlay + (optional) sonar analysis overlay.
    If VIDEO_SEQ_DIR is given, the output includes the actual camera frame (side-by-side).
    If SONAR_RESULTS is provided, displays both DVL and sonar analysis distances.
    If FFT_NET_DATA is provided, displays FFT net position overlay in cyan.

    Output is saved under EXPORTS_FOLDER / 'videos' with an 'optimized_sync' name.
    """
    import time
    import matplotlib.cm as cm

    print("OPTIMIZED SONAR VIDEO WITH THREE-SYSTEM SYNCHRONIZATION")
    print("=" * 70)
    print(f"Target Bag: {TARGET_BAG}")
    print(f"   Cone Size: {CONE_W}x{CONE_H}")
    print(f"   Range: {RANGE_MIN_M}-{DISPLAY_RANGE_MAX_M}m | FOV: {FOV_DEG}°")
    
    # --- Auto-detect video files if enabled ---
    if VIDEO_SEQ_DIR is None and AUTO_DETECT_VIDEO:
        from utils.sonar_config import VIDEO_CONFIG
        
        # Check if video overlay is globally enabled
        if VIDEO_CONFIG.get('enable_video_overlay', False):
            print(f"Auto-detecting video files for bag: {TARGET_BAG}")
            video_info = get_video_overlay_info(TARGET_BAG)
            if video_info:
                # Use the PNG frames directory
                frames_dir = video_info['video_frames_dir']
                print(f"Found video frames: {frames_dir.name}")
                VIDEO_SEQ_DIR = frames_dir  # Set the frames directory
                print(f"Using frames directory: {VIDEO_SEQ_DIR}")
            else:
                print(f"No video files found for bag {TARGET_BAG}")
        else:
            print(f"Video overlay disabled in configuration (enable_video_overlay=False)")
    
    print(f"Camera: {'enabled' if VIDEO_SEQ_DIR is not None else 'disabled'}")
    print(f"Net-line: {'enabled' if INCLUDE_NET else 'disabled'}"
          + (f" (dist tol={NET_DISTANCE_TOLERANCE}s, pitch tol={NET_PITCH_TOLERANCE}s)" if INCLUDE_NET else ""))
    print(f"Sonar Analysis: {'enabled' if SONAR_RESULTS is not None else 'disabled'}")
    print(f"FFT Data: {'enabled' if FFT_NET_DATA is not None else 'disabled'}")

    # --- Resolve exports folder and load sonar timestamps/frames ---
    exports_root = Path(EXPORTS_FOLDER) if EXPORTS_FOLDER is not None else Path(EXPORTS_DIR_DEFAULT)
    sonar_csv_file = exports_root / EXPORTS_SUBDIRS.get('by_bag', 'by_bag') / f"sensor_sonoptix_echo_image__{TARGET_BAG}_video.csv"
    if not sonar_csv_file.exists():
        print(f"ERROR: Sonar CSV not found: {sonar_csv_file}")
        return

    print(f"   Loading sonar data: {sonar_csv_file.name}")
    t0 = time.time()
    df = load_df(sonar_csv_file)
    if "ts_utc" not in df.columns:
        if "t" not in df.columns:
            print("ERROR: Missing timestamp column")
            return
        df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
    print(f"Loaded {len(df)} sonar frames in {time.time()-t0:.2f}s")

    # --- STEP 1: Load and synchronize all three systems using comparison analysis approach ---
    
    # Load DVL data if not provided explicitly (same method as comparison analysis)
    if DVL_NAV_DATA is None and INCLUDE_NET:
        try:
            # Use same loading method as comparison analysis
            import utils.net_distance_analysis as sda
            BY_BAG_FOLDER = exports_root / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
            raw_data, _ = sda.load_all_distance_data_for_bag(TARGET_BAG, BY_BAG_FOLDER)
            
            dvl_data = raw_data.get('navigation', None)
            if dvl_data is not None:
                # Ensure proper timestamp format
                if 'timestamp' not in dvl_data.columns and 'ts_utc' in dvl_data.columns:
                    dvl_data = dvl_data.copy()
                    dvl_data['timestamp'] = pd.to_datetime(dvl_data['ts_utc'])
                elif 'timestamp' in dvl_data.columns:
                    dvl_data['timestamp'] = pd.to_datetime(dvl_data['timestamp'])
                DVL_NAV_DATA = dvl_data
                print(f"   Auto-loaded DVL data: {len(DVL_NAV_DATA)} records")
            else:
                print("   No DVL navigation data found")
        except Exception as e:
            print(f"   DVL loading error: {e}")
            DVL_NAV_DATA = None

    # Synchronize all timestamps to sonar reference (same as comparison analysis)
    if df is not None and len(df) > 0:
        # Use first sonar frame as reference time
        reference_time = pd.to_datetime(df.iloc[0]["ts_utc"], utc=True, errors="coerce")
        
        # Synchronize FFT data
        if FFT_NET_DATA is not None:
            fft_sync = FFT_NET_DATA.copy()
            if 'timestamp' in fft_sync.columns:
                fft_relative_seconds = pd.to_numeric(fft_sync['timestamp'], errors='coerce')
                
                # Check if timestamps are relative (small values) or absolute
                if fft_relative_seconds.max() < 86400:  # Less than 1 day in seconds = relative
                    fft_sync['timestamp'] = reference_time + pd.to_timedelta(fft_relative_seconds, unit='s')
                    print(f"   Applied relative timestamp conversion for FFT data")
                else:
                    fft_sync['timestamp'] = pd.to_datetime(fft_sync['timestamp'])
                    print(f"   Used absolute timestamps for FFT data")
            FFT_NET_DATA = fft_sync
        
        # Synchronize DVL data  
        if DVL_NAV_DATA is not None:
            dvl_sync = DVL_NAV_DATA.copy()
            if 'timestamp' not in dvl_sync.columns and 'ts_utc' in dvl_sync.columns:
                dvl_sync['timestamp'] = pd.to_datetime(dvl_sync['ts_utc'])
            elif 'timestamp' in dvl_sync.columns:
                dvl_sync['timestamp'] = pd.to_datetime(dvl_sync['timestamp'])
            DVL_NAV_DATA = dvl_sync

        # Verify time overlap between systems
        if DVL_NAV_DATA is not None or FFT_NET_DATA is not None:
            sonar_time_range = (pd.to_datetime(df["ts_utc"]).min(), pd.to_datetime(df["ts_utc"]).max())
            print(f"   Time Range Verification:")
            print(f"     Sonar: {sonar_time_range[0]} to {sonar_time_range[1]}")
            
            if DVL_NAV_DATA is not None:
                dvl_time_range = (DVL_NAV_DATA['timestamp'].min(), DVL_NAV_DATA['timestamp'].max())
                print(f"     DVL:   {dvl_time_range[0]} to {dvl_time_range[1]}")
            
            if FFT_NET_DATA is not None:
                fft_time_range = (FFT_NET_DATA['timestamp'].min(), FFT_NET_DATA['timestamp'].max())
                print(f"     FFT:   {fft_time_range[0]} to {fft_time_range[1]}")

    # --- Optional: load navigation (NetDistance/NetPitch) - use synchronized DVL data ---
    nav_complete = DVL_NAV_DATA
    if nav_complete is not None:
        print(f"   Using synchronized DVL data: {len(nav_complete)} records")
        avail = [c for c in ["NetDistance", "NetPitch", "timestamp"] if c in nav_complete.columns]
        print(f"      Available: {avail}")
    elif INCLUDE_NET:
        # Fallback to original loading method
        nav_file = exports_root / EXPORTS_SUBDIRS.get('by_bag', 'by_bag') / f"navigation_plane_approximation__{TARGET_BAG}_data.csv"
        if nav_file.exists():
            t0 = time.time()
            nav_complete = pd.read_csv(nav_file)
            nav_complete["timestamp"] = pd.to_datetime(nav_complete["ts_utc"])
            nav_complete = nav_complete.sort_values("timestamp")
            print(f"Loaded {len(nav_complete)} navigation records in {time.time()-t0:.2f}s")
            avail = [c for c in ["NetDistance", "NetPitch", "timestamp"] if c in nav_complete.columns]
            print(f"      Available: {avail}")
        else:
            print("Navigation file not found; net-line overlay disabled")
            INCLUDE_NET = False

    # --- Optional: load camera index ---
    dfv = None
    video_idx = None
    if VIDEO_SEQ_DIR is not None:
        try:
            dfv = read_video_index(VIDEO_SEQ_DIR)
            dfv = dfv.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
            video_idx = pd.Index(dfv["ts_utc"])
            print(f"Loaded {len(dfv)} camera index entries")
        except Exception as e:
            print(f"Camera index load failed: {e}")
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
    out_name = f"{TARGET_BAG}_optimized_sync_{'withcam_' if VIDEO_SEQ_DIR is not None else ''}{'withsonar_' if SONAR_RESULTS is not None else ''}{'nonet_' if not INCLUDE_NET else ''}{ts_for_name(first_ts)}.mp4"
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

    # --- frame creation (optimized logic with FFT support) ---
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

            # --- optional net-line overlay (optimized sync with FFT support) ---
            status_text = None
            line_color = (128, 128, 128)
            if INCLUDE_NET or SONAR_RESULTS is not None or FFT_NET_DATA is not None:
                net_angle_deg = 0.0
                net_distance = None
                sonar_distance_m = None
                fft_distance_m = None
                fft_pitch_deg = None
                sync_status = "NO_DATA"

                # CRITICAL FIX: Use timezone-naive timestamps consistently
                ts_target = pd.to_datetime(df.loc[frame_idx, "ts_utc"], utc=True, errors="coerce")
                # Convert to timezone-naive if it's timezone-aware
                if hasattr(ts_target, 'tz') and ts_target.tz is not None:
                    ts_target = ts_target.tz_localize(None)

                # --- DVL Navigation Data ---
                if INCLUDE_NET:
                    if nav_complete is not None and len(nav_complete) > 0:
                        # Ensure nav_complete timestamps are timezone-naive
                        nav_timestamps = nav_complete["timestamp"]
                        if nav_timestamps.dt.tz is not None:
                            nav_timestamps = nav_timestamps.dt.tz_localize(None)
                        
                        diffs = abs(nav_timestamps - ts_target)
                        idx = diffs.idxmin()
                        min_dt = diffs.iloc[idx]
                        rec = nav_complete.loc[idx]

                        if min_dt <= pd.Timedelta(f"{NET_DISTANCE_TOLERANCE}s") and "NetDistance" in rec and pd.notna(rec["NetDistance"]):
                            net_distance = float(rec["NetDistance"])
                            sync_status = "DISTANCE_OK"

                            # Try to get pitch data - be very lenient
                            pitch_time_window = max(NET_PITCH_TOLERANCE, 2.0)  # At least 2 second window for pitch

                            # First, check if the current record has pitch
                            if "NetPitch" in rec and pd.notna(rec["NetPitch"]):
                                net_angle_deg = float(np.degrees(rec["NetPitch"]))
                                sync_status = "FULL_SYNC"
                            # If not, look for the closest pitch data within time window
                            elif min_dt <= pd.Timedelta(f"{pitch_time_window}s"):
                                # Find closest record with valid pitch
                                pitch_candidates = nav_complete[nav_complete["NetPitch"].notna()]
                                if len(pitch_candidates) > 0:
                                    pitch_diffs = abs(nav_timestamps[pitch_candidates.index] - ts_target)
                                    pitch_idx = pitch_diffs.idxmin()
                                    pitch_min_dt = pitch_diffs.iloc[pitch_idx]

                                    if pitch_min_dt <= pd.Timedelta(f"{pitch_time_window}s"):
                                        pitch_rec = pitch_candidates.loc[pitch_idx]
                                        net_angle_deg = float(np.degrees(pitch_rec["NetPitch"]))
                                        sync_status = "DISTANCE_OK_WITH_PITCH"

                # --- FFT Data Synchronization with Unit Conversion ---
                if FFT_NET_DATA is not None and len(FFT_NET_DATA) > 0:
                    try:
                        # Ensure FFT timestamps are timezone-naive
                        fft_timestamps = FFT_NET_DATA["timestamp"]
                        if hasattr(fft_timestamps.iloc[0], 'tz') and fft_timestamps.dt.tz is not None:
                            fft_timestamps = fft_timestamps.dt.tz_localize(None)
                        
                        # Find closest FFT measurement within tolerance
                        fft_diffs = abs(fft_timestamps - ts_target)
                        fft_idx = fft_diffs.idxmin()
                        min_fft_dt = fft_diffs.iloc[fft_idx]
                        fft_rec = FFT_NET_DATA.loc[fft_idx]

                        if min_fft_dt <= pd.Timedelta(f"{NET_DISTANCE_TOLERANCE}s"):
                            # Handle different possible column names and units
                            if "distance_m" in fft_rec and pd.notna(fft_rec["distance_m"]):
                                fft_distance_m = float(fft_rec["distance_m"])
                            elif "distance_cm" in fft_rec and pd.notna(fft_rec["distance_cm"]):
                                fft_distance_m = float(fft_rec["distance_cm"]) / 100.0  # cm -> m
                            elif "distance" in fft_rec and pd.notna(fft_rec["distance"]):
                                # Assume cm if no unit specified (common FFT format)
                                fft_distance_m = float(fft_rec["distance"]) / 100.0  # cm -> m
                                
                            if "pitch_deg" in fft_rec and pd.notna(fft_rec["pitch_deg"]):
                                fft_pitch_deg = float(fft_rec["pitch_deg"])
                            elif "pitch_rad" in fft_rec and pd.notna(fft_rec["pitch_rad"]):
                                fft_pitch_deg = float(np.degrees(fft_rec["pitch_rad"]))  # rad -> deg
                            elif "pitch" in fft_rec and pd.notna(fft_rec["pitch"]):
                                # Assume radians if no unit specified (common FFT format)
                                fft_pitch_deg = float(np.degrees(fft_rec["pitch"]))  # rad -> deg
                            
                            # Update sync status
                            if fft_distance_m is not None:
                                if sync_status == "NO_DATA":
                                    sync_status = "FFT_ONLY"
                                elif sync_status in ["DISTANCE_OK", "FULL_SYNC", "DISTANCE_OK_WITH_PITCH"]:
                                    sync_status = "FFT_DVL_SYNC"
                                elif sync_status in ["SONAR_ONLY", "BOTH_SYNC"]:
                                    sync_status = "ALL_THREE_SYNC"
                    except Exception as e:
                        print(f"Warning: Could not sync FFT data: {e}")

                # --- Sonar Analysis Data ---
                if SONAR_RESULTS is not None and len(SONAR_RESULTS) > 0:
                    try:
                        # Calculate pixel-to-meter conversion
                        x_min, x_max, y_min, y_max = meta_extent if meta_extent is not None else (-DISPLAY_RANGE_MAX_M, DISPLAY_RANGE_MAX_M, RANGE_MIN_M, DISPLAY_RANGE_MAX_M)
                        width_m = float(x_max - x_min)
                        height_m = float(y_max - y_min)
                        px2m_x = width_m / float(CONE_W)
                        px2m_y = height_m / float(CONE_H)
                        pixels_to_meters_avg = 0.5 * (px2m_x + px2m_y)

                        # Ensure sonar timestamps are timezone-naive
                        sonar_timestamps = SONAR_RESULTS["timestamp"]
                        if hasattr(sonar_timestamps.iloc[0], 'tz') and sonar_timestamps.dt.tz is not None:
                            sonar_timestamps = sonar_timestamps.dt.tz_localize(None)

                        # Find closest sonar analysis measurement
                        sonar_diffs = abs(sonar_timestamps - ts_target)
                        sonar_idx = sonar_diffs.idxmin()
                        min_sonar_dt = sonar_diffs.iloc[sonar_idx]
                        sonar_rec = SONAR_RESULTS.loc[sonar_idx]

                        if min_sonar_dt <= pd.Timedelta(f"{NET_DISTANCE_TOLERANCE}s") and sonar_rec.get("detection_success", False):
                            sonar_distance_px = sonar_rec["distance_pixels"]
                            sonar_distance_m = sonar_rec.get("distance_meters", sonar_distance_px * pixels_to_meters_avg)
                            sonar_angle_deg = sonar_rec.get("angle_degrees", 0.0) if pd.notna(sonar_rec.get("angle_degrees")) else 0.0
                            if sync_status == "NO_DATA":
                                sync_status = "SONAR_ONLY"
                            elif sync_status in ["DISTANCE_OK", "FULL_SYNC", "DISTANCE_OK_WITH_PITCH"]:
                                sync_status = "BOTH_SYNC"
                            elif sync_status in ["FFT_ONLY", "FFT_DVL_SYNC"]:
                                sync_status = "ALL_THREE_SYNC"
                    except Exception as e:
                        print(f"Warning: Could not sync sonar analysis data: {e}")

                # --- Draw overlays for all three systems ---
                if sync_status in ["DISTANCE_OK", "FULL_SYNC", "SONAR_ONLY", "BOTH_SYNC", "DISTANCE_OK_WITH_PITCH", "FFT_ONLY", "FFT_DVL_SYNC", "ALL_THREE_SYNC"]:
                    net_half_width = 2.0
                    
                    # Standardized visual parameters for all systems
                    line_thickness = 4
                    endpoint_radius = 6
                    center_radius = 8
                    endpoint_outline = 2
                    center_outline = 2

                    # Draw DVL net line (if available) - Yellow/Orange
                    if net_distance is not None and net_distance <= DISPLAY_RANGE_MAX_M:
                        if sync_status == "FULL_SYNC":
                            dvl_line_color = (0, 255, 255)  # Yellow - perfect sync
                            dvl_center_color = (0, 200, 200)  # Darker yellow
                            dvl_label = f"DVL: {net_distance:.2f}m @ {net_angle_deg:.1f}°"
                        elif sync_status == "DISTANCE_OK_WITH_PITCH":
                            dvl_line_color = (0, 255, 128)  # Light green - distance OK, pitch from closest
                            dvl_center_color = (0, 200, 100)  # Darker green
                            dvl_label = f"DVL: {net_distance:.2f}m @ {net_angle_deg:.1f}° (approx)"
                        else:
                            dvl_line_color = (0, 165, 255)  # Orange - distance only
                            dvl_center_color = (0, 130, 200)  # Darker orange
                            dvl_label = f"DVL: {net_distance:.2f}m @ 0.0°"

                        # Only draw angled line if we have pitch data
                        use_angle = sync_status in ["FULL_SYNC", "DISTANCE_OK_WITH_PITCH", "BOTH_SYNC", "FFT_DVL_SYNC", "ALL_THREE_SYNC"]
                        net_angle_rad = np.radians(net_angle_deg) if use_angle else 0.0

                        x1, y1 = -net_half_width, net_distance
                        x2, y2 = +net_half_width, net_distance

                        cos_a, sin_a = np.cos(net_angle_rad), np.sin(net_angle_rad)
                        rx1 = x1 * cos_a - (y1 - net_distance) * sin_a
                        ry1 = x1 * sin_a + (y1 - net_distance) * cos_a + net_distance
                        rx2 = x2 * cos_a - (y2 - net_distance) * sin_a
                        ry2 = x2 * sin_a + (y2 - net_distance) * cos_a + net_distance

                        px1, py1 = x_px(rx1), y_px(ry1)
                        px2, py2 = x_px(rx2), y_px(ry2)

                        # Standardized DVL line drawing
                        cv2.line(cone_bgr, (px1, py1), (px2, py2), dvl_line_color, line_thickness)
                        
                        # Standardized endpoint circles
                        for (px, py) in [(px1, py1), (px2, py2)]:
                            cv2.circle(cone_bgr, (px, py), endpoint_radius, (255, 255, 255), -1)
                            cv2.circle(cone_bgr, (px, py), endpoint_radius, dvl_line_color, endpoint_outline)

                        # Standardized center point
                        center_px, center_py = x_px(0), y_px(net_distance)
                        cv2.circle(cone_bgr, (center_px, center_py), center_radius, dvl_center_color, -1)
                        cv2.circle(cone_bgr, (center_px, center_py), center_radius, (255, 255, 255), center_outline)

                    # Draw FFT net line (if available) - Cyan
                    if fft_distance_m is not None and fft_distance_m <= DISPLAY_RANGE_MAX_M:
                        fft_line_color = (255, 255, 0)  # Cyan (BGR format)
                        fft_center_color = (200, 200, 0)  # Darker cyan
                        fft_label = f"FFT: {fft_distance_m:.2f}m"
                        if fft_pitch_deg is not None:
                            fft_label += f" @ {fft_pitch_deg:.1f}°"

                        # Draw angled line using FFT pitch (if available)
                        fft_angle_rad = np.radians(fft_pitch_deg) if fft_pitch_deg is not None else 0.0
                        fx1, fy1 = -net_half_width, fft_distance_m
                        fx2, fy2 = +net_half_width, fft_distance_m

                        cos_a, sin_a = np.cos(fft_angle_rad), np.sin(fft_angle_rad)
                        frx1 = fx1 * cos_a - (fy1 - fft_distance_m) * sin_a
                        fry1 = fx1 * sin_a + (fy1 - fft_distance_m) * cos_a + fft_distance_m
                        frx2 = fx2 * cos_a - (fy2 - fft_distance_m) * sin_a
                        fry2 = fx2 * sin_a + (fy2 - fft_distance_m) * cos_a + fft_distance_m

                        fpx1, fpy1 = x_px(frx1), y_px(fry1)
                        fpx2, fpy2 = x_px(frx2), y_px(fry2)

                        # Standardized FFT line drawing
                        cv2.line(cone_bgr, (fpx1, fpy1), (fpx2, fpy2), fft_line_color, line_thickness)
                        
                        # Standardized endpoint circles
                        for (px, py) in [(fpx1, fpy1), (fpx2, fpy2)]:
                            cv2.circle(cone_bgr, (px, py), endpoint_radius, (255, 255, 255), -1)
                            cv2.circle(cone_bgr, (px, py), endpoint_radius, fft_line_color, endpoint_outline)

                        # Standardized center point
                        center_px, center_py = x_px(0), y_px(fft_distance_m)
                        cv2.circle(cone_bgr, (center_px, center_py), center_radius, fft_center_color, -1)
                        cv2.circle(cone_bgr, (center_px, center_py), center_radius, (255, 255, 255), center_outline)

                    # Draw Sonar analysis line (if available) - Magenta
                    if sonar_distance_m is not None and sonar_distance_m <= DISPLAY_RANGE_MAX_M:
                        sonar_line_color = (255, 0, 255)  # Magenta
                        sonar_center_color = (200, 0, 200)  # Darker magenta
                        sonar_label = f"SONAR: {sonar_distance_m:.2f}m @ {sonar_angle_deg:.1f}°"

                        # Draw angled line using sonar angle
                        sonar_angle_rad = np.radians(sonar_angle_deg)
                        sx1, sy1 = -net_half_width, sonar_distance_m
                        sx2, sy2 = +net_half_width, sonar_distance_m

                        cos_a, sin_a = np.cos(sonar_angle_rad), np.sin(sonar_angle_rad)
                        srx1 = sx1 * cos_a - (sy1 - sonar_distance_m) * sin_a
                        sry1 = sx1 * sin_a + (sy1 - sonar_distance_m) * cos_a + sonar_distance_m
                        srx2 = sx2 * cos_a - (sy2 - sonar_distance_m) * sin_a
                        sry2 = sx2 * sin_a + (sy2 - sonar_distance_m) * cos_a + sonar_distance_m

                        spx1, spy1 = x_px(srx1), y_px(sry1)
                        spx2, spy2 = x_px(srx2), y_px(sry2)

                        # Standardized Sonar line drawing
                        cv2.line(cone_bgr, (spx1, spy1), (spx2, spy2), sonar_line_color, line_thickness)
                        
                        # Standardized endpoint circles
                        for (px, py) in [(spx1, spy1), (spx2, spy2)]:
                            cv2.circle(cone_bgr, (px, py), endpoint_radius, (255, 255, 255), -1)
                            cv2.circle(cone_bgr, (px, py), endpoint_radius, sonar_line_color, endpoint_outline)

                        # Standardized center point
                        center_px, center_py = x_px(0), y_px(sonar_distance_m)
                        cv2.circle(cone_bgr, (center_px, center_py), center_radius, sonar_center_color, -1)
                        cv2.circle(cone_bgr, (center_px, center_py), center_radius, (255, 255, 255), center_outline)

                    # --- AOI / Corridor mask overlay (if available and enabled) ---
                    from utils.sonar_config import VIDEO_CONFIG
                    if VIDEO_CONFIG.get('show_aoi_corridor', False):
                        # Attempt to locate mask data in sonar_rec (or global SONAR_RESULTS)
                        try:
                            # Sonar analysis may include raw masks (ellipse_mask, corridor_mask)
                            mask_ell = sonar_rec.get('ellipse_mask', None) if 'sonar_rec' in locals() else None
                            mask_corr = sonar_rec.get('corridor_mask', None) if 'sonar_rec' in locals() else None
                        except Exception:
                            mask_ell = None
                            mask_corr = None

                        # If masks are not present but SONAR_RESULTS contains blobs as arrays,
                        # skip gracefully. The analysis module populates masks in `current_aoi` when running live.
                        if mask_ell is not None and isinstance(mask_ell, (np.ndarray,)):
                            # Convert single-channel mask to colored overlay and alpha blend
                            a_color = VIDEO_CONFIG.get('aoi_mask_color', (0,255,0))
                            a_alpha = float(VIDEO_CONFIG.get('aoi_mask_alpha', 0.25))
                            mask_bool = (mask_ell > 0).astype(np.uint8)
                            color_layer = np.zeros_like(cone_bgr, dtype=np.uint8)
                            color_layer[mask_bool > 0] = a_color
                            cone_bgr = cv2.addWeighted(cone_bgr.astype(np.float32), 1.0, color_layer.astype(np.float32), a_alpha, 0)

                        if mask_corr is not None and isinstance(mask_corr, (np.ndarray,)):
                            c_color = VIDEO_CONFIG.get('corridor_mask_color', (0,128,255))
                            c_alpha = float(VIDEO_CONFIG.get('corridor_mask_alpha', 0.25))
                            mask_bool = (mask_corr > 0).astype(np.uint8)
                            color_layer = np.zeros_like(cone_bgr, dtype=np.uint8)
                            color_layer[mask_bool > 0] = c_color
                            cone_bgr = cv2.addWeighted(cone_bgr.astype(np.float32), 1.0, color_layer.astype(np.float32), c_alpha, 0)

                    # Create status text with all three systems
                    status_lines = []
                    if net_distance is not None:
                        status_lines.append(dvl_label if 'dvl_label' in locals() else f"DVL: {net_distance:.2f}m")
                    if fft_distance_m is not None:
                        status_lines.append(fft_label)
                    if sonar_distance_m is not None:
                        status_lines.append(sonar_label if 'sonar_label' in locals() else f"SONAR: {sonar_distance_m:.2f}m")
                    if status_lines:
                        status_text = " | ".join(status_lines)

                else:
                    if INCLUDE_NET or SONAR_RESULTS is not None or FFT_NET_DATA is not None:
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
            if (INCLUDE_NET or SONAR_RESULTS is not None or FFT_NET_DATA is not None) and status_text:
                # Updated Legend for three systems
                legend_y = 15
                cv2.putText(cone_bgr, "DVL: Yellow/Orange | FFT: Cyan | SONAR: Magenta", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(cone_bgr, "DVL: Yellow/Orange | FFT: Cyan | SONAR: Magenta", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Status with black outline for better visibility
                cv2.putText(cone_bgr, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(cone_bgr, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # footer info with three-system indicator
            frame_info = f"Frame {frame_idx}/{len(frame_indices)} | {TARGET_BAG} | 3-Sys.Sync"
            cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            return cone_bgr

        except Exception as e:
            print(f"Frame {frame_idx} error: {e}")
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
            
            # Check if we're beyond the camera video time range
            if ts_target < video_idx.min() or ts_target > video_idx.max():
                # Outside camera time range - use sonar-only mode for this frame
                composite = cone_frame
                print(f"   Frame {frame_idx}: Outside camera time range, using sonar-only")
            else:
                # Find nearest camera frame within reasonable tolerance
                idx_near = video_idx.get_indexer([ts_target], method="nearest")[0]
                ts_cam = dfv.loc[idx_near, "ts_utc"]
                dt = abs(ts_cam - ts_target)
                
                # Check if the time difference is reasonable (configurable tolerance)
                from utils.sonar_config import VIDEO_CONFIG
                max_sync_tolerance_s = VIDEO_CONFIG.get('max_sync_tolerance_seconds', 5.0)
                max_sync_tolerance = pd.Timedelta(seconds=max_sync_tolerance_s)
                if dt > max_sync_tolerance:
                    # Time difference too large - camera video probably ended
                    composite = cone_frame
                    print(f"   Frame {frame_idx}: Sync tolerance exceeded ({dt.total_seconds():.1f}s), using sonar-only")
                else:
                    # Good sync - compose with camera
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
            "include_sonar_analysis": SONAR_RESULTS is not None,
            "include_fft_data": FFT_NET_DATA is not None,
            "fft_units_converted": True,
            "three_system_sync": True,
            "units_info": {
                "distance_unit": "meters",
                "angle_unit": "degrees",
                "fft_conversions": "cm->m, rad->deg"
            },
            "sync_window_seconds": float(SYNC_WINDOW_SECONDS),
            "flip_range": bool(FLIP_RANGE),
            "flip_beams": bool(FLIP_BEAMS),
        }
        if meta_extent is not None:
            meta["extent"] = list(meta_extent)
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"\n🎉 DONE! Wrote {frames_written} frames to {out_path} @ {natural_fps:.2f} FPS")
        print(f"Three-system synchronization: DVL(Yellow/Orange) | FFT(Cyan) | Sonar(Magenta)")
        print(f"Metadata saved to: {meta_path}")
    except Exception as e:
        print(f"Warning: could not write metadata: {e}")
    
    return out_path

def generate_three_system_video(
    target_bag: str,
    exports_folder: Path,
    net_analysis_results: pd.DataFrame,
    raw_data: dict,
    fft_csv_path: Path | None = None,
    start_idx: int = 1,
    end_idx: int = 1200,
    **video_kwargs
) -> Path:
    """
    Simplified three-system video generation with automatic synchronization.
    
    Args:
        target_bag: Bag identifier
        exports_folder: Path to exports folder
        net_analysis_results: Sonar analysis results
        raw_data: Raw data dictionary (with 'navigation' key)
        fft_csv_path: Path to FFT CSV file
        start_idx: Start frame index
        end_idx: End frame index
        **video_kwargs: Additional arguments for export_optimized_sonar_video
    
    Returns:
        Path to generated video file
    """
    print("GENERATING THREE-SYSTEM VIDEO")
    print("=" * 50)
    
    # Prepare synchronized data
    net_analysis_sync, fft_net_data_sync = prepare_three_system_data(
        target_bag=target_bag,
        exports_folder=exports_folder,
        net_analysis_results=net_analysis_results,
        raw_data=raw_data,
        fft_csv_path=fft_csv_path
    )
    
    # Generate video with synchronized data
    video_path = export_optimized_sonar_video(
        TARGET_BAG=target_bag,
        EXPORTS_FOLDER=exports_folder,
        START_IDX=start_idx,
        END_IDX=end_idx,
        STRIDE=1,
        AUTO_DETECT_VIDEO=True,
        INCLUDE_NET=True,
        SONAR_RESULTS=net_analysis_sync,
        FFT_NET_DATA=fft_net_data_sync,
        NET_DISTANCE_TOLERANCE=0.5,
        NET_PITCH_TOLERANCE=2.0,
        **video_kwargs
    )
    
    print(f"\nTHREE-SYSTEM VIDEO GENERATED: {video_path}")
    print(f"   Systems synchronized:")
    print(f"   - DVL: Navigation-based robot position relative to net")
    print(f"   - FFT: High-precision signal processing net detection")
    print(f"   - Sonar: Image analysis of sonar returns")
    print(f"   - Uniform styling: All systems use consistent visual appearance")
    
    return video_path
