# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from utils.io_utils import load_df, read_video_index
from utils.sonar_utils import (
    get_sonoptix_frame, enhance_intensity, apply_flips, cone_raster_like_display_cell, 
    
)
from utils.config import (
    SONAR_VIS_DEFAULTS,
    CONE_W_DEFAULT, CONE_H_DEFAULT, CONE_FLIP_VERTICAL_DEFAULT,
    CMAP_NAME_DEFAULT, DISPLAY_RANGE_MAX_M_DEFAULT, FOV_DEG_DEFAULT,
    EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS,
)

# Define fallback configs at module level to avoid undefined variable errors
VIDEO_CONFIG = {'fps': 30}
IMAGE_PROCESSING_CONFIG = {
    'binary_threshold': 128,
    'adaptive_angle_steps': 36,
    'adaptive_base_radius': 3,
    'adaptive_max_elongation': 3.0,
    'momentum_boost': 0.8,
    'adaptive_linearity_threshold': 0.15,
    'downscale_factor': 2,
    'top_k_bins': 8,
    'min_coverage_percent': 0.5,
    'gaussian_sigma': 1.0,
    'morph_close_kernel': 0,
    'edge_dilation_iterations': 0,
    'min_contour_area': 100,
    'aoi_center_x_percent': 50,
    'aoi_center_y_percent': 60,
    'aoi_width_percent': 60,
    'aoi_height_percent': 70,
    'max_distance_change_pixels': 20
}
TRACKING_CONFIG = {
    'ellipse_expansion_factor': 0.3,
    'center_smoothing_alpha': 0.3,
    'ellipse_size_smoothing_alpha': 0.1,
    'ellipse_orientation_smoothing_alpha': 0.1,
    'ellipse_max_movement_pixels': 4.0,
    'corridor_band_k': 0.75,
    'corridor_length_factor': 1.25,
    'corridor_widen': 1.0,
    'corridor_both_directions': True
}

# Try to import actual configs, which will override the fallbacks if available
try:
    from utils.config import VIDEO_CONFIG as _VIDEO_CONFIG, IMAGE_PROCESSING_CONFIG as _IMAGE_PROCESSING_CONFIG, TRACKING_CONFIG as _TRACKING_CONFIG
    VIDEO_CONFIG.update(_VIDEO_CONFIG)
    IMAGE_PROCESSING_CONFIG.update(_IMAGE_PROCESSING_CONFIG)
    TRACKING_CONFIG.update(_TRACKING_CONFIG)
except ImportError:
    pass  # Use fallback configs defined above

# Handle image enhancement imports with fallbacks
try:
    from utils.image_enhancement import preprocess_edges, adaptive_linear_momentum_merge_fast
except ImportError:
    def preprocess_edges(frame_u8, config):
        # Fallback implementation
        binary_threshold = config.get('binary_threshold', 128)
        binary_frame = (frame_u8 > binary_threshold).astype(np.uint8) * 255
        kernel_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        raw_edges = cv2.filter2D(binary_frame, cv2.CV_32F, kernel_edge)
        raw_edges = np.clip(raw_edges, 0, 255).astype(np.uint8)
        raw_edges = (raw_edges > 0).astype(np.uint8) * 255
        return raw_edges, raw_edges

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

def find_video_files_for_bag(bag_name: str, exports_dir: Optional[Path] = None) -> dict:
    """
    Automatically find video CSV and PNG frame directories for a given bag name.
    
    Args:
        bag_name: Bag name (e.g., "2024-08-20_13-39-34")
        exports_dir: Path to exports directory (uses default if None)
        
    Returns:
        Dictionary with keys:
        - 'video_frames_dir': Path to best matching PNG frames directory or None
        - 'available_frame_dirs': List of all available frame directories for this bag
    """
    from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS, VIDEO_CONFIG
    
    if exports_dir is None:
        exports_dir = Path(EXPORTS_DIR_DEFAULT)
    
    # Clean bag name (remove _data or _video suffix if present)
    clean_bag_name = bag_name.replace('_data', '').replace('_video', '')
    
    result = {
        'sonar_csv': None,
        'video_frames_dir': None,
        'available_frame_dirs': []
    }
    
    # Find sonar CSV file
    csv_dir = exports_dir / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
    csv_pattern = f"sensor_sonoptix_echo_image__{clean_bag_name}_video.csv"
    csv_file = csv_dir / csv_pattern
    
    if csv_file.exists():
        result['sonar_csv'] = csv_file
    else:
        # Try alternative patterns
        for csv_candidate in csv_dir.glob(f"*{clean_bag_name}*video*.csv"):
            if 'sonoptix' in csv_candidate.name:
                result['sonar_csv'] = csv_candidate
                break
    
    # Find PNG frame directories
    frames_dir = exports_dir / EXPORTS_SUBDIRS.get('frames', 'frames')
    if frames_dir.exists():
        # Look for frame directories matching this bag
        for frame_dir in frames_dir.iterdir():
            if frame_dir.is_dir() and clean_bag_name in frame_dir.name:
                # Verify it has PNG frames and index.csv
                if (frame_dir / 'index.csv').exists() or list(frame_dir.glob('*.png')):
                    result['available_frame_dirs'].append(frame_dir)
        
        # Select best frame directory based on topic preference
        if result['available_frame_dirs']:
            topic_preferences = VIDEO_CONFIG.get('video_topic_preference', [
                'image_compressed_image_data',
                'ted_image', 
                'camera_image'
            ])
            
            # Try to find preferred topic
            for preferred_topic in topic_preferences:
                for frame_dir in result['available_frame_dirs']:
                    if preferred_topic in frame_dir.name:
                        result['video_frames_dir'] = frame_dir
                        break
                if result['video_frames_dir']:
                    break
            
            # If no preferred topic found, use the first available
            if not result['video_frames_dir'] and result['available_frame_dirs']:
                result['video_frames_dir'] = result['available_frame_dirs'][0]
    
    return result


def get_video_overlay_info(bag_name: str) -> Optional[dict]:
    """
    Get video overlay information if enabled in configuration.
    
    Args:
        bag_name: Bag name to find video files for
        
    Returns:
        Dictionary with 'sonar_csv' and 'video_frames_dir' paths if video overlay
        is enabled and files are found, None otherwise.
    """
    from utils.config import VIDEO_CONFIG
    
    if not VIDEO_CONFIG.get('enable_video_overlay', False):
        return None
    
    video_files = find_video_files_for_bag(bag_name)
    
    if video_files['sonar_csv'] and video_files['video_frames_dir']:
        return {
            'sonar_csv': video_files['sonar_csv'],
            'video_frames_dir': video_files['video_frames_dir']
        }
    else:
        print(f"Warning: Video overlay enabled but files not found for bag '{bag_name}'")
        if not video_files['sonar_csv']:
            print(f"  Missing sonar CSV file")
        if not video_files['video_frames_dir']:
            print(f"  Missing video frames directory")
            if video_files['available_frame_dirs']:
                print(f"  Available frame dirs: {[v.name for v in video_files['available_frame_dirs']]}")
        return None

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
    
    # STEP 1: Load FFT data if available
    fft_net_data = None
    if fft_csv_path and fft_csv_path.exists():
        try:
            from utils.fft_processing import load_relative_fft_data, convert_fft_to_xy_coordinates
            
            print("✓ Using notebook 08's FFT loading methods...")
            fft_net_data = load_relative_fft_data(fft_csv_path)
            fft_net_data = convert_fft_to_xy_coordinates(fft_net_data)
            print(f"✓ Loaded FFT data via notebook 08 method: {len(fft_net_data)} records")
            
        except Exception as e:
            print(f"✗ FFT loading error: {e}")
            print("   Continuing without FFT data (two-system mode: DVL + Sonar)")
            fft_net_data = None
    elif fft_csv_path:
        print(f"✗ FFT file not found: {fft_csv_path.name}")
        print("   Continuing without FFT data (two-system mode: DVL + Sonar)")
    else:
        print("ℹ️  No FFT data path provided")
        print("   Running in two-system mode: DVL + Sonar only")
    
    # STEP 2: Use notebook 08's synchronization approach if available
    if fft_net_data is not None and use_notebook08_sync:
        try:
            # Import notebook 08's synchronization utilities
            from utils.multi_system_sync import run_complete_three_system_analysis
            
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
    print(f"   Systems active: DVL + Sonar" + (" + FFT" if fft_net_data is not None else ""))
    
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
    print(f"   Range: {RANGE_MIN_M}-{DISPLAY_RANGE_MAX_M}m | FOV: {FOV_DEG}")
    
    # --- Auto-detect video files if enabled ---
    if VIDEO_SEQ_DIR is None and AUTO_DETECT_VIDEO:
        from utils.config import VIDEO_CONFIG
        
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
            import utils.distance_measurement as sda
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
    
    # CRITICAL FIX: Pre-calculate maximum frame dimensions to avoid size mismatches
    writer = None
    max_frame_width = CONE_W
    max_frame_height = CONE_H
    
    # If camera is enabled, calculate the maximum width including camera feed
    if VIDEO_SEQ_DIR is not None and dfv is not None:
        try:
            # Load a sample camera frame to determine its dimensions after resizing
            sample_cam_file = VIDEO_SEQ_DIR / dfv.iloc[0]["file"]
            sample_cam = load_png_bgr(sample_cam_file)
            vh, vw0 = sample_cam.shape[:2]
            scale = VIDEO_HEIGHT / vh
            camera_width_resized = int(round(vw0 * scale))
            
            # Maximum width = camera + padding + cone
            max_frame_width = camera_width_resized + PAD_BETWEEN + CONE_W
            max_frame_height = CONE_H  # Height remains constant
            
            print(f"   Video dimensions calculated:")
            print(f"     Camera: {camera_width_resized}x{VIDEO_HEIGHT}")
            print(f"     Sonar cone: {CONE_W}x{CONE_H}")
            print(f"     Max composite: {max_frame_width}x{max_frame_height}")
        except Exception as e:
            print(f"   Warning: Could not pre-calculate camera dimensions: {e}")
            print(f"   Falling back to dynamic initialization")
    
    # Initialize writer with maximum dimensions (ensure even numbers for codec)
    max_frame_width = max_frame_width + (max_frame_width % 2)  # Make even
    max_frame_height = max_frame_height + (max_frame_height % 2)  # Make even
    
    writer = cv2.VideoWriter(str(out_path), fourcc, natural_fps, (max_frame_width, max_frame_height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {out_path} with size {max_frame_width}x{max_frame_height}")
    
    print(f"   VideoWriter initialized: {max_frame_width}x{max_frame_height} @ {natural_fps:.2f} FPS")
    
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
                                fft_distance_m = float(fft_rec["distance"]) / 100.0  # cm -> m
                                
                            if "pitch_deg" in fft_rec and pd.notna(fft_rec["pitch_deg"]):
                                fft_pitch_deg = float(fft_rec["pitch_deg"])
                            elif "pitch_rad" in fft_rec and pd.notna(fft_rec["pitch_rad"]):
                                fft_pitch_deg = float(np.degrees(fft_rec["pitch_rad"]))
                            elif "pitch" in fft_rec and pd.notna(fft_rec["pitch"]):
                                fft_pitch_deg = float(np.degrees(fft_rec["pitch"]))
                            
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
                            dvl_label = f"DVL: {net_distance:.2f}m @ {net_angle_deg:.1f}"
                        elif sync_status == "DISTANCE_OK_WITH_PITCH":
                            dvl_line_color = (0, 255, 128)  # Light green - distance OK, pitch from closest
                            dvl_center_color = (0, 200, 100)  # Darker green
                            dvl_label = f"DVL: {net_distance:.2f}m @ {net_angle_deg:.1f} (approx)"
                        else:
                            dvl_line_color = (0, 165, 255)  # Orange - distance only
                            dvl_center_color = (0, 130, 200)  # Darker orange
                            dvl_label = f"DVL: {net_distance:.2f}m @ 0.0"

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
                            fft_label += f" @ {fft_pitch_deg:.1f}"

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
                        sonar_label = f"SONAR: {sonar_distance_m:.2f}m @ {sonar_angle_deg:.1f}"

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

                # Status with black outline for better visibility
                cv2.putText(cone_bgr, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(cone_bgr, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # footer info with three-system indicator
            frame_info = f"Frame {frame_idx}/{len(frame_indices)} | {TARGET_BAG} | 3-Sys.Sync"
            cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(cone_bgr, frame_info, (10, CONE_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # CRITICAL FIX: After all alpha blending operations, ensure uint8
            # The alpha blend operations return float32, but VideoWriter needs uint8
            if cone_bgr.dtype != np.uint8:
                cone_bgr = np.clip(cone_bgr, 0, 255).astype(np.uint8)

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

        # CRITICAL FIX: Ensure cone_frame is uint8 before compositing
        if cone_frame.dtype != np.uint8:
            cone_frame = np.clip(cone_frame, 0, 255).astype(np.uint8)

        # compose (camera optional)
        if VIDEO_SEQ_DIR is not None and dfv is not None and video_idx is not None:
            ts_target = pd.to_datetime(df.loc[frame_idx, "ts_utc"], utc=True, errors="coerce")
            
            # Check if we're beyond the camera video time range
            if ts_target < video_idx.min() or ts_target > video_idx.max():
                # Outside camera time range - use sonar-only mode for this frame
                composite = cone_frame
                # PAD to match expected width
                if composite.shape[1] < max_frame_width:
                    pad_width = max_frame_width - composite.shape[1]
                    pad_left = np.zeros((CONE_H, pad_width, 3), dtype=np.uint8)
                    composite = np.hstack([pad_left, composite])
                print(f"   Frame {frame_idx}: Outside camera time range, using sonar-only")
            else:
                # Find nearest camera frame within reasonable tolerance
                idx_near = video_idx.get_indexer([ts_target], method="nearest")[0]
                ts_cam = dfv.loc[idx_near, "ts_utc"]
                dt = abs(ts_cam - ts_target)
                
                # Check if the time difference is reasonable (configurable tolerance)
                from utils.config import VIDEO_CONFIG
                max_sync_tolerance_s = VIDEO_CONFIG.get('max_sync_tolerance_seconds', 5.0)
                max_sync_tolerance = pd.Timedelta(seconds=max_sync_tolerance_s)
                if dt > max_sync_tolerance:
                    # Time difference too large - camera video probably ended
                    composite = cone_frame
                    # PAD to match expected width
                    if composite.shape[1] < max_frame_width:
                        pad_width = max_frame_width - composite.shape[1]
                        pad_left = np.zeros((CONE_H, pad_width, 3), dtype=np.uint8)
                        composite = np.hstack([pad_left, composite])
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
            # sonar-only output canvas - PAD to match expected width
            composite = cone_frame
            if composite.shape[1] < max_frame_width:
                pad_width = max_frame_width - composite.shape[1]
                pad_left = np.zeros((CONE_H, pad_width, 3), dtype=np.uint8)
                composite = np.hstack([pad_left, composite])

        # FINAL CHECK: Ensure correct dimensions and uint8
        if composite.shape[:2] != (max_frame_height, max_frame_width):
            # Resize to exact dimensions if needed
            composite = cv2.resize(composite, (max_frame_width, max_frame_height))
        
        if composite.dtype != np.uint8:
            composite = np.clip(composite, 0, 255).astype(np.uint8)
            
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
        print(f"\n DONE! Wrote {frames_written} frames to {out_path} @ {natural_fps:.2f} FPS")
        print(f"Three-system synchronization: DVL(Yellow/Orange) | FFT(Cyan) | Sonar(Magenta)")
        print(f"Metadata saved to: {meta_path}")
    except Exception as e:
        print(f"Warning: could not write metadata: {e}")
    
    return out_path

def generate_three_system_video(
    target_bag: str,
    exports_folder: Path,
    net_analysis_results: pd.DataFrame | None = None,  # Now optional
    raw_data: dict | None = None,  # Now optional
    fft_csv_path: Path | None = None,
    start_idx: int = 1,
    end_idx: int = 1200,
    **video_kwargs
) -> Path:
    """
    Simplified three-system video generation with automatic data loading.
    Falls back gracefully to two-system mode if FFT data is not available.
    
    Args:
        target_bag: Bag name to process
        exports_folder: Path to exports directory
        net_analysis_results: Optional pre-loaded sonar analysis results. 
                            If None, will attempt to load from saved CSV file.
        raw_data: Optional pre-loaded raw data dictionary.
                 If None, will attempt to load DVL navigation data.
        fft_csv_path: Optional path to FFT data CSV
        start_idx: Starting frame index
        end_idx: Ending frame index
        **video_kwargs: Additional arguments passed to export_optimized_sonar_video
        
    Returns:
        Path to generated video file
    """
    print("GENERATING THREE-SYSTEM VIDEO" if fft_csv_path and fft_csv_path.exists() else "GENERATING TWO-SYSTEM VIDEO")
    print("=" * 50)
    
    # STEP 1: Load sonar analysis results if not provided
    if net_analysis_results is None:
        print("📊 Loading sonar analysis results from saved CSV...")
        
        # Look for the analysis CSV file saved by analyze_npz_sequence
        outputs_dir = exports_folder / EXPORTS_SUBDIRS.get('outputs', 'outputs')
        analysis_csv_pattern = f"{target_bag.replace('_video', '')}_analysis.csv"
        analysis_csv_path = outputs_dir / analysis_csv_pattern
        
        if analysis_csv_path.exists():
            net_analysis_results = pd.read_csv(analysis_csv_path)
            net_analysis_results['timestamp'] = pd.to_datetime(net_analysis_results['timestamp'])
            print(f"✓ Loaded {len(net_analysis_results)} sonar analysis records from {analysis_csv_path.name}")
        else:
            # Try to find any matching analysis file
            matching_files = list(outputs_dir.glob(f"*{target_bag}*_analysis.csv"))
            if matching_files:
                analysis_csv_path = matching_files[0]
                net_analysis_results = pd.read_csv(analysis_csv_path)
                net_analysis_results['timestamp'] = pd.to_datetime(net_analysis_results['timestamp'])
                print(f"✓ Loaded {len(net_analysis_results)} sonar analysis records from {analysis_csv_path.name}")
            else:
                raise FileNotFoundError(
                    f"No sonar analysis CSV found for bag '{target_bag}'.\n"
                    f"Expected location: {analysis_csv_path}\n"
                    f"Please run analyze_npz_sequence() with save_outputs=True first."
                )
    else:
        print(f"✓ Using provided sonar analysis results: {len(net_analysis_results)} records")
    
    # STEP 2: Load DVL navigation data if not provided
    if raw_data is None:
        print("📊 Loading DVL navigation data...")
        try:
            import utils.distance_measurement as sda
            BY_BAG_FOLDER = exports_folder / EXPORTS_SUBDIRS.get('by_bag', 'by_bag')
            raw_data, _ = sda.load_all_distance_data_for_bag(target_bag, BY_BAG_FOLDER)
            
            if raw_data.get('navigation') is not None:
                print(f"✓ Loaded {len(raw_data['navigation'])} DVL navigation records")
            else:
                print("⚠️  No DVL navigation data found")
                raw_data = {'navigation': None}
        except Exception as e:
            print(f"⚠️  Could not load DVL data: {e}")
            raw_data = {'navigation': None}
    else:
        print(f"✓ Using provided raw data")
    
    # Prepare synchronized data (handles missing FFT gracefully)
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
        FFT_NET_DATA=fft_net_data_sync,  # Can be None
        NET_DISTANCE_TOLERANCE=0.5,
        NET_PITCH_TOLERANCE=2.0,
        **video_kwargs
    )
    
    systems_count = 2 + (1 if fft_net_data_sync is not None else 0)
    print(f"\n{systems_count}-SYSTEM VIDEO GENERATED: {video_path}")
    print(f"   Systems synchronized:")
    print(f"   - DVL: Navigation-based robot position relative to net")
    if fft_net_data_sync is not None:
        print(f"   - FFT: High-precision signal processing net detection")
    print(f"   - Sonar: Image analysis of sonar returns")
    
    return video_path

def create_enhanced_contour_detection_video(
    npz_file_index=0, 
    frame_start=0, 
    frame_count=100,
    frame_step=5, 
    output_path='enhanced_video.mp4'
):
    """
    Create video showing contour detection pipeline using NetTracker.
    Now matches the analysis pipeline exactly.
    """
    import cv2
    import numpy as np
    from pathlib import Path
    
    from utils.sonar_utils import load_cone_run_npz, to_uint8_gray
    from utils.io_utils import get_available_npz_files
    from utils.sonar_tracking import NetTracker
    
    print("=== CONTOUR DETECTION PIPELINE VIDEO (3x3 Grid with NetTracker) ===")
    
    files = get_available_npz_files()
    if npz_file_index >= len(files):
        print(f"Error: NPZ file index {npz_file_index} not available")
        return None
    
    cones, timestamps, extent, _ = load_cone_run_npz(files[npz_file_index])
    T = len(cones)
    actual = int(min(frame_count, max(0, (T - frame_start) // max(1, frame_step))))
    
    if actual <= 0:
        print("Error: Not enough frames to process")
        return None
    
    first_u8 = to_uint8_gray(cones[frame_start])
    H, W = first_u8.shape
    
    # CRITICAL FIX: Grid is 2 rows x 3 columns
    grid_h = H * 2
    grid_w = W * 3
    
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = VIDEO_CONFIG.get('fps', 30)
    vw = cv2.VideoWriter(str(outp), fourcc, fps, (grid_w, grid_h))
    
    if not vw.isOpened():
        print("Error: Could not initialize video writer")
        return None
    
    print(f"Processing {actual} frames...")
    print(f"Grid layout (2x3):")
    print(f"  Row 1: Raw | Momentum-Merged | Edges")
    print(f"  Row 2: Search Mask | Best Contour | Distance")
    print(f"Output grid size: {grid_w}x{grid_h}")
    
    # Initialize NetTracker (same as analysis)
    config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
    tracker = NetTracker(config)
    
    print(f"Tracker config:")
    print(f"  expansion: {config['ellipse_expansion_factor']}")
    print(f"  center_alpha: {config['center_smoothing_alpha']}")
    print(f"  size_alpha: {config['ellipse_size_smoothing_alpha']}")
    print(f"  angle_alpha: {config['ellipse_orientation_smoothing_alpha']}")
    
    for i in range(actual):
        idx = frame_start + i * frame_step
        frame_u8 = to_uint8_gray(cones[idx])
        
        # 1. Binary conversion
        binary = (frame_u8 > config['binary_threshold']).astype(np.uint8) * 255
        
        # Just for showcasing the momentum merge step
        # (not used in tracking for the rest of the pipeline)
        try:
            use_advanced = config.get('use_advanced_momentum_merging', True)
            if use_advanced:
                momentum = adaptive_linear_momentum_merge_fast(binary,
                    angle_steps=config['adaptive_angle_steps'],
                    base_radius=config['adaptive_base_radius'],
                    max_elongation=config['adaptive_max_elongation'],
                    momentum_boost=config['momentum_boost'],
                    linearity_threshold=config['adaptive_linearity_threshold'],
                    downscale_factor=config['downscale_factor'],
                    top_k_bins=config['top_k_bins'],
                    min_coverage_percent=config['min_coverage_percent'],
                    gaussian_sigma=config['gaussian_sigma']
                )
            else:
                # Use basic enhancement methods (morphological dilation or Gaussian blur)
                use_dilation = config.get('basic_use_dilation', True)
                if use_dilation:
                    # Use morphological dilation to grow non-zero pixels into nearby zero pixels
                    kernel_size = config.get('basic_dilation_kernel_size', 3)
                    iterations = config.get('basic_dilation_iterations', 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    momentum = cv2.dilate(binary, kernel, iterations=iterations)
                else:
                    # Fallback to Gaussian blur
                    kernel_size = config.get('basic_gaussian_kernel_size', 3)
                    gaussian_sigma = config.get('basic_gaussian_sigma', 1.0)
                    momentum = cv2.GaussianBlur(binary, (kernel_size, kernel_size), gaussian_sigma)
            momentum_display = cv2.cvtColor(momentum, cv2.COLOR_GRAY2BGR)
        except:
            momentum = binary
            momentum_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 2. Image processing
        try:
            _, edges = preprocess_edges(binary, config)
            edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        except:
            edges = binary
            edges_display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 4. Search mask (from tracker)
        search_mask_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        search_mask = tracker._get_search_mask((H, W))
        if search_mask is not None:
            overlay = search_mask_display.copy()
            overlay[search_mask > 0] = [0, 255, 0]  # Green
            search_mask_display = cv2.addWeighted(search_mask_display, 0.7, overlay, 0.3, 0)
            
            # Draw ellipse outline (if tracker has established tracking)
            if tracker.center and tracker.size and tracker.angle is not None:
                try:
                    ell = ((int(tracker.center[0]), int(tracker.center[1])),
                           (int(tracker.size[0] * (1 + config['ellipse_expansion_factor'])),
                            int(tracker.size[1] * (1 + config['ellipse_expansion_factor']))),
                           tracker.angle)
                    cv2.ellipse(search_mask_display, ell, (255, 0, 255), 2)
                except:
                    pass
        
        # 5. Track using NetTracker
        contour = tracker.find_and_update(edges, (H, W))
        
        best_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        if contour is not None:
            cv2.drawContours(best_display, [contour], -1, (0, 255, 0), 2)
            if len(contour) >= 5:
                try:
                    ell = cv2.fitEllipse(contour)
                    cv2.ellipse(best_display, ell, (255, 0, 255), 1)
                except:
                    pass
        
        # 6. Distance visualization
        distance_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
        distance_result = tracker.calculate_distance(W, H)
        
        if distance_result is not None:
            distance_px, angle_deg = distance_result
        else:
            distance_px, angle_deg = None, None
        
        if distance_px is not None:
            # Draw center line
            cv2.line(distance_display, (W//2, 0), (W//2, H), (128, 128, 128), 1)
            
            # Draw distance point at the intersection of red line with center line
            cv2.circle(distance_display, (W//2, int(distance_px)), 8, (0, 0, 255), -1)
            cv2.circle(distance_display, (W//2, int(distance_px)), 8, (255, 255, 255), 2)
            
            # Draw the red line itself (perpendicular to major axis)
            if tracker.center and tracker.size and tracker.angle is not None:
                ang_r = np.radians(angle_deg)
                half_len = max(tracker.size) / 2
                
                p1x = int(tracker.center[0] + half_len * np.cos(ang_r))
                p1y = int(tracker.center[1] + half_len * np.sin(ang_r))
                p2x = int(tracker.center[0] - half_len * np.cos(ang_r))
                p2y = int(tracker.center[1] - half_len * np.sin(ang_r))
                
                cv2.line(distance_display, (p1x, p1y), (p2x, p2y), (0, 0, 255), 2)
            
            dist_m = None
            if extent:
                px2m = (extent[3] - extent[2]) / H
                dist_m = extent[2] + distance_px * px2m
                cv2.putText(distance_display, f"{dist_m:.2f}m", (W//2 + 15, int(distance_px)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Ensure all arrays have the same number of dimensions for np.hstack
        if binary.ndim == 2:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if momentum_display.ndim == 2:
            momentum_display = cv2.cvtColor(momentum_display, cv2.COLOR_GRAY2BGR)
        if edges_display.ndim == 2:
            edges_display = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR

        # Add text labels to each panel (single text only)
        cv2.putText(binary, "1. Raw - Binary Frame", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(momentum_display, "2. Momentum Merged", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(edges_display, "3. Edges", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(search_mask_display, "4. Search Mask", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(best_display, "5. Best Contour", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(distance_display, "6. Distance", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Stack the images horizontally
        row0 = np.hstack([binary, momentum_display, edges_display])
        row1 = np.hstack([search_mask_display, best_display, distance_display])
        grid_frame = np.vstack([row0, row1])
        
        # Ensure uint8 before writing
        if grid_frame.dtype != np.uint8:
            grid_frame = np.clip(grid_frame, 0, 255).astype(np.uint8)
        
        # Frame info (single text only)
        frame_info = f'Frame: {idx} | {tracker.get_status()}'
        cv2.putText(grid_frame, frame_info, (10, grid_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        vw.write(grid_frame)
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{actual} frames")
    
    vw.release()
    print(f"\n✓ Video saved to: {output_path}")
    print(f"Grid layout: Raw | Momentum | Edges")
    print(f"             Search Mask | Best Contour | Distance")
    
    return output_path

def save_pipeline_step_figures(
    npz_file_index=0,
    frame_start=0,
    frame_count=10,
    output_dir='pipeline_steps',
    dpi=150
):
    """
    Save individual figures for each step of the contour detection pipeline.
    Creates one figure per step per frame for detailed analysis.
    
    Args:
        npz_file_index: Index of NPZ file to process
        frame_start: Starting frame index
        frame_count: Number of consecutive frames to process
        output_dir: Directory to save figures (will be created if not exists)
        dpi: Resolution of saved figures
        
    Output Structure:
        output_dir/
            frame_0000/
                step1_binary.png
                step2_momentum.png
                step3_edges.png
                step4_search_mask.png
                step5_contour.png
                step6_distance.png
            frame_0001/
                ...
    
    Returns:
        Path to output directory
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    from utils.sonar_utils import load_cone_run_npz, to_uint8_gray
    from utils.io_utils import get_available_npz_files
    from utils.sonar_tracking import NetTracker
    
    print("=== SAVING PIPELINE STEP FIGURES ===")
    print(f"Processing {frame_count} frames starting from frame {frame_start}")
    
    # Load NPZ data
    files = get_available_npz_files()
    if npz_file_index >= len(files):
        print(f"Error: NPZ file index {npz_file_index} not available")
        return None
    
    print(f"Loading: {files[npz_file_index].name}")
    cones, timestamps, extent, _ = load_cone_run_npz(files[npz_file_index])
    T = len(cones)
    
    if frame_start + frame_count > T:
        frame_count = T - frame_start
        print(f"Adjusted frame_count to {frame_count} (available frames)")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracker
    config = {**IMAGE_PROCESSING_CONFIG, **TRACKING_CONFIG}
    tracker = NetTracker(config)
    
    print(f"Output directory: {output_path.absolute()}")
    print(f"Image resolution: {dpi} DPI")
    print(f"Tracker initialized with config:")
    print(f"  binary_threshold: {config['binary_threshold']}")
    print(f"  min_contour_area: {config.get('min_contour_area', 100)}")
    print()
    
    # Process each frame
    for i in range(frame_count):
        idx = frame_start + i
        frame_u8 = to_uint8_gray(cones[idx])
        H, W = frame_u8.shape
        
        # Create frame directory
        frame_dir = output_path / f"frame_{idx:04d}"
        frame_dir.mkdir(exist_ok=True)
        
        print(f"Processing frame {idx} ({i+1}/{frame_count})...")
        
        # ===== STEP 1: Binary Conversion =====
        binary_threshold = config['binary_threshold']
        binary = (frame_u8 > binary_threshold).astype(np.uint8) * 255
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.patch.set_facecolor('black')
        ax.imshow(binary, cmap='gray', vmin=0, vmax=255, origin='lower')
        ax.set_title(f'Step 1: Binary Conversion (threshold={binary_threshold})', 
                    color='white', fontsize=14)
        ax.axis('off')
        ax.set_facecolor('black')
        
        # Add statistics
        white_pct = 100 * np.sum(binary == 255) / binary.size
        stats_text = f"White: {white_pct:.1f}%"
        ax.text(10, 30, stats_text, color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(frame_dir / 'step1_binary.png', dpi=dpi, facecolor='black')
        plt.close(fig)
        
        # ===== STEP 2: Momentum Merging =====
        try:
            use_advanced = config.get('use_advanced_momentum_merging', True)
            if use_advanced:
                momentum = adaptive_linear_momentum_merge_fast(
                    binary,
                    angle_steps=config['adaptive_angle_steps'],
                    base_radius=config['adaptive_base_radius'],
                    max_elongation=config['adaptive_max_elongation'],
                    momentum_boost=config['momentum_boost'],
                    linearity_threshold=config['adaptive_linearity_threshold'],
                    downscale_factor=config['downscale_factor'],
                    top_k_bins=config['top_k_bins'],
                    min_coverage_percent=config['min_coverage_percent'],
                    gaussian_sigma=config['gaussian_sigma']
                )
                method = "Advanced Adaptive Momentum"
            else:
                use_dilation = config.get('basic_use_dilation', True)
                if use_dilation:
                    kernel_size = config.get('basic_dilation_kernel_size', 3)
                    iterations = config.get('basic_dilation_iterations', 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    momentum = cv2.dilate(binary, kernel, iterations=iterations)
                    method = f"Basic Dilation ({kernel_size}x{kernel_size}, {iterations} iter)"
                else:
                    kernel_size = config.get('basic_gaussian_kernel_size', 3)
                    gaussian_sigma = config.get('basic_gaussian_sigma', 1.0)
                    momentum = cv2.GaussianBlur(binary, (kernel_size, kernel_size), gaussian_sigma)
                    method = f"Gaussian Blur (σ={gaussian_sigma})"
        except:
            momentum = binary
            method = "None (fallback)"
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.patch.set_facecolor('black')
        ax.imshow(momentum, cmap='gray', vmin=0, vmax=255, origin='lower')
        ax.set_title(f'Step 2: Momentum Merging\n{method}', 
                    color='white', fontsize=14)
        ax.axis('off')
        ax.set_facecolor('black')
        
        enhanced_pct = 100 * np.sum(momentum > 0) / momentum.size
        stats_text = f"Enhanced: {enhanced_pct:.1f}%"
        ax.text(10, 30, stats_text, color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(frame_dir / 'step2_momentum.png', dpi=dpi, facecolor='black')
        plt.close(fig)
        
        # ===== STEP 3: Edge Detection =====
        try:
            _, edges = preprocess_edges(binary, config)
            method = "Advanced Edge Detection"
        except:
            kernel_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            raw_edges = cv2.filter2D(binary, cv2.CV_32F, kernel_edge)
            edges = np.clip(raw_edges, 0, 255).astype(np.uint8)
            edges = (edges > 0).astype(np.uint8) * 255
            method = "Basic Laplacian"
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.patch.set_facecolor('black')
        ax.imshow(edges, cmap='gray', vmin=0, vmax=255, origin='lower')
        ax.set_title(f'Step 3: Edge Detection\n{method}', 
                    color='white', fontsize=14)
        ax.axis('off')
        ax.set_facecolor('black')
        
        edge_pct = 100 * np.sum(edges > 0) / edges.size
        stats_text = f"Edge pixels: {edge_pct:.2f}%"
        ax.text(10, 30, stats_text, color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(frame_dir / 'step3_edges.png', dpi=dpi, facecolor='black')
        plt.close(fig)
        
        # ===== STEP 4: Search Mask (AOI) =====
        search_mask = tracker._get_search_mask((H, W))
        
        # Create RGB visualization
        search_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2RGB)
        if search_mask is not None:
            # Green overlay for search area
            overlay = search_display.copy()
            overlay[search_mask > 0] = [0, 255, 0]
            search_display = cv2.addWeighted(search_display, 0.7, overlay, 0.3, 0).astype(np.uint8)
            
            # Draw ellipse outline (if tracker has established tracking)
            if tracker.center and tracker.size and tracker.angle is not None:
                try:
                    ell = ((int(tracker.center[0]), int(tracker.center[1])),
                           (int(tracker.size[0] * (1 + config['ellipse_expansion_factor'])),
                            int(tracker.size[1] * (1 + config['ellipse_expansion_factor']))),
                           tracker.angle)
                    cv2.ellipse(search_display, ell, (255, 0, 255), 2)
                except:
                    pass
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.patch.set_facecolor('black')
        ax.imshow(search_display, origin='lower')
        ax.set_title(f'Step 4: Search Mask (AOI)\nStatus: {tracker.get_status()}', 
                    color='white', fontsize=14)
        ax.axis('off')
        ax.set_facecolor('black')
        
        if search_mask is not None:
            coverage = 100 * np.sum(search_mask > 0) / (H * W)
            stats_text = f"Coverage: {coverage:.1f}%"
            if tracker.center:
                stats_text += f"\nCenter: ({tracker.center[0]:.0f}, {tracker.center[1]:.0f})"
        else:
            stats_text = "Full-frame search"
        
        ax.text(10, 30, stats_text, color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(frame_dir / 'step4_search_mask.png', dpi=dpi, facecolor='black')
        plt.close(fig)
        
        # ===== STEP 5: Contour Detection =====
        contour = tracker.find_and_update(edges, (H, W))
        
        contour_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2RGB)
        if contour is not None:
            cv2.drawContours(contour_display, [contour], -1, (0, 255, 0), 2)
            
            if len(contour) >= 5:
                try:
                    ell = cv2.fitEllipse(contour)
                    cv2.ellipse(contour_display, ell, (255, 0, 255), 2)
                    
                    # Draw center point
                    center = (int(ell[0][0]), int(ell[0][1]))
                    cv2.circle(contour_display, center, 5, (0, 0, 255), -1)
                    cv2.circle(contour_display, center, 5, (255, 255, 255), 2)
                except:
                    pass
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.patch.set_facecolor('black')
        ax.imshow(contour_display, origin='lower')
        ax.set_title(f'Step 5: Contour Detection\n{tracker.get_status()}', 
                    color='white', fontsize=14)
        ax.axis('off')
        ax.set_facecolor('black')
        
        if contour is not None:
            stats_text = f"Contour points: {len(contour)}"
            if tracker.center and tracker.size:
                stats_text += f"\nCenter: ({tracker.center[0]:.1f}, {tracker.center[1]:.1f})"
                stats_text += f"\nSize: ({tracker.size[0]:.1f}, {tracker.size[1]:.1f})"
            if tracker.angle is not None:
                stats_text += f"\nAngle: {tracker.angle:.1f}°"
        else:
            stats_text = "No contour detected"
        
        ax.text(10, 30, stats_text, color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(frame_dir / 'step5_contour.png', dpi=dpi, facecolor='black')
        plt.close(fig)
        
        # ===== STEP 6: Distance Measurement =====
        distance_result = tracker.calculate_distance(W, H)
        
        distance_display = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2RGB)
        
        # Draw center reference line
        cv2.line(distance_display, (W//2, 0), (W//2, H), (128, 128, 128), 1)
        
        if distance_result is not None:
            distance_px, angle_deg = distance_result
            
            # Draw the perpendicular distance line (red)
            if tracker.center and tracker.size and tracker.angle is not None:
                ang_r = np.radians(angle_deg)
                half_len = max(tracker.size) / 2
                
                p1x = int(tracker.center[0] + half_len * np.cos(ang_r))
                p1y = int(tracker.center[1] + half_len * np.sin(ang_r))
                p2x = int(tracker.center[0] - half_len * np.cos(ang_r))
                p2y = int(tracker.center[1] - half_len * np.sin(ang_r))
                
                cv2.line(distance_display, (p1x, p1y), (p2x, p2y), (0, 0, 255), 3)
            
            # Draw distance measurement point
            cv2.circle(distance_display, (W//2, int(distance_px)), 10, (0, 0, 255), -1)
            cv2.circle(distance_display, (W//2, int(distance_px)), 10, (255, 255, 255), 2)
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        fig.patch.set_facecolor('black')
        ax.imshow(distance_display, origin='lower')
        ax.set_title('Step 6: Distance Measurement', 
                    color='white', fontsize=14)
        ax.axis('off')
        ax.set_facecolor('black')
        
        if distance_result is not None:
            distance_px, angle_deg = distance_result
            stats_text = f"Distance: {distance_px:.1f} px"
            stats_text += f"\nAngle: {angle_deg:.1f}°"
            
            # Convert to meters if extent available
            if extent:
                px2m = (extent[3] - extent[2]) / H
                distance_m = extent[2] + distance_px * px2m
                stats_text += f"\nDistance: {distance_m:.2f} m"
        else:
            stats_text = "No distance measurement"
        
        ax.text(10, 30, stats_text, color='white', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(frame_dir / 'step6_distance.png', dpi=dpi, facecolor='black')
        plt.close(fig)
        
        print(f"  ✓ Saved 6 figures to {frame_dir.name}/")
    
    print(f"\n✓ COMPLETE! Processed {frame_count} frames")
    print(f"Output directory: {output_path.absolute()}")
    print(f"\nDirectory structure:")
    print(f"  {output_path.name}/")
    print(f"    frame_XXXX/")
    print(f"      step1_binary.png")
    print(f"      step2_momentum.png")
    print(f"      step3_edges.png")
    print(f"      step4_search_mask.png")
    print(f"      step5_contour.png")
    print(f"      step6_distance.png")
    
    return output_path