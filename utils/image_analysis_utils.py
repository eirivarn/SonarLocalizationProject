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


def extent_px_to_m(extent: Tuple[float,float,float,float], H: int, W: int, i_row: float, j_col: float):
    x_min, x_max, y_min, y_max = extent
    x = x_min + (x_max-x_min) * (j_col/max(W-1,1))
    y = y_min + (y_max-y_min) * (i_row/max(H-1,1))
    return x, y

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

# ============================ Simple Frame Utilities ============================

def get_available_npz_files(npz_dir: str = "exports/outputs") -> List[Path]:
    """Get list of available NPZ cone files."""
    npz_dir = Path(npz_dir)
    return list(npz_dir.glob("*_cones.npz")) if npz_dir.exists() else []


def pick_and_save_frame(npz_file_index: int = 0, frame_position: str = 'middle', 
                       output_path: str = "sample_frame.png", npz_dir: str = "exports/outputs") -> Dict:
    """
    Pick a frame from NPZ file and save it locally as PNG.
    
    Args:
        npz_file_index: Index of NPZ file to use (0-based)
        frame_position: 'start', 'middle', 'end', or float (0.0-1.0) for position
        output_path: Path to save the frame image
        npz_dir: Directory containing NPZ files
        
    Returns:
        Dictionary with frame info and saved path
    """
    npz_files = get_available_npz_files(npz_dir)
    
    if not npz_files:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")
    
    if npz_file_index >= len(npz_files):
        raise IndexError(f"NPZ file index {npz_file_index} out of range. Available: 0-{len(npz_files)-1}")
    
    npz_path = npz_files[npz_file_index]
    
    # Load the data
    cones, timestamps, extent, meta = load_cone_run_npz(npz_path)
    T = cones.shape[0]
    
    # Determine frame index
    if frame_position == 'start':
        frame_index = 0
    elif frame_position == 'end':
        frame_index = T - 1
    elif frame_position == 'middle':
        frame_index = T // 2
    elif isinstance(frame_position, (int, float)):
        if frame_position <= 1.0:
            frame_index = int(frame_position * T)
        else:
            frame_index = int(frame_position)
        frame_index = max(0, min(frame_index, T-1))
    else:
        raise ValueError(f"Invalid frame_position: {frame_position}")
    
    # Extract and convert frame
    frame = cones[frame_index]
    frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    
    # Save as PNG
    output_path = Path(output_path)
    cv2.imwrite(str(output_path), frame_uint8)
    
    info = {
        'saved_path': str(output_path),
        'npz_file': npz_path.name,
        'frame_index': frame_index,
        'total_frames': T,
        'timestamp': timestamps[frame_index],
        'shape': frame_uint8.shape,
        'extent': extent
    }
    
    print(f"Frame saved to: {output_path}")
    print(f"Source: {npz_path.name}, Frame {frame_index}/{T-1}")
    print(f"Timestamp: {timestamps[frame_index].strftime('%H:%M:%S')}")
    print(f"Shape: {frame_uint8.shape}")
    
    return info


def load_saved_frame(image_path: str) -> np.ndarray:
    """
    Load a saved frame image for processing.
    
    Args:
        image_path: Path to the saved image
        
    Returns:
        Loaded image as numpy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return image


def list_npz_files(npz_dir: str = "exports/outputs") -> None:
    """Print information about available NPZ files."""
    npz_files = get_available_npz_files(npz_dir)
    
    if not npz_files:
        print(f"No NPZ files found in {npz_dir}")
        return
    
    print(f"Available NPZ files in {npz_dir}:")
    for i, npz_path in enumerate(npz_files):
        try:
            cones, timestamps, extent, meta = load_cone_run_npz(npz_path)
            print(f"  {i}: {npz_path.name}")
            print(f"     {cones.shape[0]} frames, {timestamps[0].strftime('%H:%M:%S')} to {timestamps[-1].strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"  {i}: {npz_path.name} - Error: {e}")


# ============================ Image Processing Pipeline ============================


