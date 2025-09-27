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

# Image Processing Settings
IMAGE_PROCESSING_CONFIG = {
    # DIRECTIONAL MOMENTUM MERGING - Novel approach to replace traditional blurring
    # Instead of uniform Gaussian blur, merge cells that gain momentum in line directions
    'use_momentum_merging': True,     # Enable/disable momentum-based merging
    'momentum_search_radius': 3,      # How far to search for neighboring cells to merge (keep ≤3 for speed)
    'momentum_threshold': 0.1,        # Minimum momentum to consider a direction valid (lowered for speed)
    'momentum_decay': 0.9,           # How quickly momentum decays (0.7-0.9, lower = faster)
    'momentum_boost': 5.0,           # Momentum boost when continuing in same direction
    
    # Traditional Gaussian Blur (used when momentum_merging is False)
    # EFFECTS:
    # - SMALLER values (5,5) to (15,15): Preserves fine details, may detect noise as edges
    # - LARGER values (21,21) to (35,35): Removes noise and small details, smoother contours
    # - TOO LARGE (>35): May blur away the shapes you want to detect
    # COMMON VALUES: (15,15) for detailed detection, (25,25) for noisy images
    'blur_kernel_size': (31, 31),

    # Canny Edge Detection Thresholds - Controls sensitivity to edges
    # HOW IT WORKS: Pixels with gradient > high_threshold = strong edges
    #               Pixels between low and high = weak edges (kept if connected to strong)
    #               Pixels < low_threshold = ignored
    # EFFECTS:
    # - LOWER thresholds (30, 100): Detects more edges, including noise and weak features
    # - HIGHER thresholds (70, 200): Only strong, clear edges, may miss faint shapes
    # - RATIO should be ~1:2 or 1:3 (low:high) for best results
    # TUNING TIPS:
    #   * Too many edges? Increase both values (try 70, 180)
    #   * Missing shapes? Decrease both values (try 30, 100)  
    #   * Noisy edges? Increase blur_kernel_size instead
    'canny_low_threshold': 40,       
    'canny_high_threshold': 120,     

    # Minimum Contour Area - Filters out tiny detected shapes
    # EFFECTS:
    # - SMALLER values (20-50): Detects small details, may include noise
    # - LARGER values (100-200): Only detects substantial shapes, cleaner results
    # - TOO LARGE: May exclude the shapes you're looking for
    # TUNING: Start with 50, increase if too many small noise contours appear
    # KEEN ON LINES: Lower threshold to catch thinner lines
    'min_contour_area': 100,
    
    # MORPHOLOGICAL PROCESSING - Helps connect incomplete/open edges
    # These operations help detect net lines that don't form complete closed shapes
    
    # Morphological Closing Kernel Size - Connects nearby edge pixels
    # EFFECTS:
    # - SMALLER values (3, 5): Only connects very close edge pixels, preserves detail
    # - LARGER values (7, 9): Connects farther gaps, may merge separate objects
    # - 0: Disables morphological closing
    # RECOMMENDED: 3 for thin lines, 5 for thicker connections needed
    # KEEN ON LINES: Larger kernel to connect broken line segments
    'morph_close_kernel': 5,
    
    # Edge Dilation Iterations - Thickens detected edges for better contour detection
    # EFFECTS:  
    # - 0: No dilation, keeps original edge thickness
    # - 1: Slightly thickens edges, helps with thin line detection
    # - 2-3: Significantly thickens edges, may merge nearby features
    # RECOMMENDED: 1 for most cases, 0 if you have thick clear lines already
    # KEEN ON LINES: More iterations to strengthen thin line edges
    'edge_dilation_iterations': 2,
}

# Elongation Detection Settings - Controls how "elongated-ness" is calculated
# These weights determine what makes a shape "elongated" - they should sum to ~1.0
ELONGATION_CONFIG = {
    # Basic Aspect Ratio Weight - Simple width/height ratio from bounding rectangle
    # EFFECTS:
    # - HIGHER weight (0.4-0.6): Strongly prefers shapes with high width/height ratios
    # - LOWER weight (0.1-0.2): Less emphasis on simple rectangular elongation
    # LIMITATION: Doesn't consider shape orientation - a diagonal line appears square
    # VERY KEEN ON STRAIGHT SHAPES: High weight for rectangular elongation
    'aspect_ratio_weight': 0.4,      

    # Ellipse Elongation Weight - Major/minor axis ratio from fitted ellipse
    # EFFECTS:
    # - HIGHER weight (0.5-0.7): Best for detecting true elongated shapes regardless of orientation
    # - LOWER weight (0.2-0.3): Less emphasis on true geometric elongation
    # ADVANTAGE: Considers actual shape orientation, more accurate than aspect ratio
    # RECOMMENDED: Keep this highest for best elongation detection
    # VERY KEEN ON STRAIGHT SHAPES: Maximum weight for true geometric elongation
    'ellipse_elongation_weight': 0.7, 

    # Solidity Weight - How "filled" the shape is (contour_area / convex_hull_area)
    # EFFECTS:
    # - HIGHER weight (0.2-0.4): Prefers sparse, thin shapes (lines, curves)
    # - LOWER weight (0.0-0.1): Less preference for hollow shapes
    # NOTE: Uses (1-solidity) so lower solidity = higher score
    # USEFUL FOR: Distinguishing lines/ropes from filled rectangular objects
    # KEEN ON STRAIGHT SHAPES: Higher weight to prefer thin, line-like shapes
    'solidity_weight': 0.1,          

    # Extent Weight - How efficiently the shape fills its bounding rectangle
    # EFFECTS:
    # - HIGHER weight (0.2-0.4): Prefers shapes that fill their bounding box well
    # - LOWER weight (0.0-0.1): Less emphasis on rectangular efficiency
    # USEFUL FOR: Distinguishing compact shapes from sprawling irregular ones
    # STRAIGHT SHAPES: Zero weight - we don't care about rectangular efficiency for lines
    'extent_weight': 0.0,            

    # Perimeter Weight - Normalized perimeter-to-area ratio
    # EFFECTS:
    # - HIGHER weight (0.2-0.4): Prefers shapes with complex/long perimeters
    # - LOWER weight (0.0-0.1): Less emphasis on perimeter complexity
    # USEFUL FOR: Detecting winding or irregular elongated shapes
    # NOTE: Automatically capped to prevent extreme values
    # STRAIGHT SHAPES: Zero weight - we want simple straight perimeters, not complex ones
    'perimeter_weight': 0.0,         
}

# Tracking Settings - Controls how the Area of Interest (AOI) system works
TRACKING_CONFIG = {
    # AOI Boost Factor - Score multiplier for contours found within the Area of Interest
    # EFFECTS:
    # - 1.0: No tracking preference, treats all contours equally
    # - 2.0-3.0: Moderate preference for contours near previous detection
    # - 4.0+: Strong tracking, may stick to suboptimal shapes in AOI
    # TUNING:
    #   * Tracker jumps around too much? Increase to 3.0-4.0
    #   * Tracker stuck on wrong shape? Decrease to 1.5-2.0
    #   * No tracking effect? Check if AOI is too small
    # VERY KEEN TRACKING: Set to very high value for aggressive tracking continuity
    'aoi_boost_factor': 1000.0,         

    # AOI Expansion Pixels - How much to expand the search area around previous detection
    # EFFECTS:
    # - SMALLER values (10-20): Tight tracking, may lose fast-moving objects
    # - LARGER values (40-60): Loose tracking, more forgiving of movement
    # - TOO LARGE (>80): May include too much of the image, reduces tracking benefit
    # TUNING:
    #   * Object moves fast between frames? Increase to 40-50
    #   * Tracker picks up nearby noise? Decrease to 15-25
    #   * Object barely moves? Can use smaller values like 20
    # KEEN TRACKING: Smaller expansion for tighter tracking
    'aoi_expansion_pixels': 10,      
}

# Video Settings - Controls visual output and video properties
VIDEO_CONFIG = {
    # Frames Per Second - Playback speed of output video
    # EFFECTS:
    # - LOWER fps (5-8): Slower playback, easier to see details
    # - HIGHER fps (15-20): Faster playback, more natural motion
    # - FILE SIZE: Lower fps = smaller file size
    'fps': 15,                       

    # Show All Contours - Whether to draw all detected contours in light blue
    # EFFECTS:
    # - True: Shows complete detection context, may be cluttered
    # - False: Cleaner view, only shows the selected best contour
    # USEFUL: Turn off when you want to focus only on the tracking result
    'show_all_contours': True,       

    # Show Ellipse - Whether to draw fitted ellipse on the best contour
    # EFFECTS:
    # - True: Shows the geometric interpretation of the shape
    # - False: Cleaner view without ellipse overlay
    # USEFUL: Ellipse helps visualize true shape orientation and elongation
    'show_ellipse': True,            

    # Show Bounding Box - Whether to draw rectangular bounding box
    # EFFECTS:
    # - True: Shows the axis-aligned bounds of the shape
    # - False: Cleaner view without rectangular overlay
    # USEFUL: Compare ellipse vs bounding box to see orientation effects
    'show_bounding_box': True,       

    # Text Scale - Size of overlay text and labels
    # EFFECTS:
    # - SMALLER (0.4-0.5): Less intrusive text, may be hard to read
    # - LARGER (0.8-1.0): More readable text, may clutter the image
    # RECOMMENDATION: 0.6 is usually a good balance
    'text_scale': 0.6,               
}

# ==================== PROCESSING FUNCTIONS ====================

def directional_momentum_merge(frame, search_radius=3, momentum_threshold=0.2, 
                              momentum_decay=0.8, momentum_boost=1.5):
    """
    SUPER-OPTIMIZED directional momentum-based cell merging system.
    Uses pre-computed kernels and vectorized operations for maximum speed.
    """
    # ULTRA-FAST VERSION: Use morphological operations for line enhancement
    if search_radius <= 2:
        return fast_directional_enhance(frame, momentum_threshold, momentum_boost)
    
    result = frame.astype(np.float32)
    
    # Quick gradient-based energy map
    grad_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
    energy_map = np.sqrt(grad_x**2 + grad_y**2) / 255.0
    
    # Skip processing if no significant gradients
    if np.max(energy_map) < momentum_threshold:
        return frame
    
    # Use pre-computed line kernels (cached for speed)
    kernel_size = 5  # Fixed small size for speed
    center = 2
    
    # Horizontal line kernel
    h_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    h_kernel[center, :] = [0.1, 0.2, 0.4, 0.2, 0.1]
    
    # Vertical line kernel  
    v_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    v_kernel[:, center] = [0.1, 0.2, 0.4, 0.2, 0.1]
    
    # Diagonal kernels
    d1_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    d2_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        d1_kernel[i, i] = 0.2
        d2_kernel[i, kernel_size-1-i] = 0.2
    d1_kernel[center, center] = 0.4
    d2_kernel[center, center] = 0.4
    
    # Apply all kernels (parallel convolutions)
    responses = [
        cv2.filter2D(result, -1, h_kernel),
        cv2.filter2D(result, -1, v_kernel), 
        cv2.filter2D(result, -1, d1_kernel),
        cv2.filter2D(result, -1, d2_kernel)
    ]
    
    # Maximum response enhances lines in any direction
    enhanced = np.maximum.reduce(responses)
    
    # Apply momentum boost based on gradient energy
    boost_factor = 1.0 + momentum_boost * np.clip(energy_map, 0, 1)
    result = enhanced * boost_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)

def fast_directional_enhance(frame, threshold=0.2, boost=1.5):
    """
    Ultra-fast directional enhancement using morphological operations.
    Optimized for real-time processing.
    """
    # Use morphological opening with line-shaped kernels
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # Horizontal lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # Vertical lines
    
    # Enhance horizontal and vertical structures
    h_enhanced = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, h_kernel)
    v_enhanced = cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, v_kernel)
    
    # Combine enhancements
    enhanced = np.maximum(h_enhanced, v_enhanced)
    
    # Boost based on local gradient
    grad = cv2.Laplacian(frame, cv2.CV_32F)
    grad_norm = np.abs(grad) / 255.0
    
    # Apply boost where gradients are strong
    boost_mask = grad_norm > threshold
    result = frame.astype(np.float32)
    result[boost_mask] += boost * enhanced[boost_mask]
    
    return np.clip(result, 0, 255).astype(np.uint8)

def process_frame_for_video(frame, prev_area_of_interest=None):
    """
    Process a single frame and return annotated result with area of interest tracking
    Uses the configuration settings defined above for easy adjustment.
    """
    # Get settings from config
    blur_size = IMAGE_PROCESSING_CONFIG['blur_kernel_size']
    canny_low = IMAGE_PROCESSING_CONFIG['canny_low_threshold']
    canny_high = IMAGE_PROCESSING_CONFIG['canny_high_threshold']
    min_area = IMAGE_PROCESSING_CONFIG['min_contour_area']
    aoi_boost_factor = TRACKING_CONFIG['aoi_boost_factor']
    aoi_expansion = TRACKING_CONFIG['aoi_expansion_pixels']
    
    # Apply either momentum merging or traditional blur
    if IMAGE_PROCESSING_CONFIG['use_momentum_merging']:
        # Use novel directional momentum merging system
        processed_input = directional_momentum_merge(
            frame,
            search_radius=IMAGE_PROCESSING_CONFIG['momentum_search_radius'],
            momentum_threshold=IMAGE_PROCESSING_CONFIG['momentum_threshold'],
            momentum_decay=IMAGE_PROCESSING_CONFIG['momentum_decay'],
            momentum_boost=IMAGE_PROCESSING_CONFIG['momentum_boost']
        )
    else:
        # Traditional Gaussian blur
        processed_input = cv2.GaussianBlur(frame, blur_size, 0)
    
    # Apply edge detection to processed input
    edges = cv2.Canny(processed_input, canny_low, canny_high)
    
    # ENHANCED: Handle open/incomplete edges with morphological closing
    # This helps connect nearby edge pixels to form more complete contours
    morph_kernel_size = IMAGE_PROCESSING_CONFIG['morph_close_kernel']
    dilation_iterations = IMAGE_PROCESSING_CONFIG['edge_dilation_iterations']
    
    edges_processed = edges.copy()
    
    # Apply morphological closing if enabled (kernel size > 0)
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        edges_processed = cv2.morphologyEx(edges_processed, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation if enabled (iterations > 0)
    if dilation_iterations > 0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_processed = cv2.dilate(edges_processed, kernel_dilate, iterations=dilation_iterations)
    
    # Find contours on the processed edges
    # Use RETR_LIST to get all contours, not just external ones
    contours, _ = cv2.findContours(edges_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert to color for annotations
    result_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Draw area of interest from previous frame
    current_aoi = None
    if prev_area_of_interest is not None:
        aoi_x, aoi_y, aoi_w, aoi_h = prev_area_of_interest
        # Expand the AOI
        expanded_x = max(0, aoi_x - aoi_expansion)
        expanded_y = max(0, aoi_y - aoi_expansion)
        expanded_w = min(frame.shape[1] - expanded_x, aoi_w + 2 * aoi_expansion)
        expanded_h = min(frame.shape[0] - expanded_y, aoi_h + 2 * aoi_expansion)
        
        current_aoi = (expanded_x, expanded_y, expanded_w, expanded_h)
        
        # Draw AOI rectangle in yellow (semi-transparent effect with thin line)
        cv2.rectangle(result_frame, (expanded_x, expanded_y), 
                     (expanded_x + expanded_w, expanded_y + expanded_h), 
                     (0, 255, 255), 1)
        
        # Add AOI label
        cv2.putText(result_frame, 'AOI', (expanded_x + 5, expanded_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0, 255, 255), 1)
    
    # Find most elongated contour with AOI preference
    most_elongated_contour = None
    max_elongation_score = 0
    contour_count = 0
    aoi_contour_count = 0
    
    def contour_overlaps_aoi(contour, aoi):
        """Check if contour overlaps with area of interest"""
        if aoi is None:
            return False
        
        aoi_x, aoi_y, aoi_w, aoi_h = aoi
        cx, cy, cw, ch = cv2.boundingRect(contour)
        
        # Check if bounding rectangles overlap
        return not (cx + cw < aoi_x or cx > aoi_x + aoi_w or 
                   cy + ch < aoi_y or cy > aoi_y + aoi_h)
    
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            contour_count += 1
            
            # Calculate elongation metrics
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Extent
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Ellipse elongation
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center), (minor_axis, major_axis), angle = ellipse
                    ellipse_elongation = major_axis / minor_axis if minor_axis > 0 else 0
                except:
                    ellipse_elongation = aspect_ratio
            else:
                ellipse_elongation = aspect_ratio
            
            # ENHANCED: Add straightness measurement for line detection
            straightness_score = 1.0  # Default for shapes that can't be measured
            
            if len(contour) >= 10:  # Need sufficient points for line fitting
                # Fit a line to the contour points and measure how well they align
                try:
                    # Reshape contour points for cv2.fitLine
                    points = contour.reshape(-1, 2).astype(np.float32)
                    
                    # Fit line using least squares
                    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    # Calculate distance of each point from the fitted line
                    # Line equation: (x-x0)/vx = (y-y0)/vy
                    distances = []
                    for point in points:
                        px, py = point[0], point[1]
                        # Distance from point to line
                        # Using formula: |ax + by + c| / sqrt(a² + b²)
                        # Line in form: vy*x - vx*y + (vx*y0 - vy*x0) = 0
                        a, b, c = vy, -vx, vx*y0 - vy*x0
                        dist = abs(a*px + b*py + c) / np.sqrt(a*a + b*b)
                        distances.append(dist)
                    
                    # Calculate straightness as inverse of average distance from line
                    avg_distance = np.mean(distances)
                    max_distance = max(w, h) * 0.1  # 10% of bounding box dimension as reference
                    
                    # Straightness score: 1.0 for perfect line, approaches 0 for curved shapes
                    straightness_score = max(0.1, 1.0 - (avg_distance / max(max_distance, 1.0)))
                    
                except:
                    straightness_score = 0.5  # Default if line fitting fails
            
            # Composite elongation score using configurable weights + straightness bonus
            composite_elongation = (
                aspect_ratio * ELONGATION_CONFIG['aspect_ratio_weight'] +
                ellipse_elongation * ELONGATION_CONFIG['ellipse_elongation_weight'] +
                (1 - solidity) * ELONGATION_CONFIG['solidity_weight'] +
                extent * ELONGATION_CONFIG['extent_weight'] +
                min(aspect_ratio / 10, 0.5) * ELONGATION_CONFIG['perimeter_weight']
            )
            
            # BOOST FOR STRAIGHT SHAPES: Apply straightness multiplier
            # Very keen on straight shapes - multiply by straightness score
            composite_elongation *= (0.5 + 1.5 * straightness_score)  # Range: 0.5x to 2.0x boost
            
            elongation_score = area * composite_elongation
            
            # BOOST SCORE IF IN AREA OF INTEREST
            is_in_aoi = contour_overlaps_aoi(contour, current_aoi)
            if is_in_aoi:
                elongation_score *= aoi_boost_factor
                aoi_contour_count += 1
            
            if elongation_score > max_elongation_score:
                max_elongation_score = elongation_score
                most_elongated_contour = contour
    
    # Draw all contours in light blue (if enabled)
    if VIDEO_CONFIG['show_all_contours'] and contours:
        cv2.drawContours(result_frame, contours, -1, (255, 200, 100), 1)
    
    # Determine next AOI - KEEP LAST KNOWN POSITION IF NO CONTOUR FOUND
    next_aoi = prev_area_of_interest  # Default: keep previous AOI
    detection_status = "LOST"  # Default status
    
    if most_elongated_contour is not None:
        # Draw the main contour in green
        cv2.drawContours(result_frame, [most_elongated_contour], -1, (0, 255, 0), 2)
        
        # Calculate bounding rectangle once
        x, y, w, h = cv2.boundingRect(most_elongated_contour)
        
        # Draw bounding rectangle (if enabled)
        if VIDEO_CONFIG['show_bounding_box']:
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        # Update AOI to new position
        next_aoi = (x, y, w, h)
        
        # Draw fitted ellipse (if enabled and possible)
        if VIDEO_CONFIG['show_ellipse'] and len(most_elongated_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(most_elongated_contour)
                cv2.ellipse(result_frame, ellipse, (255, 0, 255), 1)
                
                # DRAW RED LINE THROUGH MAJOR AXIS ENDPOINTS (rotated 90 degrees)
                # Extract ellipse parameters
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse
                
                # Convert angle from degrees to radians and rotate 90 degrees
                angle_rad = np.radians(angle + 90)
                
                # Calculate half-length of major axis (same length, rotated orientation)
                half_major = major_axis / 2.0
                
                # Calculate endpoints of major axis rotated 90 degrees
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                # Endpoint 1 (in direction of rotated major axis)
                end1_x = int(center_x + half_major * cos_angle)
                end1_y = int(center_y + half_major * sin_angle)
                
                # Endpoint 2 (opposite direction - rotated 90 degrees)
                end2_x = int(center_x - half_major * cos_angle)
                end2_y = int(center_y - half_major * sin_angle)
                
                # Draw red line connecting the two furthest points on the ellipse
                cv2.line(result_frame, (end1_x, end1_y), (end2_x, end2_y), (0, 0, 255), 2)
                
            except:
                pass
        
        # Add text info
        area = cv2.contourArea(most_elongated_contour)
        text_scale = VIDEO_CONFIG['text_scale']
        cv2.putText(result_frame, f'Area: {area:.0f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2)
        cv2.putText(result_frame, f'Score: {max_elongation_score:.0f}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2)
        
        # Determine detection status
        is_tracked = contour_overlaps_aoi(most_elongated_contour, current_aoi)
        detection_status = "TRACKED" if is_tracked else "NEW"
        
    else:
        # NO CONTOUR FOUND - Show "LOST" status and maintain AOI
        cv2.putText(result_frame, 'No contour detected', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (100, 100, 255), 2)
        
        # If we have a previous AOI, draw a dashed rectangle to show "searching" area
        if prev_area_of_interest is not None:
            aoi_x, aoi_y, aoi_w, aoi_h = prev_area_of_interest
            # Draw dashed rectangle effect by drawing multiple small lines
            dash_length = 10
            gap_length = 5
            
            # Top and bottom lines
            for x in range(aoi_x, aoi_x + aoi_w, dash_length + gap_length):
                end_x = min(x + dash_length, aoi_x + aoi_w)
                cv2.line(result_frame, (x, aoi_y), (end_x, aoi_y), (0, 100, 255), 2)  # Top
                cv2.line(result_frame, (x, aoi_y + aoi_h), (end_x, aoi_y + aoi_h), (0, 100, 255), 2)  # Bottom
            
            # Left and right lines
            for y in range(aoi_y, aoi_y + aoi_h, dash_length + gap_length):
                end_y = min(y + dash_length, aoi_y + aoi_h)
                cv2.line(result_frame, (aoi_x, y), (aoi_x, end_y), (0, 100, 255), 2)  # Left
                cv2.line(result_frame, (aoi_x + aoi_w, y), (aoi_x + aoi_w, end_y), (0, 100, 255), 2)  # Right
            
            # Add "SEARCHING" label
            cv2.putText(result_frame, 'SEARCHING', (aoi_x + 5, aoi_y + aoi_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], (0, 100, 255), 2)
    
    # Show detection status with appropriate color
    status_colors = {
        "TRACKED": (0, 255, 0),    # Green
        "NEW": (0, 165, 255),      # Orange
        "LOST": (0, 100, 255)      # Red
    }
    cv2.putText(result_frame, detection_status, (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, VIDEO_CONFIG['text_scale'], status_colors[detection_status], 2)
    
    # Add frame info
    info_y = result_frame.shape[0] - 40
    small_text = VIDEO_CONFIG['text_scale'] * 0.8
    cv2.putText(result_frame, f'Total: {contour_count}', (10, info_y), 
               cv2.FONT_HERSHEY_SIMPLEX, small_text, (255, 255, 255), 1)
    cv2.putText(result_frame, f'In AOI: {aoi_contour_count}', (10, info_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, small_text, (255, 255, 255), 1)
    
    return result_frame, next_aoi

def get_red_line_distance_and_angle(frame):
    """
    Extract the distance (length) and angle of the red line from a single frame.
    
    Args:
        frame: Grayscale image as numpy array (uint8)
        
    Returns:
        tuple: (distance, angle) where:
            - distance: Length of the major axis (red line length)
            - angle: Angle of the major axis in degrees (rotated 90 degrees from ellipse major axis)
            - Returns (None, None) if no suitable contour is found
    """
    # Get settings from config
    blur_size = IMAGE_PROCESSING_CONFIG['blur_kernel_size']
    canny_low = IMAGE_PROCESSING_CONFIG['canny_low_threshold']
    canny_high = IMAGE_PROCESSING_CONFIG['canny_high_threshold']
    min_area = IMAGE_PROCESSING_CONFIG['min_contour_area']
    
    # Apply either momentum merging or traditional blur
    if IMAGE_PROCESSING_CONFIG['use_momentum_merging']:
        processed_input = directional_momentum_merge(
            frame,
            search_radius=IMAGE_PROCESSING_CONFIG['momentum_search_radius'],
            momentum_threshold=IMAGE_PROCESSING_CONFIG['momentum_threshold'],
            momentum_decay=IMAGE_PROCESSING_CONFIG['momentum_decay'],
            momentum_boost=IMAGE_PROCESSING_CONFIG['momentum_boost']
        )
    else:
        processed_input = cv2.GaussianBlur(frame, blur_size, 0)
    
    # Apply edge detection
    edges = cv2.Canny(processed_input, canny_low, canny_high)
    
    # Apply morphological processing
    morph_kernel_size = IMAGE_PROCESSING_CONFIG['morph_close_kernel']
    dilation_iterations = IMAGE_PROCESSING_CONFIG['edge_dilation_iterations']
    
    edges_processed = edges.copy()
    
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        edges_processed = cv2.morphologyEx(edges_processed, cv2.MORPH_CLOSE, kernel)
    
    if dilation_iterations > 0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_processed = cv2.dilate(edges_processed, kernel_dilate, iterations=dilation_iterations)
    
    # Find contours
    contours, _ = cv2.findContours(edges_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find most elongated contour (same logic as in process_frame_for_video)
    most_elongated_contour = None
    max_elongation_score = 0
    
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Calculate elongation metrics (simplified version)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Extent
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Ellipse elongation
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center), (minor_axis, major_axis), angle = ellipse
                    ellipse_elongation = major_axis / minor_axis if minor_axis > 0 else 0
                except:
                    ellipse_elongation = aspect_ratio
            else:
                ellipse_elongation = aspect_ratio
            
            # Straightness measurement
            straightness_score = 1.0
            if len(contour) >= 10:
                try:
                    points = contour.reshape(-1, 2).astype(np.float32)
                    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    distances = []
                    for point in points:
                        px, py = point[0], point[1]
                        a, b, c = vy, -vx, vx*y0 - vy*x0
                        dist = abs(a*px + b*py + c) / np.sqrt(a*a + b*b)
                        distances.append(dist)
                    
                    avg_distance = np.mean(distances)
                    max_distance = max(w, h) * 0.1
                    straightness_score = max(0.1, 1.0 - (avg_distance / max(max_distance, 1.0)))
                except:
                    straightness_score = 0.5
            
            # Composite elongation score
            composite_elongation = (
                aspect_ratio * ELONGATION_CONFIG['aspect_ratio_weight'] +
                ellipse_elongation * ELONGATION_CONFIG['ellipse_elongation_weight'] +
                (1 - solidity) * ELONGATION_CONFIG['solidity_weight'] +
                extent * ELONGATION_CONFIG['extent_weight'] +
                min(aspect_ratio / 10, 0.5) * ELONGATION_CONFIG['perimeter_weight']
            )
            
            composite_elongation *= (0.5 + 1.5 * straightness_score)
            elongation_score = area * composite_elongation
            
            if elongation_score > max_elongation_score:
                max_elongation_score = elongation_score
                most_elongated_contour = contour
    
    # Extract distance and angle from the most elongated contour
    if most_elongated_contour is not None and len(most_elongated_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(most_elongated_contour)
            (center_x, center_y), (minor_axis, major_axis), angle = ellipse
            
            # Distance is the major axis length (same as red line length)
            distance = major_axis
            
            # Angle is the ellipse angle + 90 degrees (matching the red line rotation)
            red_line_angle = angle + 90
            
            # Normalize angle to [0, 360) range
            red_line_angle = red_line_angle % 360
            
            return distance, red_line_angle
            
        except:
            return None, None
    else:
        return None, None

def visualize_processing_steps(frame_index=50, npz_file_index=0, figsize=(15, 5)):
    """
    Visualize the image processing pipeline: Original -> Blur -> Edges -> Contours
    Shows how the current config settings affect each step.
    """
    import matplotlib.pyplot as plt
    
    # Load data
    available_files = get_available_npz_files()
    if npz_file_index >= len(available_files):
        print(f"Error: NPZ file index {npz_file_index} not available")
        return
    
    npz_file = available_files[npz_file_index]
    print(f"Loading from: {npz_file}")
    
    cones, ts, extent, meta = load_cone_run_npz(npz_file)
    
    if frame_index >= len(cones):
        print(f"Error: Frame {frame_index} not available (max: {len(cones)-1})")
        return
    
    # Get frame and convert to uint8
    frame = cones[frame_index]
    frame_uint8 = (frame * 255).astype(np.uint8)
    
    # Get settings from config
    blur_size = IMAGE_PROCESSING_CONFIG['blur_kernel_size']
    canny_low = IMAGE_PROCESSING_CONFIG['canny_low_threshold']
    canny_high = IMAGE_PROCESSING_CONFIG['canny_high_threshold']
    min_area = IMAGE_PROCESSING_CONFIG['min_contour_area']
    
    # Apply processing steps
    if IMAGE_PROCESSING_CONFIG['use_momentum_merging']:
        # Use momentum merging system
        processed_input = directional_momentum_merge(
            frame_uint8,
            search_radius=IMAGE_PROCESSING_CONFIG['momentum_search_radius'],
            momentum_threshold=IMAGE_PROCESSING_CONFIG['momentum_threshold'],
            momentum_decay=IMAGE_PROCESSING_CONFIG['momentum_decay'],
            momentum_boost=IMAGE_PROCESSING_CONFIG['momentum_boost']
        )
        processing_name = "Momentum Merge"
        processing_params = f"radius={IMAGE_PROCESSING_CONFIG['momentum_search_radius']}"
    else:
        # Traditional blur
        processed_input = cv2.GaussianBlur(frame_uint8, blur_size, 0)
        processing_name = "Gaussian Blur"
        processing_params = f"Kernel: {blur_size}"
    
    edges = cv2.Canny(processed_input, canny_low, canny_high)
    
    # Apply morphological processing to handle open edges
    morph_kernel_size = IMAGE_PROCESSING_CONFIG['morph_close_kernel']
    dilation_iterations = IMAGE_PROCESSING_CONFIG['edge_dilation_iterations']
    
    edges_processed = edges.copy()
    
    # Apply morphological closing if enabled
    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        edges_processed = cv2.morphologyEx(edges_processed, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation if enabled
    if dilation_iterations > 0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges_processed = cv2.dilate(edges_processed, kernel_dilate, iterations=dilation_iterations)
    
    # Find contours on the processed edges using RETR_LIST to catch open shapes
    contours, _ = cv2.findContours(edges_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    # Create contour visualization
    contour_vis = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
    if filtered_contours:
        cv2.drawContours(contour_vis, filtered_contours, -1, (0, 255, 0), 2)
        print(f"Found {len(filtered_contours)} contours (area >= {min_area})")
    else:
        print(f"No contours found with area >= {min_area}")
    
    # Create 5-panel visualization to show morphological processing
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    # Panel 1: Original
    axes[0].imshow(frame_uint8, cmap='gray')
    axes[0].set_title(f'Original Frame {frame_index}')
    axes[0].axis('off')
    
    # Panel 2: Processed Input (Momentum or Blur)
    axes[1].imshow(processed_input, cmap='gray')
    axes[1].set_title(f'{processing_name}\n{processing_params}')
    axes[1].axis('off')
    
    # Panel 3: Raw Edges
    axes[2].imshow(edges, cmap='gray')
    axes[2].set_title(f'Canny Edges\nThresholds: {canny_low}, {canny_high}')
    axes[2].axis('off')
    
    # Panel 4: Processed Edges (after morphological operations)
    axes[3].imshow(edges_processed, cmap='gray')
    morph_title = f'Processed Edges\nClose: {morph_kernel_size}, Dilate: {dilation_iterations}'
    axes[3].set_title(morph_title)
    axes[3].axis('off')
    
    # Panel 5: Contours
    axes[4].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[4].set_title(f'Filtered Contours\nMin Area: {min_area}\nFound: {len(filtered_contours)}')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print current settings for reference
    print("\n=== Current Processing Settings ===")
    if IMAGE_PROCESSING_CONFIG['use_momentum_merging']:
        print(f"Momentum merging: radius={IMAGE_PROCESSING_CONFIG['momentum_search_radius']}, "
              f"threshold={IMAGE_PROCESSING_CONFIG['momentum_threshold']}, "
              f"decay={IMAGE_PROCESSING_CONFIG['momentum_decay']}")
    else:
        print(f"Gaussian blur kernel: {blur_size}")
    print(f"Canny thresholds: ({canny_low}, {canny_high})")
    print(f"Morphological closing kernel: {morph_kernel_size}")
    print(f"Edge dilation iterations: {dilation_iterations}")
    print(f"Min contour area: {min_area}")
    print(f"Total contours found: {len(contours)}")
    print(f"Contours after area filter: {len(filtered_contours)}")
    
    # Show improvement from morphological processing
    if morph_kernel_size > 0 or dilation_iterations > 0:
        original_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        original_filtered = [c for c in original_contours if cv2.contourArea(c) >= min_area]
        print(f"Improvement: {len(original_filtered)} -> {len(filtered_contours)} contours after morphological processing")

# Create video from multiple frames with configurable settings
def create_contour_detection_video(npz_file_index=0, frame_start=0, frame_count=100, 
                                 frame_step=5, output_path='contour_detection_video.mp4'):
    """
    Create a video showing contour detection across multiple frames.
    All image processing settings are configured in the CONFIG sections above.
    """
    print("=== CONTOUR DETECTION VIDEO CREATION ===")
    print(f"Creating video with {frame_count} frames, stepping by {frame_step}...")
    
    # Display current configuration
    print("\nCurrent Settings:")
    if IMAGE_PROCESSING_CONFIG['use_momentum_merging']:
        print(f"  Image Processing: MOMENTUM MERGING (radius={IMAGE_PROCESSING_CONFIG['momentum_search_radius']}, "
              f"threshold={IMAGE_PROCESSING_CONFIG['momentum_threshold']}, decay={IMAGE_PROCESSING_CONFIG['momentum_decay']}), "
              f"canny=({IMAGE_PROCESSING_CONFIG['canny_low_threshold']}, {IMAGE_PROCESSING_CONFIG['canny_high_threshold']}), "
              f"min_area={IMAGE_PROCESSING_CONFIG['min_contour_area']}")
    else:
        print(f"  Image Processing: blur={IMAGE_PROCESSING_CONFIG['blur_kernel_size']}, "
              f"canny=({IMAGE_PROCESSING_CONFIG['canny_low_threshold']}, {IMAGE_PROCESSING_CONFIG['canny_high_threshold']}), "
              f"min_area={IMAGE_PROCESSING_CONFIG['min_contour_area']}")
    print(f"  Tracking: boost={TRACKING_CONFIG['aoi_boost_factor']}x, "
          f"expansion={TRACKING_CONFIG['aoi_expansion_pixels']}px")
    print(f"  Video: fps={VIDEO_CONFIG['fps']}, "
          f"show_contours={VIDEO_CONFIG['show_all_contours']}, "
          f"show_ellipse={VIDEO_CONFIG['show_ellipse']}")
    
    # Load NPZ data
    available_files = get_available_npz_files()
    if npz_file_index >= len(available_files):
        print(f"Error: NPZ file index {npz_file_index} not available")
        return None
    
    npz_file = available_files[npz_file_index]
    print(f"Loading from: {npz_file}")
    
    # Load the data - FIXED: load_cone_run_npz returns a tuple, not a dict
    cones, ts, extent, meta = load_cone_run_npz(npz_file)
    total_frames = len(cones)
    
    print(f"Total frames in NPZ: {total_frames}")
    
    # Adjust frame parameters
    actual_frame_count = min(frame_count, (total_frames - frame_start) // frame_step)
    print(f"Will process {actual_frame_count} frames")
    
    if actual_frame_count <= 0:
        print("Error: Not enough frames to process")
        return None
    
    # Get first frame to determine video dimensions - need to convert to uint8
    first_frame = cones[frame_start]
    # Convert from [0,1] float to [0,255] uint8
    first_frame_uint8 = (first_frame * 255).astype(np.uint8)
    height, width = first_frame_uint8.shape
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = VIDEO_CONFIG['fps']
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return None
    
    print("Processing frames with configurable settings...")
    
    # Initialize tracking with enhanced statistics
    current_aoi = None  # Area of interest from previous frame
    tracked_frames = 0
    new_detections = 0
    lost_frames = 0
    
    # Process frames
    for i in range(actual_frame_count):
        frame_idx = frame_start + (i * frame_step)
        
        if frame_idx >= total_frames:
            break
            
        # Get frame and convert to uint8
        frame = cones[frame_idx]
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Process frame with configurable settings
        processed_frame, next_aoi = process_frame_for_video(frame_uint8, current_aoi)
        
        # Update tracking statistics
        if next_aoi is not None:
            if current_aoi is not None:
                # Check if position changed significantly (new detection vs tracked)
                if current_aoi == next_aoi:
                    lost_frames += 1  # AOI maintained, no new detection
                else:
                    tracked_frames += 1  # AOI updated with new detection
            else:
                new_detections += 1  # First detection
        else:
            lost_frames += 1  # No AOI at all
        
        # Update AOI for next frame (next_aoi might be same as current_aoi if no detection)
        current_aoi = next_aoi
        
        # Add frame number
        cv2.putText(processed_frame, f'Frame: {frame_idx}', (width - 120, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame to video
        video_writer.write(processed_frame)
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{actual_frame_count} frames")
    
    # Release video writer
    video_writer.release()
    
    print(f"\n=== VIDEO CREATION COMPLETE ===")
    print(f"Video saved to: {output_path}")
    print(f"Video specs: {width}x{height}, {fps} fps, {actual_frame_count} frames")
    print(f"Tracking stats:")
    print(f"  - Tracked frames: {tracked_frames}")
    print(f"  - New detections: {new_detections}")
    print(f"  - Lost/searching frames: {lost_frames}")
    total_detections = tracked_frames + new_detections
    if total_detections > 0:
        print(f"  - Detection success rate: {total_detections/actual_frame_count*100:.1f}%")
        print(f"  - Tracking continuity: {tracked_frames/total_detections*100:.1f}%")
    
    return output_path