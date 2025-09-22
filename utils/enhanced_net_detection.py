# Enhanced Net Detection Utilities
# =================================
# Sophisticated net detection with tracking, ROI, and parallel line analysis

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path
import math
import utils.image_analysis_utils as iau


class NetTracker:
    """Global net tracking state manager"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking state for new video/analysis"""
        self.last_position = None
        self.last_orientation = None
        self.confidence_history = []
        self.search_radius = 50  # Much smaller search radius
        self.position_history = []  # Track position over time
        self.stable_frames = 0  # Count stable detections
        self.max_jump_distance = 30  # Maximum allowed movement per frame
        self.roi_padding = 80  # ROI size around last position


# Global tracker instance
net_tracker = NetTracker()


def detect_net_with_tracking(frame, blur_ksize=70, blur_sigma=10.0, 
                           thr_percentile=80, min_area_px=200,
                           pre_blur_ksize=15, pre_blur_sigma=3.0,
                           strong_blur_ksize=25, strong_blur_sigma=5.0,
                           verbose=False):
    """
    Enhanced net detection with tight ROI tracking and conservative position updates
    
    Parameters:
    -----------
    frame : numpy.ndarray
        Input sonar frame
    blur_ksize : int
        Main blur kernel size
    blur_sigma : float
        Main blur sigma
    thr_percentile : int
        Percentile threshold for segmentation
    min_area_px : int
        Minimum area for blob detection
    pre_blur_ksize : int
        Pre-blur kernel size
    pre_blur_sigma : float
        Pre-blur sigma
    strong_blur_ksize : int
        Strong initial blur kernel size
    strong_blur_sigma : float
        Strong initial blur sigma
    verbose : bool
        Print detection details
        
    Returns:
    --------
    dict : Detection results with tracking info
    """
    if verbose:
        print("🔍 Running enhanced net detection with tight tracking...")
    
    # Step 0: Multiple blur stages for noise reduction
    strong_blurred = iau.gaussian_blur01(frame, ksize=strong_blur_ksize, sigma=strong_blur_sigma)
    pre_blurred = iau.gaussian_blur01(strong_blurred, ksize=pre_blur_ksize, sigma=pre_blur_sigma)
    
    # Step 1: Use blob preprocessing with additional blur
    blurred = iau.gaussian_blur01(pre_blurred, ksize=blur_ksize, sigma=blur_sigma)
    
    # Step 1.5: Create ROI mask if we have previous position
    roi_mask = None
    search_area = None
    if net_tracker.last_position is not None:
        H, W = frame.shape
        last_x, last_y = net_tracker.last_position
        padding = net_tracker.roi_padding
        
        # Define ROI boundaries (with bounds checking)
        x1 = max(0, int(last_x - padding))
        y1 = max(0, int(last_y - padding))
        x2 = min(W, int(last_x + padding))
        y2 = min(H, int(last_y + padding))
        
        # Create ROI mask
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
        search_area = (x1, y1, x2, y2)
        
        # Apply ROI to blurred image
        blurred_roi = blurred.copy()
        blurred_roi[roi_mask == 0] = 0  # Zero out areas outside ROI
        blurred = blurred_roi
    
    mask = iau.segment_percentile(blurred, p=thr_percentile, 
                                 min_area_px=min_area_px, open_close=1)
    
    # Step 2: Convert to uint8 for OpenCV processing
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Step 3: Enhanced edge detection for all orientations
    edges = cv2.Canny(mask_uint8, 30, 100, apertureSize=3)
    
    # Step 4: Multi-angle Hough line detection (only in ROI if available)
    lines_all = []
    
    # Detect lines at multiple resolutions and angles
    for rho_res in [1, 2]:
        for theta_res in [np.pi/180, np.pi/360]:
            lines = cv2.HoughLinesP(edges, rho=rho_res, theta=theta_res,
                                   threshold=25,  # Even lower threshold for ROI
                                   minLineLength=40,  # Shorter minimum length
                                   maxLineGap=25)    # Larger gap tolerance
            if lines is not None:
                lines_all.extend(lines)
    
    # Step 5: Filter lines to ROI if tracking
    if search_area is not None and lines_all:
        x1, y1, x2, y2 = search_area
        filtered_lines = []
        for line in lines_all:
            lx1, ly1, lx2, ly2 = line[0]
            # Check if line midpoint is in ROI
            mid_x, mid_y = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            if x1 <= mid_x <= x2 and y1 <= mid_y <= y2:
                filtered_lines.append(line)
        lines_all = filtered_lines
    
    # Step 6: Analyze lines and find parallel pairs
    line_detections = []
    parallel_pairs = []
    
    if lines_all:
        for line in lines_all:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            # Normalize angle to 0-180 range
            if angle < 0:
                angle += 180
            
            line_detections.append({
                'start': (x1, y1),
                'end': (x2, y2),
                'length': length,
                'angle': angle,
                'midpoint': ((x1+x2)/2, (y1+y2)/2)
            })
        
        # Find parallel line pairs (potential net edges)
        for i, line1 in enumerate(line_detections):
            for j, line2 in enumerate(line_detections[i+1:], i+1):
                angle_diff = abs(line1['angle'] - line2['angle'])
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                # Check if lines are parallel (within 8 degrees - tighter)
                if angle_diff < 8 and line1['length'] > 60 and line2['length'] > 60:
                    # Calculate distance between parallel lines
                    mid1 = line1['midpoint']
                    mid2 = line2['midpoint']
                    distance = np.sqrt((mid1[0]-mid2[0])**2 + (mid1[1]-mid2[1])**2)
                    
                    # Check if distance suggests a net width (15-150 pixels - tighter range)
                    if 15 < distance < 150:
                        center = ((mid1[0]+mid2[0])/2, (mid1[1]+mid2[1])/2)
                        
                        # If tracking, heavily penalize candidates far from last position
                        confidence = (line1['length'] + line2['length']) / distance
                        
                        if net_tracker.last_position is not None:
                            last_pos = net_tracker.last_position
                            distance_from_last = np.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                            
                            # Very strong penalty for distance from last position
                            if distance_from_last > net_tracker.max_jump_distance:
                                confidence *= 0.1  # Heavy penalty
                            else:
                                confidence *= 2.0  # Reward for staying close
                        
                        parallel_pairs.append({
                            'line1': line1,
                            'line2': line2,
                            'distance': distance,
                            'avg_angle': (line1['angle'] + line2['angle']) / 2,
                            'avg_length': (line1['length'] + line2['length']) / 2,
                            'center': center,
                            'confidence': confidence
                        })
    
    # Step 7: Select best net candidate using conservative tracking
    best_net = None
    
    if parallel_pairs:
        # Sort by confidence (now heavily weighted by tracking)
        parallel_pairs.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Select best candidate
        candidate = parallel_pairs[0]
        
        # Conservative position update
        if net_tracker.last_position is not None:
            last_pos = net_tracker.last_position
            new_pos = candidate['center']
            distance_moved = np.sqrt((new_pos[0]-last_pos[0])**2 + (new_pos[1]-last_pos[1])**2)
            
            # Only accept if movement is reasonable
            if distance_moved <= net_tracker.max_jump_distance:
                best_net = candidate
                
                # Smooth position update (blend old and new position)
                alpha = 0.7  # How much to trust new position (0.7 = 70% new, 30% old)
                smooth_x = alpha * new_pos[0] + (1-alpha) * last_pos[0]
                smooth_y = alpha * new_pos[1] + (1-alpha) * last_pos[1]
                
                # Update tracking with smoothed position
                net_tracker.last_position = (smooth_x, smooth_y)
                net_tracker.last_orientation = candidate['avg_angle']
                net_tracker.stable_frames += 1
            else:
                # Keep previous position if movement too large
                if verbose:
                    print(f"   ⚠️ Large movement detected ({distance_moved:.1f}px), keeping previous position")
        else:
            # First detection
            best_net = candidate
            net_tracker.last_position = candidate['center']
            net_tracker.last_orientation = candidate['avg_angle']
            net_tracker.stable_frames = 1
        
        # Update confidence history
        if best_net:
            net_tracker.confidence_history.append(best_net['confidence'])
            net_tracker.position_history.append(net_tracker.last_position)
            
            # Keep only last 10 values
            if len(net_tracker.confidence_history) > 10:
                net_tracker.confidence_history = net_tracker.confidence_history[-10:]
            if len(net_tracker.position_history) > 10:
                net_tracker.position_history = net_tracker.position_history[-10:]
    
    # Step 8: Find contours that match the detected net (using actual net position)
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    net_contour = None
    if net_tracker.last_position is not None and contours:  # Use tracked position, not just current detection
        net_center = net_tracker.last_position
        
        # Find contour closest to tracked net center
        min_distance = float('inf')
        for contour in contours:
            if cv2.contourArea(contour) > 300:  # Lower minimum area threshold
                # Get contour center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    distance = np.sqrt((cx - net_center[0])**2 + (cy - net_center[1])**2)
                    if distance < min_distance and distance < 60:  # Tighter range
                        min_distance = distance
                        net_contour = contour
    
    return {
        'net_detected': best_net is not None or net_tracker.last_position is not None,
        'net_info': best_net,
        'parallel_pairs': parallel_pairs,
        'all_lines': line_detections,
        'net_contour': net_contour,
        'mask': mask,
        'edges': edges,
        'blurred': blurred,
        'roi_info': search_area,
        'tracking_info': {
            'position': net_tracker.last_position,
            'orientation': net_tracker.last_orientation,
            'confidence_avg': np.mean(net_tracker.confidence_history) if net_tracker.confidence_history else 0,
            'stable_frames': net_tracker.stable_frames,
            'search_radius': net_tracker.search_radius
        }
    }


def create_enhanced_detection_video(npz_path, output_filename='enhanced_net_detection.mp4',
                                  fps=15, max_frames=200, detection_params=None):
    """
    Create video with enhanced net detection including tracking visualization
    
    Parameters:
    -----------
    npz_path : str
        Path to NPZ file
    output_filename : str
        Output video filename
    fps : int
        Video frames per second
    max_frames : int
        Maximum number of frames to process
    detection_params : dict
        Detection parameters (blur_ksize, thr_percentile, etc.)
        
    Returns:
    --------
    str : Path to created video
    """
    # Default detection parameters
    if detection_params is None:
        detection_params = {
            'blur_ksize': 70,
            'blur_sigma': 10.0,
            'thr_percentile': 80,
            'min_area_px': 200,
            'pre_blur_ksize': 15,
            'pre_blur_sigma': 3.0,
            'strong_blur_ksize': 25,
            'strong_blur_sigma': 5.0
        }
    
    # Ensure output goes to exports/outputs/ folder
    output_path = Path("exports/outputs") / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Creating enhanced net detection video: {output_path}")
    
    # Reset tracking for new video
    net_tracker.reset()
    
    # Load data using utils
    cones, ts, extent, meta = iau.load_cone_run_npz(npz_path)
    T, H, W = cones.shape
    
    if max_frames is not None:
        T = min(T, max_frames)
        cones = cones[:T]
    
    print(f"   Processing {T} frames at {fps} FPS")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W*2, H))
    
    # Use same colormap as blob analysis
    cmap = plt.cm.viridis
    
    for t in range(T):
        if t % 50 == 0:
            print(f"   Processing frame {t}/{T}...")
        
        frame = cones[t]
        
        # Create base visualization
        frame_normalized = np.nan_to_num(frame, nan=0.0)
        rgb_left = (cmap(frame_normalized)[:, :, :3] * 255).astype(np.uint8)
        rgb_right = rgb_left.copy()
        
        # Run enhanced detection
        result = detect_net_with_tracking(frame, **detection_params)
        
        # Draw all detected lines (faded)
        for line in result['all_lines']:
            start = tuple(map(int, line['start']))
            end = tuple(map(int, line['end']))
            cv2.line(rgb_right, start, end, (100, 100, 100), 1)  # Gray lines
        
        # Draw ROI rectangle if tracking
        if result['roi_info'] is not None:
            x1, y1, x2, y2 = result['roi_info']
            cv2.rectangle(rgb_right, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green ROI box
        
        # Draw parallel pairs (yellow)
        for pair in result['parallel_pairs']:
            line1 = pair['line1']
            line2 = pair['line2']
            
            start1 = tuple(map(int, line1['start']))
            end1 = tuple(map(int, line1['end']))
            start2 = tuple(map(int, line2['start']))
            end2 = tuple(map(int, line2['end']))
            
            cv2.line(rgb_right, start1, end1, (0, 255, 255), 2)  # Yellow
            cv2.line(rgb_right, start2, end2, (0, 255, 255), 2)  # Yellow
        
        # Draw detected net (bright red) - use tracked position
        if result['tracking_info']['position'] is not None:
            tracked_pos = result['tracking_info']['position']
            
            # Draw small red dot at tracked position (much more stable)
            center = tuple(map(int, tracked_pos))
            cv2.circle(rgb_right, center, 4, (0, 0, 255), -1)  # Smaller red dot
            
            # Draw small search radius (much smaller green circle)
            search_radius = result['tracking_info']['search_radius']
            cv2.circle(rgb_right, center, search_radius, (0, 255, 0), 1)  # Thin green circle
            
            # If we have actual line detection, draw those too
            if result['net_detected'] and result['net_info'] is not None:
                net = result['net_info']
                line1 = net['line1']
                line2 = net['line2']
                
                start1 = tuple(map(int, line1['start']))
                end1 = tuple(map(int, line1['end']))
                start2 = tuple(map(int, line2['start']))
                end2 = tuple(map(int, line2['end']))
                
                cv2.line(rgb_right, start1, end1, (0, 0, 255), 3)  # Red lines
                cv2.line(rgb_right, start2, end2, (0, 0, 255), 3)  # Red lines
            
            # Draw net contour if found (this should be stable)
            if result['net_contour'] is not None:
                cv2.drawContours(rgb_right, [result['net_contour']], -1, (255, 0, 255), 2)  # Magenta
        
        # Combine frames
        combined = np.hstack([rgb_left, rgb_right])
        
        # Add frame info
        cv2.putText(combined, f"Frame {t+1}/{T}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        net_status = "NET TRACKED" if result['tracking_info']['position'] is not None else "NO NET"
        color = (0, 255, 0) if result['tracking_info']['position'] is not None else (0, 0, 255)
        cv2.putText(combined, net_status, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show tracking info
        if result['tracking_info']['position'] is not None:
            stable_frames = result['tracking_info']['stable_frames']
            cv2.putText(combined, f"Stable: {stable_frames} frames", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if result['net_detected'] and result['net_info'] is not None:
                confidence = result['net_info']['confidence']
                cv2.putText(combined, f"Confidence: {confidence:.1f}", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add labels
        cv2.putText(combined, "ORIGINAL", (W//2-40, H-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "ENHANCED NET DETECTION", (W+W//2-120, H-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        video_writer.write(combined)
    
    video_writer.release()
    print(f"✅ Enhanced video saved: {output_path}")
    return str(output_path)


def select_npz_file(npz_file=None):
    """
    Select NPZ file for detection analysis
    
    Parameters:
    -----------
    npz_file : str or None
        - If None: auto-discover and use first available NPZ file
        - If str: specific NPZ filename (e.g. "2024-08-20_13-39-34_cones.npz")
        - If str with path: full path to NPZ file
        
    Returns:
    --------
    str : Path to selected NPZ file
    """
    npz_dir = Path("exports/outputs")
    available_npz = list(npz_dir.glob("*_cones.npz")) if npz_dir.exists() else []
    
    if not available_npz:
        print(f"❌ No NPZ files found in exports/outputs/")
        print(f"   Please run the 01_Data_Export notebook first to create NPZ files")
        raise FileNotFoundError("No cone NPZ files available. Run 01_Data_Export notebook first.")
    
    print(f"🔍 Found {len(available_npz)} NPZ files:")
    for i, npz_file_path in enumerate(available_npz):
        print(f"   {i+1}. {npz_file_path.name}")
    
    if npz_file is None:
        # Auto-select first file
        selected_npz = str(available_npz[0])
        print(f"\n✅ Auto-selected: {Path(selected_npz).name}")
    else:
        # User specified a file
        if "/" in npz_file or "\\" in npz_file:
            # Full path provided
            selected_npz = npz_file
            if not Path(selected_npz).exists():
                raise FileNotFoundError(f"NPZ file not found: {selected_npz}")
        else:
            # Just filename provided, look in exports/outputs
            candidate_path = npz_dir / npz_file
            if candidate_path.exists():
                selected_npz = str(candidate_path)
            else:
                # Try to find matching filename
                matches = [f for f in available_npz if npz_file in f.name]
                if matches:
                    selected_npz = str(matches[0])
                    print(f"🔍 Found matching file: {Path(selected_npz).name}")
                else:
                    print(f"❌ Could not find NPZ file matching: {npz_file}")
                    print(f"   Available files:")
                    for f in available_npz:
                        print(f"     • {f.name}")
                    raise FileNotFoundError(f"NPZ file not found: {npz_file}")
        
        print(f"\n✅ Selected: {Path(selected_npz).name}")
    
    return selected_npz


def run_comparison_blob_analysis(npz_path, blob_params=None):
    """
    Run the standard blob analysis for comparison
    
    Parameters:
    -----------
    npz_path : str
        Path to NPZ file
    blob_params : dict
        Blob analysis parameters
        
    Returns:
    --------
    pandas.DataFrame : Blob detection results
    """
    if blob_params is None:
        blob_params = {
            'blur_ksize': 70,
            'blur_sigma': 10.0,
            'thr_percentile': 80,
            'min_area_px': 80,
            'open_close': 1,
            'save_csv': "exports/outputs/largest_blob.csv",
            'save_mp4': "exports/outputs/largest_blob.mp4",
            'fps': 15,
            'progress': True
        }
    
    print(f"\n📊 Running largest blob analysis for comparison...")
    try:
        df = iau.analyze_run_largest_blob(npz_path, **blob_params)
        
        print(f"✅ Largest blob analysis complete! Found {len(df)} detections")
        print(f"📄 Results saved to: {blob_params['save_csv']}")
        print(f"🎬 Video saved to: {blob_params['save_mp4']}")
        
        # Show summary
        if len(df) > 0:
            print(f"\n📈 Detection Summary:")
            print(f"   Frames with detections: {len(df)}")
            print(f"   Average area: {df['area_px'].mean():.0f} pixels")
            print(f"   Largest detection: {df['area_px'].max():.0f} pixels")
        
        return df
        
    except Exception as e:
        print(f"⚠️  Largest blob analysis failed: {e}")
        print(f"   Continuing with enhanced net detection...")
        return None


def test_enhanced_detection(cones, detection_params=None, test_frames=2):
    """
    Test enhanced detection on sample frames
    
    Parameters:
    -----------
    cones : numpy.ndarray
        Loaded cone data
    detection_params : dict
        Detection parameters
    test_frames : int
        Number of frames to test
        
    Returns:
    --------
    list : Detection results for each test frame
    """
    if detection_params is None:
        detection_params = {
            'blur_ksize': 70,
            'blur_sigma': 10.0,
            'thr_percentile': 80,
            'min_area_px': 200,
            'pre_blur_ksize': 15,
            'pre_blur_sigma': 3.0,
            'strong_blur_ksize': 25,
            'strong_blur_sigma': 5.0,
            'verbose': True
        }
    
    print("🧪 Testing enhanced net detection with tracking...")
    print("Using stronger blurring and parallel line analysis")
    
    results = []
    
    for frame_idx in range(min(test_frames, len(cones))):
        test_frame = cones[frame_idx]
        print(f"\n📍 Testing on frame {frame_idx} (shape: {test_frame.shape})...")
        
        # Test enhanced detection
        print(f"\n🔍 Enhanced Net Detection:")
        result = detect_net_with_tracking(test_frame, **detection_params)
        
        print(f"   Total lines detected: {len(result['all_lines'])}")
        print(f"   Parallel pairs found: {len(result['parallel_pairs'])}")
        print(f"   Net detected: {'✅ YES' if result['net_detected'] else '❌ NO'}")
        
        if result['net_detected'] and result['net_info'] is not None:
            net = result['net_info']
            print(f"\n🎯 NET DETAILS:")
            print(f"   Position: ({net['center'][0]:.1f}, {net['center'][1]:.1f})")
            print(f"   Orientation: {net['avg_angle']:.1f}°")
            print(f"   Width: {net['distance']:.1f} pixels")
            print(f"   Confidence: {net['confidence']:.2f}")
            print(f"   Line 1 length: {net['line1']['length']:.1f} pixels")
            print(f"   Line 2 length: {net['line2']['length']:.1f} pixels")
            
            if result['net_contour'] is not None:
                contour_area = cv2.contourArea(result['net_contour'])
                print(f"   Contour area: {contour_area:.0f} pixels")
        
        if result['parallel_pairs']:
            print(f"\n📊 ALL PARALLEL PAIRS:")
            for i, pair in enumerate(result['parallel_pairs'][:3]):  # Show top 3
                print(f"   Pair {i+1}: width={pair['distance']:.1f}px, "
                      f"angle={pair['avg_angle']:.1f}°, confidence={pair['confidence']:.2f}")
        
        results.append(result)
        
        # Test tracking between frames
        if frame_idx > 0 and len(results) > 1:
            prev_result = results[frame_idx-1]
            if (result['net_detected'] and result['net_info'] is not None and 
                prev_result['net_detected'] and prev_result['net_info'] is not None):
                pos1 = prev_result['net_info']['center']
                pos2 = result['net_info']['center']
                movement = np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
                print(f"   Movement from frame {frame_idx-1}: {movement:.1f} pixels")
                print(f"   Tracking confidence: {result['tracking_info']['confidence_avg']:.2f}")
    
    print(f"\n🎯 Enhanced detection ready!")
    print(f"🔧 Features tested:")
    print(f"   • Triple-stage blurring ✅")
    print(f"   • Multi-angle line detection ✅")
    print(f"   • Parallel line analysis ✅")
    print(f"   • Frame-to-frame tracking ✅")
    print(f"   • Single net selection ✅")
    
    return results
