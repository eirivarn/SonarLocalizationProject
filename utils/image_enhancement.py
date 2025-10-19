# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
Sonar Image Processing Algorithms

Advanced image processing functions for sonar data including structure tensor
analysis and adaptive enhancement.
"""

import cv2
import numpy as np
from typing import Tuple, Dict

# Global kernel cache
_KERNEL_CACHE = {}
_CACHE_MAX_SIZE = 200

def _get_cache_key(kernel_type, *params):
    """Generate cache key for kernel caching."""
    return (kernel_type,) + tuple(float(p) for p in params)

def _cache_kernel(cache_key, kernel):
    """Add kernel to cache with size management."""
    global _KERNEL_CACHE
    if len(_KERNEL_CACHE) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_KERNEL_CACHE))
        del _KERNEL_CACHE[oldest_key]
    _KERNEL_CACHE[cache_key] = kernel

def create_oriented_gradient_kernel_fast(angle_degrees, size):
    """Fast gradient kernel creation."""
    if size % 2 == 0:
        size += 1
    
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Create simple 3-point gradient along the direction
    if size >= 3:
        for i in range(size):
            offset = i - center
            if abs(offset) <= 1:
                kernel[center, i] = offset * cos_a
        
        for i in range(size):
            offset = i - center  
            if abs(offset) <= 1:
                kernel[i, center] += offset * sin_a
    
    # Ensure zero sum
    kernel = kernel - np.mean(kernel)
    return kernel

def create_circular_kernel_fast(radius):
    """Fast circular kernel with weights."""
    size = 2 * radius + 1
    center = radius
    y, x = np.ogrid[:size, :size]
    
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    kernel = (dist <= radius).astype(np.float32)
    
    return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

def create_elliptical_kernel_fast(base_radius, elongation_factor, angle_degrees):
    """Fast elliptical kernel creation."""
    elongated_radius = int(base_radius * elongation_factor)
    size = 2 * elongated_radius + 1
    center = size // 2
    
    y, x = np.ogrid[:size, :size]
    y_c, x_c = y - center, x - center
    
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    x_rot = x_c * cos_a + y_c * sin_a
    y_rot = -x_c * sin_a + y_c * cos_a
    
    ellipse_dist = (x_rot / elongated_radius) ** 2 + (y_rot / base_radius) ** 2
    kernel = (ellipse_dist <= 1.0).astype(np.float32)
    
    return kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel

def compute_structure_tensor_field_fast(grad_x: np.ndarray, grad_y: np.ndarray, 
                                       sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast vectorized structure tensor computation for entire image.
    
    Args:
        grad_x, grad_y: Gradient components
        sigma: Gaussian smoothing parameter
        
    Returns:
        orientation_map: Dominant orientation at each pixel (0-180)
        coherency_map: Coherency (linearity measure) at each pixel (0-1)
    """
    # Structure tensor components
    Jxx = grad_x * grad_x
    Jyy = grad_y * grad_y  
    Jxy = grad_x * grad_y
    
    # Apply Gaussian smoothing (vectorized)
    kernel_size = max(3, int(4 * sigma + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel_gauss = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = kernel_gauss @ kernel_gauss.T
    
    Jxx_smooth = cv2.filter2D(Jxx, -1, kernel_2d)
    Jyy_smooth = cv2.filter2D(Jyy, -1, kernel_2d)
    Jxy_smooth = cv2.filter2D(Jxy, -1, kernel_2d)
    
    # Vectorized eigenvalue analysis
    trace = Jxx_smooth + Jyy_smooth
    det = Jxx_smooth * Jyy_smooth - Jxy_smooth * Jxy_smooth
    
    # Compute orientation (vectorized)
    orientation_map = np.zeros_like(Jxx_smooth)
    coherency_map = np.zeros_like(Jxx_smooth)
    
    # Mask for valid regions (non-zero gradients)
    valid_mask = np.abs(Jxy_smooth) > 1e-6
    
    # Vectorized orientation computation
    orientation_map[valid_mask] = 0.5 * np.arctan2(2 * Jxy_smooth[valid_mask], 
                                                   Jxx_smooth[valid_mask] - Jyy_smooth[valid_mask])
    orientation_map = (orientation_map * 180 / np.pi + 180) % 180
    
    # Handle near-zero cases
    horizontal_mask = (~valid_mask) & (Jxx_smooth > Jyy_smooth)
    vertical_mask = (~valid_mask) & (Jxx_smooth <= Jyy_smooth)
    orientation_map[horizontal_mask] = 0    # Horizontal
    orientation_map[vertical_mask] = 90     # Vertical
    
    # Vectorized coherency computation (safely)
    valid_coherency_mask = (trace > 1e-6) & (det >= 0)
    coherency_map[valid_coherency_mask] = ((trace[valid_coherency_mask] - 
                                          2 * np.sqrt(det[valid_coherency_mask])) / 
                                         trace[valid_coherency_mask])
    coherency_map = np.clip(coherency_map, 0, 1)
    
    return orientation_map, coherency_map

def adaptive_linear_momentum_merge_fast(
    frame: np.ndarray,
    angle_steps: int = 36,
    base_radius: int = 3,
    max_elongation: float = 3.0,
    momentum_boost: float = 0.8,
    linearity_threshold: float = 0.15,
    downscale_factor: int = 2,
    top_k_bins: int = 8,
    min_coverage_percent: float = 0.5,
    gaussian_sigma: float = 1.0
) -> np.ndarray:
    """
    ADVANCED OPTIMIZED version using structure tensors and sophisticated filtering:
    1. Structure tensor-based orientation detection (replaces oriented gradient bank)
    2. Top-K bin selection for processing only most significant angles
    3. Separable Gaussian blur instead of circular kernels  
    4. ROI-based convolution processing
    5. Aggressive bin filtering by coverage and linearity
    6. Quantized angle management throughout pipeline
    
    Args:
        frame: Input grayscale image (0-255)
        angle_steps: Number of angle bins for quantization
        base_radius: Base kernel radius for enhancement
        max_elongation: Maximum elongation factor for elliptical kernels
        momentum_boost: Enhancement strength multiplier
        linearity_threshold: Minimum linearity for processing
        downscale_factor: Factor for downsampling during analysis
        top_k_bins: Maximum number of angle bins to process
        min_coverage_percent: Minimum pixel coverage (%) for bin processing
        gaussian_sigma: Sigma for Gaussian blur enhancement
    
    Returns:
        Enhanced grayscale image (0-255)
    """
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    result = frame.astype(np.float32)
    h, w = result.shape
    
    # ADVANCED OPTIMIZATION 1: Early exit for low contrast images
    frame_std = np.std(result)
    if frame_std < 5.0:
        # Apply separable Gaussian blur for mild enhancement
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.3 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Downsampled dimensions for structure tensor analysis
    h_small = max(h // downscale_factor, 32)
    w_small = max(w // downscale_factor, 32)
    frame_small = cv2.resize(result, (w_small, h_small), interpolation=cv2.INTER_AREA)
    
    # ADVANCED OPTIMIZATION 2: Structure tensor-based orientation detection
    # First compute gradients for structure tensor
    grad_x = cv2.Sobel(frame_small, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame_small, cv2.CV_64F, 0, 1, ksize=3)
    
    # Use existing structure tensor function for superior orientation and linearity detection
    orientations, linearity_map_small = compute_structure_tensor_field_fast(
        grad_x, grad_y, 
        sigma=1.5  # Sigma for Gaussian smoothing in structure tensor
    )
    
    # ADVANCED OPTIMIZATION 3: Quantize orientations to integer angle bins
    # Convert orientations (-π/2 to π/2) to angle bins (0 to angle_steps-1)
    orientations_normalized = (orientations + np.pi/2) / np.pi  # Normalize to [0, 1]
    direction_bin_map_small = np.round(orientations_normalized * (angle_steps - 1)).astype(np.int32)
    direction_bin_map_small = np.clip(direction_bin_map_small, 0, angle_steps - 1)
    
    # Normalize linearity map
    max_linearity = np.max(linearity_map_small)
    if max_linearity > 0:
        linearity_map_small = linearity_map_small / max_linearity
    else:
        # No linearity detected - apply Gaussian blur enhancement
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # ADVANCED OPTIMIZATION 4: Upsample maps to full resolution
    linearity_map = cv2.resize(linearity_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
    direction_bin_map = cv2.resize(direction_bin_map_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    
    # Create binary mask for linear regions
    linear_mask = linearity_map > linearity_threshold
    
    if np.sum(linear_mask) == 0:
        # No linear patterns - apply Gaussian blur enhancement
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # ADVANCED OPTIMIZATION 5: Aggressive bin filtering and Top-K selection
    unique_bins, bin_counts = np.unique(direction_bin_map[linear_mask], return_counts=True)
    
    # Calculate coverage percentages and total linearity per bin
    total_linear_pixels = np.sum(linear_mask)
    bin_coverage_percent = (bin_counts / total_linear_pixels) * 100.0
    
    # Calculate total linearity strength per bin
    bin_linearity_totals = []
    for bin_idx in unique_bins:
        bin_mask = (direction_bin_map == bin_idx) & linear_mask
        total_linearity = np.sum(linearity_map[bin_mask])
        bin_linearity_totals.append(total_linearity)
    bin_linearity_totals = np.array(bin_linearity_totals)
    
    # Filter bins by minimum coverage
    coverage_filter = bin_coverage_percent >= min_coverage_percent
    filtered_bins = unique_bins[coverage_filter]
    filtered_linearity = bin_linearity_totals[coverage_filter]
    
    if len(filtered_bins) == 0:
        # No bins meet coverage criteria - apply Gaussian blur
        enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
        final_result = result + momentum_boost * 0.5 * enhanced
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    # ADVANCED OPTIMIZATION 6: Select top-K bins by linearity strength
    if len(filtered_bins) > top_k_bins:
        top_k_indices = np.argsort(filtered_linearity)[-top_k_bins:]
        significant_bins = filtered_bins[top_k_indices]
    else:
        significant_bins = filtered_bins
    
    # Base enhancement with separable Gaussian blur (faster than circular convolution)
    enhanced = cv2.GaussianBlur(result, (2*base_radius+1, 2*base_radius+1), gaussian_sigma)
    
    # ADVANCED OPTIMIZATION 7: ROI-based convolution processing
    for angle_bin in significant_bins:
        # Convert quantized bin back to angle for kernel creation
        angle_degrees = float(angle_bin * 180.0 / angle_steps)
        
        # Create binary mask for this bin
        bin_mask = (direction_bin_map == angle_bin) & linear_mask
        
        if np.sum(bin_mask) == 0:
            continue
            
        # ADVANCED OPTIMIZATION 8: ROI bounding box calculation
        # Find bounding box of the mask to limit convolution area
        rows, cols = np.where(bin_mask)
        if len(rows) == 0:
            continue
            
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        
        # Expand ROI by kernel radius to account for convolution border effects
        kernel_margin = base_radius + 2
        roi_row_start = max(0, row_min - kernel_margin)
        roi_row_end = min(h, row_max + kernel_margin + 1)
        roi_col_start = max(0, col_min - kernel_margin)
        roi_col_end = min(w, col_max + kernel_margin + 1)
        
        # Extract ROI for processing
        roi_frame = result[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_mask = bin_mask[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_linearity = linearity_map[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        
        if roi_frame.size == 0:
            continue
        
        # Calculate average elongation for this bin within ROI
        avg_elongation = 1 + (max_elongation - 1) * np.mean(roi_linearity[roi_mask])
        
        # ADVANCED OPTIMIZATION 9: Cached elliptical kernel with ROI convolution
        cache_key = _get_cache_key('ellipse', base_radius, avg_elongation, angle_degrees)
        if cache_key not in _KERNEL_CACHE:
            _cache_kernel(cache_key, create_elliptical_kernel_fast(base_radius, avg_elongation, angle_degrees))
        ellipse_kernel = _KERNEL_CACHE[cache_key]
        
        # Apply elliptical convolution only to ROI
        roi_enhanced = cv2.filter2D(roi_frame, -1, ellipse_kernel)
        
        # ADVANCED OPTIMIZATION 10: Masked blending within ROI
        blend_weights = roi_linearity * roi_mask.astype(np.float32)
        
        # Update enhanced image only within ROI
        roi_current = enhanced[roi_row_start:roi_row_end, roi_col_start:roi_col_end]
        roi_blended = roi_current * (1.0 - blend_weights) + roi_enhanced * blend_weights
        enhanced[roi_row_start:roi_row_end, roi_col_start:roi_col_end] = roi_blended
    
    # Final enhancement combination with adaptive clipping
    final_result = result + momentum_boost * enhanced
    
    # ADVANCED OPTIMIZATION 11: Adaptive soft clipping based on image statistics
    clip_upper = 255.0 * (1.0 + momentum_boost * 0.2)
    final_result = np.clip(final_result, 0.0, clip_upper)
    
    # Smooth saturation using tanh for natural look
    final_result = 255.0 * np.tanh(final_result / 255.0)
    
    return np.clip(final_result, 0.0, 255.0).astype(np.uint8)

def get_momentum_merged_frame(frame_u8: np.ndarray, config: Dict) -> np.ndarray:
    """
    Returns the momentum-merged frame (after adaptive linear momentum merging).
    This is the frame you want to visualize after the raw frame.
    """
    binary_threshold = config.get('binary_threshold', 128)
    binary_frame = (frame_u8 > binary_threshold).astype(np.uint8) * 255

    use_advanced = config.get('use_advanced_momentum_merging', True)
    if use_advanced:
        merged = adaptive_linear_momentum_merge_fast(
            binary_frame,
            angle_steps=config.get('adaptive_angle_steps', 36),
            base_radius=config.get('adaptive_base_radius', 3),
            max_elongation=config.get('adaptive_max_elongation', 3.0),
            momentum_boost=config.get('momentum_boost', 0.8),
            linearity_threshold=config.get('adaptive_linearity_threshold', 0.15),
        )
    else:
        kernel_size = config.get('basic_gaussian_kernel_size', 5)
        gaussian_sigma = config.get('basic_gaussian_sigma', 1.0)
        momentum_boost = config.get('basic_momentum_boost', 0.5)
        enhanced = cv2.GaussianBlur(binary_frame, (kernel_size, kernel_size), gaussian_sigma)
        merged = binary_frame + momentum_boost * enhanced
        merged = np.clip(merged, 0, 255).astype(np.uint8)
    return merged

def preprocess_edges(frame_u8: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess sonar frame for edge detection with optional advanced enhancement.
    
    Args:
        frame_u8: Input sonar frame (uint8)
        config: IMAGE_PROCESSING_CONFIG dictionary
        
    Returns:
        Tuple of (raw_edges, processed_edges)
    """
    
    # STEP 1: Convert to binary frame (signal-strength independent)
    binary_threshold = config.get('binary_threshold', 128)
    binary_frame = (frame_u8 > binary_threshold).astype(np.uint8) * 255
    
    # STEP 2: Apply enhancement (advanced or basic)
    use_advanced = config.get('use_advanced_momentum_merging', True)
    
    if use_advanced:
        # Use advanced adaptive linear momentum merging with structure tensor analysis
        enhanced_binary = adaptive_linear_momentum_merge_fast(
            binary_frame,
            angle_steps=config.get('adaptive_angle_steps', 36),
            base_radius=config.get('adaptive_base_radius', 3),
            max_elongation=config.get('adaptive_max_elongation', 3.0),
            momentum_boost=config.get('momentum_boost', 0.8),
            linearity_threshold=config.get('adaptive_linearity_threshold', 0.15),
        )
    else:
        # Use basic Gaussian blur enhancement (faster)
        kernel_size = config.get('basic_gaussian_kernel_size', 5)
        gaussian_sigma = config.get('basic_gaussian_sigma', 1.0)
        momentum_boost = config.get('basic_momentum_boost', 0.5)
        
        enhanced = cv2.GaussianBlur(binary_frame, (kernel_size, kernel_size), gaussian_sigma)
        enhanced_binary = binary_frame + momentum_boost * enhanced
        enhanced_binary = np.clip(enhanced_binary, 0, 255).astype(np.uint8)
    
    # STEP 3: Extract edges from enhanced binary frame
    kernel_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    
    # Raw edges
    raw_edges = cv2.filter2D(binary_frame, cv2.CV_32F, kernel_edge)
    raw_edges = np.clip(raw_edges, 0, 255).astype(np.uint8)
    raw_edges = (raw_edges > 0).astype(np.uint8) * 255
    
    # Enhanced edges
    enhanced_edges = cv2.filter2D(enhanced_binary, cv2.CV_32F, kernel_edge)
    enhanced_edges = np.clip(enhanced_edges, 0, 255).astype(np.uint8)
    enhanced_edges = (enhanced_edges > 0).astype(np.uint8) * 255
    
    # Post-process edges with morphological operations
    mks = int(config.get('morph_close_kernel', 0))
    if mks > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mks, mks))
        enhanced_edges = cv2.morphologyEx(enhanced_edges, cv2.MORPH_CLOSE, kernel)
    
    dil = int(config.get('edge_dilation_iterations', 0))
    if dil > 0:
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced_edges = cv2.dilate(enhanced_edges, kernel2, iterations=dil)
    
    return raw_edges, enhanced_edges