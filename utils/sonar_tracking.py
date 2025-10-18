# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
Sonar Tracking Utilities

This module contains tracking and AOI (Area of Interest) management functions:
- Elliptical AOI creation with temporal smoothing
- Center position smoothing
- Corridor-based region segmentation
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def smooth_center_position(current_center: Optional[Tuple[float, float]], 
                          new_center: Tuple[float, float], 
                          smoothing_alpha: float) -> Tuple[float, float]:
    """
    Smooth center position using exponential moving average.
    
    Args:
        current_center: Current smoothed center (None for first frame)
        new_center: New detected center
        smoothing_alpha: 0.0 = instant jump, 1.0 = maximum smoothing
        
    Returns:
        New smoothed center position
    """
    if current_center is None:
        return new_center
    
    curr_x, curr_y = current_center
    new_x, new_y = new_center
    
    smoothed_x = curr_x + (1 - smoothing_alpha) * (new_x - curr_x)
    smoothed_y = curr_y + (1 - smoothing_alpha) * (new_y - curr_y)
    
    return (smoothed_x, smoothed_y)

def create_smooth_elliptical_aoi(
    contour: np.ndarray, 
    expansion_factor: float, 
    image_shape: Tuple[int, int], 
    previous_ellipse: Optional[Tuple] = None, 
    position_smoothing_alpha: float = 0.3,
    size_smoothing_alpha: float = 0.1, 
    orientation_smoothing_alpha: float = 0.1, 
    max_movement_pixels: float = 4.0
) -> Tuple[np.ndarray, Tuple[float, float], Tuple]:
    """
    Create elliptical AOI with temporal smoothing.
    
    Args:
        contour: Input contour
        expansion_factor: Ellipse expansion (e.g., 0.3 = 30% larger)
        image_shape: (height, width)
        previous_ellipse: Previous ellipse parameters or None
        position_smoothing_alpha: Position smoothing factor
        size_smoothing_alpha: Size smoothing factor
        orientation_smoothing_alpha: Orientation smoothing factor
        max_movement_pixels: Maximum center movement per frame
        
    Returns:
        Tuple of (ellipse_mask, center, ellipse_params)
    """
    H, W = image_shape
    
    # Fit ellipse to current contour
    if len(contour) >= 5:
        current_ellipse = cv2.fitEllipse(contour)
        (curr_center_x, curr_center_y), (curr_width, curr_height), curr_angle = current_ellipse
    else:
        # Fallback to circular AOI
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            curr_center_x = int(moments['m10'] / moments['m00'])
            curr_center_y = int(moments['m01'] / moments['m00'])
        else:
            curr_center_x, curr_center_y = W//2, H//2
        
        area = cv2.contourArea(contour)
        radius = max(10, np.sqrt(area) * 2)
        curr_width = curr_height = radius * 2
        curr_angle = 0
    
    # Apply temporal smoothing if previous ellipse exists
    if previous_ellipse is not None:
        (prev_center, prev_axes, prev_angle) = previous_ellipse
        prev_center_x, prev_center_y = prev_center
        prev_width, prev_height = prev_axes
        
        # Smooth center with movement limit
        center_dx = curr_center_x - prev_center_x
        center_dy = curr_center_y - prev_center_y
        center_distance = np.sqrt(center_dx**2 + center_dy**2)
        
        if center_distance > max_movement_pixels:
            scale = max_movement_pixels / center_distance
            center_dx *= scale
            center_dy *= scale
        
        smoothed_center_x = prev_center_x + (1 - position_smoothing_alpha) * center_dx
        smoothed_center_y = prev_center_y + (1 - position_smoothing_alpha) * center_dy
        
        # Smooth size
        smoothed_width = prev_width + (1 - size_smoothing_alpha) * (curr_width - prev_width)
        smoothed_height = prev_height + (1 - size_smoothing_alpha) * (curr_height - prev_height)
        
        # Smooth angle (handle wraparound)
        angle_diff = curr_angle - prev_angle
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        smoothed_angle = prev_angle + (1 - orientation_smoothing_alpha) * angle_diff
        
        center_x, center_y = smoothed_center_x, smoothed_center_y
        width, height = smoothed_width, smoothed_height
        angle = smoothed_angle
    else:
        center_x, center_y = curr_center_x, curr_center_y
        width, height = curr_width, curr_height
        angle = curr_angle
    
    # Expand ellipse
    expanded_width = width * (1 + expansion_factor)
    expanded_height = height * (1 + expansion_factor)
    expanded_ellipse = ((center_x, center_y), (expanded_width, expanded_height), angle)
    
    # Create mask
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, expanded_ellipse, 255, -1)
    
    ellipse_params = ((center_x, center_y), (width, height), angle)
    
    return mask, (center_x, center_y), ellipse_params

# Corridor helper functions
def _ellipse_params_from_cv2(ellipse):
    """Normalize cv2.fitEllipse to (cx, cy, a, b, theta_rad)."""
    (cx, cy), (w, h), ang_deg = ellipse
    if h > w:
        w, h = h, w
        ang_deg = ang_deg + 90.0
    a = 0.5 * float(w)
    b = 0.5 * float(h)
    theta = np.deg2rad(ang_deg % 180.0)
    return float(cx), float(cy), a, b, theta

def _unit_axes(theta):
    """Get major and minor axis unit vectors."""
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    return u, v

def _poly_mask(shape_hw, poly_xy):
    """Create filled polygon mask."""
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    poly_i32 = np.round(poly_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly_i32], 255)
    return mask

def _oriented_rect_polygon(p0, u, v, half_width, length):
    """Create oriented rectangle polygon."""
    p1 = p0 + length * u
    n = half_width * v
    return np.stack([p0 + n, p0 - n, p1 - n, p1 + n], axis=0)

def _oriented_trapezoid_polygon(p0, u, v, half_w_near, half_w_far, length):
    """Create oriented trapezoid polygon."""
    p1 = p0 + length * u
    n0 = half_w_near * v
    n1 = half_w_far * v
    return np.stack([p0 + n0, p0 - n0, p1 - n1, p1 + n1], axis=0)

def build_aoi_corridor_mask(
    image_shape_hw,
    ellipse,
    *,
    band_k=0.55,
    length_px=None,
    length_factor=1.25,
    widen=1.0,
    both_directions=True,
    include_inside_ellipse=True
):
    """Build corridor mask around ellipse major axis."""
    if isinstance(ellipse, tuple) and len(ellipse) == 3:
        cx, cy, a, b, theta = _ellipse_params_from_cv2(ellipse)
    else:
        cx, cy, a, b, theta = ellipse
    u, v = _unit_axes(theta)

    p_plus = np.array([cx, cy], dtype=np.float32) + a * u
    p_minus = np.array([cx, cy], dtype=np.float32) - a * u

    half_w_near = band_k * max(b, 1.0)
    if length_px is None:
        length_px = float(length_factor * max(a, 1.0))
    half_w_far = float(widen * half_w_near)

    H, W = image_shape_hw
    out = np.zeros((H, W), dtype=np.uint8)

    if include_inside_ellipse:
        ell_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(
            ell_mask,
            (int(round(cx)), int(round(cy))),
            (int(round(a)), int(round(b))),
            np.rad2deg(theta),
            0, 360, 255, thickness=-1
        )
        out = cv2.bitwise_or(out, ell_mask)

    if widen > 1.0:
        poly_plus = _oriented_trapezoid_polygon(p_plus, u, v, half_w_near, half_w_far, length_px)
        mask_plus = _poly_mask((H, W), poly_plus)
        out = cv2.bitwise_or(out, mask_plus)
        if both_directions:
            poly_minus = _oriented_trapezoid_polygon(p_minus, -u, v, half_w_near, half_w_far, length_px)
            mask_minus = _poly_mask((H, W), poly_minus)
            out = cv2.bitwise_or(out, mask_minus)
    else:
        poly_plus = _oriented_rect_polygon(p_plus, u, v, half_w_near, length_px)
        mask_plus = _poly_mask((H, W), poly_plus)
        out = cv2.bitwise_or(out, mask_plus)
        if both_directions:
            poly_minus = _oriented_rect_polygon(p_minus, -u, v, half_w_near, length_px)
            mask_minus = _poly_mask((H, W), poly_minus)
            out = cv2.bitwise_or(out, mask_minus)

    return out

def split_contour_by_corridor(
    contour,
    ellipse,
    image_shape_hw,
    *,
    band_k=0.55,
    length_px=None,
    length_factor=1.25,
    widen=1.0,
    both_directions=True
):
    """Split contour into inside ellipse, corridor, and other regions."""
    H, W = image_shape_hw

    if isinstance(ellipse, tuple) and len(ellipse) == 3:
        (cx, cy), (w, h), ang = ellipse
        a = 0.5 * float(w)
        b = 0.5 * float(h)
        theta = np.deg2rad(ang % 180.0)
    else:
        cx, cy, a, b, theta = ellipse

    # Ellipse mask
    ell_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(
        ell_mask,
        (int(round(cx)), int(round(cy))),
        (int(round(a)), int(round(b))),
        np.rad2deg(theta),
        0, 360, 255, thickness=-1
    )

    # Corridor mask
    corr_mask = build_aoi_corridor_mask(
        image_shape_hw, ellipse,
        band_k=band_k,
        length_px=length_px,
        length_factor=length_factor,
        widen=widen,
        both_directions=both_directions,
        include_inside_ellipse=False
    )

    P = contour.reshape(-1, 2).astype(np.float32)
    xs = np.clip(np.round(P[:, 0]).astype(int), 0, W-1)
    ys = np.clip(np.round(P[:, 1]).astype(int), 0, H-1)

    inside_mask_pts = ell_mask[ys, xs] > 0
    corridor_mask_pts = (corr_mask[ys, xs] > 0) & (~inside_mask_pts)
    other_mask_pts = ~(inside_mask_pts | corridor_mask_pts)

    def _pts_to_contour(pts_bool):
        pts = P[pts_bool]
        if pts.size == 0:
            return np.zeros((0, 1, 2), dtype=contour.dtype)
        return pts.reshape(-1, 1, 2).astype(contour.dtype)

    inside_contour = _pts_to_contour(inside_mask_pts)
    corridor_contour = _pts_to_contour(corridor_mask_pts)
    other_contour = _pts_to_contour(other_mask_pts)

    return inside_contour, corridor_contour, other_contour, ell_mask, corr_mask
