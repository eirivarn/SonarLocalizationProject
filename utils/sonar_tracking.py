# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
Clean slate: Simple net tracker with explicit smoothing.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

class NetTracker:
    """
    Simple net tracker. One class, clear logic.
    
    Smoothing formula (same for all parameters):
        new = old * (1 - alpha) + measured * alpha
        
    Alpha interpretation:
        alpha = 0.0 → 100% old (infinite smoothing)
        alpha = 0.5 → 50/50 blend
        alpha = 1.0 → 100% new (no smoothing)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Current ellipse state 
        self.center: Optional[Tuple[float, float]] = None
        self.size: Optional[Tuple[float, float]] = None  # (width, height)
        self.angle: Optional[float] = None  # degrees
        
        # Tracking
        self.last_distance: Optional[float] = None
        self.frames_lost: int = 0
        
    def find_and_update(self, edges: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Find net in edges, update tracking state.
        Returns best contour or None.
        """
        H, W = image_shape
        
        # Get search mask
        search_mask = self._get_search_mask((H, W))
        
        # Apply mask to edges
        if search_mask is not None:
            masked_edges = cv2.bitwise_and(edges, search_mask)
        else:
            masked_edges = edges
        
        # Find contours
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best
        best = self._find_best_contour(contours)
        
        # Update state
        if best is not None and len(best) >= 5:
            self._update_from_contour(best)
            self.frames_lost = 0
        else:
            self.frames_lost += 1
            if self.frames_lost > self.config.get('max_frames_without_detection', 30):
                self._reset()
        
        return best
    
    def calculate_distance(self, image_width: int, image_height: int) -> Tuple[Optional[float], Optional[float]]:
        """Calculate distance and angle from contour."""
        if self.center is None or self.angle is None:
            return self.last_distance, None
        
        try:
            cx, cy = self.center
            w, h = self.size if self.size else (1, 1)
            
            # Return the raw angle directly - this is the major axis angle
            major_axis_angle = self.angle
            
            # CRITICAL FIX: The red line is perpendicular to major axis
            # So the red line angle is major_axis_angle + 90°
            red_line_angle = (major_axis_angle + 90.0) % 360.0
            
            # Calculate intersection with center line using RED LINE angle (not major axis)
            center_x = image_width / 2
            ang_r = np.radians(red_line_angle)
            cos_ang = np.cos(ang_r)
            
            if abs(cos_ang) > 1e-6:
                t = (center_x - cx) / cos_ang
                intersect_y = cy + t * np.sin(ang_r)
                distance = intersect_y
            else:
                distance = cy
            
            distance = np.clip(distance, 0, image_height - 1)
            
            # Smooth distance
            if self.last_distance is not None:
                max_change = self.config.get('max_distance_change_pixels', 20)
                change = abs(distance - self.last_distance)
                if change > max_change:
                    direction = 1 if distance > self.last_distance else -1
                    distance = self.last_distance + (direction * max_change)
            
            self.last_distance = distance
            
            # Return the red line angle (already perpendicular)
            return float(distance), float(red_line_angle)
        except:
            return self.last_distance, None
    
    def _get_search_mask(self, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get search mask (ellipse + corridor)."""
        if self.center is None or self.size is None or self.angle is None:
            return None
        
        H, W = image_shape
        
        # Get expansion
        expansion = self.config.get('ellipse_expansion_factor', 0.5)
        
        # Expand if losing track
        if self.frames_lost > 0:
            decay = self.config.get('aoi_decay_factor', 0.98)
            growth = 1.0 + (1.0 - decay) * self.frames_lost
            expansion *= growth
        
        # Create ellipse mask
        w, h = self.size
        expanded_size = (w * (1 + expansion), h * (1 + expansion))
        
        mask = np.zeros((H, W), dtype=np.uint8)
        
        try:
            ellipse = (
                (int(self.center[0]), int(self.center[1])),
                (int(expanded_size[0]), int(expanded_size[1])),
                self.angle
            )
            cv2.ellipse(mask, ellipse, 255, -1)
        except:
            return None
        
        # Add corridor
        if self.config.get('use_corridor_splitting', True):
            try:
                corridor = self._make_corridor_mask((H, W))
                if corridor is not None:
                    mask = cv2.bitwise_or(mask, corridor)
            except:
                pass
        
        return mask
    
    def _make_corridor_mask(self, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Make corridor mask along major axis."""
        if self.center is None or self.size is None or self.angle is None:
            return None
        
        H, W = image_shape
        cx, cy = self.center
        w, h = self.size
        
        # Corridor parameters
        band_k = self.config.get('corridor_band_k', 2.0)
        length_factor = self.config.get('corridor_length_factor', 2.0)
        
        # Dimensions
        half_width = band_k * min(w, h) / 2.0
        half_length = length_factor * max(w, h) / 2.0
        
        # Rectangle in local coordinates
        local_pts = np.array([
            [-half_length, -half_width],
            [+half_length, -half_width],
            [+half_length, +half_width],
            [-half_length, +half_width],
        ], dtype=np.float32)
        
        # Rotate by major axis angle
        major_angle = self.angle
        if h > w:
            major_angle = (self.angle + 90.0) % 360.0
        
        ang_r = np.radians(major_angle)
        cos_a, sin_a = np.cos(ang_r), np.sin(ang_r)
        
        # Rotation matrix
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = local_pts @ R.T
        
        # Translate to center
        world_pts = rotated + np.array([cx, cy])
        
        # Draw
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [world_pts.astype(np.int32)], 255)
        
        return mask
    
    def _find_best_contour(self, contours) -> Optional[np.ndarray]:
        """Find best contour by area."""
        min_area = self.config.get('min_contour_area', 200)
        if self.center is not None:
            min_area *= 0.3  # Lower threshold when tracking
        
        best = None
        best_score = 0.0
        
        for c in contours:
            if c is None or len(c) < 5:
                continue
            
            try:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                
                score = area
                
                # Proximity bonus
                if self.center is not None:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        dist = np.sqrt((cx - self.center[0])**2 + (cy - self.center[1])**2)
                        proximity = max(0.1, 1.0 - dist / 100.0)
                        score *= proximity
                
                if score > best_score:
                    best = c
                    best_score = score
            except:
                continue
        
        return best
    
    def _update_from_contour(self, contour: np.ndarray):
        """Update ellipse state from detected contour with smoothing."""
        # Fit ellipse to contour
        try:
            (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
        except:
            return
        
        # Get alphas from config
        alpha_center = self.config.get('center_smoothing_alpha', 0.4)
        alpha_size = self.config.get('ellipse_size_smoothing_alpha', 0.01)
        alpha_angle = self.config.get('ellipse_orientation_smoothing_alpha', 0.2)
        
        # Smooth center
        if self.center is None:
            self.center = (cx, cy)
        else:
            old_cx, old_cy = self.center
            new_cx = old_cx * (1 - alpha_center) + cx * alpha_center
            new_cy = old_cy * (1 - alpha_center) + cy * alpha_center
            
            # Limit movement
            max_move = self.config.get('ellipse_max_movement_pixels', 30.0)
            dx = new_cx - old_cx
            dy = new_cy - old_cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > max_move:
                scale = max_move / dist
                new_cx = old_cx + dx * scale
                new_cy = old_cy + dy * scale
            
            self.center = (new_cx, new_cy)
        
        # Smooth size
        if self.size is None:
            self.size = (w, h)
        else:
            old_w, old_h = self.size
            new_w = old_w * (1 - alpha_size) + w * alpha_size
            new_h = old_h * (1 - alpha_size) + h * alpha_size
            self.size = (new_w, new_h)
        
        # Smooth angle (handle wrap-around)
        if self.angle is None:
            self.angle = angle
        else:
            angle_diff = angle - self.angle
            if angle_diff > 90:
                angle_diff -= 180
            elif angle_diff < -90:
                angle_diff += 180
            self.angle = self.angle + angle_diff * alpha_angle
    
    def _reset(self):
        """Reset to initial state."""
        self.center = None
        self.size = None
        self.angle = None
        self.last_distance = None
        self.frames_lost = 0
    
    def get_status(self) -> str:
        """Get tracking status."""
        if self.center is None:
            return "LOST"
        elif self.frames_lost == 0:
            return "TRACKED"
        else:
            max_frames = self.config.get('max_frames_without_detection', 30)
            return f"SEARCHING ({self.frames_lost}/{max_frames})"
