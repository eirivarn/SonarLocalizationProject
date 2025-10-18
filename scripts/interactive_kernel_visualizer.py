#!/usr/bin/env python3
"""
Simplified Interactive Kernel Visualizer for SOLAQUA

A more reliable version that focuses on core functionality.
Click on the sonar image to see the kernel at that position.

Usage:
    python scripts/simple_kernel_visualizer.py [--npz_index 0] [--frame_index 0]

Controls:
    - Click: Show kernel at clicked position
    - Space: Toggle between original and enhanced frame
    - 'c': Toggle CV2 enhancement methods
    - 'q': Quit

Author: SOLAQUA Project
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Rectangle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.contour_analysis import (
    create_elliptical_kernel_fast, 
    create_circular_kernel_fast,
    create_oriented_gradient_kernel_fast,
    adaptive_linear_momentum_merge_fast,
    compute_structure_tensor_field_fast,
    get_available_npz_files,
    load_cone_run_npz,
    to_uint8_gray
)
from utils.config import IMAGE_PROCESSING_CONFIG


class SimpleKernelVisualizer:
    """Simplified kernel visualizer using click events instead of hover."""
    
    def __init__(self, frame):
        self.frame = frame
        self.current_pos = (frame.shape[1]//2, frame.shape[0]//2)
        self.show_enhanced = False
        
        # Initialize zoom parameters
        self.current_zoom_params = {
            'zoom_x1': 0, 'zoom_x2': 100,
            'zoom_y1': 0, 'zoom_y2': 100,
            'zoom_scale': 4
        }
        
        # Configuration
        self.config = IMAGE_PROCESSING_CONFIG.copy()
        
        # Pre-compute enhanced frame
        self.enhanced_frame = self._compute_enhanced_frame()
        
        print("Setting up visualization...")
        self.setup_figure()
        
    def _compute_enhanced_frame(self):
        """Compute enhanced frame using adaptive linear momentum merging."""
        # Use advanced optimized version with structure tensors
        return adaptive_linear_momentum_merge_fast(
            self.frame,
            angle_steps=self.config.get('adaptive_angle_steps', 36),
            base_radius=self.config.get('adaptive_base_radius', 3),
            max_elongation=self.config.get('adaptive_max_elongation', 3.0),
            momentum_boost=self.config.get('momentum_boost', 0.8),
            linearity_threshold=self.config.get('adaptive_linearity_threshold', 0.15),
            downscale_factor=self.config.get('downscale_factor', 2),
            top_k_bins=self.config.get('top_k_bins', 8),
            min_coverage_percent=self.config.get('min_coverage_percent', 0.5),
            gaussian_sigma=self.config.get('gaussian_sigma', 1.0)
        )
    
    def setup_figure(self):
        """Setup matplotlib figure with zoomed area view."""
        self.fig, ((self.ax_main, self.ax_kernel), 
                  (self.ax_zoom, self.ax_params), 
                  (self.ax_config, self.ax_empty)) = plt.subplots(3, 2, figsize=(14, 12))
        
        # Hide the empty subplot
        self.ax_empty.axis('off')
        
        self.fig.suptitle('Interactive Kernel Visualizer - Click to explore!', fontsize=14, fontweight='bold')
        
        # Initial display
        self.update_display()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        print("Figure setup complete")
        
    def get_kernel_at_position(self, x, y):
        """Get merging kernel information at specified position."""
        
        h, w = self.frame.shape
        
        # Ensure position is within bounds
        x = max(10, min(w-10, x))
        y = max(10, min(h-10, y))
        
        # Custom adaptive kernel analysis using optimized structure tensor
        frame_float = self.frame.astype(np.float64)
        
        # Compute gradients for entire frame
        grad_x = cv2.Sobel(frame_float, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame_float, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        base_radius = self.config.get('adaptive_base_radius', 2)
        
        # Use optimized structure tensor computation for entire image
        orientation_map, coherency_map = compute_structure_tensor_field_fast(
            grad_x, grad_y, sigma=1.0)
        
        # Extract values at the clicked position
        struct_angle = float(orientation_map[y, x])
        struct_coherency = float(coherency_map[y, x])
        
        # Extract local region around the point for additional RANSAC analysis
        window_size = base_radius * 3
        y_start = max(0, y - window_size)
        y_end = min(frame_float.shape[0], y + window_size + 1)
        x_start = max(0, x - window_size)
        x_end = min(frame_float.shape[1], x + window_size + 1)
        
        local_region = frame_float[y_start:y_end, x_start:x_end]
        local_grad_mag = grad_mag[y_start:y_end, x_start:x_end]
        
        # Method 2: Direct line fitting using RANSAC on edge points (for additional validation)
        def compute_ransac_line_orientation(region, grad_mag_local):
            """Fit line to strong edge points using RANSAC."""
            # Check if there are any gradients at all
            valid_gradients = grad_mag_local[grad_mag_local > 0]
            if len(valid_gradients) < 4:  # Need at least 4 points with gradients
                return 0, 0
            
            # Threshold for strong edges
            try:
                edge_threshold = np.percentile(valid_gradients, 75)
            except (IndexError, ValueError):
                return 0, 0
            
            edge_points = np.where(grad_mag_local > edge_threshold)
            
            if len(edge_points[0]) < 4:  # Need at least 4 points for line fitting
                return 0, 0
            
            # Convert to (x,y) coordinates
            points = np.column_stack([edge_points[1], edge_points[0]])
            
            try:
                # Fit line using cv2.fitLine (robust to outliers)
                [vx, vy, cx, cy] = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Convert to angle (line direction)
                line_angle = np.arctan2(vy, vx) * 180 / np.pi
                line_angle = (line_angle + 180) % 180  # Keep in 0-180 range
                
                # Compute line strength (how well points fit the line)
                line_strength = min(1.0, len(edge_points[0]) / (window_size * window_size * 0.1))
                
                return line_angle, line_strength
            except:
                return 0, 0
        
        # Apply RANSAC method for additional validation
        ransac_angle, ransac_strength = compute_ransac_line_orientation(
            local_region, local_grad_mag)
        
        # Combine the two methods for robustness
        if struct_coherency > 0.3 and ransac_strength > 0.2:
            # Both methods agree - use weighted average
            angle_diff = min(abs(struct_angle - ransac_angle), 
                           abs(struct_angle - ransac_angle + 180),
                           abs(struct_angle - ransac_angle - 180))
            if angle_diff < 30:  # Methods agree
                detected_line_angle = (struct_angle + ransac_angle) / 2
                line_confidence = min(1.0, struct_coherency + ransac_strength)
                detection_method = "Combined"
            else:  # Methods disagree - use more confident one
                if struct_coherency > ransac_strength:
                    detected_line_angle = struct_angle
                    line_confidence = struct_coherency
                    detection_method = "Structure Tensor"
                else:
                    detected_line_angle = ransac_angle
                    line_confidence = ransac_strength
                    detection_method = "RANSAC"
        elif struct_coherency > 0.3:
            detected_line_angle = struct_angle
            line_confidence = struct_coherency
            detection_method = "Structure Tensor"
        elif ransac_strength > 0.2:
            detected_line_angle = ransac_angle
            line_confidence = ransac_strength
            detection_method = "RANSAC"
        else:
            detected_line_angle = 0
            line_confidence = 0
            detection_method = "None"
        
        # Normalize to 0-180 range
        detected_line_angle = detected_line_angle % 180
        
        # Test linearity using the detected line angle (if confident enough)
        angles = np.linspace(0, 180, self.config.get('adaptive_angle_steps', 9), endpoint=False)
        responses = []
        kernel_size = max(3, base_radius + 1)
        
        for angle in angles:
            kernel_test = create_oriented_gradient_kernel_fast(angle, kernel_size)
            response = cv2.filter2D(grad_mag, cv2.CV_64F, kernel_test)
            responses.append(response[y, x])
        
        # Calculate linearity (variance of responses)
        linearity = np.var(responses)
        linearity_norm = linearity / 1000.0  # Rough normalization
        linearity_threshold = self.config.get('adaptive_linearity_threshold', 0.3)
        
        # Determine kernel type and create kernel
        combined_confidence = linearity_norm * 0.5 + line_confidence * 0.5
        
        if combined_confidence > linearity_threshold and line_confidence > 0.2:
            # Strong line detected - use the detected line angle
            kernel_angle = float(detected_line_angle)  # Ensure scalar
            
            # Calculate elongation based on combined confidence
            max_elongation = self.config.get('adaptive_max_elongation', 4)
            elongation = float(min(1.0 + combined_confidence * 4.0, max_elongation))
            
            # Create elliptical kernel aligned with detected line
            kernel = create_elliptical_kernel_fast(base_radius, elongation, kernel_angle)
            kernel_type = f"Elliptical ({elongation:.1f}x, {kernel_angle:.0f}°)"
            method_info = f"Line-Guided ({detection_method})"
            
        elif linearity_norm > linearity_threshold:
            # Fallback to response-based detection if line detection fails
            best_angle_idx = np.argmax(responses)
            kernel_angle = float(angles[best_angle_idx])  # Ensure scalar
            
            max_elongation = self.config.get('adaptive_max_elongation', 4)
            elongation = float(min(1.0 + linearity_norm * 3.0, max_elongation))
            
            kernel = create_elliptical_kernel_fast(base_radius, elongation, kernel_angle)
            kernel_type = f"Elliptical ({elongation:.1f}x, {kernel_angle:.0f}°)"
            method_info = "Response-Based (Fallback)"
            
        else:
            # No strong linearity - use circular kernel
            kernel = create_circular_kernel_fast(base_radius)
            kernel_type = "Circular"
            kernel_angle = 0.0
            method_info = "Isotropic"
        
        return {
            'kernel': kernel,
            'type': kernel_type,
            'linearity': float(linearity_norm),
            'angle': float(kernel_angle),
            'line_angle': float(detected_line_angle),
            'line_confidence': float(line_confidence),
            'detection_method': detection_method,
            'struct_coherency': float(struct_coherency),
            'ransac_strength': float(ransac_strength),
            'combined_confidence': float(combined_confidence),
            'method': f'Advanced-Adaptive ({method_info})'
        }
    
    def update_display(self):
        """Update all display elements."""
        
        print(f"Updating display at position ({self.current_pos[0]}, {self.current_pos[1]})")
        
        # Main image
        self.ax_main.clear()
        display_frame = self.enhanced_frame if self.show_enhanced else self.frame
        self.ax_main.imshow(display_frame, cmap='viridis', aspect='equal')
        
        # Add square outline matching kernel size at current position
        x, y = self.current_pos
        
        # Get kernel info to determine size
        kernel_info = self.get_kernel_at_position(x, y)
        kernel_shape = kernel_info['kernel'].shape
        
        # Calculate square dimensions (use the larger dimension for a square)
        kernel_size = max(kernel_shape[0], kernel_shape[1])
        half_size = kernel_size // 2
        
        # Calculate square corners
        x1 = x - half_size
        x2 = x + half_size
        y1 = y - half_size 
        y2 = y + half_size
        
        # Draw square outline with thin edge
        square = Rectangle((x1, y1), kernel_size, kernel_size, 
                         linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.8)
        self.ax_main.add_patch(square)
        
        # Add center point
        self.ax_main.plot(x, y, 'r.', markersize=4)
        
        title_suffix = " (Enhanced)" if self.show_enhanced else " (Original)"
        method_suffix = " [Adaptive]"
        self.ax_main.set_title(f'Sonar Frame{title_suffix}{method_suffix}')
        
        # Get kernel info at current position
        kernel_info = self.get_kernel_at_position(*self.current_pos)
        
        # Kernel visualization
        self.ax_kernel.clear()
        kernel_display = cv2.resize(kernel_info['kernel'], None, 
                                  fx=8, fy=8, 
                                  interpolation=cv2.INTER_NEAREST)
        
        im = self.ax_kernel.imshow(kernel_display, cmap='hot', aspect='equal')
        self.ax_kernel.set_title(f"Kernel at ({x}, {y})")
        
        # Zoomed area view
        self.ax_zoom.clear()
        
        # Extract the area within the red square with some padding
        zoom_padding = 10
        zoom_x1 = max(0, x1 - zoom_padding)
        zoom_x2 = min(self.frame.shape[1], x2 + zoom_padding)
        zoom_y1 = max(0, y1 - zoom_padding)
        zoom_y2 = min(self.frame.shape[0], y2 + zoom_padding)
        
        # Store zoom parameters for use in params display
        self.current_zoom_params = {
            'zoom_x1': zoom_x1, 'zoom_x2': zoom_x2,
            'zoom_y1': zoom_y1, 'zoom_y2': zoom_y2,
            'zoom_scale': 4
        }
        
        # Extract zoomed region from current display frame
        zoomed_region = display_frame[zoom_y1:zoom_y2, zoom_x1:zoom_x2]
        
        if zoomed_region.size > 0:
            # Scale up the zoomed region for better visibility
            zoom_scale = self.current_zoom_params['zoom_scale']
            zoomed_scaled = cv2.resize(zoomed_region, None, 
                                     fx=zoom_scale, fy=zoom_scale, 
                                     interpolation=cv2.INTER_NEAREST)
            
            self.ax_zoom.imshow(zoomed_scaled, cmap='viridis', aspect='equal')
            
            # Draw kernel area outline in zoomed view
            # Adjust coordinates for the zoomed and scaled view
            kernel_x_center = (x - zoom_x1) * zoom_scale
            kernel_y_center = (y - zoom_y1) * zoom_scale
            kernel_size_scaled = kernel_size * zoom_scale
            
            # Draw the kernel area
            kernel_rect = Rectangle((kernel_x_center - kernel_size_scaled//2, 
                                   kernel_y_center - kernel_size_scaled//2), 
                                   kernel_size_scaled, kernel_size_scaled,
                                   linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            self.ax_zoom.add_patch(kernel_rect)
            
            # Add center point
            self.ax_zoom.plot(kernel_x_center, kernel_y_center, 'r.', markersize=8)
            
            self.ax_zoom.set_title(f"Zoomed Area ({zoom_scale}x) - Kernel Region")
        else:
            self.ax_zoom.text(0.5, 0.5, 'Invalid zoom region', 
                            transform=self.ax_zoom.transAxes, 
                            ha='center', va='center', fontsize=12)
            self.ax_zoom.set_title("Zoom View")
        
        # Kernel parameters
        self.ax_params.clear()
        self.ax_params.axis('off')
        
        # Format detailed line detection info if available
        line_info = ""
        if 'line_angle' in kernel_info:
            line_info = f"""
Line Detection:
  Detected Angle: {kernel_info['line_angle']:.1f}°
  Detection Method: {kernel_info.get('detection_method', 'Unknown')}
  Line Confidence: {kernel_info['line_confidence']:.3f}
  Struct Coherency: {kernel_info.get('struct_coherency', 0):.3f}
  RANSAC Strength: {kernel_info.get('ransac_strength', 0):.3f}
  Combined Conf.: {kernel_info.get('combined_confidence', 0):.3f}"""

        params_text = f"""Kernel Info:
Type: {kernel_info['type']}
Method: {kernel_info['method']}
Linearity: {kernel_info['linearity']:.3f}
Kernel Angle: {kernel_info['angle']:.1f}°{line_info}
Size: {kernel_info['kernel'].shape}

Position: ({x}, {y})
Zoom Region: ({self.current_zoom_params['zoom_x1']},{self.current_zoom_params['zoom_y1']}) to ({self.current_zoom_params['zoom_x2']},{self.current_zoom_params['zoom_y2']})
Zoom Scale: {self.current_zoom_params['zoom_scale']}x"""
        
        self.ax_params.text(0.05, 0.95, params_text, transform=self.ax_params.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Configuration panel
        self.ax_config.clear()
        self.ax_config.axis('off')
        
        config_text = f"""Adaptive Enhancement
Base Radius: {self.config.get('adaptive_base_radius', 2)}
Max Elongation: {self.config.get('adaptive_max_elongation', 4)}
Linearity Threshold: {self.config.get('adaptive_linearity_threshold', 0.3)}

Views:
- Main: Full sonar image
- Kernel: Applied filter
- Zoom: 4x magnified area

Controls:
[Click] Select position
[Space] Toggle enhanced
[q] Quit"""
        
        self.ax_config.text(0.05, 0.95, config_text, transform=self.ax_config.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes == self.ax_main and event.xdata is not None and event.ydata is not None:
            self.current_pos = (int(event.xdata), int(event.ydata))
            print(f"Clicked at ({self.current_pos[0]}, {self.current_pos[1]})")
            self.update_display()
    
    def on_key_press(self, event):
        """Handle key press events."""
        print(f"Key pressed: {event.key}")
        
        if event.key == ' ':  # Space - toggle enhanced/original
            self.show_enhanced = not self.show_enhanced
            self.update_display()
            
        elif event.key == 'q':  # Quit
            print("Quitting...")
            plt.close('all')


def load_sonar_frame(npz_index=0, frame_index=0):
    """Load a sonar frame from NPZ files."""
    
    try:
        files = get_available_npz_files()
        if npz_index >= len(files):
            print(f"Error: NPZ index {npz_index} not available. Found {len(files)} files.")
            return None
            
        print(f"Loading NPZ file {npz_index}: {files[npz_index].name}")
        
        cones, timestamps, extent, metadata = load_cone_run_npz(files[npz_index])
        
        if frame_index >= len(cones):
            print(f"Error: Frame index {frame_index} not available. Found {len(cones)} frames.")
            return None
            
        frame = to_uint8_gray(cones[frame_index])
        print(f"Loaded frame {frame_index}: {frame.shape} pixels")
        
        return frame
        
    except Exception as e:
        print(f"Error loading sonar frame: {e}")
        return None


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Simple Interactive Kernel Visualizer for SOLAQUA')
    parser.add_argument('--npz_index', type=int, default=0, 
                      help='NPZ file index to load (default: 0)')
    parser.add_argument('--frame_index', type=int, default=0,
                      help='Frame index within NPZ file (default: 0)')
    
    args = parser.parse_args()
    
    print("SOLAQUA Simple Kernel Visualizer")
    print("=" * 50)
    
    # Load sonar frame
    frame = load_sonar_frame(args.npz_index, args.frame_index)
    if frame is None:
        return 1
    
    print("Starting simple visualizer...")
    print("   Click on the sonar image to explore kernels!")
    
    # Create and show visualizer
    try:
        visualizer = SimpleKernelVisualizer(frame)
        print("Window should be visible. Click anywhere on the left image!")
        plt.show()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        
    except Exception as e:
        print(f"Error in visualizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())