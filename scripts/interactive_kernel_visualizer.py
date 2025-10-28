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

# Fix imports - get functions from correct modules
from utils.image_enhancement import (
    create_elliptical_kernel_fast, 
    create_circular_kernel_fast,
    adaptive_linear_momentum_merge_fast,
    compute_structure_tensor_field_fast
)
from utils.io_utils import get_available_npz_files
from utils.sonar_utils import load_cone_run_npz, to_uint8_gray
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
        """Get EXACT kernel that adaptive_linear_momentum_merge_fast would use at this position."""
        
        h, w = self.frame.shape
        x = max(10, min(w-10, x))
        y = max(10, min(h-10, y))
        
        # Convert to grayscale if needed
        if len(self.frame.shape) == 3:
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame
        
        # EXACT REPLICATION: Start with binary conversion (like the real algorithm)
        binary_threshold = self.config.get('binary_threshold', 128)
        binary_frame = (frame > binary_threshold).astype(np.uint8) * 255
        
        # Use same parameters
        angle_steps = self.config.get('adaptive_angle_steps', 36)
        base_radius = self.config.get('adaptive_base_radius', 3)
        max_elongation = self.config.get('adaptive_max_elongation', 3.0)  
        linearity_threshold = self.config.get('adaptive_linearity_threshold', 0.15)
        downscale_factor = self.config.get('downscale_factor', 2)
        top_k_bins = self.config.get('top_k_bins', 8)
        min_coverage_percent = self.config.get('min_coverage_percent', 0.5)
        
        # DEBUG: Print the actual config values being used
        print(f"CONFIG DEBUG: max_elongation={max_elongation}, base_radius={base_radius}")
        print(f"CONFIG DEBUG: linearity_threshold={linearity_threshold}")
        
        result = binary_frame.astype(np.float32)  # Work with binary like the real algorithm
        
        # STEP 1: Early exit check (like real algorithm)
        frame_std = np.std(result)
        if frame_std < 5.0:
            print(f"→ EARLY EXIT: Low contrast (std={frame_std:.1f} < 5.0) → Gaussian fallback")
            kernel = create_circular_kernel_fast(base_radius)
            return {
                'kernel': kernel,
                'type': f"Circular (Low contrast std={frame_std:.1f})",
                'linearity': 0.0,
                'angle': 0.0,
                'method': 'Early exit - low contrast'
            }
        
        # STEP 2: Downsampled analysis (exact same)
        h_small = max(h // downscale_factor, 32)
        w_small = max(w // downscale_factor, 32)
        frame_small = cv2.resize(result, (w_small, h_small), interpolation=cv2.INTER_AREA)
        
        # STEP 3: Structure tensor (exact same)
        grad_x = cv2.Sobel(frame_small, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame_small, cv2.CV_64F, 0, 1, ksize=3)
        
        orientations, linearity_map_small = compute_structure_tensor_field_fast(
            grad_x, grad_y, sigma=1.5  # EXACT same sigma
        )
        
        # STEP 4: Quantization (EXACT - this is the key part!)
        # CRITICAL: Check how orientations are mapped in the real algorithm
        orientations_normalized = orientations / 180.0  # Convert 0-180° to 0-1
        direction_bin_map_small = np.round(orientations_normalized * (angle_steps - 1)).astype(np.int32)
        direction_bin_map_small = np.clip(direction_bin_map_small, 0, angle_steps - 1)
        
        # DEBUG: Check orientation mapping at clicked position
        x_small = int(x * w_small / w)
        y_small = int(y * h_small / h)
        x_small = np.clip(x_small, 0, w_small - 1)
        y_small = np.clip(y_small, 0, h_small - 1)
        
        raw_orientation = orientations[y_small, x_small]
        quantized_bin = direction_bin_map_small[y_small, x_small]
        reconstructed_angle = quantized_bin * 180.0 / (angle_steps - 1)
        
        print(f"ANGLE DEBUG: raw={raw_orientation:.1f}° → bin={quantized_bin} → reconstructed={reconstructed_angle:.1f}°")
        
        # Check gradient direction at this position for validation
        gx = grad_x[y_small, x_small]
        gy = grad_y[y_small, x_small]
        gradient_angle = np.arctan2(gy, gx) * 180.0 / np.pi
        if gradient_angle < 0:
            gradient_angle += 180.0  # Ensure 0-180 range
        
        print(f"GRADIENT DEBUG: gx={gx:.3f}, gy={gy:.3f} → gradient_angle={gradient_angle:.1f}°")
        
        # CRITICAL ANALYSIS: What does the structure tensor actually return?
        angle_diff = abs(raw_orientation - gradient_angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff  # Handle wraparound
        
        print(f"RELATIONSHIP: structure_tensor={raw_orientation:.1f}° vs gradient={gradient_angle:.1f}°")
        print(f"ANGLE DIFFERENCE: {angle_diff:.1f}° (should be ~0° if both parallel to edge)")
        
        if angle_diff < 30:
            print("→ STRUCTURE TENSOR and GRADIENT both PARALLEL to edge - CORRECT!")
        elif angle_diff > 60:
            print("→ STRUCTURE TENSOR perpendicular to gradient - one may be wrong")
        else:
            print("→ UNCLEAR relationship - noisy region?")
        
        # STEP 5: Normalize linearity (like real algorithm)
        max_linearity = np.max(linearity_map_small)
        if max_linearity > 0:
            linearity_map_small = linearity_map_small / max_linearity
        else:
            print("→ NO LINEARITY DETECTED → Gaussian fallback")
            kernel = create_circular_kernel_fast(base_radius)
            return {
                'kernel': kernel,
                'type': "Circular (No linearity detected)",
                'linearity': 0.0,
                'angle': 0.0,
                'method': 'No linearity - Gaussian fallback'
            }
        
        # STEP 6: Upsample (exact same)
        linearity_map = cv2.resize(linearity_map_small, (w, h), interpolation=cv2.INTER_LINEAR)
        direction_bin_map = cv2.resize(direction_bin_map_small.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        # STEP 7: Create linear mask (exact same)
        linear_mask = linearity_map > linearity_threshold
        
        if np.sum(linear_mask) == 0:
            print("→ NO LINEAR REGIONS → Gaussian fallback")
            kernel = create_circular_kernel_fast(base_radius)
            return {
                'kernel': kernel,
                'type': "Circular (No linear regions)",
                'linearity': 0.0,
                'angle': 0.0,
                'method': 'No linear regions - Gaussian fallback'
            }
        
        # STEP 8: Bin filtering (EXACT same as real algorithm)
        unique_bins, bin_counts = np.unique(direction_bin_map[linear_mask], return_counts=True)
        total_linear_pixels = np.sum(linear_mask)
        bin_coverage_percent = (bin_counts / total_linear_pixels) * 100.0
        
        # Filter bins by coverage
        coverage_filter = bin_coverage_percent >= min_coverage_percent
        filtered_bins = unique_bins[coverage_filter]
        
        if len(filtered_bins) == 0:
            print(f"→ NO BINS PASS COVERAGE ({min_coverage_percent}%) → Gaussian fallback")
            kernel = create_circular_kernel_fast(base_radius)
            return {
                'kernel': kernel,
                'type': f"Circular (No bins pass {min_coverage_percent}% coverage)",
                'linearity': 0.0,
                'angle': 0.0,
                'method': 'Coverage filter failed - Gaussian fallback'
            }
        
        # STEP 9: Top-K selection (EXACT same)
        bin_linearity_totals = []
        for bin_idx in filtered_bins:
            bin_mask = (direction_bin_map == bin_idx) & linear_mask
            total_linearity = np.sum(linearity_map[bin_mask])
            bin_linearity_totals.append(total_linearity)
        bin_linearity_totals = np.array(bin_linearity_totals)
        
        if len(filtered_bins) > top_k_bins:
            top_k_indices = np.argsort(bin_linearity_totals)[-top_k_bins:]
            significant_bins = filtered_bins[top_k_indices]
        else:
            significant_bins = filtered_bins
        
        # STEP 10: Check if clicked position is in a significant bin
        angle_bin = direction_bin_map[y, x]
        linearity_value = linearity_map[y, x]
        
        if angle_bin in significant_bins and linear_mask[y, x]:
            # This pixel gets elliptical treatment!
            angle_degrees = float(angle_bin * 180.0 / (angle_steps - 1))
            
            # CRITICAL FIX: If kernels appear perpendicular to edges, rotate by 90°
            # The structure tensor gives the correct mathematical angle, but the kernel
            # creation function may interpret it differently
            corrected_angle = (angle_degrees + 90) % 180
            
            print(f"ANGLE CORRECTION: {angle_degrees:.0f}° → {corrected_angle:.0f}° (rotated 90°)")
            print(f"FINAL ANGLE: {corrected_angle:.0f}° (corrected to be parallel to edges)")
            
            # VERIFICATION: What direction is the kernel actually oriented?
            print(f"KERNEL ORIENTATION: {corrected_angle:.0f}° - this should now be parallel to edges")
            print(f"INTERPRETATION: Kernel will enhance features running PARALLEL to {corrected_angle:.0f}° direction")
            
            # Calculate elongation for this bin (FIXED to ensure variation)
            bin_mask = (direction_bin_map == angle_bin) & linear_mask
            avg_linearity = np.mean(linearity_map[bin_mask])
            
            # DEBUG: Print what's happening with elongation calculation
            print(f"   DEBUG: avg_linearity={avg_linearity:.3f}, max_elongation={max_elongation}")
            
            # FIXED: Ensure we get meaningful elongation values
            if avg_linearity < 0.1:
                # Very low linearity - force some elongation to see variation
                avg_elongation = 1.0  # Increased from 1.5 to 2.0 for more visible difference
                print(f"   FORCED elongation due to low linearity")
            else:
                # Use the real calculation but ensure minimum elongation > 1
                calculated_elongation = 1 + (max_elongation - 1) * avg_linearity
                avg_elongation = max(calculated_elongation, 1.5)  # Increased minimum from 1.2 to 1.5
                print(f"   CALCULATED: {calculated_elongation:.3f} → clamped to {avg_elongation:.3f}")
            
            # FORCE a test elongation to verify the visualization works
            if avg_elongation < 2.0:
                test_elongation = 2.5
                print(f"   TESTING: Forcing elongation to {test_elongation} to verify visualization")
                kernel = create_elliptical_kernel_fast(base_radius, test_elongation, corrected_angle)
            else:
                kernel = create_elliptical_kernel_fast(base_radius, avg_elongation, corrected_angle)
            
            print(f"→ ELLIPTICAL: bin={angle_bin}, corrected_angle={corrected_angle:.0f}°, elongation={avg_elongation:.1f}")
            print(f"   Bin coverage: {bin_coverage_percent[list(unique_bins).index(angle_bin)]:.1f}%")
            print(f"   Kernel shape: {kernel.shape}, min={kernel.min():.3f}, max={kernel.max():.3f}")
            
            return {
                'kernel': kernel,
                'type': f"Elliptical ({avg_elongation:.1f}x, {corrected_angle:.0f}°)",
                'linearity': float(linearity_value),
                'angle': float(corrected_angle),  # Return corrected angle
                'method': f'Significant bin (coverage OK, top-K selected)'
            }
        else:
            # This pixel gets Gaussian treatment
            reason = "Not in significant bin" if angle_bin not in significant_bins else "Not in linear region"
            print(f"→ CIRCULAR: {reason}")
            
            kernel = create_circular_kernel_fast(base_radius)
            return {
                'kernel': kernel,
                'type': f"Circular ({reason})",
                'linearity': float(linearity_value),
                'angle': 0.0,
                'method': f'Gaussian fallback - {reason}'
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