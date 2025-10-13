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

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sonar_image_analysis import (
    create_elliptical_kernel_fast, 
    create_circular_kernel_fast,
    create_oriented_gradient_kernel_fast,
    adaptive_linear_momentum_merge_fast,
    apply_cv2_enhancement_method,
    get_available_npz_files,
    load_cone_run_npz,
    to_uint8_gray
)
from utils.sonar_config import IMAGE_PROCESSING_CONFIG


class SimpleKernelVisualizer:
    """Simplified kernel visualizer using click events instead of hover."""
    
    def __init__(self, frame, use_cv2=False):
        self.frame = frame
        self.use_cv2 = use_cv2
        self.current_pos = (frame.shape[1]//2, frame.shape[0]//2)
        self.show_enhanced = False
        
        # Configuration
        self.config = IMAGE_PROCESSING_CONFIG.copy()
        
        # Pre-compute enhanced frame
        self.enhanced_frame = self._compute_enhanced_frame()
        
        print("🎨 Setting up visualization...")
        self.setup_figure()
        
    def _compute_enhanced_frame(self):
        """Compute enhanced frame using current settings."""
        if self.use_cv2:
            return apply_cv2_enhancement_method(
                self.frame,
                method=self.config.get('cv2_method', 'morphological'),
                kernel_size=self.config.get('cv2_kernel_size', 5)
            )
        else:
            return adaptive_linear_momentum_merge_fast(
                self.frame,
                base_radius=self.config.get('adaptive_base_radius', 2),
                max_elongation=self.config.get('adaptive_max_elongation', 4),
                linearity_threshold=self.config.get('adaptive_linearity_threshold', 0.3),
                momentum_boost=self.config.get('momentum_boost', 1.5),
                angle_steps=self.config.get('adaptive_angle_steps', 9)
            )
    
    def setup_figure(self):
        """Setup matplotlib figure with simpler layout."""
        self.fig, ((self.ax_main, self.ax_kernel), 
                  (self.ax_params, self.ax_config)) = plt.subplots(2, 2, figsize=(14, 10))
        
        self.fig.suptitle('Simple Kernel Visualizer - Click to explore!', fontsize=14, fontweight='bold')
        
        # Initial display
        self.update_display()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        print("✅ Figure setup complete")
        
    def get_kernel_at_position(self, x, y):
        """Get merging kernel information at specified position."""
        
        h, w = self.frame.shape
        
        # Ensure position is within bounds
        x = max(10, min(w-10, x))
        y = max(10, min(h-10, y))
        
        if self.use_cv2:
            # For CV2 methods, show the actual kernel used
            kernel_size = self.config.get('cv2_kernel_size', 5)
            method = self.config.get('cv2_method', 'morphological')
            
            if method == 'morphological':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                kernel_type = f"CV2 Morphological ({kernel_size}x{kernel_size})"
            else:
                # Simplified for other methods
                kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
                kernel_type = f"CV2 {method.title()}"
            
            return {
                'kernel': kernel.astype(np.float32),
                'type': kernel_type,
                'linearity': 0.0,
                'angle': 0.0,
                'method': 'CV2'
            }
        
        else:
            # Custom adaptive kernel analysis
            frame_float = self.frame.astype(np.float64)
            
            # Compute gradients
            grad_x = cv2.Sobel(frame_float, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(frame_float, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Test linearity at the specified position
            angles = np.linspace(0, 180, self.config.get('adaptive_angle_steps', 9), endpoint=False)
            responses = []
            
            base_radius = self.config.get('adaptive_base_radius', 2)
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
            if linearity_norm > linearity_threshold:
                # Linear structure detected - create elliptical kernel
                best_angle_idx = np.argmax(responses)
                best_angle = angles[best_angle_idx]
                
                # Calculate elongation based on linearity strength
                max_elongation = self.config.get('adaptive_max_elongation', 4)
                elongation = min(1.0 + linearity_norm * 5.0, max_elongation)
                
                # Create elliptical kernel
                kernel = create_elliptical_kernel_fast(base_radius, elongation, best_angle)
                kernel_type = f"Elliptical ({elongation:.1f}x, {best_angle:.0f}°)"
                
            else:
                # No strong linearity - use circular kernel
                kernel = create_circular_kernel_fast(base_radius)
                kernel_type = "Circular"
                best_angle = 0.0
            
            return {
                'kernel': kernel,
                'type': kernel_type,
                'linearity': linearity_norm,
                'angle': best_angle,
                'method': 'Adaptive'
            }
    
    def update_display(self):
        """Update all display elements."""
        
        print(f"🔄 Updating display at position ({self.current_pos[0]}, {self.current_pos[1]})")
        
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
        from matplotlib.patches import Rectangle
        square = Rectangle((x1, y1), kernel_size, kernel_size, 
                         linewidth=1.5, edgecolor='red', facecolor='none', alpha=0.8)
        self.ax_main.add_patch(square)
        
        # Add center point
        self.ax_main.plot(x, y, 'r.', markersize=4)
        
        title_suffix = " (Enhanced)" if self.show_enhanced else " (Original)"
        method_suffix = " [CV2]" if self.use_cv2 else " [Adaptive]"
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
        
        # Kernel parameters
        self.ax_params.clear()
        self.ax_params.axis('off')
        
        params_text = f"""Type: {kernel_info['type']}
Method: {kernel_info['method']}
Linearity: {kernel_info['linearity']:.3f}
Angle: {kernel_info['angle']:.1f}°
Size: {kernel_info['kernel'].shape}
Position: ({x}, {y})"""
        
        self.ax_params.text(0.05, 0.95, params_text, transform=self.ax_params.transAxes,
                          fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        # Configuration panel
        self.ax_config.clear()
        self.ax_config.axis('off')
        
        if self.use_cv2:
            config_text = f"""CV2 Enhancement
Method: {self.config.get('cv2_method', 'morphological')}
Kernel Size: {self.config.get('cv2_kernel_size', 5)}

Controls:
[Click] Select position
[Space] Toggle enhanced
[c] Switch to Adaptive
[q] Quit"""
        else:
            config_text = f"""Adaptive Enhancement
Base Radius: {self.config.get('adaptive_base_radius', 2)}
Max Elongation: {self.config.get('adaptive_max_elongation', 4)}
Linearity Threshold: {self.config.get('adaptive_linearity_threshold', 0.3)}

Controls:
[Click] Select position
[Space] Toggle enhanced
[c] Switch to CV2
[q] Quit"""
        
        self.ax_config.text(0.05, 0.95, config_text, transform=self.ax_config.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes == self.ax_main and event.xdata is not None and event.ydata is not None:
            self.current_pos = (int(event.xdata), int(event.ydata))
            print(f"👆 Clicked at ({self.current_pos[0]}, {self.current_pos[1]})")
            self.update_display()
    
    def on_key_press(self, event):
        """Handle key press events."""
        print(f"⌨️ Key pressed: {event.key}")
        
        if event.key == ' ':  # Space - toggle enhanced/original
            self.show_enhanced = not self.show_enhanced
            self.update_display()
            
        elif event.key == 'c':  # Toggle CV2/Adaptive
            self.use_cv2 = not self.use_cv2
            print(f"🔄 Switched to {'CV2' if self.use_cv2 else 'Adaptive'} mode")
            self.enhanced_frame = self._compute_enhanced_frame()
            self.update_display()
            
        elif event.key == 'q':  # Quit
            print("👋 Quitting...")
            plt.close('all')


def load_sonar_frame(npz_index=0, frame_index=0):
    """Load a sonar frame from NPZ files."""
    
    try:
        files = get_available_npz_files()
        if npz_index >= len(files):
            print(f"❌ Error: NPZ index {npz_index} not available. Found {len(files)} files.")
            return None
            
        print(f"📁 Loading NPZ file {npz_index}: {files[npz_index].name}")
        
        cones, timestamps, extent, metadata = load_cone_run_npz(files[npz_index])
        
        if frame_index >= len(cones):
            print(f"❌ Error: Frame index {frame_index} not available. Found {len(cones)} frames.")
            return None
            
        frame = to_uint8_gray(cones[frame_index])
        print(f"🖼️ Loaded frame {frame_index}: {frame.shape} pixels")
        
        return frame
        
    except Exception as e:
        print(f"❌ Error loading sonar frame: {e}")
        return None


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Simple Interactive Kernel Visualizer for SOLAQUA')
    parser.add_argument('--npz_index', type=int, default=0, 
                      help='NPZ file index to load (default: 0)')
    parser.add_argument('--frame_index', type=int, default=0,
                      help='Frame index within NPZ file (default: 0)')
    parser.add_argument('--use_cv2', action='store_true',
                      help='Start with CV2 enhancement methods')
    
    args = parser.parse_args()
    
    print("🌊 SOLAQUA Simple Kernel Visualizer")
    print("=" * 50)
    
    # Load sonar frame
    frame = load_sonar_frame(args.npz_index, args.frame_index)
    if frame is None:
        return 1
    
    print("🎮 Starting simple visualizer...")
    print("   Click on the sonar image to explore kernels!")
    
    # Create and show visualizer
    try:
        visualizer = SimpleKernelVisualizer(frame, use_cv2=args.use_cv2)
        print("👀 Window should be visible. Click anywhere on the left image!")
        plt.show()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Error in visualizer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())