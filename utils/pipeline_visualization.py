"""
SOLAQUA Video Generation Pipeline Analysis

This module provides visualization functions for each step of the video generation pipeline
used in create_enhanced_contour_detection_video_with_processor().

Each function produces clean, saveable figures showing the exact processing that occurs
in video generation.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# Import the actual video generation functions
from .sonar_image_analysis import (
    # Data loading functions
    get_available_npz_files, 
    load_cone_run_npz, 
    to_uint8_gray,
    
    # Core processor (same as used in video generation)
    SonarDataProcessor,
    
    # Processing functions
    preprocess_edges,
    select_best_contour_core,
    
    # Configuration used in videos
    IMAGE_PROCESSING_CONFIG,
    VIDEO_CONFIG
)


class PipelineVisualizer:
    """
    Visualizes each step of the SOLAQUA video generation pipeline.
    
    This class provides methods to create clean, saveable visualizations
    of the exact processing steps used in video generation.
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (15, 10), font_size: int = 12):
        """
        Initialize the pipeline visualizer.
        
        Args:
            figure_size: Default figure size for visualizations
            font_size: Default font size for plots
        """
        self.figure_size = figure_size
        self.font_size = font_size
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = figure_size
        plt.rcParams['font.size'] = font_size
        
        # Initialize processor
        self.processor = None
        self.current_frame = None
        self.current_extent = None
        self.frame_idx = 0
        self.data_available = False
        
    def load_data(self, npz_data_path: str, npz_file_index: int = 0, 
                  frame_start: int = 0, frame_count: int = 100, 
                  frame_step: int = 5) -> bool:
        """
        Load data exactly as done in video generation function.
        
        Args:
            npz_data_path: Path to NPZ data files
            npz_file_index: Index of file to select (default: 0)
            frame_start: Starting frame index
            frame_count: Number of frames to process
            frame_step: Step size between frames
            
        Returns:
            bool: True if real data loaded successfully, False if using synthetic
        """
        print("STEP 1: Data Loading (create_enhanced_contour_detection_video_with_processor)")
        print("=" * 80)
        
        try:
            # Get available files (same function call)
            files = get_available_npz_files(npz_data_path)
            if not files:
                print("⚠️  No NPZ files found, using synthetic data")
                raise FileNotFoundError("No real data")
            
            # Select file (same as video function parameter)
            selected_file = files[npz_file_index]
            print(f"📁 Selected NPZ file: {selected_file.name}")
            
            # Load data (exact same function call)
            cones, timestamps, extent, meta = load_cone_run_npz(selected_file)
            T = len(cones)
            print(f"✅ Loaded {T} frames")
            print(f"📐 Spatial extent: {extent}")
            
            # Video generation parameters (same defaults as function)
            frame_count = min(frame_count, T)  # Same logic as video function
            actual_frames = int(min(frame_count, max(0, (T - frame_start)) // max(1, frame_step)))
            
            print(f"🎬 Video parameters:")
            print(f"   - Frame start: {frame_start}")
            print(f"   - Frame count: {frame_count}")
            print(f"   - Frame step: {frame_step}")
            print(f"   - Actual frames to process: {actual_frames}")
            
            # Select a representative frame for analysis
            self.frame_idx = frame_start + (actual_frames // 2) * frame_step
            self.current_frame = to_uint8_gray(cones[self.frame_idx])  # Exact same conversion
            self.current_extent = extent
            
            print(f"\n🔍 Analyzing frame {self.frame_idx} (middle of sequence)")
            print(f"📊 Frame shape: {self.current_frame.shape}")
            print(f"📊 Pixel range: {self.current_frame.min()} - {self.current_frame.max()}")
            
            self.data_available = True
            
        except Exception as e:
            print(f"⚠️  Could not load real data: {e}")
            print("📝 Using synthetic frame for demonstration")
            
            # Create synthetic frame matching typical sonar dimensions
            self.current_frame = np.random.randint(0, 255, (400, 600), dtype=np.uint8)
            self.current_extent = (0, 100, 0, 50)
            self.frame_idx = 0
            self.data_available = False
        
        print(f"\n✅ Data loading complete - ready for video generation pipeline analysis")
        return self.data_available
    
    def initialize_processor(self) -> SonarDataProcessor:
        """
        Initialize processor exactly as done in video generation function.
        
        Returns:
            SonarDataProcessor: The initialized processor
        """
        print("STEP 2: Processor Initialization")
        print("=" * 50)
        
        # Create processor exactly as done in video generation function
        print("Creating SonarDataProcessor (same as video generation)...")
        self.processor = SonarDataProcessor()
        
        # Reset tracking (exact same call as in video function)
        self.processor.reset_tracking()
        print("✅ Processor tracking reset")
        
        print(f"\n🔧 Processor Configuration:")
        print(f"Image processing config: {type(self.processor.img_config).__name__}")
        print(f"Tracking initialized: {self.processor.last_center is None}")
        print(f"Current AOI: {self.processor.current_aoi}")
        
        print(f"\n📋 VIDEO_CONFIG settings (used for annotations):")
        for key, value in VIDEO_CONFIG.items():
            print(f"   - {key}: {value}")
        
        print(f"\n📋 IMAGE_PROCESSING_CONFIG settings:")
        for key, value in IMAGE_PROCESSING_CONFIG.items():
            if isinstance(value, dict):
                print(f"   - {key}:")
                for subkey, subvalue in value.items():
                    print(f"     • {subkey}: {subvalue}")
            else:
                print(f"   - {key}: {value}")
        
        print(f"\n✅ Processor ready - same configuration as video generation system")
        return self.processor
    
    def visualize_data_loading(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the data loading step.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.current_frame is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        fig = plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.imshow(self.current_frame, cmap='viridis', aspect='auto')
        plt.title(f'Input Frame (Frame {self.frame_idx})', fontsize=14, fontweight='bold')
        plt.xlabel('Lateral Position (pixels)')
        plt.ylabel('Range (pixels)')
        plt.colorbar(label='Intensity')
        
        if self.data_available and self.current_extent is not None:
            plt.subplot(2, 2, 2)
            plt.imshow(self.current_frame, cmap='viridis', aspect='auto', extent=self.current_extent)
            plt.title('Real-World Coordinates', fontsize=14, fontweight='bold')
            plt.xlabel('Lateral Position (m)')
            plt.ylabel('Range (m)')
            plt.colorbar(label='Intensity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Data loading visualization saved: {save_path}")
        
        plt.show()
    
    def visualize_edge_detection(self, save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize the edge detection pipeline step.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Tuple of (raw_edges, processed_edges)
        """
        if self.current_frame is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Creating Figure 1: Edge Detection Pipeline")
        
        # Run the exact preprocessing used in analyze_frame()
        edges_raw, edges_proc = preprocess_edges(self.current_frame, IMAGE_PROCESSING_CONFIG)
        
        # Create clean edge detection visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Step 1: Edge Detection Pipeline', fontsize=16, fontweight='bold')
        
        # Original frame
        axes[0].imshow(self.current_frame, cmap='viridis', aspect='auto')
        axes[0].set_title('Original Sonar Frame', fontweight='bold')
        axes[0].set_xlabel('Lateral (pixels)')
        axes[0].set_ylabel('Range (pixels)')
        
        # Raw Canny edges
        axes[1].imshow(edges_raw, cmap='gray', aspect='auto')
        axes[1].set_title('Raw Canny Edges', fontweight='bold')
        axes[1].set_xlabel('Lateral (pixels)')
        axes[1].set_ylabel('Range (pixels)')
        
        # Enhanced edges with momentum merging
        axes[2].imshow(edges_proc, cmap='gray', aspect='auto')
        axes[2].set_title('Enhanced Edges (with Momentum Merge)', fontweight='bold')
        axes[2].set_xlabel('Lateral (pixels)')
        axes[2].set_ylabel('Range (pixels)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure 1 saved: {save_path}")
        
        plt.show()
        return edges_raw, edges_proc
    
    def visualize_contour_detection(self, edges_proc: np.ndarray, 
                                  save_path: Optional[str] = None) -> Tuple[List, Any, Dict, Dict]:
        """
        Visualize the contour detection and selection step.
        
        Args:
            edges_proc: Processed edge image from edge detection step
            save_path: Optional path to save the figure
            
        Returns:
            Tuple of (contours, best_contour, features, stats)
        """
        if self.current_frame is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.processor is None:
            raise ValueError("Processor not initialized. Call initialize_processor() first.")
        
        print("Creating Figure 2: Contour Detection")
        
        # Find contours using exact same method as video generation
        contours, hierarchy = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = IMAGE_PROCESSING_CONFIG.get('min_contour_area', 200)
        large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # Run the actual contour selection
        best_contour, features, stats = select_best_contour_core(
            contours, self.processor.last_center, self.processor.current_aoi, IMAGE_PROCESSING_CONFIG
        )
        
        # Create clean contour visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Step 2: Contour Detection & Selection', fontsize=16, fontweight='bold')
        
        # All contours
        vis_all = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_all, contours, -1, (0, 255, 255), 1)
        axes[0].imshow(cv2.cvtColor(vis_all, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'All Contours\\n({len(contours)} found)', fontweight='bold')
        axes[0].set_xlabel('Lateral (pixels)')
        axes[0].set_ylabel('Range (pixels)')
        
        # Filtered contours
        vis_filtered = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_filtered, large_contours, -1, (255, 165, 0), 2)
        axes[1].imshow(cv2.cvtColor(vis_filtered, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Area Filtered\\n({len(large_contours)} above {min_area}px²)', fontweight='bold')
        axes[1].set_xlabel('Lateral (pixels)')
        axes[1].set_ylabel('Range (pixels)')
        
        # Best contour selection
        vis_best = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        if best_contour is not None:
            cv2.drawContours(vis_best, [best_contour], -1, (0, 255, 0), 3)
            if features:
                cx, cy = int(features['centroid_x']), int(features['centroid_y'])
                cv2.circle(vis_best, (cx, cy), 5, (0, 0, 255), -1)
        axes[2].imshow(cv2.cvtColor(vis_best, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Best Contour Selected\\n(Green=contour, Red=center)', fontweight='bold')
        axes[2].set_xlabel('Lateral (pixels)')
        axes[2].set_ylabel('Range (pixels)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure 2 saved: {save_path}")
        
        plt.show()
        return contours, best_contour, features, stats
    
    def visualize_elliptical_aoi(self, best_contour: Any, features: Dict,
                               save_path: Optional[str] = None) -> Any:
        """
        Visualize the elliptical AOI and tracking step.
        
        Args:
            best_contour: Best contour from contour detection step
            features: Features dictionary from contour detection
            save_path: Optional path to save the figure
            
        Returns:
            Analysis result from processor
        """
        if self.current_frame is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.processor is None:
            raise ValueError("Processor not initialized. Call initialize_processor() first.")
        
        print("Creating Figure 3: Elliptical AOI and Tracking")
        
        # Run analyze_frame to get the elliptical AOI
        result = self.processor.analyze_frame(
            self.current_frame, 
            self.current_extent if self.data_available else None
        )
        
        # Create elliptical AOI visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Step 3: Elliptical AOI & Smoothed Center Tracking', fontsize=16, fontweight='bold')
        
        # Show original frame
        axes[0].imshow(self.current_frame, cmap='viridis', aspect='auto')
        axes[0].set_title('Original Frame', fontweight='bold')
        axes[0].set_xlabel('Lateral (pixels)')
        axes[0].set_ylabel('Range (pixels)')
        
        # Show detection with rectangular AOI (legacy comparison)
        vis_rect = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        if best_contour is not None:
            cv2.drawContours(vis_rect, [best_contour], -1, (0, 255, 0), 2)
            if features:
                x, y, w, h = features['rect']
                expansion = 25
                cv2.rectangle(vis_rect, (x-expansion, y-expansion), 
                             (x+w+expansion, y+h+expansion), (255, 255, 0), 2)
                cv2.putText(vis_rect, 'Rectangular AOI', (x-expansion, y-expansion-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        axes[1].imshow(cv2.cvtColor(vis_rect, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Legacy Rectangular AOI\\n(Old method)', fontweight='bold')
        axes[1].set_xlabel('Lateral (pixels)')
        axes[1].set_ylabel('Range (pixels)')
        
        # Show detection with elliptical AOI
        vis_ellipse = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        if result.detection_success and result.best_contour is not None:
            cv2.drawContours(vis_ellipse, [result.best_contour], -1, (0, 255, 0), 2)
        
        # Draw elliptical AOI if available
        if self.processor.current_aoi is not None and isinstance(self.processor.current_aoi, dict):
            aoi_type = self.processor.current_aoi.get('type', 'rectangular')
            
            if aoi_type == 'elliptical':
                # Draw elliptical AOI mask outline
                aoi_mask = self.processor.current_aoi['mask']
                aoi_contours, _ = cv2.findContours(aoi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if aoi_contours:
                    cv2.drawContours(vis_ellipse, aoi_contours, -1, (0, 255, 255), 3)
                
                # Draw ellipse center (current detection) - yellow dot
                ellipse_center = self.processor.current_aoi['center']
                cv2.circle(vis_ellipse, (int(ellipse_center[0]), int(ellipse_center[1])), 
                          6, (0, 255, 255), -1)
                cv2.putText(vis_ellipse, 'Detection Center', 
                           (int(ellipse_center[0]) + 10, int(ellipse_center[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw smoothed center (tracking center) - red dot
                if 'smoothed_center' in self.processor.current_aoi and self.processor.current_aoi['smoothed_center']:
                    smoothed = self.processor.current_aoi['smoothed_center']
                    cv2.circle(vis_ellipse, (int(smoothed[0]), int(smoothed[1])), 
                              8, (0, 0, 255), -1)
                    cv2.putText(vis_ellipse, 'Smoothed Center', 
                               (int(smoothed[0]) + 10, int(smoothed[1]) + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.putText(vis_ellipse, 'Elliptical AOI', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        axes[2].imshow(cv2.cvtColor(vis_ellipse, cv2.COLOR_BGR2RGB))
        axes[2].set_title('New Elliptical AOI\\n(Yellow=AOI, Yellow dot=detection, Red dot=smoothed)', fontweight='bold')
        axes[2].set_xlabel('Lateral (pixels)')
        axes[2].set_ylabel('Range (pixels)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure 3 saved: {save_path}")
        
        plt.show()
        return result
    
    def visualize_distance_angle_measurement(self, result: Any, 
                                           save_path: Optional[str] = None) -> None:
        """
        Visualize the distance and angle measurement step.
        
        Args:
            result: Analysis result from processor
            save_path: Optional path to save the figure
        """
        if self.current_frame is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if self.processor is None:
            raise ValueError("Processor not initialized. Call initialize_processor() first.")
        
        print("Creating Figure 4: Distance and Angle Measurement")
        
        # Create clean distance/angle visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Step 4: Distance & Angle Measurement (SOLAQUA Coordinate System)', 
                    fontsize=16, fontweight='bold')
        
        H, W = self.current_frame.shape
        
        # UNIFIED CENTER POINT - use the same center for both visualizations
        center_x, center_y = W//2, result.distance_pixels  # Default fallback
        
        # Get center point (prioritize smoothed center if available)
        if (isinstance(self.processor.current_aoi, dict) and 
            'smoothed_center' in self.processor.current_aoi and 
            self.processor.current_aoi['smoothed_center']):
            center_x, center_y = self.processor.current_aoi['smoothed_center']
            center_source = "Smoothed Center"
        elif result.best_contour is not None and len(result.best_contour) >= 5:
            # Use ellipse center as backup
            try:
                (cx, cy), _, _ = cv2.fitEllipse(result.best_contour)
                center_x, center_y = cx, cy
                center_source = "Ellipse Center"
            except:
                center_source = "Default Center"
        else:
            center_source = "Default Center"
        
        print(f"Using {center_source}: ({center_x:.1f}, {center_y:.1f})")
        
        # Distance measurement visualization
        vis_distance = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        if result.detection_success and result.distance_pixels is not None:
            # Draw the detection contour
            cv2.drawContours(vis_distance, [result.best_contour], -1, (0, 255, 0), 2)
            
            # SOLAQUA coordinate system: sonar is at TOP (y=0), distance measured forward (downward in pixels)
            sonar_x, sonar_y = W//2, 0  # Sonar at top center
            target_pixel_y = int(result.distance_pixels)  # Direct pixel distance from sonar
            
            # Draw UNIFIED net center
            cv2.circle(vis_distance, (int(center_x), int(center_y)), 8, (0, 0, 255), -1)
            cv2.circle(vis_distance, (int(center_x), int(center_y)), 8, (255, 255, 255), 1)  # White outline
            
            # Draw distance measurement - from sonar position to target y-coordinate
            cv2.line(vis_distance, (sonar_x, sonar_y), (sonar_x, target_pixel_y), 
                    (0, 0, 255), 3)  # Vertical distance line from sonar
            
            # Add distance annotations
            cv2.putText(vis_distance, f'Distance: {result.distance_pixels:.0f} px', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if result.distance_meters is not None:
                cv2.putText(vis_distance, f'Distance: {result.distance_meters:.2f} m', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mark sonar position at TOP
            cv2.circle(vis_distance, (sonar_x, sonar_y), 6, (255, 255, 0), -1)
            cv2.putText(vis_distance, 'Sonar (Vehicle)', (sonar_x + 10, sonar_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add center label
            cv2.putText(vis_distance, 'Net Center', (int(center_x) + 10, int(center_y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add coordinate system info
            cv2.putText(vis_distance, 'Forward →', (10, H - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        axes[0].imshow(cv2.cvtColor(vis_distance, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Distance Measurement\\n(Forward range from vehicle)', fontweight='bold')
        axes[0].set_xlabel('Lateral (pixels)')
        axes[0].set_ylabel('Forward Range (pixels)')
        
        # Angle measurement visualization (net orientation, not bearing)
        vis_angle = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
        if result.detection_success and result.angle_degrees is not None:
            # Draw the detection contour
            cv2.drawContours(vis_angle, [result.best_contour], -1, (0, 255, 0), 2)
            
            # Draw SAME UNIFIED net center as left panel
            cv2.circle(vis_angle, (int(center_x), int(center_y)), 8, (0, 0, 255), -1)
            cv2.circle(vis_angle, (int(center_x), int(center_y)), 8, (255, 255, 255), 1)  # White outline
            
            # Get ellipse information for angle visualization
            if result.best_contour is not None and len(result.best_contour) >= 5:
                try:
                    # Get ellipse parameters (same as used in angle calculation)
                    (cx, cy), (minor_axis, major_axis), angle = cv2.fitEllipse(result.best_contour)
                    
                    # Draw ellipse major axis to show orientation (using ellipse center for axis calculation)
                    angle_rad = np.radians(angle + 90.0)  # Major axis direction (same as in calculation)
                    axis_length = 60  # Length for visualization
                    dx = axis_length * np.cos(angle_rad)
                    dy = axis_length * np.sin(angle_rad)
                    
                    # Draw major axis line through the ellipse center (not necessarily same as tracking center)
                    p1 = (int(cx - dx), int(cy - dy))
                    p2 = (int(cx + dx), int(cy + dy))
                    cv2.line(vis_angle, p1, p2, (255, 0, 255), 3)
                    
                    # Draw reference horizontal line through unified center
                    ref_length = 40
                    cv2.line(vis_angle, (int(center_x - ref_length), int(center_y)), 
                            (int(center_x + ref_length), int(center_y)), (255, 255, 255), 2)
                    
                    # Angle annotation
                    angle_text = f'Net Orientation: {result.angle_degrees:.1f}'
                    cv2.putText(vis_angle, angle_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    # Labels
                    cv2.putText(vis_angle, 'Net Center', (int(center_x) + 10, int(center_y) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(vis_angle, 'Major Axis', (p2[0] + 5, p2[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.putText(vis_angle, 'Reference', (int(center_x + ref_length) + 5, int(center_y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                except Exception as e:
                    cv2.putText(vis_angle, f'Angle calc error: {e}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        axes[1].imshow(cv2.cvtColor(vis_angle, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Angle Measurement\\n(Net orientation from major axis)', fontweight='bold')
        axes[1].set_xlabel('Lateral (pixels)')
        axes[1].set_ylabel('Forward Range (pixels)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure 4 saved: {save_path}")
        
        plt.show()
    
    def run_complete_pipeline_analysis(self, npz_data_path: str, 
                                     output_dir: str = "/tmp",
                                     npz_file_index: int = 0) -> Dict[str, Any]:
        """
        Run the complete pipeline analysis with all visualization steps.
        
        Args:
            npz_data_path: Path to NPZ data files
            output_dir: Directory to save visualization figures
            npz_file_index: Index of NPZ file to analyze
            
        Returns:
            Dictionary containing all analysis results
        """
        print("SOLAQUA Video Generation Pipeline Analysis")
        print("=" * 60)
        print("✅ Video generation functions loaded!")
        print("✅ Using EXACT same code as create_enhanced_contour_detection_video_with_processor()")
        
        results = {}
        
        # Step 1: Load data
        data_loaded = self.load_data(npz_data_path, npz_file_index)
        self.visualize_data_loading(f"{output_dir}/solaqua_step0_data_loading.png")
        results['data_loaded'] = data_loaded
        
        # Step 2: Initialize processor
        processor = self.initialize_processor()
        results['processor'] = processor
        
        # Step 3: Edge detection
        edges_raw, edges_proc = self.visualize_edge_detection(f"{output_dir}/solaqua_step1_edge_detection.png")
        results['edges_raw'] = edges_raw
        results['edges_processed'] = edges_proc
        
        # Step 4: Contour detection
        contours, best_contour, features, stats = self.visualize_contour_detection(
            edges_proc, f"{output_dir}/solaqua_step2_contour_detection.png"
        )
        results['contours'] = contours
        results['best_contour'] = best_contour
        results['features'] = features
        results['stats'] = stats
        
        # Step 5: Elliptical AOI
        analysis_result = self.visualize_elliptical_aoi(
            best_contour, features, f"{output_dir}/solaqua_step3_elliptical_aoi.png"
        )
        results['analysis_result'] = analysis_result
        
        # Step 6: Distance and angle measurement
        self.visualize_distance_angle_measurement(
            analysis_result, f"{output_dir}/solaqua_step4_distance_angle.png"
        )
        
        print("\\n" + "=" * 60)
        print("✅ Complete pipeline analysis finished!")
        print("✅ All visualization figures saved to:", output_dir)
        
        return results


def run_pipeline_analysis(npz_data_path: str = "/Volumes/LaCie/SOLAQUA/exports/outputs",
                         output_dir: str = "/tmp",
                         npz_file_index: int = 0) -> Dict[str, Any]:
    """
    Convenience function to run complete pipeline analysis.
    
    Args:
        npz_data_path: Path to NPZ data files
        output_dir: Directory to save visualization figures  
        npz_file_index: Index of NPZ file to analyze
        
    Returns:
        Dictionary containing all analysis results
    """
    visualizer = PipelineVisualizer()
    return visualizer.run_complete_pipeline_analysis(npz_data_path, output_dir, npz_file_index)


def test_adaptive_merging_parameters(npz_data_path: str = "/Volumes/LaCie/SOLAQUA/exports/outputs",
                                    npz_file_index: int = 0,
                                    save_dir: str = "/tmp") -> None:
    """
    Test different parameter combinations for adaptive linear merging.
    
    This function loads a frame and shows comparison of different parameter settings:
    1. Original frame
    2. Conservative settings (low elongation)
    3. Balanced settings (default)
    4. Aggressive settings (high elongation)
    
    Args:
        npz_data_path: Path to NPZ data files
        npz_file_index: Index of file to test
        save_dir: Directory to save comparison images
    """
    from .sonar_image_analysis import adaptive_linear_momentum_merge_fast, preprocess_edges
    import matplotlib.pyplot as plt
    
    print("🧪 Testing Adaptive Linear Merging Parameter Combinations")
    print("=" * 60)
    
    # Load test data
    visualizer = PipelineVisualizer()
    data_loaded = visualizer.load_data(npz_data_path, npz_file_index)
    
    if not data_loaded or visualizer.current_frame is None:
        print("❌ Could not load test data")
        return
    
    frame = visualizer.current_frame
    print(f"📊 Test frame shape: {frame.shape}")
    
    # Test different parameter combinations
    test_configs = [
        {"base_radius": 1, "max_elongation": 2, "linearity_threshold": 0.4, "name": "Conservative"},
        {"base_radius": 2, "max_elongation": 4, "linearity_threshold": 0.3, "name": "Balanced (Default)"},
        {"base_radius": 3, "max_elongation": 6, "linearity_threshold": 0.2, "name": "Aggressive"},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n🔬 Testing configuration: {config['name']}")
        
        # Apply adaptive linear merging with these parameters
        print("   Running adaptive linear merging...")
        enhanced = adaptive_linear_momentum_merge_fast(
            frame,
            base_radius=config["base_radius"],
            max_elongation=config["max_elongation"], 
            linearity_threshold=config["linearity_threshold"],
            momentum_boost=10.0,
            angle_steps=9
        )
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Adaptive Linear Merging: {config["name"]} Configuration', 
                    fontsize=16, fontweight='bold')
        
        # Original frame
        axes[0, 0].imshow(frame, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Original Frame', fontweight='bold')
        axes[0, 0].set_xlabel('Lateral (pixels)')
        axes[0, 0].set_ylabel('Range (pixels)')
        
        # Enhanced frame
        axes[0, 1].imshow(enhanced, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'Enhanced Frame\\n'
                           f'radius={config["base_radius"]}, elongation={config["max_elongation"]}, '
                           f'threshold={config["linearity_threshold"]}', fontweight='bold')
        axes[0, 1].set_xlabel('Lateral (pixels)')
        axes[0, 1].set_ylabel('Range (pixels)')
        
        # Edge detection comparison
        import cv2
        original_edges = cv2.Canny(frame, 60, 180)
        enhanced_edges = cv2.Canny(enhanced, 60, 180)
        
        axes[1, 0].imshow(original_edges, cmap='gray', aspect='auto')
        axes[1, 0].set_title('Original Edges', fontweight='bold')
        axes[1, 0].set_xlabel('Lateral (pixels)')
        axes[1, 0].set_ylabel('Range (pixels)')
        
        axes[1, 1].imshow(enhanced_edges, cmap='gray', aspect='auto')
        axes[1, 1].set_title('Enhanced Edges', fontweight='bold')
        axes[1, 1].set_xlabel('Lateral (pixels)')
        axes[1, 1].set_ylabel('Range (pixels)')
        
        plt.tight_layout()
        
        # Save comparison
        save_path = f"{save_dir}/adaptive_merging_test_{config['name'].lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved comparison: {save_path}")
        
        plt.show()
    
    print(f"\n🎯 Testing complete! Check {save_dir} for comparison images.")
    print("\n💡 Parameter tuning guide:")
    print("   - adaptive_base_radius: Controls base merging distance (1-5)")
    print("   - adaptive_max_elongation: Max ellipse elongation (2-8)")  
    print("   - adaptive_linearity_threshold: Sensitivity to linear patterns (0.1-0.5)")
    print("   - momentum_boost: Enhancement strength (1-20)")


if __name__ == "__main__":
    # Example usage
    # results = run_pipeline_analysis()
    
    # Test the adaptive merging parameters
    test_adaptive_merging_parameters()
