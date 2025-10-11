"""
Pipeline Visualization Utilities for SOLAQUA

This module provides step-by-step visualization of the sonar image processing pipeline,
specifically breaking down the create_enhanced_contour_detection_video_with_processor function.
Uses actual functions from sonar_image_analysis.py for authentic pipeline behavior.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, List, Tuple, Any, Optional

# Import the actual sonar processing functions
from utils.sonar_image_analysis import (
    SonarDataProcessor, 
    preprocess_edges, 
    select_best_contour_core,
    compute_contour_features,
    _distance_angle_from_contour,
    IMAGE_PROCESSING_CONFIG
)


class PipelineStepVisualizer:
    """Visualizes each step in the sonar image processing pipeline"""
    
    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.step_counter = 0
        
    def reset_counter(self):
        """Reset the step counter"""
        self.step_counter = 0
        
    def increment_step(self):
        """Increment and return the step counter"""
        self.step_counter += 1
        return self.step_counter
        
    def step_1_load_original_frame(self, frame: np.ndarray, frame_number: int) -> Dict[str, Any]:
        """
        Step 1: Load and display the original sonar frame
        
        Args:
            frame: Original sonar frame
            frame_number: Current frame number
            
        Returns:
            Dictionary with step results and metadata
        """
        step_num = self.increment_step()
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Display original frame
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Step {step_num}: Original Sonar Frame #{frame_number}', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        
        # Add frame info
        height, width = frame.shape[:2]
        ax.text(10, height-10, f'Frame: {frame_number}\nSize: {width}x{height}\nType: {frame.dtype}', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'step': step_num,
            'description': 'Original sonar frame loaded',
            'frame_number': frame_number,
            'frame_shape': frame.shape,
            'frame_dtype': str(frame.dtype),
            'frame_stats': {
                'min': float(np.min(frame)),
                'max': float(np.max(frame)),
                'mean': float(np.mean(frame))
            }
        }
    
    def step_2_processor_initialization(self, processor_config: Dict) -> Dict[str, Any]:
        """
        Step 2: Show processor configuration and initialization
        
        Args:
            processor_config: Configuration dictionary for the processor
            
        Returns:
            Dictionary with step results and metadata
        """
        step_num = self.increment_step()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.axis('off')
        
        # Display configuration as text
        config_text = "SonarDataProcessor Configuration:\n\n"
        for key, value in processor_config.items():
            if isinstance(value, dict):
                config_text += f"{key}:\n"
                for sub_key, sub_value in value.items():
                    config_text += f"  {sub_key}: {sub_value}\n"
            else:
                config_text += f"{key}: {value}\n"
        
        ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        ax.set_title(f'Step {step_num}: Processor Configuration', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'step': step_num,
            'description': 'Processor configuration loaded',
            'config': processor_config
        }
    
    def step_3_roi_extraction(self, original_frame: np.ndarray, roi_frame: np.ndarray, 
                             roi_params: Dict) -> Dict[str, Any]:
        """
        Step 3: Region of Interest (ROI) extraction
        
        Args:
            original_frame: Original frame
            roi_frame: Extracted ROI frame
            roi_params: ROI parameters (center, radius, etc.)
            
        Returns:
            Dictionary with step results and metadata
        """
        step_num = self.increment_step()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Original frame with ROI overlay
        ax1.imshow(original_frame, cmap='gray')
        if 'center' in roi_params and 'radius' in roi_params:
            circle = Circle(roi_params['center'], roi_params['radius'], 
                          fill=False, color='red', linewidth=2)
            ax1.add_patch(circle)
            ax1.plot(roi_params['center'][0], roi_params['center'][1], 'r+', markersize=10)
        
        ax1.set_title('Original Frame with ROI', fontsize=14)
        ax1.set_xlabel('X Coordinate (pixels)')
        ax1.set_ylabel('Y Coordinate (pixels)')
        
        # Extracted ROI
        ax2.imshow(roi_frame, cmap='gray')
        ax2.set_title('Extracted ROI', fontsize=14)
        ax2.set_xlabel('X Coordinate (pixels)')
        ax2.set_ylabel('Y Coordinate (pixels)')
        
        fig.suptitle(f'Step {step_num}: ROI Extraction', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'step': step_num,
            'description': 'Region of Interest extracted',
            'roi_params': roi_params,
            'roi_shape': roi_frame.shape,
            'roi_stats': {
                'min': float(np.min(roi_frame)),
                'max': float(np.max(roi_frame)),
                'mean': float(np.mean(roi_frame))
            }
        }
    
    def step_4_preprocessing(self, roi_frame: np.ndarray, preprocessed_frame: np.ndarray,
                           preprocessing_params: Dict) -> Dict[str, Any]:
        """
        Step 4: Image preprocessing (denoising, enhancement)
        
        Args:
            roi_frame: Original ROI frame
            preprocessed_frame: Preprocessed frame
            preprocessing_params: Parameters used for preprocessing
            
        Returns:
            Dictionary with step results and metadata
        """
        step_num = self.increment_step()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)
        
        # Original ROI
        ax1.imshow(roi_frame, cmap='gray')
        ax1.set_title('Original ROI', fontsize=14)
        ax1.set_xlabel('X Coordinate (pixels)')
        ax1.set_ylabel('Y Coordinate (pixels)')
        
        # Preprocessed frame
        ax2.imshow(preprocessed_frame, cmap='gray')
        ax2.set_title('Preprocessed Frame', fontsize=14)
        ax2.set_xlabel('X Coordinate (pixels)')
        ax2.set_ylabel('Y Coordinate (pixels)')
        
        # Difference
        diff = cv2.absdiff(roi_frame, preprocessed_frame)
        im3 = ax3.imshow(diff, cmap='hot')
        ax3.set_title('Preprocessing Difference', fontsize=14)
        ax3.set_xlabel('X Coordinate (pixels)')
        ax3.set_ylabel('Y Coordinate (pixels)')
        plt.colorbar(im3, ax=ax3)
        
        fig.suptitle(f'Step {step_num}: Image Preprocessing', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'step': step_num,
            'description': 'Image preprocessing applied',
            'preprocessing_params': preprocessing_params,
            'noise_reduction': float(np.mean(diff))
        }
    
    def step_5_contour_detection_and_analysis(self, processed_frame: np.ndarray, 
                                            contours: List, best_contour: Optional[np.ndarray],
                                            analysis_params: Dict) -> Dict[str, Any]:
        """
        Step 5: Contour detection and best contour analysis
        
        Args:
            processed_frame: Processed frame for contour detection
            contours: All detected contours
            best_contour: Selected best contour
            analysis_params: Analysis parameters
            
        Returns:
            Dictionary with step results and metadata
        """
        step_num = self.increment_step()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)
        
        # Original processed frame
        ax1.imshow(processed_frame, cmap='gray')
        ax1.set_title('Processed Frame for Contour Detection', fontsize=14)
        ax1.set_xlabel('X Coordinate (pixels)')
        ax1.set_ylabel('Y Coordinate (pixels)')
        
        # All contours
        all_contours_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
        if contours:
            cv2.drawContours(all_contours_frame, contours, -1, (255, 255, 0), 2)  # Yellow
        ax2.imshow(all_contours_frame)
        ax2.set_title(f'All Contours ({len(contours)})', fontsize=14)
        ax2.set_xlabel('X Coordinate (pixels)')
        ax2.set_ylabel('Y Coordinate (pixels)')
        
        # Best contour selected
        best_contour_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
        if contours:
            cv2.drawContours(best_contour_frame, contours, -1, (100, 100, 100), 1)  # Gray all
        if best_contour is not None:
            cv2.drawContours(best_contour_frame, [best_contour], -1, (0, 255, 0), 3)  # Green best
        ax3.imshow(best_contour_frame)
        ax3.set_title('Best Contour Selected', fontsize=14)
        ax3.set_xlabel('X Coordinate (pixels)')
        ax3.set_ylabel('Y Coordinate (pixels)')
        
        fig.suptitle(f'Step {step_num}: Contour Detection and Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'step': step_num,
            'description': 'Contours detected and best contour selected',
            'analysis_params': analysis_params,
            'total_contours': len(contours),
            'has_best_contour': best_contour is not None
        }
    
    def step_6_geometric_analysis(self, frame: np.ndarray, contour: np.ndarray, 
                                ellipse_data: Dict, distance_data: Dict) -> Dict[str, Any]:
        """
        Step 6: Geometric analysis - ellipse fitting and distance calculation
        
        Args:
            frame: Input frame
            contour: Contour for analysis
            ellipse_data: Ellipse fitting results
            distance_data: Distance calculation results
            
        Returns:
            Dictionary with step results and metadata
        """
        step_num = self.increment_step()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Frame with contour and ellipse
        geometric_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else frame.copy()
        
        # Draw contour only if it's not empty
        if contour is not None and len(contour) > 0:
            try:
                cv2.drawContours(geometric_frame, [contour], -1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Warning: Could not draw contour: {e}")
        
        # Draw ellipse if available
        if 'ellipse' in ellipse_data:
            ellipse = ellipse_data['ellipse']
            cv2.ellipse(geometric_frame, ellipse, (255, 0, 255), 2)  # Magenta
            
            # Draw major axis line
            if 'major_axis_line' in ellipse_data:
                p1, p2 = ellipse_data['major_axis_line']
                cv2.line(geometric_frame, p1, p2, (0, 0, 255), 3)  # Red
        
        ax1.imshow(geometric_frame)
        ax1.set_title('Ellipse Fitting and Major Axis', fontsize=14)
        ax1.set_xlabel('X Coordinate (pixels)')
        ax1.set_ylabel('Y Coordinate (pixels)')
        
        # Distance measurement visualization
        distance_frame = geometric_frame.copy()
        if 'distance_point' in distance_data:
            point = distance_data['distance_point']
            cv2.circle(distance_frame, point, 8, (255, 0, 0), -1)  # Blue dot
            
            # Add distance text
            if 'distance_pixels' in distance_data:
                dist_text = f"Distance: {distance_data['distance_pixels']:.1f}px"
                cv2.putText(distance_frame, dist_text, (point[0] + 15, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        ax2.imshow(distance_frame)
        ax2.set_title('Distance Measurement', fontsize=14)
        ax2.set_xlabel('X Coordinate (pixels)')
        ax2.set_ylabel('Y Coordinate (pixels)')
        
        fig.suptitle(f'Step {step_num}: Geometric Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'step': step_num,
            'description': 'Geometric analysis with ellipse fitting and distance measurement',
            'ellipse_data': ellipse_data,
            'distance_data': distance_data
        }
    
    def step_7_final_output_generation(self, original_frame: np.ndarray, final_frame: np.ndarray,
                                     analysis_results: Dict) -> Dict[str, Any]:
        """
        Step 7: Final output frame generation with all annotations
        
        Args:
            original_frame: Original input frame
            final_frame: Final annotated output frame
            analysis_results: Complete analysis results
            
        Returns:
            Dictionary with step results and metadata
        """
        step_num = self.increment_step()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Original frame
        ax1.imshow(original_frame, cmap='gray' if len(original_frame.shape) == 2 else None)
        ax1.set_title('Original Frame', fontsize=14)
        ax1.set_xlabel('X Coordinate (pixels)')
        ax1.set_ylabel('Y Coordinate (pixels)')
        
        # Final annotated frame
        ax2.imshow(final_frame, cmap='gray' if len(final_frame.shape) == 2 else None)
        ax2.set_title('Final Annotated Frame', fontsize=14)
        ax2.set_xlabel('X Coordinate (pixels)')
        ax2.set_ylabel('Y Coordinate (pixels)')
        
        # Add summary statistics
        summary_text = "Processing Summary:\n"
        if 'detection_found' in analysis_results:
            summary_text += f"Detection: {'Yes' if analysis_results['detection_found'] else 'No'}\n"
        if 'distance_pixels' in analysis_results:
            summary_text += f"Distance: {analysis_results['distance_pixels']:.1f}px\n"
        if 'processing_time' in analysis_results:
            summary_text += f"Time: {analysis_results['processing_time']:.3f}s\n"
        
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                fontsize=10, verticalalignment='top')
        
        fig.suptitle(f'Step {step_num}: Final Output Generation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'step': step_num,
            'description': 'Final annotated frame generated',
            'analysis_results': analysis_results,
            'pipeline_complete': True
        }
    
    def generate_pipeline_summary(self, all_step_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the entire pipeline
        
        Args:
            all_step_results: List of results from all pipeline steps
            
        Returns:
            Pipeline summary dictionary
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.axis('off')
        
        summary_text = "SONAR IMAGE PROCESSING PIPELINE SUMMARY\n"
        summary_text += "=" * 50 + "\n\n"
        
        for result in all_step_results:
            step_num = result.get('step', 'Unknown')
            description = result.get('description', 'No description')
            summary_text += f"Step {step_num}: {description}\n"
            
            # Add key metrics for each step
            if 'frame_shape' in result:
                summary_text += f"  Frame size: {result['frame_shape']}\n"
            if 'total_contours' in result:
                summary_text += f"  Contours found: {result['total_contours']}\n"
            if 'detection_found' in result.get('analysis_results', {}):
                summary_text += f"  Detection: {'Yes' if result['analysis_results']['detection_found'] else 'No'}\n"
            
            summary_text += "\n"
        
        # Add overall statistics
        summary_text += "\nOVERALL STATISTICS:\n"
        summary_text += "-" * 20 + "\n"
        total_steps = len(all_step_results)
        summary_text += f"Total processing steps: {total_steps}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.title('Pipeline Processing Summary', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        return {
            'total_steps': total_steps,
            'step_results': all_step_results
        }


def simulate_enhanced_contour_detection_frame_processing(frame: np.ndarray, extent: Tuple[float,float,float,float] = None) -> Tuple[np.ndarray, Dict]:
    """
    Simulate the enhanced contour detection frame processing using REAL sonar analysis functions
    
    Args:
        frame: Input sonar frame  
        extent: Optional spatial extent for real-world coordinate conversion
        
    Returns:
        Tuple of (processed_frame, analysis_results) using actual SOLAQUA algorithms
    """
    print("🔧 Using REAL SonarDataProcessor for frame processing...")
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    # Use actual SonarDataProcessor
    processor = SonarDataProcessor(IMAGE_PROCESSING_CONFIG)
    result = processor.analyze_frame(gray_frame, extent)
    
    # Create visualization frame
    vis_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Get preprocessing results for visualization
    _, edges_proc = processor.preprocess_frame(gray_frame)
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw all contours in light blue
    if contours:
        cv2.drawContours(vis_frame, contours, -1, (255, 200, 100), 1)
    
    # Draw AOI if available
    if processor.current_aoi is not None:
        ax, ay, aw, ah = processor.current_aoi
        cv2.rectangle(vis_frame, (ax, ay), (ax+aw, ay+ah), (0, 255, 255), 2)
        cv2.putText(vis_frame, 'AOI', (ax + 5, ay + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    
    # Draw best contour and analysis results
    if result.detection_success and result.best_contour is not None:
        best_contour = result.best_contour
        
        # Draw best contour (green)
        cv2.drawContours(vis_frame, [best_contour], -1, (0, 255, 0), 2)
        
        # Draw bounding box
        if result.contour_features and 'rect' in result.contour_features:
            x, y, w, h = result.contour_features['rect']
            cv2.rectangle(vis_frame, (x,y), (x+w, y+h), (0,0,255), 1)
        
        # Draw ellipse and red line if possible
        if len(best_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(best_contour)
                (cx, cy), (minor, major), ang = ellipse
                
                # Draw the ellipse (magenta)
                cv2.ellipse(vis_frame, ellipse, (255, 0, 255), 1)
                
                # 90°-rotated major-axis line (red)
                ang_r = np.radians(ang + 90.0)
                half = major * 0.5
                p1 = (int(cx + half*np.cos(ang_r)), int(cy + half*np.sin(ang_r)))
                p2 = (int(cx - half*np.cos(ang_r)), int(cy - half*np.sin(ang_r)))
                cv2.line(vis_frame, p1, p2, (0,0,255), 2)
                
                # Blue dot at intersection with center beam
                if result.distance_pixels is not None:
                    H, W = gray_frame.shape
                    center_x = W // 2
                    dot_y = int(result.distance_pixels)
                    cv2.circle(vis_frame, (center_x, dot_y), 4, (255, 0, 0), -1)
                    
                    # Distance text
                    if result.distance_meters is not None:
                        text = f"D: {result.distance_meters:.2f}m"
                        cv2.putText(vis_frame, text, (center_x + 10, dot_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
            except cv2.error:
                pass  # Skip ellipse drawing if it fails
    
    # Create analysis results dictionary
    analysis_results = {
        'detection_found': result.detection_success,
        'distance_pixels': result.distance_pixels,
        'angle_degrees': result.angle_degrees,
        'distance_meters': result.distance_meters,
        'contour_features': result.contour_features or {},
        'tracking_status': result.tracking_status,
        'total_contours': len(contours),
        'processing_method': 'SonarDataProcessor.analyze_frame',
        'processor_config': IMAGE_PROCESSING_CONFIG,
        'processing_time': 0.0,  # Would need actual timing measurement
        'frame_result': result  # Include the full FrameAnalysisResult
    }
    
    return vis_frame, analysis_results


def demonstrate_full_pipeline(frame: np.ndarray, frame_number: int = 0, extent: Tuple[float,float,float,float] = None) -> Dict[str, Any]:
    """
    Demonstrate the complete pipeline with step-by-step visualization using REAL sonar processing functions
    
    Args:
        frame: Input sonar frame
        frame_number: Frame number for tracking
        extent: Spatial extent (x_min, x_max, y_min, y_max) for real-world coordinate conversion
        
    Returns:
        Complete pipeline results using actual SOLAQUA processing functions
    """
    visualizer = PipelineStepVisualizer()
    all_results = []
    
    print("Starting SOLAQUA Pipeline Visualization (Using Real Functions)...")
    print("=" * 60)
    
    # Step 1: Load original frame
    result1 = visualizer.step_1_load_original_frame(frame, frame_number)
    all_results.append(result1)
    
    # Step 2: Initialize actual SonarDataProcessor 
    processor = SonarDataProcessor(IMAGE_PROCESSING_CONFIG)
    processor_config = {
        'processor_type': 'SonarDataProcessor',
        'config': IMAGE_PROCESSING_CONFIG,
        'tracking_enabled': True
    }
    result2 = visualizer.step_2_processor_initialization(processor_config)
    all_results.append(result2)
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    # Step 3: Use actual preprocessing from sonar_image_analysis
    print("🔧 Using actual preprocess_edges() function...")
    blurred, edges_proc = preprocess_edges(gray_frame, IMAGE_PROCESSING_CONFIG)
    
    preprocessing_params = {
        'function_used': 'preprocess_edges',
        'config': IMAGE_PROCESSING_CONFIG,
        'blur_applied': True,
        'edge_detection': 'Canny'
    }
    result3 = visualizer.step_3_roi_extraction(gray_frame, blurred, preprocessing_params)
    all_results.append(result3)
    
    # Step 4: Edge detection result 
    result4 = visualizer.step_4_preprocessing(blurred, edges_proc, preprocessing_params)
    all_results.append(result4)
    
    # Step 5: Use actual contour detection and selection
    print("🔧 Using actual select_best_contour_core() function...")
    contours, _ = cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the real contour selection function
    best_contour, contour_features, selection_stats = select_best_contour_core(
        contours, 
        last_center=None, 
        aoi=None, 
        cfg_img=IMAGE_PROCESSING_CONFIG
    )
    
    analysis_params = {
        'function_used': 'select_best_contour_core',
        'total_contours_found': len(contours),
        'selection_method': 'real_algorithm',
        'config': IMAGE_PROCESSING_CONFIG
    }
    result5 = visualizer.step_5_contour_detection_and_analysis(edges_proc, contours, best_contour, analysis_params)
    all_results.append(result5)
    
    # Step 6: Use actual geometric analysis functions
    ellipse_data = {}
    distance_data = {}
    
    if best_contour is not None:
        print("🔧 Using actual compute_contour_features() and _distance_angle_from_contour() functions...")
        
        # Use real feature computation
        if contour_features is None:
            contour_features = compute_contour_features(best_contour)
        
        # Use real distance/angle calculation
        H, W = gray_frame.shape
        distance_pixels, angle_degrees = _distance_angle_from_contour(best_contour, W, H)
        
        # Convert to real-world coordinates if extent provided
        distance_meters = None
        if distance_pixels is not None and extent is not None:
            x_min, x_max, y_min, y_max = extent
            height_m = y_max - y_min
            px2m_y = height_m / H
            distance_meters = y_min + distance_pixels * px2m_y
        
        # Ellipse fitting (if possible)
        if len(best_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(best_contour)
                (cx, cy), (minor, major), angle = ellipse
                
                # Calculate major axis line (rotated 90 degrees)
                angle_rad = np.radians(angle + 90)
                half_major = major * 0.5
                p1 = (int(cx + half_major * np.cos(angle_rad)), int(cy + half_major * np.sin(angle_rad)))
                p2 = (int(cx - half_major * np.cos(angle_rad)), int(cy - half_major * np.sin(angle_rad)))
                
                ellipse_data = {
                    'ellipse': ellipse,
                    'center': (cx, cy),
                    'major_axis_line': (p1, p2),
                    'angle': angle,
                    'aspect_ratio': minor/major if major > 0 else 0
                }
            except cv2.error as e:
                print(f"Ellipse fitting failed: {e}")
                ellipse_data = {'error': str(e)}
        
        distance_data = {
            'distance_pixels': distance_pixels,
            'angle_degrees': angle_degrees,
            'distance_meters': distance_meters,
            'function_used': '_distance_angle_from_contour'
        }
        if distance_pixels is not None:
            distance_data['distance_point'] = (W // 2, int(distance_pixels))
    
    # Handle best_contour safely - avoid numpy array truth value ambiguity
    contour_for_analysis = best_contour if best_contour is not None else np.array([])
    result6 = visualizer.step_6_geometric_analysis(gray_frame, contour_for_analysis, 
                                                  ellipse_data, distance_data)
    all_results.append(result6)
    
    # Step 7: Final output - use actual processor analyze_frame method
    print("🔧 Using actual SonarDataProcessor.analyze_frame() method...")
    final_analysis = processor.analyze_frame(gray_frame, extent)
    
    # Convert FrameAnalysisResult to dictionary for compatibility
    analysis_results = {
        'detection_found': final_analysis.detection_success,
        'distance_pixels': final_analysis.distance_pixels,
        'angle_degrees': final_analysis.angle_degrees, 
        'distance_meters': final_analysis.distance_meters,
        'contour_features': final_analysis.contour_features,
        'tracking_status': final_analysis.tracking_status,
        'processing_time': 0.0,  # Would need timing if required
        'processor_used': 'SonarDataProcessor.analyze_frame'
    }
    
    result7 = visualizer.step_7_final_output_generation(frame, gray_frame, analysis_results)
    all_results.append(result7)
    
    # Generate summary
    print("\nGenerating Pipeline Summary...")
    summary = visualizer.generate_pipeline_summary(all_results)
    
    return {
        'all_step_results': all_results,
        'summary': summary,
        'final_frame': gray_frame,
        'analysis_results': analysis_results,
        'processor_result': final_analysis,  # Include the actual FrameAnalysisResult
        'real_functions_used': [
            'preprocess_edges',
            'select_best_contour_core', 
            'compute_contour_features',
            '_distance_angle_from_contour',
            'SonarDataProcessor.analyze_frame'
        ]
    }


# Example usage functions
def load_sample_frame(frame_path: str = None) -> np.ndarray:
    """
    Load a sample sonar frame for demonstration
    
    Args:
        frame_path: Path to frame image, if None creates a synthetic frame
        
    Returns:
        Sample frame array
    """
    if frame_path and cv2.imread(frame_path) is not None:
        return cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Create synthetic sonar frame for demonstration
        frame = np.zeros((400, 600), dtype=np.uint8)
        
        # Add some noise
        noise = np.random.normal(20, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add some objects (ellipses to simulate fish/nets)
        cv2.ellipse(frame, (150, 250), (30, 15), 45, 0, 360, 180, -1)
        cv2.ellipse(frame, (450, 180), (25, 10), -30, 0, 360, 200, -1)
        cv2.ellipse(frame, (300, 320), (40, 20), 0, 0, 360, 160, -1)
        
        # Add some linear features (ropes/nets)
        cv2.line(frame, (100, 100), (500, 150), 120, 3)
        cv2.line(frame, (200, 350), (400, 300), 140, 2)
        
        return frame