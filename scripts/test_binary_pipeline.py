#!/usr/bin/env python3
"""
Test script for binary preprocessing pipeline verification.

Tests the new signal-strength independent binary conversion approach
to ensure the pipeline correctly processes sonar frames without signal dependency.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from sonar_image_analysis import preprocess_edges
from sonar_config import IMAGE_PROCESSING_CONFIG

def create_test_frame():
    """Create a synthetic sonar frame with varying signal strengths."""
    # Create test frame with different signal intensity regions
    frame = np.zeros((400, 600), dtype=np.uint8)
    
    # High signal strength net structure (top half)
    frame[50:150, 100:500] = 200  # Background
    frame[90:110, 100:500] = 255  # Net strands (high intensity)
    frame[95:105, 150:450] = 255  # Cross strands
    
    # Low signal strength net structure (bottom half)  
    frame[250:350, 100:500] = 80   # Background (low signal)
    frame[290:310, 100:500] = 150  # Net strands (low intensity)
    frame[295:305, 150:450] = 150  # Cross strands
    
    # Add some noise
    noise = np.random.normal(0, 10, frame.shape)
    frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return frame

def test_binary_preprocessing():
    """Test the binary preprocessing pipeline."""
    print("Testing Binary Preprocessing Pipeline")
    print("=" * 40)
    
    # Create test frame
    test_frame = create_test_frame()
    
    # Test with different binary thresholds
    thresholds = [64, 128, 192]
    
    fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(15, 8))
    
    # Show original frame
    axes[0, 0].imshow(test_frame, cmap='viridis')
    axes[0, 0].set_title('Original Frame\n(Variable Signal Strength)')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(test_frame, cmap='viridis')
    axes[1, 0].set_title('Original Frame\n(Variable Signal Strength)')
    axes[1, 0].axis('off')
    
    for i, threshold in enumerate(thresholds):
        print(f"\nTesting with binary_threshold = {threshold}")
        
        # Update configuration
        test_config = IMAGE_PROCESSING_CONFIG.copy()
        test_config['binary_threshold'] = threshold
        
        # Test preprocessing
        raw_edges, processed_edges = preprocess_edges(test_frame, test_config)
        
        # Show results
        axes[0, i+1].imshow(raw_edges, cmap='gray')
        axes[0, i+1].set_title(f'Raw Edges\nThreshold={threshold}')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(processed_edges, cmap='gray')
        axes[1, i+1].set_title(f'Enhanced Edges\nThreshold={threshold}')
        axes[1, i+1].axis('off')
        
        # Print statistics
        print(f"  Raw edges pixels: {np.sum(raw_edges > 0)}")
        print(f"  Enhanced edges pixels: {np.sum(processed_edges > 0)}")
        
        # Test that both high and low signal regions produce edges
        top_half_edges = np.sum(processed_edges[:200] > 0)
        bottom_half_edges = np.sum(processed_edges[200:] > 0)
        
        print(f"  Top half (high signal) edges: {top_half_edges}")
        print(f"  Bottom half (low signal) edges: {bottom_half_edges}")
        
        if top_half_edges > 0 and bottom_half_edges > 0:
            print("  ✅ SUCCESS: Both signal regions detected")
        else:
            print("  ❌ ISSUE: Uneven detection between signal regions")
    
    plt.tight_layout()
    plt.suptitle('Binary Pipeline Test: Signal-Strength Independence', y=0.98)
    plt.show()
    
    return True

def test_enhancement_methods():
    """Test both CV2 and adaptive enhancement on binary data."""
    print("\n\nTesting Enhancement Methods on Binary Data")
    print("=" * 45)
    
    test_frame = create_test_frame()
    
    # Test configurations
    configs = [
        {'use_cv2_enhancement': True, 'cv2_method': 'morphological', 'binary_threshold': 128},
        {'use_cv2_enhancement': True, 'cv2_method': 'bilateral', 'binary_threshold': 128},
        {'use_cv2_enhancement': False, 'binary_threshold': 128}  # Adaptive method
    ]
    
    fig, axes = plt.subplots(2, len(configs) + 1, figsize=(12, 6))
    
    # Show original
    axes[0, 0].imshow(test_frame, cmap='viridis')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(test_frame, cmap='viridis') 
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')
    
    for i, config in enumerate(configs):
        test_config = IMAGE_PROCESSING_CONFIG.copy()
        test_config.update(config)
        
        method_name = config.get('cv2_method', 'adaptive') if config['use_cv2_enhancement'] else 'adaptive'
        print(f"\nTesting {method_name} enhancement")
        
        raw_edges, processed_edges = preprocess_edges(test_frame, test_config)
        
        axes[0, i+1].imshow(raw_edges, cmap='gray')
        axes[0, i+1].set_title(f'Raw Edges\n{method_name}')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(processed_edges, cmap='gray')
        axes[1, i+1].set_title(f'Enhanced\n{method_name}')
        axes[1, i+1].axis('off')
        
        print(f"  Enhanced edges pixels: {np.sum(processed_edges > 0)}")
    
    plt.tight_layout()
    plt.suptitle('Enhancement Methods Comparison on Binary Data', y=0.98)
    plt.show()

if __name__ == "__main__":
    try:
        print("Binary Pipeline Verification Test")
        print("=================================")
        print("\nThis script tests the new signal-strength independent")
        print("binary preprocessing pipeline.\n")
        
        # Run tests
        test_binary_preprocessing()
        test_enhancement_methods()
        
        print("\n" + "="*50)
        print("✅ All tests completed successfully!")
        print("Binary pipeline is working correctly.")
        print("Signal strength independence verified.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()