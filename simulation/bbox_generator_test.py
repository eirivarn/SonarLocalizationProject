"""
Test script for oriented bounding box data generation.

Generates a small number of samples and validates the output.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bbox_data_generator import generate_sample, visualize_sample

def test_bbox_generation():
    """Test generating samples with oriented bounding boxes."""
    print("=" * 80)
    print("ORIENTED BBOX GENERATION TEST")
    print("=" * 80)
    
    num_test_samples = 10
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating {num_test_samples} test samples...")
    
    samples_with_net = 0
    samples_without_net = 0
    
    for i in range(num_test_samples):
        print(f"\n--- Sample {i} ---")
        
        # Generate sample
        sample_data = generate_sample(
            scene_id=i,
            sample_id=i
        )
        
        # Check output
        assert 'image' in sample_data, "Missing 'image' in output"
        assert 'valid_mask' in sample_data, "Missing 'valid_mask' in output"
        assert 'net_visible' in sample_data, "Missing 'net_visible' in output"
        assert 'bbox' in sample_data, "Missing 'bbox' in output"
        
        # Validate image shape
        img_shape = sample_data['image'].shape
        print(f"  Image shape: {img_shape}")
        assert len(img_shape) == 2, f"Expected 2D image, got shape {img_shape}"
        assert img_shape == (512, 512), f"Expected (512, 512), got {img_shape}"
        
        # Check net visibility
        net_visible = sample_data['net_visible']
        bbox = sample_data['bbox']
        
        if net_visible:
            samples_with_net += 1
            print(f"  Net VISIBLE")
            
            # Validate bbox
            assert bbox is not None, "bbox should not be None when net is visible"
            assert isinstance(bbox, dict), f"bbox should be dict, got {type(bbox)}"
            
            required_keys = ['center_y', 'center_z', 'width', 'height', 'angle']
            for key in required_keys:
                assert key in bbox, f"Missing key '{key}' in bbox"
            
            # Print bbox parameters
            print(f"  BBox center: ({bbox['center_y']:.2f}, {bbox['center_z']:.2f}) m")
            print(f"  BBox size: {bbox['width']:.2f} × {bbox['height']:.2f} m")
            print(f"  BBox angle: {bbox['angle']:+.1f}°")
            
            # Validate ranges
            assert -90 <= bbox['angle'] <= 90, f"Angle {bbox['angle']} out of range [-90, 90]"
            assert bbox['width'] > 0, f"Width {bbox['width']} should be positive"
            assert bbox['height'] > 0, f"Height {bbox['height']} should be positive"
            assert 0 <= bbox['center_y'] <= 20, f"center_y {bbox['center_y']} out of expected range"
            
        else:
            samples_without_net += 1
            print(f"  Net NOT visible")
            assert bbox is None, "bbox should be None when net is not visible"
        
        # Save visualization
        viz_path = output_dir / f"test_sample_{i:03d}.png"
        visualize_sample(sample_data, viz_path)
        print(f"  Saved visualization: {viz_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total samples: {num_test_samples}")
    print(f"Samples with net: {samples_with_net} ({100*samples_with_net/num_test_samples:.1f}%)")
    print(f"Samples without net: {samples_without_net} ({100*samples_without_net/num_test_samples:.1f}%)")
    print(f"\n✓ All tests passed!")
    print(f"✓ Visualizations saved to: {output_dir}")


def test_bbox_loading():
    """Test loading and validating saved .npz files."""
    print("\n" + "=" * 80)
    print("TESTING .NPZ SAVE/LOAD")
    print("=" * 80)
    
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate and save sample
    print("\nGenerating sample...")
    sample_data = generate_sample(scene_id=100, sample_id=0)
    
    sample_path = output_dir / "test_sample.npz"
    print(f"Saving to: {sample_path}")
    np.savez_compressed(sample_path, **sample_data)
    
    # Load and validate
    print("Loading from disk...")
    loaded = np.load(sample_path, allow_pickle=True)
    
    print(f"  Keys: {list(loaded.keys())}")
    print(f"  Image shape: {loaded['image'].shape}")
    print(f"  Net visible: {loaded['net_visible']}")
    
    if loaded['net_visible']:
        bbox = loaded['bbox'].item()  # Extract dict from 0-d array
        print(f"  BBox: center=({bbox['center_y']:.2f}, {bbox['center_z']:.2f}), "
              f"size={bbox['width']:.2f}×{bbox['height']:.2f}, angle={bbox['angle']:+.1f}°")
    else:
        print(f"  BBox: None")
    
    # Visualize loaded sample
    viz_path = output_dir / "test_loaded_sample.png"
    visualize_sample(loaded, viz_path)
    print(f"\n✓ Successfully saved and loaded .npz file")
    print(f"✓ Visualization: {viz_path}")


def test_bbox_sizes():
    """Test different scenes to see bbox variation."""
    print("\n" + "=" * 80)
    print("TESTING BBOX VARIATION ACROSS SCENES")
    print("=" * 80)
    
    output_dir = Path("test_outputs/bbox_variation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_scenes = 5
    
    for scene_idx in range(num_scenes):
        print(f"\n--- Scene {scene_idx} ---")
        
        # Try multiple samples to find one with visible net
        for attempt in range(20):
            sample_data = generate_sample(
                scene_id=200 + scene_idx * 100 + attempt,
                sample_id=attempt
            )
            
            if sample_data['net_visible']:
                bbox = sample_data['bbox']
                print(f"  Generated bbox: {bbox['width']:.2f} × {bbox['height']:.2f} m @ {bbox['angle']:+.1f}°")
                
                # Save visualization
                viz_path = output_dir / f"scene_{scene_idx:02d}.png"
                visualize_sample(sample_data, viz_path)
                print(f"  Saved: {viz_path}")
                break
        else:
            print(f"  Warning: No visible net found in 20 attempts for scene {scene_idx}")
    
    print(f"\n✓ Tested bbox variation across scenes")
    print(f"✓ Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    try:
        test_bbox_generation()
        test_bbox_loading()
        test_bbox_sizes()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
