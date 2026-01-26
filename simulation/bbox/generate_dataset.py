"""
Generate and save bbox training dataset.

Pre-generates samples from simulation and saves them to disk for faster training.
"""import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_bbox_world import generate_real_sample_with_bbox
from core import DATA_GEN_CONFIG


def generate_bbox_dataset(
    output_dir,
    num_samples,
    scene_id_start=0,
    split_name='train'
):
    """
    Generate and save bbox dataset.
    
    Args:
        output_dir: Directory to save dataset
        num_samples: Number of samples to generate
        scene_id_start: Starting scene ID
        split_name: 'train', 'val', or 'test'
    """
    output_dir = Path(output_dir)
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} samples for {split_name} set...")
    print(f"Saving to: {split_dir}")
    
    # Use 100 samples per scene
    samples_per_scene = 100
    
    metadata = {
        'num_samples': num_samples,
        'scene_id_start': scene_id_start,
        'samples_per_scene': samples_per_scene,
        'split': split_name,
        'config': {
            'image_size': DATA_GEN_CONFIG['image_size'],
            'range_m': DATA_GEN_CONFIG.get('range_m', 20.0),
        }
    }
    
    for idx in tqdm(range(num_samples), desc=f"Generating {split_name}"):
        # Determine scene_id and sample_id
        scene_id = scene_id_start + (idx // samples_per_scene)
        sample_id = idx % samples_per_scene
        
        # Generate sample
        sample_data = generate_real_sample_with_bbox(scene_id, sample_id)
        
        # Prepare save data
        save_data = {
            'image': sample_data['image'].astype(np.float32),
            'valid_mask': sample_data['valid_mask'].astype(np.float32),
            'segmentation': sample_data['segmentation'].astype(np.uint8),
            'net_visible': sample_data['net_visible'],
            'scene_id': scene_id,
            'sample_id': sample_id,
        }
        
        # Add bbox if net is visible
        if sample_data['net_visible']:
            bbox = sample_data['bbox']
            save_data['bbox'] = {
                'center_y': bbox['center_y'],
                'center_z': bbox['center_z'],
                'width': bbox['width'],
                'height': bbox['height'],
                'angle': bbox['angle'],
            }
        else:
            save_data['bbox'] = None
        
        # Save sample
        save_path = split_dir / f'sample_{idx:06d}.npz'
        np.savez_compressed(save_path, **save_data)
    
    # Save metadata
    with open(split_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved {num_samples} samples to {split_dir}")
    
    # Print statistics
    print("\nDataset statistics:")
    visible_count = 0
    for idx in range(num_samples):
        data = np.load(split_dir / f'sample_{idx:06d}.npz', allow_pickle=True)
        if data['net_visible']:
            visible_count += 1
    
    print(f"  Net visible: {visible_count}/{num_samples} ({100*visible_count/num_samples:.1f}%)")


def generate_all_splits(
    output_dir='datasets/bbox_detection',
    train_samples=5000,
    val_samples=500,
    test_samples=500
):
    """
    Generate train, validation, and test splits.
    """
    output_dir = Path(output_dir)
    
    print("=" * 80)
    print("GENERATING BBOX DETECTION DATASET")
    print("=" * 80)
    
    # Train set (scenes 0-49)
    generate_bbox_dataset(
        output_dir=output_dir,
        num_samples=train_samples,
        scene_id_start=0,
        split_name='train'
    )
    
    print()
    
    # Validation set (scenes 1000-1009)
    generate_bbox_dataset(
        output_dir=output_dir,
        num_samples=val_samples,
        scene_id_start=1000,
        split_name='val'
    )
    
    print()
    
    # Test set (scenes 2000-2009)
    generate_bbox_dataset(
        output_dir=output_dir,
        num_samples=test_samples,
        scene_id_start=2000,
        split_name='test'
    )
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Dataset saved to: {output_dir}")
    print(f"  Train: {train_samples} samples")
    print(f"  Val:   {val_samples} samples")
    print(f"  Test:  {test_samples} samples")


if __name__ == '__main__':
    generate_all_splits(
        output_dir='datasets/bbox_detection',
        train_samples=5000,
        val_samples=500,
        test_samples=500
    )
