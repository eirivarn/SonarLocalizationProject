"""
Dataset for bounding box detection training.

Can either generate samples on-the-fly or load from pre-generated dataset.
"""import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_bbox_world import generate_real_sample_with_bbox


class BBoxDataset(Dataset):
    """
    Dataset that provides sonar images with bounding box labels.
    
    Can work in two modes:
    1. Load from saved dataset (fast, recommended)
    2. Generate on-the-fly from simulation (slow, for testing)
    """
    
    def __init__(self, num_samples=None, scene_id_start=0, transform=None, 
                 dataset_dir=None, split='train'):
        """
        Args:
            num_samples: Number of samples (for on-the-fly generation)
            scene_id_start: Starting scene ID (for on-the-fly generation)
            transform: Optional transforms to apply
            dataset_dir: Directory with saved dataset (if None, generate on-the-fly)
            split: 'train', 'val', or 'test' (used if dataset_dir is provided)
        """
        self.transform = transform
        self.use_saved = dataset_dir is not None
        
        if self.use_saved:
            # Load from saved dataset
            self.dataset_dir = Path(dataset_dir) / split
            if not self.dataset_dir.exists():
                raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
            
            # Count samples
            self.sample_files = sorted(self.dataset_dir.glob('sample_*.npz'))
            self.num_samples = len(self.sample_files)
            
            if self.num_samples == 0:
                raise ValueError(f"No samples found in {self.dataset_dir}")
            
            print(f"Loaded {self.num_samples} samples from {self.dataset_dir}")
        else:
            # Generate on-the-fly
            if num_samples is None:
                raise ValueError("num_samples required for on-the-fly generation")
            
            self.num_samples = num_samples
            self.scene_id_start = scene_id_start
            self.samples_per_scene = 100
            
            print(f"Dataset will generate {self.num_samples} samples on-the-fly")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample with image and bbox label.
        
        Returns:
            image: (1, 512, 512) tensor
            target: dict with bbox parameters
        """
        if self.use_saved:
            return self._load_saved_sample(idx)
        else:
            return self._generate_sample(idx)
    
    def _load_saved_sample(self, idx):
        """Load pre-generated sample from disk."""
        # Load data
        data = np.load(self.sample_files[idx], allow_pickle=True)
        
        image = data['image']
        net_visible = bool(data['net_visible'])
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / image.max()
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, 512, 512)
        
        # Prepare target
        if net_visible:
            bbox = data['bbox'].item()  # Extract dict from numpy array
            
            # Convert angle to sin/cos encoding
            angle_rad = np.deg2rad(bbox['angle'])
            sin_angle = np.sin(angle_rad)
            cos_angle = np.cos(angle_rad)
            
            target = {
                'net_visible': torch.tensor(True),
                'center': torch.tensor([bbox['center_y'], bbox['center_z']], dtype=torch.float32),
                'size': torch.tensor([bbox['width'], bbox['height']], dtype=torch.float32),
                'angle': torch.tensor([sin_angle, cos_angle], dtype=torch.float32),
            }
        else:
            target = {
                'net_visible': torch.tensor(False),
                'center': torch.zeros(2, dtype=torch.float32),
                'size': torch.zeros(2, dtype=torch.float32),
                'angle': torch.tensor([0.0, 1.0], dtype=torch.float32),
            }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def _generate_sample(self, idx):
        """Generate sample on-the-fly from simulation."""
        # Determine scene_id and sample_id
        scene_id = self.scene_id_start + (idx // self.samples_per_scene)
        sample_id = idx % self.samples_per_scene
        
        # Generate sample from simulation
        sample_data = generate_real_sample_with_bbox(scene_id, sample_id)
        
        # Extract image
        image = sample_data['image']
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / image.max()
        
        # Convert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, 512, 512)
        
        # Prepare target
        net_visible = sample_data['net_visible']
        
        if net_visible:
            bbox = sample_data['bbox']
            
            # Convert angle to sin/cos encoding
            angle_rad = np.deg2rad(bbox['angle'])
            sin_angle = np.sin(angle_rad)
            cos_angle = np.cos(angle_rad)
            
            target = {
                'net_visible': torch.tensor(True),
                'center': torch.tensor([bbox['center_y'], bbox['center_z']], dtype=torch.float32),
                'size': torch.tensor([bbox['width'], bbox['height']], dtype=torch.float32),
                'angle': torch.tensor([sin_angle, cos_angle], dtype=torch.float32),
            }
        else:
            target = {
                'net_visible': torch.tensor(False),
                'center': torch.zeros(2, dtype=torch.float32),
                'size': torch.zeros(2, dtype=torch.float32),
                'angle': torch.tensor([0.0, 1.0], dtype=torch.float32),
            }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def collate_bbox_batch(batch):
    """
    Custom collate function to handle dict targets.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        images: (B, 1, 512, 512) tensor
        targets: dict with batched tensors
    """
    images = torch.stack([item[0] for item in batch])
    
    targets = {
        'net_visible': torch.stack([item[1]['net_visible'] for item in batch]),
        'center': torch.stack([item[1]['center'] for item in batch]),
        'size': torch.stack([item[1]['size'] for item in batch]),
        'angle': torch.stack([item[1]['angle'] for item in batch]),
    }
    
    return images, targets
