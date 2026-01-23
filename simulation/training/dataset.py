"""PyTorch Dataset for heatmap-based net detection."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_generator import generate_heatmap_target, generate_direction_target, apply_rotation_augmentation
from config import SONAR_CONFIG, DATA_GEN_CONFIG


class SonarNetDataset(Dataset):
    """Dataset for net detection with heatmap targets."""
    
    def __init__(self, data_dir: str, split='train', augment=True, coordconv=True):
        """
        Args:
            data_dir: Path to dataset root directory
            split: 'train', 'val', or 'test'
            augment: Whether to apply augmentation
            coordconv: Whether to include coordinate channels
        """
        self.data_dir = Path(data_dir) / split
        self.augment = augment and (split == 'train')
        self.coordconv = coordconv
        
        # Get all sample files
        self.sample_files = sorted(list(self.data_dir.glob("sample_*.npz")))
        
        # Load metadata
        metadata_path = self.data_dir / 'dataset_metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        self.range_m = SONAR_CONFIG['range_m']
        self.fov_deg = SONAR_CONFIG['fov_deg']
        self.image_size = DATA_GEN_CONFIG['image_size']
        self.heatmap_sigma = DATA_GEN_CONFIG['heatmap_sigma_pixels']
        self.direction_radius = DATA_GEN_CONFIG['direction_radius_pixels']
        
        print(f"Loaded {len(self.sample_files)} samples from {split}")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx: int):
        """Get a single sample with heatmap targets."""
        # Load sample
        sample_path = self.sample_files[idx]
        data = np.load(sample_path)
        
        # Convert to dict for augmentation
        sample_dict = {
            'image': data['image'],
            'valid_mask': data['valid_mask'],
            'y_map': data['y_map'],
            'z_map': data['z_map'],
            'p': data['p'] if data['net_visible'] else None,
            't': data['t'] if data['net_visible'] else None,
            'net_visible': bool(data['net_visible']),
            'distance_m': float(data['distance_m']) if data['net_visible'] else None,
            'orientation_deg': float(data['orientation_deg']) if data['net_visible'] else None,
            'scene_id': int(data['scene_id']),
            'sample_id': int(data['sample_id']),
        }
        
        # Apply augmentation
        if self.augment:
            sample_dict = self._augment(sample_dict)
        
        # Build input tensor
        input_channels = [sample_dict['image'], sample_dict['valid_mask']]
        if self.coordconv:
            input_channels.extend([sample_dict['y_map'], sample_dict['z_map']])
        
        input_tensor = torch.from_numpy(np.stack(input_channels, axis=0)).float()
        
        # Generate targets
        if sample_dict['net_visible']:
            py, pz = sample_dict['p']
            ty, tz = sample_dict['t']
            
            heatmap = generate_heatmap_target(
                py, pz, 
                image_size=self.image_size,
                sigma=self.heatmap_sigma,
                range_m=self.range_m,
                fov_deg=self.fov_deg
            )
            
            # Mask heatmap by valid region
            heatmap = heatmap * sample_dict['valid_mask']
            
            direction_map, direction_mask = generate_direction_target(
                py, pz, ty, tz,
                image_size=self.image_size,
                radius=self.direction_radius,
                range_m=self.range_m,
                fov_deg=self.fov_deg
            )
            
            # Also mask direction supervision by valid region
            direction_mask = direction_mask * sample_dict['valid_mask']
            
        else:
            # No net visible - zero targets
            heatmap = np.zeros(self.image_size, dtype=np.float32)
            direction_map = np.zeros((2,) + self.image_size, dtype=np.float32)
            direction_mask = np.zeros(self.image_size, dtype=np.float32)
        
        # Return as dictionary for easier training loop handling
        return {
            'input': input_tensor,
            'heatmap': torch.from_numpy(heatmap).unsqueeze(0).float(),  # (1, H, W)
            'direction': torch.from_numpy(direction_map).float(),        # (2, H, W)
            'direction_mask': torch.from_numpy(direction_mask).float(),  # (H, W)
            'visibility': torch.tensor(sample_dict['net_visible']).float().unsqueeze(0),  # (1,)
            'hitpoint': torch.from_numpy(sample_dict['p']).float() if sample_dict['p'] is not None else torch.zeros(2),
        }
    
    def _augment(self, sample_dict):
        """Apply augmentation to sample."""
        # Rotation augmentation (most important)
        if np.random.rand() < 0.8:
            angle = np.random.uniform(-20, 20)
            sample_dict = apply_rotation_augmentation(sample_dict, angle)
        
        # Intensity augmentations
        image = sample_dict['image']
        
        # Random gain
        if np.random.rand() < 0.7:
            gain = np.random.uniform(0.5, 2.0)
            image = np.clip(image * gain, 0, 1)
        
        # Gamma perturbation
        if np.random.rand() < 0.5:
            gamma = np.random.uniform(0.7, 1.4)
            image = np.power(image, gamma)
        
        # Additive noise
        if np.random.rand() < 0.6:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)
        
        # Speckle noise
        if np.random.rand() < 0.5:
            speckle = np.random.gamma(5.0, 1.0/5.0, image.shape).astype(np.float32)
            image = np.clip(image * speckle, 0, 1)
        
        sample_dict['image'] = image
        
        return sample_dict


def collate_fn(batch):
    """Custom collate function for DataLoader to batch dictionaries."""
    return {
        'input': torch.stack([item['input'] for item in batch]),
        'heatmap': torch.stack([item['heatmap'] for item in batch]),
        'direction': torch.stack([item['direction'] for item in batch]),
        'direction_mask': torch.stack([item['direction_mask'] for item in batch]),
        'visibility': torch.stack([item['visibility'] for item in batch]),
        'hitpoint': torch.stack([item['hitpoint'] for item in batch]),
    }
