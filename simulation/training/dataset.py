"""
PyTorch Dataset for sonar net detection.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict


class SonarNetDataset(Dataset):
    """
    PyTorch Dataset for sonar net detection.
    
    Loads pre-generated sonar images and labels from disk.
    """
    
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Path to dataset directory (train/val/test)
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all samples
        self.image_files = sorted(self.data_dir.glob("sample_*.npy"))
        self.label_files = sorted(self.data_dir.glob("sample_*.json"))
        
        assert len(self.image_files) == len(self.label_files), \
            "Mismatch between number of images and labels"
        
        # Filter to only samples where net is visible
        valid_indices = []
        for idx in range(len(self.label_files)):
            with open(self.label_files[idx], 'r') as f:
                label_data = json.load(f)
            # Only include if net is visible AND both distance and orientation are not None
            if (label_data.get('net_visible', False) and 
                label_data['distance_m'] is not None and
                label_data['orientation_deg'] is not None):
                valid_indices.append(idx)
        
        self.image_files = [self.image_files[i] for i in valid_indices]
        self.label_files = [self.label_files[i] for i in valid_indices]
        
        print(f"Loaded {len(self.image_files)} samples from {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get a single sample.
        
        Returns:
            image: (1, H, W) tensor
            label: dict with 'distance' and 'orientation' (sin, cos)
        """
        # Load image
        image = np.load(self.image_files[idx])  # (1024, 256)
        
        # Load label
        with open(self.label_files[idx], 'r') as f:
            label_data = json.load(f)
        
        # Convert to tensor
        image = torch.from_numpy(image).unsqueeze(0)  # (1, 1024, 256)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert orientation to sin/cos encoding
        orientation_rad = np.radians(label_data['orientation_deg'])
        
        label = {
            'distance': torch.tensor(label_data['distance_m'], dtype=torch.float32),
            'orientation_sin': torch.tensor(np.sin(orientation_rad), dtype=torch.float32),
            'orientation_cos': torch.tensor(np.cos(orientation_rad), dtype=torch.float32),
            'orientation_deg': torch.tensor(label_data['orientation_deg'], dtype=torch.float32),
        }
        
        return image, label
    
    def get_metadata(self, idx: int) -> Dict:
        """Get full metadata for a sample."""
        with open(self.label_files[idx], 'r') as f:
            return json.load(f)


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of (image, label) tuples
        
    Returns:
        images: (B, 1, H, W) tensor
        labels: dict with (B,) tensors
    """
    images = torch.stack([item[0] for item in batch])
    
    labels = {
        'distance': torch.stack([item[1]['distance'] for item in batch]),
        'orientation_sin': torch.stack([item[1]['orientation_sin'] for item in batch]),
        'orientation_cos': torch.stack([item[1]['orientation_cos'] for item in batch]),
        'orientation_deg': torch.stack([item[1]['orientation_deg'] for item in batch]),
    }
    
    return images, labels


if __name__ == '__main__':
    """Test dataset loading."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DATASET_DIR
    
    # Test loading
    train_dir = DATASET_DIR / 'train'
    
    if not train_dir.exists():
        print(f"Dataset not found at {train_dir}")
        print("Run data_generator.py first!")
        sys.exit(1)
    
    dataset = SonarNetDataset(train_dir)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    image, label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Distance: {label['distance']:.2f}m")
    print(f"  Orientation: {label['orientation_deg']:.1f}°")
    print(f"  Orientation (sin/cos): ({label['orientation_sin']:.3f}, {label['orientation_cos']:.3f})")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    print(f"\nTesting DataLoader:")
    images, labels = next(iter(loader))
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch distances: {labels['distance']}")
    print(f"  Batch orientations: {labels['orientation_deg']}")
    
    print("\n✓ Dataset loading works!")
