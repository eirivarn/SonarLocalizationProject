"""
Baseline CNN model for net detection.

Simple architecture:
- Input: (1, 1024, 256) grayscale sonar image
- Output: [distance_m, orientation_deg]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """Simple CNN for net distance and orientation regression."""
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = {
                'conv_channels': [16, 32, 64, 128],
                'kernel_size': 3,
                'pool_size': 2,
                'dropout': 0.2,
                'fc_dims': [256, 128],
            }
        
        self.config = config
        in_channels = 1  # Grayscale
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for out_channels in config['conv_channels']:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=config['kernel_size'], 
                         padding=config['kernel_size']//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(config['pool_size']),
                nn.Dropout2d(config['dropout'])
            ))
            in_channels = out_channels
        
        # Calculate size after convolutions
        # Starting: 1024 x 256
        # After each pool(2): divide by 2
        h = 1024 // (2 ** len(config['conv_channels']))
        w = 256 // (2 ** len(config['conv_channels']))
        flatten_size = config['conv_channels'][-1] * h * w
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_in = flatten_size
        for fc_out in config['fc_dims']:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_in, fc_out),
                nn.ReLU(inplace=True),
                nn.Dropout(config['dropout'])
            ))
            fc_in = fc_out
        
        # Output heads
        self.distance_head = nn.Linear(fc_in, 1)   # Distance in meters
        self.orientation_head = nn.Linear(fc_in, 2)  # sin(θ), cos(θ) encoding
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, 1024, 256) sonar images
            
        Returns:
            distance: (B, 1) distance in meters
            orientation: (B, 2) [sin(θ), cos(θ)]
        """
        # Convolutional feature extraction
        for conv in self.conv_layers:
            x = conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
        
        # Output heads
        distance = self.distance_head(x).squeeze(-1)  # (B,)
        orientation = self.orientation_head(x)  # (B, 2)
        
        # Normalize orientation to unit circle
        orientation = F.normalize(orientation, p=2, dim=1)
        
        return {
            'distance': distance,
            'orientation': orientation
        }
    
    def predict(self, x):
        """
        Convenience method that returns interpretable outputs.
        
        Args:
            x: (B, 1, 1024, 256) sonar images
            
        Returns:
            distance_m: (B,) distance in meters
            orientation_deg: (B,) orientation in degrees [-90, 90)
        """
        outputs = self.forward(x)
        distance = outputs['distance']
        orientation = outputs['orientation']
        
        # Distance is already in meters
        distance_m = distance
        
        # Convert sin/cos to degrees
        orientation_rad = torch.atan2(orientation[:, 0], orientation[:, 1])
        orientation_deg = torch.rad2deg(orientation_rad)
        
        return distance_m, orientation_deg


def test_model():
    """Test the baseline model."""
    model = BaselineCNN()
    
    # Test input
    x = torch.randn(4, 1, 1024, 256)
    
    # Forward pass
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Distance output shape: {outputs['distance'].shape}")
    print(f"Orientation output shape: {outputs['orientation'].shape}")
    
    # Test predict method
    dist_m, orient_deg = model.predict(x)
    print(f"\nPredicted distances: {dist_m}")
    print(f"Predicted orientations: {orient_deg}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params:,}")


if __name__ == '__main__':
    test_model()
