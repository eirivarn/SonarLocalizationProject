"""
CNN model for oriented bounding box detection.

Input: (1, 512, 512) Cartesian sonar image
Output: Bounding box parameters [center_y, center_z, width, height, angle]
        + confidence score for net visibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BBoxCNN(nn.Module):
    """CNN for oriented bounding box regression."""
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = {
                'conv_channels': [32, 64, 128, 256],
                'kernel_size': 3,
                'pool_size': 2,
                'dropout': 0.3,
                'fc_dims': [512, 256],
            }
        
        self.config = config
        in_channels = 1  # Grayscale sonar image
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.ModuleList()
        for out_channels in config['conv_channels']:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=config['kernel_size'], 
                         padding=config['kernel_size']//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 
                         kernel_size=config['kernel_size'], 
                         padding=config['kernel_size']//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(config['pool_size']),
                nn.Dropout2d(config['dropout'])
            ))
            in_channels = out_channels
        
        # Calculate size after convolutions
        # Starting: 512 x 512
        # After each pool(2): divide by 2
        num_pools = len(config['conv_channels'])
        h = 512 // (2 ** num_pools)
        w = 512 // (2 ** num_pools)
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
        self.visibility_head = nn.Linear(fc_in, 1)  # Binary: net visible or not
        self.center_head = nn.Linear(fc_in, 2)      # (center_y, center_z) in meters
        self.size_head = nn.Linear(fc_in, 2)        # (width, height) in meters
        self.angle_head = nn.Linear(fc_in, 2)       # (sin(angle), cos(angle)) encoding
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, 512, 512) Cartesian sonar images
            
        Returns:
            dict with:
                visibility: (B,) logits for net visibility
                center: (B, 2) [center_y, center_z] in meters
                size: (B, 2) [width, height] in meters (positive values)
                angle: (B, 2) [sin(θ), cos(θ)] normalized to unit circle
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
        visibility = self.visibility_head(x).squeeze(-1)  # (B,)
        center = self.center_head(x)  # (B, 2)
        size = F.relu(self.size_head(x)) + 0.1  # (B, 2) - ensure positive, minimum 0.1m
        angle = self.angle_head(x)  # (B, 2)
        
        # Normalize angle to unit circle
        angle = F.normalize(angle, p=2, dim=1)
        
        return {
            'visibility': visibility,
            'center': center,
            'size': size,
            'angle': angle
        }
    
    def predict(self, x):
        """
        Convenience method that returns interpretable bbox parameters.
        
        Args:
            x: (B, 1, 512, 512) sonar images
            
        Returns:
            dict with:
                net_visible: (B,) boolean tensor
                bbox: dict with center_y, center_z, width, height, angle (degrees)
        """
        outputs = self.forward(x)
        
        # Convert visibility logits to boolean
        net_visible = torch.sigmoid(outputs['visibility']) > 0.5
        
        # Convert angle encoding to degrees
        sin_angle = outputs['angle'][:, 0]
        cos_angle = outputs['angle'][:, 1]
        angle_rad = torch.atan2(sin_angle, cos_angle)
        angle_deg = torch.rad2deg(angle_rad)
        
        # Normalize angle to [-90, 90]
        angle_deg = torch.where(angle_deg > 90, angle_deg - 180, angle_deg)
        angle_deg = torch.where(angle_deg < -90, angle_deg + 180, angle_deg)
        
        return {
            'net_visible': net_visible,
            'bbox': {
                'center_y': outputs['center'][:, 0],
                'center_z': outputs['center'][:, 1],
                'width': outputs['size'][:, 0],
                'height': outputs['size'][:, 1],
                'angle': angle_deg
            }
        }


class BBoxLoss(nn.Module):
    """
    Combined loss for bounding box detection.
    
    Losses:
    1. Binary cross-entropy for visibility
    2. Smooth L1 for center position (only if net visible)
    3. Smooth L1 for size (only if net visible)
    4. Cosine similarity for angle (only if net visible)
    """
    
    def __init__(self, weights=None):
        super().__init__()
        
        if weights is None:
            weights = {
                'visibility': 1.0,
                'center': 5.0,
                'size': 3.0,
                'angle': 2.0
            }
        
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from model forward()
            targets: dict with:
                net_visible: (B,) boolean tensor
                center: (B, 2) [center_y, center_z]
                size: (B, 2) [width, height]
                angle: (B, 2) [sin(θ), cos(θ)]
        
        Returns:
            dict with total_loss and individual loss components
        """
        batch_size = predictions['visibility'].shape[0]
        device = predictions['visibility'].device
        
        # Visibility loss (all samples)
        visibility_target = targets['net_visible'].float()
        visibility_loss = self.bce(predictions['visibility'], visibility_target)
        
        # For other losses, only compute on samples where net is visible
        visible_mask = targets['net_visible']
        num_visible = visible_mask.sum()
        
        if num_visible > 0:
            # Center loss
            center_pred = predictions['center'][visible_mask]
            center_target = targets['center'][visible_mask]
            center_loss = self.smooth_l1(center_pred, center_target).mean()
            
            # Size loss
            size_pred = predictions['size'][visible_mask]
            size_target = targets['size'][visible_mask]
            size_loss = self.smooth_l1(size_pred, size_target).mean()
            
            # Angle loss (cosine similarity loss)
            angle_pred = predictions['angle'][visible_mask]
            angle_target = targets['angle'][visible_mask]
            # 1 - cosine similarity (ranges from 0 to 2, 0 = perfect match)
            angle_loss = 1 - F.cosine_similarity(angle_pred, angle_target, dim=1).mean()
        else:
            # No visible nets in batch - set these losses to zero
            center_loss = torch.tensor(0.0, device=device)
            size_loss = torch.tensor(0.0, device=device)
            angle_loss = torch.tensor(0.0, device=device)
        
        # Weighted combination
        total_loss = (
            self.weights['visibility'] * visibility_loss +
            self.weights['center'] * center_loss +
            self.weights['size'] * size_loss +
            self.weights['angle'] * angle_loss
        )
        
        return {
            'total_loss': total_loss,
            'visibility_loss': visibility_loss.item(),
            'center_loss': center_loss.item() if isinstance(center_loss, torch.Tensor) else 0.0,
            'size_loss': size_loss.item() if isinstance(size_loss, torch.Tensor) else 0.0,
            'angle_loss': angle_loss.item() if isinstance(angle_loss, torch.Tensor) else 0.0,
        }
