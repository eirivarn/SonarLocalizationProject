"""
Custom loss functions for net detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetDetectionLoss(nn.Module):
    """
    Combined loss for distance and orientation prediction.
    
    Loss = w_dist * MSE(distance) + w_orient * CircularLoss(orientation)
    """
    
    def __init__(self, distance_weight=1.0, orientation_weight=1.0):
        super().__init__()
        self.distance_weight = distance_weight
        self.orientation_weight = orientation_weight
    
    def forward(self, pred, target):
        """
        Args:
            pred: dict with 'distance' (B,) and 'orientation' (B, 2) [sin, cos]
            target: dict with 'distance' (B,), 'orientation_sin' (B,), 'orientation_cos' (B,)
            
        Returns:
            loss: scalar tensor
            metrics: dict with individual loss components
        """
        # Distance loss (MSE)
        distance_loss = F.mse_loss(pred['distance'], target['distance'])
        
        # Orientation loss (circular MSE on sin/cos)
        pred_sin = pred['orientation'][:, 0]
        pred_cos = pred['orientation'][:, 1]
        
        target_sin = target['orientation_sin']
        target_cos = target['orientation_cos']
        
        # MSE on both sin and cos
        orientation_loss = F.mse_loss(pred_sin, target_sin) + F.mse_loss(pred_cos, target_cos)
        
        # Combined loss
        total_loss = (
            self.distance_weight * distance_loss +
            self.orientation_weight * orientation_loss
        )
        
        metrics = {
            'loss': total_loss.item(),
            'distance_loss': distance_loss.item(),
            'orientation_loss': orientation_loss.item(),
        }
        
        return total_loss, metrics


def compute_orientation_error(pred_sin, pred_cos, target_sin, target_cos):
    """
    Compute angular error in degrees.
    
    Args:
        pred_sin, pred_cos: Predicted sin/cos (B,)
        target_sin, target_cos: Target sin/cos (B,)
        
    Returns:
        errors: Angular errors in degrees (B,)
    """
    # Compute predicted and target angles
    pred_angle = torch.atan2(pred_sin, pred_cos)
    target_angle = torch.atan2(target_sin, target_cos)
    
    # Compute angular difference (handle wraparound)
    diff = pred_angle - target_angle
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    # Convert to degrees
    errors = torch.abs(torch.rad2deg(diff))
    
    return errors


def compute_metrics(pred, target):
    """
    Compute evaluation metrics.
    
    Args:
        pred: dict with 'distance' and 'orientation' [sin, cos]
        target: dict with 'distance', 'orientation_sin', 'orientation_cos'
        
    Returns:
        metrics: dict with MAE for distance and orientation
    """
    # Distance MAE
    distance_mae = torch.mean(torch.abs(pred['distance'] - target['distance']))
    
    # Orientation MAE
    orientation_errors = compute_orientation_error(
        pred['orientation'][:, 0],
        pred['orientation'][:, 1],
        target['orientation_sin'],
        target['orientation_cos']
    )
    orientation_mae = torch.mean(orientation_errors)
    
    metrics = {
        'distance_mae': distance_mae.item(),
        'orientation_mae': orientation_mae.item(),
    }
    
    return metrics


if __name__ == '__main__':
    """Test loss functions."""
    
    # Create dummy predictions and targets
    batch_size = 4
    
    pred = {
        'distance': torch.tensor([1.5, 2.0, 3.5, 1.0]),
        'orientation': torch.tensor([
            [0.0, 1.0],   # 0°
            [0.707, 0.707],  # 45°
            [1.0, 0.0],   # 90°
            [-0.707, 0.707], # 135°
        ])
    }
    
    target = {
        'distance': torch.tensor([1.6, 2.1, 3.4, 0.9]),
        'orientation_sin': torch.tensor([0.0, 0.707, 1.0, -0.707]),
        'orientation_cos': torch.tensor([1.0, 0.707, 0.0, 0.707]),
    }
    
    # Test loss
    criterion = NetDetectionLoss(distance_weight=1.0, orientation_weight=1.0)
    loss, loss_metrics = criterion(pred, target)
    
    print("Loss Metrics:")
    print(f"  Total loss: {loss:.4f}")
    print(f"  Distance loss: {loss_metrics['distance_loss']:.4f}")
    print(f"  Orientation loss: {loss_metrics['orientation_loss']:.4f}")
    
    # Test metrics
    metrics = compute_metrics(pred, target)
    
    print("\nEvaluation Metrics:")
    print(f"  Distance MAE: {metrics['distance_mae']:.4f}m")
    print(f"  Orientation MAE: {metrics['orientation_mae']:.4f}°")
    
    # Test orientation error computation
    errors = compute_orientation_error(
        pred['orientation'][:, 0],
        pred['orientation'][:, 1],
        target['orientation_sin'],
        target['orientation_cos']
    )
    print(f"\nIndividual orientation errors: {errors}")
    
    print("\n✓ Loss functions work!")
