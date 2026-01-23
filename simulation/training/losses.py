"""Loss functions for heatmap-based net detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for heatmap prediction."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted logits
            target: (B, 1, H, W) target heatmap [0, 1]
        """
        pred = pred.sigmoid()
        
        # Focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt).pow(self.gamma)
        
        # BCE
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal loss
        loss = self.alpha * focal_weight * bce
        
        return loss.mean()


class CosineDirectionLoss(nn.Module):
    """Cosine-based loss for direction prediction with masking."""
    
    def __init__(self, norm_penalty=0.1):
        super().__init__()
        self.norm_penalty = norm_penalty
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, 2, H, W) predicted direction
            target: (B, 2, H, W) target direction
            mask: (B, H, W) supervision mask
        """
        # Only compute loss where mask is 1
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Normalize predictions
        pred_norm = torch.sqrt((pred ** 2).sum(dim=1, keepdim=True) + 1e-6)
        pred_normalized = pred / pred_norm
        
        # Cosine similarity
        cosine_sim = (pred_normalized * target).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Cosine loss (1 - similarity)
        cosine_loss = (1 - cosine_sim) * mask
        cosine_loss = cosine_loss.sum() / mask.sum()
        
        # Unit norm penalty
        norm_loss = ((pred_norm - 1) ** 2) * mask
        norm_loss = norm_loss.sum() / mask.sum()
        
        return cosine_loss + self.norm_penalty * norm_loss


class NetDetectionLoss(nn.Module):
    """Combined loss for heatmap + direction + visibility."""
    
    def __init__(self, 
                 heatmap_weight=1.0,
                 direction_weight=0.5,
                 visibility_weight=0.2,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.direction_weight = direction_weight
        self.visibility_weight = visibility_weight
        
        self.heatmap_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.direction_loss_fn = CosineDirectionLoss(norm_penalty=0.1)
        self.visibility_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: dict with 'heatmap', 'direction', 'visibility'
            target: dict with 'heatmap', 'direction', 'direction_mask', 'net_visible'
        """
        # Heatmap loss
        heatmap_loss = self.heatmap_loss_fn(pred['heatmap'], target['heatmap'])
        
        # Direction loss (masked)
        direction_loss = self.direction_loss_fn(
            pred['direction'], 
            target['direction'], 
            target['direction_mask']
        )
        
        # Visibility loss
        visibility_loss = self.visibility_loss_fn(
            pred['visibility'].squeeze(-1),
            target['net_visible']
        )
        
        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.direction_weight * direction_loss +
            self.visibility_weight * visibility_loss
        )
        
        metrics = {
            'loss': total_loss.item(),
            'heatmap_loss': heatmap_loss.item(),
            'direction_loss': direction_loss.item(),
            'visibility_loss': visibility_loss.item(),
        }
        
        return total_loss, metrics
