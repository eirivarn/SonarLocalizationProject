"""
Training script for sonar net detection using heatmap-based approach.

Implements:
- U-Net with GroupNorm for small batch stability
- Focal loss for sparse heatmap supervision
- Cosine loss for rotation-invariant direction
- Curriculum learning with 3 phases
- Mixed precision training
- Comprehensive evaluation metrics
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet import UNet
from training.losses import NetDetectionLoss
from training.dataset import SonarNetDataset, collate_fn
from training.metrics import compute_metrics, visualize_predictions
import config


class Trainer:
    """Main trainer class for net detection."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / f"run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Save args
        with open(self.output_dir / "args.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Setup model
        self.model = UNet(
            n_channels=4,  # intensity, valid_mask, y_map, z_map
            num_groups=8,
            base_channels=32
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup loss
        self.criterion = NetDetectionLoss(
            heatmap_weight=args.heatmap_weight,
            direction_weight=args.direction_weight,
            visibility_weight=args.visibility_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if args.mixed_precision else None
        
        # Setup datasets (must be done before scheduler)
        self.setup_datasets()
        
        # Setup learning rate scheduler (cosine with warmup)
        # Note: created after setup_datasets() which sets args.steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * args.steps_per_epoch,
            pct_start=args.warmup_pct,
            anneal_strategy='cos'
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
        self.current_epoch = 0
        
    def setup_datasets(self):
        """Setup training and validation datasets."""
        dataset_path = Path(config.DATASET_PATH)
        
        print(f"Loading datasets from: {dataset_path}")
        
        # Create datasets using train/val directory structure
        self.train_dataset = SonarNetDataset(
            data_dir=str(dataset_path),
            split='train',
            augment=True
        )
        
        self.val_dataset = SonarNetDataset(
            data_dir=str(dataset_path),
            split='val',
            augment=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Update steps per epoch
        self.args.steps_per_epoch = len(self.train_loader)
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': [],
            'heatmap_loss': [],
            'direction_loss': [],
            'visibility_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input'].to(self.device)
            
            # Build target dict
            targets = {
                'heatmap': batch['heatmap'].to(self.device),
                'direction': batch['direction'].to(self.device),
                'direction_mask': batch['direction_mask'].to(self.device),
                'net_visible': batch['visibility'].to(self.device).squeeze(-1)
            }
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss, metrics = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss, metrics = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        return avg_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        epoch_metrics = {
            'loss': [],
            'heatmap_loss': [],
            'direction_loss': [],
            'visibility_loss': []
        }
        
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            inputs = batch['input'].to(self.device)
            
            # Build target dict
            targets = {
                'heatmap': batch['heatmap'].to(self.device),
                'direction': batch['direction'].to(self.device),
                'direction_mask': batch['direction_mask'].to(self.device),
                'net_visible': batch['visibility'].to(self.device).squeeze(-1)
            }
            
            # Forward pass
            outputs = self.model(inputs)
            loss, metrics = self.criterion(outputs, targets)
            
            # Track metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # Store predictions and targets for detailed metrics
            all_predictions.append({
                'heatmap': outputs['heatmap'].cpu(),
                'direction': outputs['direction'].cpu(),
                'visibility': outputs['visibility'].cpu(),
                'hitpoint': batch['hitpoint']
            })
            all_targets.append({
                'heatmap': targets['heatmap'].cpu(),
                'direction': targets['direction'].cpu(),
                'visibility': targets['net_visible'].cpu(),
                'hitpoint': batch['hitpoint']
            })
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        
        # Compute detailed metrics (PCK, angular error, etc.)
        detailed_metrics = compute_metrics(all_predictions, all_targets)
        avg_metrics.update(detailed_metrics)
        
        return avg_metrics, all_predictions, all_targets
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'args': vars(self.args)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print("-" * 80)
        
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            print(f"\nEpoch {epoch} - Train:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # Validate
            if (epoch + 1) % self.args.val_interval == 0:
                val_metrics, predictions, targets = self.validate()
                self.val_history.append(val_metrics)
                
                print(f"\nEpoch {epoch} - Validation:")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Save checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", is_best=is_best)
                
                # Visualize predictions
                if (epoch + 1) % self.args.viz_interval == 0:
                    viz_path = self.viz_dir / f"epoch_{epoch}.png"
                    visualize_predictions(
                        predictions,
                        targets,
                        save_path=viz_path,
                        num_samples=8
                    )
            
            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Results saved to: {self.output_dir}")




def main():
    parser = argparse.ArgumentParser(description="Train sonar net detection model")
    
    # Data
    parser.add_argument('--output_dir', type=str, default='outputs/training',
                        help='Output directory for checkpoints and logs')
    
    # Model
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_pct', type=float, default=0.1,
                        help='Percentage of training for warmup')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    # Loss weights
    parser.add_argument('--heatmap_weight', type=float, default=1.0,
                        help='Weight for heatmap loss')
    parser.add_argument('--direction_weight', type=float, default=0.5,
                        help='Weight for direction loss')
    parser.add_argument('--visibility_weight', type=float, default=0.2,
                        help='Weight for visibility loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    
    # Augmentation
    parser.add_argument('--rotation_range', type=float, default=20.0,
                        help='Rotation augmentation range (degrees)')
    parser.add_argument('--gain_range', nargs=2, type=float, default=[0.5, 2.0],
                        help='Gain augmentation range')
    parser.add_argument('--gamma_range', nargs=2, type=float, default=[0.7, 1.4],
                        help='Gamma augmentation range')
    parser.add_argument('--noise_std', type=float, default=0.02,
                        help='Additive noise std')
    parser.add_argument('--speckle_std', type=float, default=0.1,
                        help='Speckle noise std')
    
    # Logging
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--viz_interval', type=int, default=10,
                        help='Visualize predictions every N epochs')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Dummy args for compatibility
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='Steps per epoch (set automatically)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create trainer and train
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
