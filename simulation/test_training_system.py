"""
Quick test of the training system with configurable parameters.

Similar to full training but with options for smaller/faster runs.
Use this to verify the pipeline works before running full 200-epoch training.

Example:
    # Quick test (2-3 minutes)
    python test_training_system.py --samples 32 --epochs 3 --batch_size 4
    
    # Medium test (10-15 minutes)
    python test_training_system.py --samples 500 --epochs 10 --batch_size 16
    
    # Full-scale test (similar to real training)
    python test_training_system.py --samples 10000 --epochs 50 --batch_size 8
"""
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from models.unet import UNet
from training.losses import NetDetectionLoss
from training.dataset import SonarNetDataset, collate_fn
from training.metrics import compute_metrics, visualize_predictions
import config


class QuickTrainer:
    """Simplified trainer for quick testing."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        
        print("="*80)
        print("QUICK TRAINING TEST")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Samples: {args.train_samples} train, {args.val_samples} val")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Base channels: {args.base_channels}")
        
        # Setup model
        self.model = UNet(
            n_channels=4,
            num_groups=8,
            base_channels=args.base_channels
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Setup loss
        self.criterion = NetDetectionLoss(
            heatmap_weight=1.0,
            direction_weight=0.5,
            visibility_weight=0.2,
            focal_alpha=0.25,
            focal_gamma=2.0
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup scheduler
        total_steps = args.epochs * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
    
    def setup_datasets(self):
        """Setup train, val, and test datasets with limited samples."""
        dataset_path = Path(config.DATASET_PATH)
        
        # Load full datasets
        full_train = SonarNetDataset(
            data_dir=str(dataset_path),
            split='train',
            augment=self.args.augment
        )
        
        full_val = SonarNetDataset(
            data_dir=str(dataset_path),
            split='val',
            augment=False
        )
        
        full_test = SonarNetDataset(
            data_dir=str(dataset_path),
            split='test',
            augment=False
        )
        
        # Create subsets
        train_indices = list(range(min(self.args.train_samples, len(full_train))))
        val_indices = list(range(min(self.args.val_samples, len(full_val))))
        test_indices = list(range(min(self.args.test_samples, len(full_test))))
        
        self.train_dataset = Subset(full_train, train_indices)
        self.val_dataset = Subset(full_val, val_indices)
        self.test_dataset = Subset(full_test, test_indices)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        print(f"\nDataset:")
        print(f"  Train: {len(self.train_dataset)} samples, {len(self.train_loader)} batches")
        print(f"  Val: {len(self.val_dataset)} samples, {len(self.val_loader)} batches")
        print(f"  Test: {len(self.test_dataset)} samples, {len(self.test_loader)} batches")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': [],
            'heatmap_loss': [],
            'direction_loss': [],
            'visibility_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in pbar:
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = {
                'heatmap': batch['heatmap'].to(self.device),
                'direction': batch['direction'].to(self.device),
                'direction_mask': batch['direction_mask'].to(self.device),
                'net_visible': batch['visibility'].to(self.device).squeeze(-1)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss, metrics = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
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
    
    @torch.no_grad()
    def test_and_visualize(self, output_dir):
        """Test on test set and generate visualizations."""
        import matplotlib.pyplot as plt
        
        self.model.eval()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("TESTING & GENERATING VISUALIZATIONS")
        print("="*80)
        print(f"Output directory: {output_dir}")
        
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        for batch in tqdm(self.test_loader, desc="Testing"):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = {
                'heatmap': batch['heatmap'].to(self.device),
                'direction': batch['direction'].to(self.device),
                'direction_mask': batch['direction_mask'].to(self.device),
                'net_visible': batch['visibility'].to(self.device).squeeze(-1)
            }
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Store for visualization
            all_inputs.append(inputs.cpu())
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
        
        # Compute metrics
        test_metrics = compute_metrics(all_predictions, all_targets)
        
        print(f"\nTest Set Metrics:")
        print(f"  Detection Rate: {test_metrics.get('detection_rate', 0):.1f}%")
        if 'PCK@1m' in test_metrics:
            print(f"  PCK@1m: {test_metrics['PCK@1m']:.1f}%")
            print(f"  PCK@2m: {test_metrics['PCK@2m']:.1f}%")
            print(f"  Distance Error: {test_metrics.get('mean_distance_error', 999):.2f}m")
            print(f"  Angular Error: {test_metrics.get('mean_angular_error', 180):.1f}°")
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        num_viz = min(20, len(self.test_dataset))
        
        from training.metrics import extract_peak_from_heatmap, pixels_to_meters
        
        for idx in tqdm(range(num_viz), desc="Saving images"):
            batch_idx = idx // self.args.batch_size
            in_batch_idx = idx % self.args.batch_size
            
            if batch_idx >= len(all_inputs):
                break
                
            # Get data
            input_tensor = all_inputs[batch_idx][in_batch_idx]
            pred_heatmap = all_predictions[batch_idx]['heatmap'][in_batch_idx]
            pred_direction = all_predictions[batch_idx]['direction'][in_batch_idx]
            target_heatmap = all_targets[batch_idx]['heatmap'][in_batch_idx]
            target_direction = all_targets[batch_idx]['direction'][in_batch_idx]
            target_hitpoint = all_targets[batch_idx]['hitpoint'][in_batch_idx].numpy()
            
            # Extract intensity channel
            intensity = input_tensor[0].numpy()
            
            # Extract predicted peak
            peak_px, conf, valid = extract_peak_from_heatmap(
                pred_heatmap.unsqueeze(0), threshold=0.1
            )
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.patch.set_facecolor('#2a2a2a')
            
            # Coordinate extent
            range_m = 20.0
            fov_deg = 120.0
            z_max = range_m * np.sin(np.deg2rad(fov_deg / 2)) * 1.05
            extent = [-z_max, z_max, 0, range_m]
            
            # Plot 1: Input image with GT direction
            ax = axes[0]
            ax.set_facecolor('#2a2a2a')
            ax.imshow(intensity, cmap='gray', extent=extent, origin='lower', aspect='auto')
            ax.plot(target_hitpoint[1], target_hitpoint[0], 'ro', markersize=8, label='GT Hitpoint')
            
            # Draw GT direction arrow
            gt_dir_y = target_direction[0].item()
            gt_dir_z = target_direction[1].item()
            arrow_scale = 3.0  # meters
            ax.arrow(target_hitpoint[1], target_hitpoint[0],
                    gt_dir_z * arrow_scale, gt_dir_y * arrow_scale,
                    head_width=0.8, head_length=0.5, fc='red', ec='red',
                    alpha=0.7, width=0.2, label='GT Direction')
            
            ax.set_title('Input with GT Direction', color='white', fontsize=12)
            ax.set_xlabel('Z (m)', color='white')
            ax.set_ylabel('Y (m)', color='white')
            ax.tick_params(colors='white')
            ax.legend(loc='upper right', framealpha=0.8)
            ax.grid(True, alpha=0.2, color='white')
            
            # Plot 2: Predicted heatmap with direction
            ax = axes[1]
            ax.set_facecolor('#2a2a2a')
            im = ax.imshow(pred_heatmap[0].numpy(), cmap='hot', extent=extent, 
                          origin='lower', aspect='auto', vmin=0, vmax=1)
            if valid[0]:
                pred_point_m = pixels_to_meters(peak_px[0:1])[0].numpy()
                ax.plot(pred_point_m[1], pred_point_m[0], 'g*', markersize=15, 
                       label=f'Pred (conf={conf[0]:.2f})')
                
                # Draw predicted direction arrow
                pred_dir_y = pred_direction[0].item()
                pred_dir_z = pred_direction[1].item()
                arrow_scale = 3.0  # meters
                ax.arrow(pred_point_m[1], pred_point_m[0],
                        pred_dir_z * arrow_scale, pred_dir_y * arrow_scale,
                        head_width=0.8, head_length=0.5, fc='lime', ec='lime',
                        alpha=0.7, width=0.2, label='Pred Direction')
                
            ax.set_title('Predicted Heatmap & Direction', color='white', fontsize=12)
            ax.set_xlabel('Z (m)', color='white')
            ax.set_ylabel('Y (m)', color='white')
            ax.tick_params(colors='white')
            ax.legend(loc='upper right', framealpha=0.8)
            ax.grid(True, alpha=0.2, color='white')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot 3: Overlay with both directions
            ax = axes[2]
            ax.set_facecolor('#2a2a2a')
            ax.imshow(intensity, cmap='gray', extent=extent, origin='lower', aspect='auto')
            ax.imshow(pred_heatmap[0].numpy(), cmap='hot', alpha=0.5, extent=extent,
                     origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.plot(target_hitpoint[1], target_hitpoint[0], 'ro', markersize=8, label='GT')
            
            # Draw GT direction arrow
            gt_dir_y = target_direction[0].item()
            gt_dir_z = target_direction[1].item()
            arrow_scale = 3.0
            ax.arrow(target_hitpoint[1], target_hitpoint[0],
                    gt_dir_z * arrow_scale, gt_dir_y * arrow_scale,
                    head_width=0.8, head_length=0.5, fc='red', ec='red',
                    alpha=0.7, width=0.2)
            
            if valid[0]:
                pred_point_m = pixels_to_meters(peak_px[0:1])[0].numpy()
                ax.plot(pred_point_m[1], pred_point_m[0], 'g*', markersize=15, label='Pred')
                
                # Draw predicted direction arrow
                pred_dir_y = pred_direction[0].item()
                pred_dir_z = pred_direction[1].item()
                ax.arrow(pred_point_m[1], pred_point_m[0],
                        pred_dir_z * arrow_scale, pred_dir_y * arrow_scale,
                        head_width=0.8, head_length=0.5, fc='lime', ec='lime',
                        alpha=0.7, width=0.2)
                
                # Calculate errors
                error = np.linalg.norm(pred_point_m - target_hitpoint)
                gt_dir_vec = np.array([gt_dir_y, gt_dir_z])
                pred_dir_vec = np.array([pred_dir_y, pred_dir_z])
                angle_error = np.rad2deg(np.arccos(np.clip(np.dot(gt_dir_vec, pred_dir_vec), -1, 1)))
                ax.set_title(f'Overlay | Pos Error: {error:.2f}m | Angle Error: {angle_error:.1f}°', 
                           color='white', fontsize=12)
            else:
                ax.set_title('Overlay | No Detection', color='white', fontsize=12)
            ax.set_xlabel('Z (m)', color='white')
            ax.set_ylabel('Y (m)', color='white')
            ax.tick_params(colors='white')
            ax.legend(loc='upper right', framealpha=0.8)
            ax.grid(True, alpha=0.2, color='white')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'test_sample_{idx:03d}.png', 
                       dpi=100, bbox_inches='tight', facecolor='#2a2a2a')
            plt.close()
        
        print(f"✓ Saved {num_viz} visualizations to {output_dir}")
        return test_metrics
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*80}")
        print("TRAINING")
        print("="*80)
        
        for epoch in range(self.args.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            print(f"\nEpoch {epoch+1} - Train:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Heatmap: {train_metrics['heatmap_loss']:.4f}")
            print(f"  Direction: {train_metrics['direction_loss']:.4f}")
            print(f"  Visibility: {train_metrics['visibility_loss']:.4f}")
            
            # Validate
            if (epoch + 1) % self.args.val_interval == 0 or (epoch + 1) == self.args.epochs:
                val_metrics, predictions, targets = self.validate()
                self.val_history.append(val_metrics)
                
                print(f"\nEpoch {epoch+1} - Validation:")
                print(f"  Loss: {val_metrics['loss']:.4f}")
                print(f"  Detection Rate: {val_metrics.get('detection_rate', 0):.1f}%")
                if 'PCK@1m' in val_metrics:
                    print(f"  PCK@1m: {val_metrics['PCK@1m']:.1f}%")
                    print(f"  PCK@2m: {val_metrics['PCK@2m']:.1f}%")
                    print(f"  Distance Error: {val_metrics.get('mean_distance_error', 999):.2f}m")
                    print(f"  Angular Error: {val_metrics.get('mean_angular_error', 180):.1f}°")
                
                # Track best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    print(f"  ✓ New best validation loss!")
        
        # Final summary
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        if self.val_history:
            final_metrics = self.val_history[-1]
            print(f"\nFinal Metrics:")
            print(f"  Detection Rate: {final_metrics.get('detection_rate', 0):.1f}%")
            if 'PCK@1m' in final_metrics:
                print(f"  PCK@1m: {final_metrics['PCK@1m']:.1f}%")
                print(f"  PCK@2m: {final_metrics['PCK@2m']:.1f}%")
                print(f"  Distance Error: {final_metrics.get('mean_distance_error', 999):.2f}m")
                print(f"  Angular Error: {final_metrics.get('mean_angular_error', 180):.1f}°")
        
        # Test on test set and generate visualizations
        if self.args.generate_visualizations:
            viz_dir = Path('test_visualizations')
            test_metrics = self.test_and_visualize(viz_dir)
        
        print("\n✅ Test successful! Training pipeline is working correctly.")
        print("\nReady for full training with:")
        print("  python training/train.py --epochs 200 --batch_size 8")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Quick test of training system")
    
    # Dataset
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training samples (default: 1000)')
    parser.add_argument('--val_samples', type=int, default=200,
                        help='Number of validation samples (default: 200)')
    parser.add_argument('--test_samples', type=int, default=200,
                        help='Number of test samples for visualization (default: 200)')
    
    # Deprecated argument for backwards compatibility
    parser.add_argument('--samples', type=int, default=None,
                        help='(Deprecated) Sets both train and val samples. Use --train_samples and --val_samples instead')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--base_channels', type=int, default=16,
                        help='Base channels for U-Net (default: 16, use 32 for full size)')
    
    # Options
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validate every N epochs (default: 1)')
    parser.add_argument('--generate_visualizations', action='store_true',
                        help='Generate test set visualizations after training')
    
    args = parser.parse_args()
    
    # Handle deprecated --samples argument
    if args.samples is not None:
        print(f"Warning: --samples is deprecated. Using {args.samples} for train, val, and test.")
        args.train_samples = args.samples
        args.val_samples = args.samples
        args.test_samples = min(50, args.samples)
    
    # Run test
    trainer = QuickTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

