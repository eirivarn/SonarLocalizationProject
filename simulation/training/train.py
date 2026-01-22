"""
Training script for sonar net detection models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import json
import time
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline import BaselineCNN
from training.dataset import SonarNetDataset, collate_fn
from training.losses import NetDetectionLoss, compute_metrics
from config import DATASET_DIR, TRAINING_CONFIG


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_distance_mae = 0
    total_orientation_mae = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for images, labels in pbar:
        # Move to device
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(images)
        
        # Compute loss
        loss, loss_metrics = criterion(pred, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        metrics = compute_metrics(pred, labels)
        
        # Update stats
        total_loss += loss_metrics['loss']
        total_distance_mae += metrics['distance_mae']
        total_orientation_mae += metrics['orientation_mae']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_metrics['loss']:.4f}",
            'dist_mae': f"{metrics['distance_mae']:.3f}m",
            'orient_mae': f"{metrics['orientation_mae']:.1f}°"
        })
    
    n = len(loader)
    return {
        'loss': total_loss / n,
        'distance_mae': total_distance_mae / n,
        'orientation_mae': total_orientation_mae / n,
    }


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_distance_mae = 0
    total_orientation_mae = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            # Move to device
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward pass
            pred = model(images)
            
            # Compute loss
            loss, loss_metrics = criterion(pred, labels)
            
            # Compute metrics
            metrics = compute_metrics(pred, labels)
            
            # Update stats
            total_loss += loss_metrics['loss']
            total_distance_mae += metrics['distance_mae']
            total_orientation_mae += metrics['orientation_mae']
    
    n = len(loader)
    return {
        'loss': total_loss / n,
        'distance_mae': total_distance_mae / n,
        'orientation_mae': total_orientation_mae / n,
    }


def train(args):
    """Main training loop."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'runs')
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SonarNetDataset(DATASET_DIR / 'train')
    val_dataset = SonarNetDataset(DATASET_DIR / 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    if args.model == 'baseline':
        model = BaselineCNN()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print(f"  Model: {args.model}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = NetDetectionLoss(
        distance_weight=args.distance_weight,
        orientation_weight=args.orientation_weight
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_distance_mae = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Log metrics
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Distance MAE/train', train_metrics['distance_mae'], epoch)
        writer.add_scalar('Distance MAE/val', val_metrics['distance_mae'], epoch)
        writer.add_scalar('Orientation MAE/train', train_metrics['orientation_mae'], epoch)
        writer.add_scalar('Orientation MAE/val', val_metrics['orientation_mae'], epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dist MAE: {train_metrics['distance_mae']:.3f}m, "
              f"Orient MAE: {train_metrics['orientation_mae']:.1f}°")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dist MAE: {val_metrics['distance_mae']:.3f}m, "
              f"Orient MAE: {val_metrics['orientation_mae']:.1f}°")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')
        
        # Save best (by validation loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, output_dir / 'best_loss.pth')
            print(f"  → Saved best loss model ({val_metrics['loss']:.4f})")
        
        # Save best (by distance MAE)
        if val_metrics['distance_mae'] < best_distance_mae:
            best_distance_mae = val_metrics['distance_mae']
            torch.save(checkpoint, output_dir / 'best_distance.pth')
            print(f"  → Saved best distance model ({val_metrics['distance_mae']:.3f}m)")
    
    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best distance MAE: {best_distance_mae:.3f}m")
    print(f"  Models saved to: {output_dir}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train sonar net detection model')
    
    # Model
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline'], help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['num_epochs'],
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--distance-weight', type=float, default=1.0,
                        help='Weight for distance loss')
    parser.add_argument('--orientation-weight', type=float, default=1.0,
                        help='Weight for orientation loss')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 80)
    print("SONAR NET DETECTION - TRAINING")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
