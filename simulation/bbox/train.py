"""
Training script for oriented bounding box detection.

Trains a CNN to predict bounding boxes from sonar images using
ground truth from semantic segmentation.
"""import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.bbox_cnn import BBoxCNN, BBoxLoss
from bbox.dataset import BBoxDataset, collate_bbox_batch
from core import SONAR_CONFIG, DATA_GEN_CONFIG


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_components = {
        'visibility': 0.0,
        'center': 0.0,
        'size': 0.0,
        'angle': 0.0
    }
    
    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        # Move to device
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        
        # Compute loss
        loss_dict = criterion(predictions, targets)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        loss_components['visibility'] += loss_dict['visibility_loss']
        loss_components['center'] += loss_dict['center_loss']
        loss_components['size'] += loss_dict['size_loss']
        loss_components['angle'] += loss_dict['angle_loss']
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Average losses
    num_batches = len(dataloader)
    total_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return total_loss, loss_components


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    loss_components = {
        'visibility': 0.0,
        'center': 0.0,
        'size': 0.0,
        'angle': 0.0
    }
    
    # Metrics
    correct_visibility = 0
    total_samples = 0
    center_errors = []
    size_errors = []
    angle_errors = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, targets in pbar:
            # Move to device
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Accumulate losses
            total_loss += loss.item()
            loss_components['visibility'] += loss_dict['visibility_loss']
            loss_components['center'] += loss_dict['center_loss']
            loss_components['size'] += loss_dict['size_loss']
            loss_components['angle'] += loss_dict['angle_loss']
            
            # Compute metrics
            pred_visible = torch.sigmoid(predictions['visibility']) > 0.5
            correct_visibility += (pred_visible == targets['net_visible']).sum().item()
            total_samples += len(images)
            
            # For visible nets, compute errors
            visible_mask = targets['net_visible']
            if visible_mask.sum() > 0:
                # Center error (Euclidean distance)
                center_pred = predictions['center'][visible_mask]
                center_target = targets['center'][visible_mask]
                center_error = torch.norm(center_pred - center_target, dim=1)
                center_errors.extend(center_error.cpu().numpy())
                
                # Size error (relative)
                size_pred = predictions['size'][visible_mask]
                size_target = targets['size'][visible_mask]
                size_error = torch.abs(size_pred - size_target) / (size_target + 1e-6)
                size_errors.extend(size_error.cpu().numpy().mean(axis=1))
                
                # Angle error (degrees)
                angle_pred = predictions['angle'][visible_mask]
                angle_target = targets['angle'][visible_mask]
                # Convert to angles
                pred_angle_rad = torch.atan2(angle_pred[:, 0], angle_pred[:, 1])
                target_angle_rad = torch.atan2(angle_target[:, 0], angle_target[:, 1])
                angle_error = torch.abs(pred_angle_rad - target_angle_rad)
                # Handle wrap-around
                angle_error = torch.min(angle_error, 2*np.pi - angle_error)
                angle_errors.extend(torch.rad2deg(angle_error).cpu().numpy())
    
    # Average losses
    num_batches = len(dataloader)
    total_loss /= num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    # Compute metrics
    visibility_acc = correct_visibility / total_samples
    
    metrics = {
        'visibility_accuracy': visibility_acc,
        'center_error_mean': np.mean(center_errors) if center_errors else 0.0,
        'center_error_std': np.std(center_errors) if center_errors else 0.0,
        'size_error_mean': np.mean(size_errors) if size_errors else 0.0,
        'angle_error_mean': np.mean(angle_errors) if angle_errors else 0.0,
        'angle_error_std': np.std(angle_errors) if angle_errors else 0.0,
    }
    
    return total_loss, loss_components, metrics


def train_bbox_model(
    num_epochs=50,
    batch_size=16,
    learning_rate=0.001,
    dataset_dir='datasets/bbox_detection',
    save_dir='models/checkpoints',
    device=None
):
    """
    Train the bounding box detection model.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        dataset_dir: Directory with saved dataset (or None to generate on-the-fly)
        save_dir: Directory to save checkpoints
        device: Device to train on (cuda/cpu)
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    if dataset_dir:
        # Load from saved dataset
        train_dataset = BBoxDataset(dataset_dir=dataset_dir, split='train')
        val_dataset = BBoxDataset(dataset_dir=dataset_dir, split='val')
    else:
        # Generate on-the-fly (slower)
        print("WARNING: Generating samples on-the-fly. Consider pre-generating with generate_bbox_dataset.py")
        train_dataset = BBoxDataset(num_samples=5000, scene_id_start=0)
        val_dataset = BBoxDataset(num_samples=500, scene_id_start=1000)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_bbox_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_bbox_batch
    )
    
    # Create model
    print("Creating model...")
    model = BBoxCNN().to(device)
    criterion = BBoxLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_components, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Visibility Acc: {val_metrics['visibility_accuracy']:.3f}")
        print(f"Center Error: {val_metrics['center_error_mean']:.3f} ± {val_metrics['center_error_std']:.3f} m")
        print(f"Angle Error: {val_metrics['angle_error_mean']:.2f} ± {val_metrics['angle_error_std']:.2f}°")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, save_dir / 'best_model.pth')
            print("✓ Saved best model")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, save_dir / 'final_model.pth')
    
    # Save history
    with open(save_dir / 'history.json', 'w') as f:
        # Convert numpy types to Python types for JSON
        history_json = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_metrics': [
                {k: float(v) for k, v in m.items()}
                for m in history['val_metrics']
            ]
        }
        json.dump(history_json, f, indent=2)
    
    # Plot training curves
    plot_training_history(history, save_dir / 'training_curves.png')
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")
    print(f"✓ Models saved to: {save_dir}")


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Visibility accuracy
    vis_acc = [m['visibility_accuracy'] for m in history['val_metrics']]
    axes[0, 1].plot(epochs, vis_acc)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Net Visibility Accuracy')
    axes[0, 1].grid(True)
    
    # Center error
    center_err = [m['center_error_mean'] for m in history['val_metrics']]
    axes[1, 0].plot(epochs, center_err)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Error (m)')
    axes[1, 0].set_title('Center Position Error')
    axes[1, 0].grid(True)
    
    # Angle error
    angle_err = [m['angle_error_mean'] for m in history['val_metrics']]
    axes[1, 1].plot(epochs, angle_err)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error (degrees)')
    axes[1, 1].set_title('Orientation Angle Error')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Train the model
    # First generate dataset: python generate_bbox_dataset.py
    # Then train: python train_bbox.py
    
    train_bbox_model(
        num_epochs=50,
        batch_size=16,
        learning_rate=0.001,
        dataset_dir='datasets/bbox_detection',  # Set to None to generate on-the-fly
    )
