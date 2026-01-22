"""
Evaluation and testing utilities.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline import BaselineCNN
from training.dataset import SonarNetDataset, collate_fn
from training.losses import compute_metrics
from config import DATASET_DIR


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    model = BaselineCNN()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_dataset(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    
    all_pred_distances = []
    all_true_distances = []
    all_pred_orientations = []
    all_true_orientations = []
    all_distance_errors = []
    all_orientation_errors = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            # Move to device
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward pass
            pred = model(images)
            
            # Get predictions
            pred_distance, pred_orientation = model.predict(pred)
            
            # Store results
            all_pred_distances.extend(pred_distance.cpu().numpy())
            all_true_distances.extend(labels['distance'].cpu().numpy())
            all_pred_orientations.extend(pred_orientation.cpu().numpy())
            all_true_orientations.extend(labels['orientation_deg'].cpu().numpy())
            
            # Compute errors
            distance_errors = torch.abs(pred_distance - labels['distance'])
            all_distance_errors.extend(distance_errors.cpu().numpy())
            
            # Orientation errors (handle wraparound)
            orientation_diff = pred_orientation - labels['orientation_deg']
            orientation_diff = torch.atan2(
                torch.sin(torch.deg2rad(orientation_diff)),
                torch.cos(torch.deg2rad(orientation_diff))
            )
            orientation_errors = torch.abs(torch.rad2deg(orientation_diff))
            all_orientation_errors.extend(orientation_errors.cpu().numpy())
    
    # Convert to numpy
    results = {
        'pred_distances': np.array(all_pred_distances),
        'true_distances': np.array(all_true_distances),
        'pred_orientations': np.array(all_pred_orientations),
        'true_orientations': np.array(all_true_orientations),
        'distance_errors': np.array(all_distance_errors),
        'orientation_errors': np.array(all_orientation_errors),
    }
    
    # Compute statistics
    stats = {
        'distance_mae': np.mean(results['distance_errors']),
        'distance_rmse': np.sqrt(np.mean(results['distance_errors']**2)),
        'distance_median': np.median(results['distance_errors']),
        'distance_max': np.max(results['distance_errors']),
        'orientation_mae': np.mean(results['orientation_errors']),
        'orientation_rmse': np.sqrt(np.mean(results['orientation_errors']**2)),
        'orientation_median': np.median(results['orientation_errors']),
        'orientation_max': np.max(results['orientation_errors']),
    }
    
    return results, stats


def plot_results(results, stats, output_path=None):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distance: Predicted vs True
    ax = axes[0, 0]
    ax.scatter(results['true_distances'], results['pred_distances'], alpha=0.5, s=10)
    min_val = min(results['true_distances'].min(), results['pred_distances'].min())
    max_val = max(results['true_distances'].max(), results['pred_distances'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
    ax.set_xlabel('True Distance (m)')
    ax.set_ylabel('Predicted Distance (m)')
    ax.set_title('Distance: Predicted vs True')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance Error Distribution
    ax = axes[0, 1]
    ax.hist(results['distance_errors'], bins=50, edgecolor='black')
    ax.axvline(stats['distance_mae'], color='r', linestyle='--', label=f"MAE: {stats['distance_mae']:.3f}m")
    ax.axvline(stats['distance_median'], color='g', linestyle='--', label=f"Median: {stats['distance_median']:.3f}m")
    ax.set_xlabel('Distance Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Distance Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance Error vs True Distance
    ax = axes[0, 2]
    ax.scatter(results['true_distances'], results['distance_errors'], alpha=0.5, s=10)
    ax.axhline(stats['distance_mae'], color='r', linestyle='--', label=f"MAE: {stats['distance_mae']:.3f}m")
    ax.set_xlabel('True Distance (m)')
    ax.set_ylabel('Distance Error (m)')
    ax.set_title('Distance Error vs True Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Orientation: Predicted vs True
    ax = axes[1, 0]
    ax.scatter(results['true_orientations'], results['pred_orientations'], alpha=0.5, s=10)
    ax.plot([0, 360], [0, 360], 'r--', label='Perfect')
    ax.set_xlabel('True Orientation (°)')
    ax.set_ylabel('Predicted Orientation (°)')
    ax.set_title('Orientation: Predicted vs True')
    ax.set_xlim([0, 360])
    ax.set_ylim([0, 360])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Orientation Error Distribution
    ax = axes[1, 1]
    ax.hist(results['orientation_errors'], bins=50, edgecolor='black')
    ax.axvline(stats['orientation_mae'], color='r', linestyle='--', label=f"MAE: {stats['orientation_mae']:.1f}°")
    ax.axvline(stats['orientation_median'], color='g', linestyle='--', label=f"Median: {stats['orientation_median']:.1f}°")
    ax.set_xlabel('Orientation Error (°)')
    ax.set_ylabel('Count')
    ax.set_title('Orientation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Orientation Error vs True Orientation
    ax = axes[1, 2]
    ax.scatter(results['true_orientations'], results['orientation_errors'], alpha=0.5, s=10)
    ax.axhline(stats['orientation_mae'], color='r', linestyle='--', label=f"MAE: {stats['orientation_mae']:.1f}°")
    ax.set_xlabel('True Orientation (°)')
    ax.set_ylabel('Orientation Error (°)')
    ax.set_title('Orientation Error vs True Orientation')
    ax.set_xlim([0, 360])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, device)
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train loss: {checkpoint.get('train_metrics', {}).get('loss', 'unknown')}")
    print(f"  Val loss: {checkpoint.get('val_metrics', {}).get('loss', 'unknown')}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = SonarNetDataset(DATASET_DIR / args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"  Samples: {len(dataset)}")
    
    # Evaluate
    print("\nEvaluating...")
    results, stats = evaluate_dataset(model, loader, device)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset: {args.split} ({len(dataset)} samples)")
    print(f"\nDistance Metrics:")
    print(f"  MAE:    {stats['distance_mae']:.4f}m")
    print(f"  RMSE:   {stats['distance_rmse']:.4f}m")
    print(f"  Median: {stats['distance_median']:.4f}m")
    print(f"  Max:    {stats['distance_max']:.4f}m")
    print(f"\nOrientation Metrics:")
    print(f"  MAE:    {stats['orientation_mae']:.2f}°")
    print(f"  RMSE:   {stats['orientation_rmse']:.2f}°")
    print(f"  Median: {stats['orientation_median']:.2f}°")
    print(f"  Max:    {stats['orientation_max']:.2f}°")
    print("=" * 80)
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save stats
        with open(output_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save full results
        np.savez(
            output_dir / 'results.npz',
            **results
        )
        
        # Plot
        plot_results(results, stats, output_dir / 'evaluation.png')
        
        print(f"\n✓ Results saved to {output_dir}")
    else:
        plot_results(results, stats)


if __name__ == '__main__':
    main()
