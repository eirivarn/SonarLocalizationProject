"""
Bounding box detection module for net detection in sonar images.

Exports:
- Dataset generation: generate_bbox_dataset
- Dataset loading: BBoxDataset, collate_bbox_batch
- Bbox computation: compute_bbox_from_segmentation, polar_to_cartesian_segmentation
- Training: train_bbox_model (from train.py)
"""
from .bbox_from_segmentation import (
    polar_to_cartesian_segmentation,
    compute_bbox_from_segmentation,
    create_scene_with_materials,
    visualize_bbox_sample
)
from .dataset import BBoxDataset, collate_bbox_batch
from .generate_dataset import generate_bbox_dataset

__all__ = [
    # Bbox computation
    'polar_to_cartesian_segmentation',
    'compute_bbox_from_segmentation',
    'create_scene_with_materials',
    'visualize_bbox_sample',
    # Dataset
    'BBoxDataset',
    'collate_bbox_batch',
    'generate_bbox_dataset',
]
