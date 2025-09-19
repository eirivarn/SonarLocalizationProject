# zero_shot_utils.py
"""
Zero-shot model utilities for sonar cone analysis.
Supports SAM (Segment Anything Model), CLIP, and other zero-shot models.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import json

import numpy as np
import pandas as pd
import cv2

# Optional imports for different zero-shot models
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline, CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# ============================ Model Management ============================

class ZeroShotModelManager:
    """Manages loading and caching of zero-shot models."""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
    
    def load_sam_model(self, model_type: str = "vit_b", checkpoint_path: Optional[str] = None) -> Optional[Any]:
        """Load SAM (Segment Anything Model)."""
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
        
        key = f"sam_{model_type}"
        if key not in self.models:
            if checkpoint_path is None:
                # You'll need to download SAM checkpoints manually
                raise ValueError("SAM checkpoint path required. Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.models[key] = SamPredictor(sam)
        
        return self.models[key]
    
    def load_clip_model(self, model_name: str = "openai/clip-vit-base-patch32") -> Tuple[Any, Any]:
        """Load CLIP model for image-text similarity."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        if model_name not in self.models:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            self.models[model_name] = model
            self.processors[model_name] = processor
        
        return self.models[model_name], self.processors[model_name]
    
    def load_detection_pipeline(self, model_name: str = "facebook/detr-resnet-50") -> Any:
        """Load object detection pipeline."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        if model_name not in self.models:
            pipe = pipeline("object-detection", model=model_name)
            self.models[model_name] = pipe
        
        return self.models[model_name]

# Global model manager instance
model_manager = ZeroShotModelManager()

# ============================ SAM Integration ============================

def sam_segment_cone(
    cone_img: np.ndarray,
    *,
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    input_points: Optional[List[Tuple[int, int]]] = None,
    input_labels: Optional[List[int]] = None,
    input_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    multimask_output: bool = True
) -> Dict[str, Any]:
    """
    Use SAM to segment objects in a sonar cone image.
    
    Args:
        cone_img: Cone image as float32 array [0,1] or uint8 [0,255]
        model_type: SAM model type ("vit_b", "vit_l", "vit_h")
        checkpoint_path: Path to SAM checkpoint
        input_points: List of (x, y) points to prompt segmentation
        input_labels: 1 for foreground, 0 for background points
        input_boxes: List of (x1, y1, x2, y2) bounding boxes
        multimask_output: Whether to return multiple mask predictions
    
    Returns:
        Dict with masks, scores, and logits
    """
    predictor = model_manager.load_sam_model(model_type, checkpoint_path)
    
    # Convert to RGB uint8 if needed
    if cone_img.dtype == np.float32:
        img_rgb = (np.clip(cone_img, 0, 1) * 255).astype(np.uint8)
    else:
        img_rgb = cone_img.astype(np.uint8)
    
    # SAM expects 3-channel RGB
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    # Set image for SAM
    predictor.set_image(img_rgb)
    
    # Prepare inputs
    point_coords = np.array(input_points) if input_points else None
    point_labels = np.array(input_labels) if input_labels else None
    box = np.array(input_boxes) if input_boxes else None
    
    # Predict masks
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=multimask_output
    )
    
    return {
        "masks": masks,
        "scores": scores,
        "logits": logits,
        "input_points": input_points,
        "input_labels": input_labels,
        "input_boxes": input_boxes
    }

def sam_auto_segment_cone(
    cone_img: np.ndarray,
    *,
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    points_per_side: int = 16,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.85,
    crop_n_layers: int = 1,
    crop_n_points_downscale_factor: int = 2,
    min_mask_region_area: int = 50
) -> List[Dict[str, Any]]:
    """
    Use SAM's automatic mask generation on a sonar cone.
    
    Returns:
        List of mask dictionaries with segmentation, area, bbox, etc.
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError("segment-anything not available for automatic mask generation")
    
    # Load base SAM model (not predictor)
    if checkpoint_path is None:
        raise ValueError("SAM checkpoint path required")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area
    )
    
    # Convert to RGB uint8 if needed
    if cone_img.dtype == np.float32:
        img_rgb = (np.clip(cone_img, 0, 1) * 255).astype(np.uint8)
    else:
        img_rgb = cone_img.astype(np.uint8)
    
    # SAM expects 3-channel RGB
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    masks = mask_generator.generate(img_rgb)
    return masks

# ============================ CLIP Integration ============================

def clip_score_cone_objects(
    cone_img: np.ndarray,
    text_queries: List[str],
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    masks: Optional[List[np.ndarray]] = None,
    return_features: bool = False
) -> Dict[str, Any]:
    """
    Use CLIP to score how well sonar cone regions match text descriptions.
    
    Args:
        cone_img: Cone image as float32 [0,1] or uint8
        text_queries: List of text descriptions (e.g., ["fish", "debris", "rock"])
        model_name: CLIP model to use
        masks: Optional list of binary masks to score specific regions
        return_features: Whether to return raw image/text features
    
    Returns:
        Dict with similarity scores, best matches, and optionally features
    """
    model, processor = model_manager.load_clip_model(model_name)
    
    # Convert to RGB uint8 if needed with robust NaN handling
    if cone_img.dtype == np.float32:
        # Handle NaN values safely
        clean_img = np.nan_to_num(cone_img, nan=0.0, posinf=1.0, neginf=0.0)
        img_rgb = (np.clip(clean_img, 0, 1) * 255).astype(np.uint8)
    else:
        img_rgb = np.nan_to_num(cone_img, nan=0).astype(np.uint8)
    
    # Ensure RGB format
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    # If no masks provided, score the whole image
    if masks is None:
        images_to_score = [img_rgb]
        mask_info = [{"type": "full_image", "area": img_rgb.shape[0] * img_rgb.shape[1]}]
    else:
        images_to_score = []
        mask_info = []
        
        for i, mask in enumerate(masks):
            # Extract masked region
            if mask.dtype != bool:
                mask = mask.astype(bool)
            
            # Get bounding box of mask
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            
            y1, y2 = rows.min(), rows.max() + 1
            x1, x2 = cols.min(), cols.max() + 1
            
            # Crop to bounding box
            cropped = img_rgb[y1:y2, x1:x2].copy()
            
            # Apply mask within crop
            mask_crop = mask[y1:y2, x1:x2]
            cropped[~mask_crop] = 0  # Set background to black
            
            images_to_score.append(cropped)
            mask_info.append({
                "type": "masked_region",
                "mask_id": i,
                "bbox": (x1, y1, x2, y2),
                "area": int(mask.sum())
            })
    
    # Process all images and texts
    if not images_to_score:
        return {"error": "No valid regions to score"}
    
    inputs = processor(
        text=text_queries,
        images=images_to_score,
        return_tensors="pt",
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Get similarity scores (images x texts)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # Softmax over text queries
    
    # Convert to numpy
    scores = probs.cpu().numpy()  # Shape: (n_images, n_texts)
    
    results = {
        "text_queries": text_queries,
        "scores": scores.tolist(),  # Convert to list for JSON serialization
        "mask_info": mask_info,
        "best_matches": []
    }
    
    # Find best match for each image region
    for i, region_scores in enumerate(scores):
        best_idx = int(np.argmax(region_scores))
        best_score = float(region_scores[best_idx])
        results["best_matches"].append({
            "region_id": i,
            "best_text": text_queries[best_idx],
            "best_score": best_score,
            "all_scores": region_scores.tolist()
        })
    
    if return_features:
        results["image_features"] = outputs.image_embeds.cpu().numpy().tolist()
        results["text_features"] = outputs.text_embeds.cpu().numpy().tolist()
    
    return results

# ============================ Object Detection Integration ============================

def detect_objects_in_cone(
    cone_img: np.ndarray,
    *,
    model_name: str = "facebook/detr-resnet-50",
    confidence_threshold: float = 0.5,
    target_classes: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Use zero-shot object detection on sonar cone images.
    
    Args:
        cone_img: Cone image as float32 [0,1] or uint8
        model_name: HuggingFace model for object detection
        confidence_threshold: Minimum confidence for detections
        target_classes: Optional list of classes to filter for
    
    Returns:
        List of detection dictionaries with bbox, score, label
    """
    detector = model_manager.load_detection_pipeline(model_name)
    
    # Convert to RGB uint8 if needed
    if cone_img.dtype == np.float32:
        img_rgb = (np.clip(cone_img, 0, 1) * 255).astype(np.uint8)
    else:
        img_rgb = cone_img.astype(np.uint8)
    
    # Ensure RGB format
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL Image format for transformers
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    
    # Run detection
    detections = detector(pil_img)
    
    # Filter and format results
    filtered_detections = []
    for det in detections:
        if det["score"] >= confidence_threshold:
            if target_classes is None or det["label"] in target_classes:
                # Convert bbox format if needed
                bbox = det["box"]
                filtered_detections.append({
                    "label": det["label"],
                    "score": float(det["score"]),
                    "bbox": {
                        "x1": int(bbox["xmin"]),
                        "y1": int(bbox["ymin"]),
                        "x2": int(bbox["xmax"]),
                        "y2": int(bbox["ymax"])
                    }
                })
    
    return filtered_detections

# ============================ Analysis Functions ============================

def analyze_cone_run_with_zero_shot(
    npz_path: str | Path,
    *,
    model_type: str = "sam",  # "sam", "clip", "detection"
    model_config: Dict[str, Any] = None,
    text_queries: Optional[List[str]] = None,
    save_results_csv: Optional[str | Path] = None,
    save_visualization_mp4: Optional[str | Path] = None,
    fps: float = 15.0,
    progress: bool = True
) -> pd.DataFrame:
    """
    Apply zero-shot models to a full cone run.
    
    Args:
        npz_path: Path to sonar cone .npz file
        model_type: Type of model ("sam", "clip", "detection")
        model_config: Model-specific configuration
        text_queries: For CLIP, list of text descriptions
        save_results_csv: Optional path to save results CSV
        save_visualization_mp4: Optional path to save overlay video
        fps: Video frame rate
        progress: Show progress bar
    
    Returns:
        DataFrame with analysis results per frame
    """
    # Import here to avoid circular imports
    from .image_analysis_utils import load_cone_run_npz, _cmap_rgb, gray01_to_rgb
    
    # Load cone data
    cones, ts, extent, meta = load_cone_run_npz(npz_path)
    T, H, W = cones.shape
    
    if model_config is None:
        model_config = {}
    
    # Setup visualization if requested
    vw = None
    if save_visualization_mp4:
        Path(save_visualization_mp4).parent.mkdir(parents=True, exist_ok=True)
        lut = _cmap_rgb("viridis")
        rgb0 = gray01_to_rgb(cones[0], lut)
        vw = cv2.VideoWriter(
            str(save_visualization_mp4), 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            fps, 
            (W, H)
        )
    
    results = []
    
    for t in range(T):
        if progress and t % 50 == 0:
            print(f"Processing frame {t}/{T}")
        
        cone = cones[t]
        timestamp = ts[t]
        
        frame_result = {
            "t_idx": t,
            "ts": str(timestamp),
            "model_type": model_type
        }
        
        try:
            if model_type == "sam":
                # SAM segmentation
                sam_result = sam_segment_cone(cone, **model_config)
                frame_result.update({
                    "n_masks": len(sam_result["masks"]),
                    "best_score": float(np.max(sam_result["scores"])) if len(sam_result["scores"]) > 0 else 0.0,
                    "masks_data": sam_result["masks"].tolist() if save_results_csv else None
                })
                
                # Visualization
                if vw is not None:
                    rgb = gray01_to_rgb(cone, lut)
                    # Overlay best mask
                    if len(sam_result["masks"]) > 0:
                        best_mask = sam_result["masks"][np.argmax(sam_result["scores"])]
                        overlay = np.zeros_like(rgb)
                        overlay[best_mask] = (255, 0, 0)  # Red overlay
                        cv2.addWeighted(overlay, 0.3, rgb, 0.7, 0, dst=rgb)
                    vw.write(rgb)
                    
            elif model_type == "clip":
                if text_queries is None:
                    raise ValueError("text_queries required for CLIP analysis")
                
                # CLIP scoring
                clip_result = clip_score_cone_objects(cone, text_queries, **model_config)
                frame_result.update({
                    "text_queries": text_queries,
                    "best_match": clip_result["best_matches"][0] if clip_result["best_matches"] else None,
                    "all_scores": clip_result["scores"][0] if clip_result["scores"] else None
                })
                
                # Visualization
                if vw is not None:
                    rgb = gray01_to_rgb(cone, lut)
                    # Add text overlay with best match
                    if clip_result["best_matches"]:
                        best = clip_result["best_matches"][0]
                        text = f"{best['best_text']}: {best['best_score']:.2f}"
                        cv2.putText(rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    vw.write(rgb)
                    
            elif model_type == "detection":
                # Object detection
                detections = detect_objects_in_cone(cone, **model_config)
                frame_result.update({
                    "n_detections": len(detections),
                    "detections": detections
                })
                
                # Visualization
                if vw is not None:
                    rgb = gray01_to_rgb(cone, lut)
                    # Draw detection boxes
                    for det in detections:
                        bbox = det["bbox"]
                        cv2.rectangle(rgb, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 255, 0), 2)
                        cv2.putText(rgb, f"{det['label']}: {det['score']:.2f}", 
                                  (bbox["x1"], bbox["y1"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    vw.write(rgb)
            
        except Exception as e:
            frame_result["error"] = str(e)
            if vw is not None:
                # Write original frame on error
                rgb = gray01_to_rgb(cone, lut)
                vw.write(rgb)
        
        results.append(frame_result)
    
    if vw is not None:
        vw.release()
        if progress:
            print(f"Saved visualization to {save_visualization_mp4}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if save_results_csv:
        Path(save_results_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_results_csv, index=False)
        if progress:
            print(f"Saved results to {save_results_csv}")
    
    return df

# ============================ Utility Functions ============================

def check_model_availability() -> Dict[str, bool]:
    """Check which zero-shot models are available."""
    return {
        "torch": TORCH_AVAILABLE,
        "transformers": TRANSFORMERS_AVAILABLE,
        "segment_anything": SAM_AVAILABLE
    }

def get_installation_commands() -> Dict[str, str]:
    """Get pip install commands for missing dependencies."""
    return {
        "torch": "pip install torch torchvision",
        "transformers": "pip install transformers",
        "segment_anything": "pip install git+https://github.com/facebookresearch/segment-anything.git"
    }

# ============================ Public API ============================

__all__ = [
    # Model management
    "ZeroShotModelManager",
    "model_manager",
    
    # SAM functions
    "sam_segment_cone",
    "sam_auto_segment_cone",
    
    # CLIP functions
    "clip_score_cone_objects",
    
    # Detection functions
    "detect_objects_in_cone",
    
    # Analysis functions
    "analyze_cone_run_with_zero_shot",
    
    # Utilities
    "check_model_availability",
    "get_installation_commands"
]
