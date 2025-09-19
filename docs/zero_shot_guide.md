# Zero-Shot Models for Sonar Analysis - Getting Started Guide

## Overview

This guide shows you how to apply zero-shot models to your sonar cone data. Zero-shot models can analyze images without needing specific training data for your domain.

## What You Already Have

✅ **Sonar cone data**: Your `.npz` files contain sonar data converted to cone format  
✅ **Cone processing functions**: `utils/sonar_utils.py` has functions to create cone images  
✅ **Classical analysis**: Your existing blob detection in `utils/image_analysis_utils.py`

## What's New

🆕 **Zero-shot utilities**: `utils/zero_shot_utils.py` provides access to modern AI models  
🆕 **Demo notebook**: `07_zero_shot_analysis.ipynb` shows complete examples  

## Quick Start

### 1. Install Dependencies

```bash
# Basic requirements (for CLIP and object detection)
pip install transformers torch torchvision

# Optional: For SAM (Segment Anything Model)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Simple CLIP Example

```python
import utils.image_analysis_utils as iau
import utils.zero_shot_utils as zsu

# Load your cone data
cones, ts, extent, meta = iau.load_cone_run_npz("exports/outputs/sonar_cones_run.npz")

# Pick a frame to analyze
test_frame = cones[100]  # or any frame index

# Define what you're looking for
marine_queries = ["fish", "rock", "debris", "empty water", "marine animal"]

# Run CLIP analysis
result = zsu.clip_score_cone_objects(test_frame, marine_queries)

# See results
best_match = result["best_matches"][0]
print(f"This looks like: {best_match['best_text']} (confidence: {best_match['best_score']:.2f})")
```

### 3. Analyze Entire Dataset

```python
# Run on all frames
df_results = zsu.analyze_cone_run_with_zero_shot(
    "exports/outputs/sonar_cones_run.npz",
    model_type="clip",
    text_queries=marine_queries,
    save_results_csv="exports/outputs/clip_results.csv",
    save_visualization_mp4="exports/outputs/clip_video.mp4"
)

print(f"Analyzed {len(df_results)} frames")
print(df_results['best_text'].value_counts())  # What was found
```

## Available Models

### 1. **CLIP (Text-Image Matching)**
- **What it does**: Tells you how well your image matches text descriptions
- **Good for**: "What is this?" questions
- **Example queries**: "fish", "rock formation", "underwater debris"
- **Output**: Confidence scores for each text description

### 2. **SAM (Segment Anything)**
- **What it does**: Finds object boundaries/masks
- **Good for**: Precise object segmentation
- **Input**: Click points or bounding boxes
- **Output**: Binary masks showing object boundaries
- **Note**: Requires downloading model checkpoints

### 3. **Object Detection (DETR)**
- **What it does**: Finds and labels objects with bounding boxes
- **Good for**: "Where are the objects?" questions
- **Output**: Bounding boxes + class labels + confidence scores

## Comparison with Classical Methods

| Method | Pros | Cons |
|--------|------|------|
| **Classical** (your existing) | Fast, no dependencies, tunable | Needs parameter tuning, limited to intensity patterns |
| **CLIP** | Natural language queries, no training needed | Requires internet models, may not understand sonar well |
| **SAM** | Very accurate segmentation | Needs manual prompts, large models |
| **Object Detection** | Automatic object finding | Limited to trained object classes |

## Typical Workflow

1. **Start with classical**: Use your existing blob detection for initial analysis
2. **Add CLIP**: Classify the regions found by classical methods
3. **Refine with SAM**: Get precise boundaries for interesting objects
4. **Compare results**: See where methods agree/disagree

## Example: Hybrid Approach

```python
# 1. Classical detection finds regions
classical_df = iau.analyze_run_largest_blob(
    "exports/outputs/sonar_cones_run.npz",
    blur_ksize=70, thr_percentile=80
)

# 2. CLIP classifies the whole image context
clip_df = zsu.analyze_cone_run_with_zero_shot(
    "exports/outputs/sonar_cones_run.npz",
    model_type="clip",
    text_queries=["fish", "rock", "empty water"]
)

# 3. Find frames where both methods detect something interesting
classical_frames = set(classical_df['t_idx'])
interesting_clip = clip_df[clip_df['best_text'] != 'empty water']
interesting_frames = set(interesting_clip['t_idx'])

# Frames detected by both methods
high_confidence = classical_frames & interesting_frames
print(f"High confidence detections: {len(high_confidence)} frames")
```

## Tips for Marine Sonar Data

### Good CLIP Queries for Sonar:
- **Objects**: "fish", "school of fish", "marine animal", "rock", "debris"
- **Patterns**: "sonar echo", "acoustic shadow", "backscatter"
- **Context**: "seafloor", "water column", "empty water"
- **Specific**: "fish school sonar reflection" vs just "fish"

### Prompt Engineering:
- Be specific: "sonar image of fish" vs "fish"
- Try variations: "school of fish", "fish group", "fish swarm"
- Include negatives: test against "empty water", "noise"

### Integration Strategies:
1. **Filter then classify**: Classical detection → CLIP classification
2. **Classify then segment**: CLIP finds interesting frames → SAM segments objects
3. **Ensemble**: Combine scores from multiple approaches

## Performance Notes

- **CLIP**: ~1-2 seconds per frame on CPU, ~0.1s on GPU
- **SAM**: ~3-5 seconds per frame, requires manual prompts
- **Object Detection**: ~0.5-1 second per frame
- **Classical**: ~0.01 seconds per frame

## Next Steps

1. **Try the demo notebook**: Open `07_zero_shot_analysis.ipynb`
2. **Install dependencies**: Start with just `transformers torch`
3. **Test on your data**: Use your existing NPZ files
4. **Compare methods**: See how zero-shot compares to your classical approach
5. **Refine queries**: Adjust text descriptions based on results

## Troubleshooting

**"transformers not found"**: Run `pip install transformers torch`
**"segment_anything not found"**: SAM is optional, CLIP works without it
**"CUDA out of memory"**: Use CPU mode or smaller models
**"Bad results"**: Try different text queries or combine with classical methods

The key insight is that zero-shot models give you a new vocabulary for describing what's in your sonar data, complementing your existing intensity-based analysis!
