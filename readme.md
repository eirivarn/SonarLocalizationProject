# SOLAQUA — Sonar Image Analysis for Aquaculture

This repository contains tools and notebooks for processing sonar data, extracting net distance estimates from sonar imagery, and comparing them with other measurements.

**Key Features:**
- **Signal-strength independent** image analysis using binary preprocessing
- **Elliptical AOI tracking** with temporal smoothing for stable net detection
- **Real-time distance measurement** with pixel-to-meter conversion
- **Interactive comparison** between sonar and DVL measurements
- **Automated video generation** with synchronized overlays

---

## Quick Start

1. **Setup Environment:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Export Data from ROS Bags:**
```bash
# Export CSVs, videos, and NPZ cones into the exports/ directory
python3 scripts/solaqua_export.py --data-dir /Volumes/LaCie/SOLAQUA/raw_data \
    --exports-dir /Volumes/LaCie/SOLAQUA/exports --all
```

3. **Run Image Analysis:**
```bash
jupyter lab 06_image_analysis.ipynb
```

The notebook demonstrates:
- Loading NPZ sonar data
- Analyzing net distance over time using computer vision
- Converting pixel distances to real-world meters
- Interactive comparison between sonar and DVL measurements
- Generating synchronized overlay videos

---

## Repository Structure

### Core Scripts
- `scripts/solaqua_export.py` — High-level export tool for CSV/Parquet, MP4, PNG sequences, and NPZ cones
- `scripts/interactive_kernel_visualizer.py` — Interactive sonar visualization tool

### Analysis Pipeline (`utils/`)
- `sonar_image_analysis.py` — **Core image analysis pipeline** with binary preprocessing, contour detection, and elliptical tracking
- `sonar_utils.py` — Sonar enhancement, TVG compensation, and polar→Cartesian rasterization
- `net_distance_analysis.py` — DVL and navigation data loaders for comparison
- `sonar_config.py` — Configuration parameters for all processing stages
- `dataset_export_utils.py` — ROS bag parsing and data export utilities

### Notebooks
- `06_image_analysis.ipynb` — **Main analysis notebook** with complete pipeline demonstration
- `01_Sonar_Visualizer.ipynb` — Basic sonar visualization
- `02_Ping360_Visualizer.ipynb` — Ping360-specific visualization
- `03_rov_navigation_analysis.ipynb` — Navigation analysis
- `04_synchronized_sonar_net_distance_analysis.ipynb` — Synchronized analysis
- `05_LEGACY_video_with_sonar_display.ipynb` — Legacy video generation
- `07_pipeline_visualization.ipynb` — Pipeline visualization

### Documentation
- `docs/sonar_pipeline_explanation.md` — **Detailed technical documentation** of the complete analysis pipeline
- `docs/dataset_runs.md` — Dataset-specific analysis results
- `README.md` — This file

---

## Advanced Image Analysis Pipeline

The SOLAQUA system uses a sophisticated computer vision pipeline for robust net detection:

### 1. Signal-Independent Preprocessing
- **Binary conversion** eliminates signal strength variability
- **Adaptive morphological enhancement** detects linear structures
- **Edge detection** without traditional Canny pipeline

### 2. Contour Detection & Selection
- **Geometric scoring** favors elongated contours typical of nets
- **AOI overlap requirements** ensure temporal consistency
- **Spatial penalties** prevent false detections

### 3. Elliptical AOI Tracking
- **Temporal smoothing** prevents erratic ellipse movement
- **Movement constraints** limit center jumps between frames
- **Parameter smoothing** maintains stable ellipse shape

### 4. Distance Measurement
- **Pixel-to-meter conversion** using NPZ extent metadata
- **Stability constraints** prevent measurement jumps
- **Real-time analysis** across video sequences

### 5. DVL Integration
- **Timestamp synchronization** between sonar and navigation data
- **Interactive comparison** with Plotly visualizations
- **Statistical analysis** of sonar vs DVL correlations

**Configuration:** All parameters are tunable in `utils/sonar_config.py`

---

## Configuration & Tuning

### Image Processing Parameters (`IMAGE_PROCESSING_CONFIG`)
```python
'binary_threshold': 128,              # Binary conversion threshold
'adaptive_base_radius': 2,            # Base kernel radius for enhancement
'adaptive_max_elongation': 4,         # Maximum kernel elongation
'min_contour_area': 100,              # Minimum contour size
```

### Tracking Parameters (`TRACKING_CONFIG`)
```python
'use_elliptical_aoi': True,           # Enable elliptical tracking
'ellipse_expansion_factor': 0.3,      # AOI expansion (30%)
'center_smoothing_alpha': 0.1,        # Center smoothing (lower = smoother)
'ellipse_max_movement_pixels': 2.0,   # Maximum center movement per frame
```

### Data Directories
- **Raw Data:** Configure `DATA_DIR` in notebooks or `sonar_config.py`
- **Exports:** Default to `exports/` with subdirectories for organization
- **External Drives:** Support for `/Volumes/LaCie/SOLAQUA/` layout

---

## Developer Notes

- **Pixel→Meter Conversion:** Uses NPZ extent metadata for geometric accuracy
- **Temporal Stability:** Elliptical AOI tracking prevents measurement jumps
- **Signal Independence:** Binary preprocessing works across different sonar conditions
- **Performance:** Optimized for real-time processing with cached kernels

### Extending the Pipeline
- Modify `SonarDataProcessor` class for custom analysis
- Add new contour features in `compute_contour_features()`
- Tune parameters in `sonar_config.py` for different environments

---

## Dependencies

**Required:**
- Python 3.10+
- `numpy`, `pandas`, `opencv-python`
- `plotly`, `matplotlib`
- `scipy`, `scikit-image`
- `rosbags` (ROS bag parsing)

**Optional:**
- `jupyterlab` (notebooks)
- `pyarrow` (Parquet support)

Install with: `pip install -r requirements.txt`

---

## License

MIT — see `LICENSE` for the full text.
