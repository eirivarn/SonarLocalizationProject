# SOLAQUA — Sonar Image Analysis for Aquaculture

This repository contains tools and notebooks for processing sonar data, extracting net distance estimates from sonar imagery, and comparing them with other measurements.

**Key Features:**
- **Signal-strength independent** image analysis using binary preprocessing
- **Elliptical AOI tracking** with temporal smoothing for stable net detection
- **Real-time distance measurement** with pixel-to-meter conversion
- **Interactive comparison** between sonar and DVL measurements
- **Automated video generation** with synchronized overlays

---

## Usage Guide

### Standard Analysis Workflow

1. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Export from ROS Bags**
   
   Export sonar, navigation, and sensor data to CSV/NPZ formats:
   ```bash
   python3 scripts/solaqua_export.py \
     --data-dir /path/to/raw_data \
     --exports-dir /path/to/exports \
     --all
   ```
   
   Or export specific components:
   ```bash
   # Export only NPZ sonar cones
   python3 scripts/solaqua_export.py --data-dir ./raw_data --npz
   
   # Export CSV tables
   python3 scripts/solaqua_export.py --data-dir ./raw_data --csv
   
   # Generate video outputs
   python3 scripts/solaqua_export.py --data-dir ./raw_data --video
   ```

3. **Run Analysis Notebooks**
   
   Launch Jupyter and open analysis notebooks:
   ```bash
   jupyter lab
   ```
   
   Key notebooks:
   - `06_full_net_analysis.ipynb` — Complete pipeline with net tracking and DVL comparison
   - `04_image_analysis.ipynb` — Image processing and visualization
   - `01_Sonar_Visualizer.ipynb` — Interactive sonar data exploration

4. **Generate Distance Measurements**
   
   Compute net distance from exported data:
   ```bash
   python3 scripts/compute_net_distance.py \
     --npz-dir exports/npz_cones \
     --output outputs/net_distance.csv
   ```

### ROS Integration

**Live ROS1 Streaming** (real-time net detection):
```bash
# Inside ROS container
roscore  # Terminal 1
rosbag play --clock bags/your_bag.bag  # Terminal 2
rosrun solaqua_tools sonar_live_net_node.py  # Terminal 3
```
See [ros_ws/README_sonar_live.md](ros_ws/README_sonar_live.md) for complete setup.

**ROS2 Container Deployment**:
```bash
# Build Docker image
docker build -f Dockerfile -t solaqua-ros2 .

# Run with data mounts
docker run --rm -it --network host \
  -v /path/to/raw_data:/workspace/raw_data \
  -v /path/to/exports:/workspace/exports \
  solaqua-ros2:latest bash

# Inside container: inspect and play bags
ros2 bag info /workspace/raw_data/your_bag.bag
ros2 bag play /workspace/raw_data/your_bag.bag --clock
```
See [docs/ros2_container_streaming.md](docs/ros2_container_streaming.md) for detailed ROS2 workflows.

### Documentation

- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** — Comprehensive command reference and workflow examples
- **[sonar_pipeline_explanation.md](docs/sonar_pipeline_explanation.md)** — Complete technical documentation of the sonar processing pipeline, algorithms, and implementation
- **[ANALYSIS_METHODOLOGY.md](docs/ANALYSIS_METHODOLOGY.md)** — Multi-system comparison methodology for FFT, Sonar, and DVL measurements
- **[dataset_runs.md](docs/dataset_runs.md)** — Dataset catalog with timestamps and experiment descriptions
- **[ros2_container_streaming.md](docs/ros2_container_streaming.md)** — ROS2 deployment and bag processing guide
- **[rov_fft_guide.md](docs/rov_fft_guide.md)** — FFT-based relative pose estimation workflow

---

## Repository Overview

The workspace contains analysis notebooks (numbered), processing scripts in `scripts/`, core utilities in `utils/`, and comprehensive documentation in `docs/`. Raw data and ROS bags are stored in dedicated directories, while processed outputs go to `exports/` and `outputs/`

---

## Technical Details

The SOLAQUA system uses a sophisticated computer vision pipeline for robust net detection:

- **Signal-Independent Preprocessing:** Binary conversion eliminates signal strength variability
- **Contour Detection & Selection:** Geometric scoring favors elongated structures with temporal consistency
- **Elliptical AOI Tracking:** Temporal smoothing prevents erratic movement and maintains stable detection
- **Distance Measurement:** Pixel-to-meter conversion using metadata with stability constraints
- **DVL Integration:** Timestamp synchronization for comparison with navigation data

Source to `utils/` for implementation details and `docs/` for comprehensive technical documentation.

**Configuration:** All processing parameters are tunable via the configuration module in `utils/`

---

## Dependencies

- Python 3.10+
- Core: `numpy`, `pandas`, `opencv-python`, `scipy`, `scikit-image`
- Visualization: `plotly`, `matplotlib`
- ROS: `rosbags` for bag parsing
- Optional: `jupyterlab`, `pyarrow`

Install with: `pip install -r requirements.txt`

---

## License

MIT — see LICENSE for details.
