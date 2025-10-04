# SOLAQUA Technical Documentation

## Complete Sonar Processing Pipeline

This document provides a deep technical explanation of the SOLAQUA sonar processing pipeline, including the mathematical methods, algorithmic choices, and implementation details. It complements the main [README](../readme.md) which focuses on usage and workflows.

**Target Audience:** Developers, researchers, and advanced users who want to understand the underlying algorithms or extend the functionality.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Extraction](#2-data-extraction-from-ros-bags)
3. [Sonar Enhancement](#3-sonar-preprocessing--enhancement)
4. [Polar-to-Cartesian Rasterization](#4-polar-to-cartesian-rasterization)
5. [Image Analysis & Net Detection](#5-image-analysis--net-detection)
6. [Distance Measurement](#6-distance-measurement--pixel-to-meter-conversion)
7. [DVL Data Integration](#7-dvl-data-integration)
8. [Video Generation](#8-synchronized-video-generation)
9. [Configuration Parameters](#9-configuration-parameters)
10. [Code Reference Map](#10-code-reference-map)

---

## 1. Pipeline Overview

The SOLAQUA pipeline transforms raw sonar data into actionable distance measurements through the following stages:

```
ROS Bags → CSV/NPZ Export → Enhancement → Rasterization → 
  → Image Analysis → Distance Estimation → Comparison with DVL → Video Output
```

### Data Flow

1. **Input:** ROS bag files containing:
   - Sonoptix/Ping360 sonar messages (polar coordinates)
   - Nucleus1000 DVL sensor data
   - Camera footage (compressed images)
   - Navigation and guidance data

2. **Processing:**
   - Extract and tabulate all sensor data
   - Apply TVG compensation and enhancement to sonar
   - Rasterize polar sonar data to Cartesian coordinates
   - Detect net structures using computer vision
   - Estimate distances using ellipse fitting
   - Synchronize with DVL measurements

3. **Output:**
   - CSV/Parquet tables for each sensor topic
   - NPZ files with preprocessed cone images
   - Distance measurement time series
   - Comparison plots
   - Annotated videos with overlay metrics

---

## 2. Data Extraction from ROS Bags

### 2.1 Bag File Structure

SOLAQUA datasets typically contain two types of bags per collection session:

- **`*_data.bag`**: Telemetry, sensors, navigation
  - Topics: DVL sensors, sonar, navigation, guidance, USBL, etc.
  
- **`*_video.bag`**: Camera footage
  - Topics: Compressed image streams, camera info

### 2.2 Export Process

**Implementation:** `solaqua_export.py` → `utils/dataset_export_utils.py`

The export process:

1. **Enumerate topics** in each bag using `rosbags` library
2. **Extract messages** topic by topic
3. **Normalize timestamps** to UTC with `ts_utc` field
4. **Parse nested fields** (JSON, arrays) into flat columns
5. **Write CSV/Parquet** files named: `{topic}__{bag_id}_data.{ext}`

**Key Functions:**
- `dataset_export_utils.export_all_topics()` - Main export orchestrator
- `dataset_export_utils.flatten_message()` - Recursively flattens ROS messages
- `sonar_utils.load_df()` - Loads CSV/Parquet with automatic format detection

### 2.3 Sonar Data Schema

Sonoptix messages contain either:

**Option A:** `image` field (nested list)
```python
{
  "t": 1234567890.123,
  "image": [[255, 254, ...], [250, 248, ...], ...]  # HxW array
}
```

**Option B:** `data` field with metadata
```python
{
  "t": 1234567890.123,
  "data": [255, 254, 250, ...],  # Flat array
  "dim_labels": ["range", "beam"],
  "dim_sizes": [1024, 256]
}
```

**Extraction:** `utils/sonar_utils.py::get_sonoptix_frame(df, idx)`

This function:
1. Checks for `image` column first
2. Falls back to `data` + `dim_*` reconstruction
3. Returns `np.ndarray` of shape `(H, W)` where:
   - `H` = range bins (distance samples)
   - `W` = beam angles

---

## 3. Sonar Preprocessing & Enhancement

### 3.1 Why Enhancement?

Raw sonar data suffers from:
- **Geometric spreading loss:** Signal weakens as `1/r²` (amplitude) or `1/r⁴` (power)
- **Water absorption:** Exponential attenuation with distance
- **Dynamic range issues:** Near-field saturates, far-field is weak
- **Variable gain:** Time-varying gain (TVG) applied in hardware

**Goal:** Normalize intensity for consistent visualization and analysis.

### 3.2 Enhancement Pipeline

**Implementation:** `utils/sonar_utils.py::enhance_intensity(M, r0, rmax, ...)`

#### Step 1: Range Grid Construction
```python
H, W = M.shape  # range bins × beam angles
r = np.linspace(range_min_m, range_max_m, H)
```

#### Step 2: TVG Compensation

Two modes available:

**Amplitude TVG:**
```python
gain_tvg = (r / r0) ** 2
```

**Power TVG:**
```python
gain_tvg = (r / r0) ** 4
```

Where `r0` is a small reference distance to avoid division by zero.

#### Step 3: Absorption Correction (Optional)

Apply frequency-dependent absorption:
```python
gain_abs = np.exp(alpha_db_per_m * r / r_max)
```

Where `alpha_db_per_m` is the absorption coefficient in dB/m.

Combined gain:
```python
total_gain = gain_tvg * gain_abs
M_compensated = M * total_gain[:, np.newaxis]  # Broadcast across beams
```

#### Step 4: Logarithmic Scaling

Convert to decibels (zero-aware):
```python
M_db = 20 * np.log10(M_compensated + eps_log)
```

The `eps_log` parameter (default `1e-5`) prevents log(0).

#### Step 5: Robust Normalization

```python
p_low, p_high = np.percentile(M_db[M_db > -inf], [1.0, 99.5])
M_clipped = np.clip(M_db, p_low, p_high)
M_norm = (M_clipped - p_low) / (p_high - p_low)  # → [0, 1]
```

#### Step 6: Gamma Correction

```python
M_enhanced = M_norm ** gamma  # gamma ≈ 0.6-0.9 for visibility
```

**Configuration:** Parameters in `utils/sonar_config.py::ENHANCE_DEFAULTS`

---

## 4. Polar-to-Cartesian Rasterization

### 4.1 Coordinate System

**Polar (native sonar):**
- Rows: Range bins (r)
- Columns: Beam angles (θ)
- Origin: Sonar transducer

**Cartesian (cone display):**
- X-axis: Starboard (positive right)
- Y-axis: Forward
- Origin: ROV center or sonar mounting point

### 4.2 Rasterization Algorithm

**Implementation:** `utils/sonar_utils.py::cone_raster_like_display_cell()`

#### Setup

```python
# Define field of view
θ_left = -FOV_DEG / 2
θ_right = +FOV_DEG / 2
θ_angles = np.linspace(θ_left, θ_right, W)  # W = beam count

# Create output grid
img_h, img_w = 700, 900  # pixels
x_grid = np.linspace(-range_max, range_max, img_w)
y_grid = np.linspace(-range_max, range_max, img_h)
X, Y = np.meshgrid(x_grid, y_grid)
```

#### Inverse Mapping

For each output pixel `(x, y)`:

```python
# Convert to polar
r = np.sqrt(x² + y²)
θ = np.arctan2(x, y)  # Note: atan2(x, y) for sonar convention

# Check bounds
if r < range_min or r > range_max or |θ| > FOV_DEG/2:
    pixel_value = 0  # Outside field of view
else:
    # Map to source indices
    r_idx = (r - range_min) / (range_max - range_min) * H
    θ_idx = (θ - θ_left) / (θ_right - θ_left) * W
    
    # Bilinear interpolation
    pixel_value = interpolate(M_enhanced, r_idx, θ_idx)
```

#### Optimizations

- Uses NumPy vectorization for all pixels simultaneously
- Scipy's `map_coordinates()` for efficient bilinear interpolation
- Pre-computed polar coordinate grids stored in NPZ for reuse

**Output:** Cone image where:
- Apex (transducer) is at bottom center
- Coverage spreads upward in a wedge shape
- Black areas indicate regions outside FOV

### 4.3 NPZ Storage Format

**File:** `exports/outputs/{bag_id}_cones.npz`

```python
{
    "cones": np.ndarray (T, H, W, dtype=float32),  # Time series of cone images
    "ts": np.ndarray (T,),  # Timestamps (UTC)
    "extent": tuple (x_min, x_max, y_min, y_max),  # Spatial bounds in meters
    "meta": dict {
        "fov_deg": float,
        "range_min_m": float,
        "range_max_m": float,
        ...
    }
}
```

**Critical:** The `extent` tuple enables precise pixel→meter conversion for distance measurements.

---

## 5. Image Analysis & Net Detection

### 5.1 Net Structure Characteristics

Underwater fishing nets in sonar imagery typically exhibit:
- **Elongated shape:** High aspect ratio contours
- **Consistent orientation:** Major axis approximately horisontal
- **Area of Interest (AOI):** Tracking previous detections improves robustness

### 5.2 Preprocessing Pipeline

**Implementation:** `utils/sonar_image_analysis.py::preprocess_edges()`

The preprocessing can optionally apply momentum-based directional enhancement before standard edge detection.

#### Step 1: Input Preparation

Two modes are available, controlled by `IMAGE_PROCESSING_CONFIG['use_momentum_merging']`:

**Option A: Momentum-based Directional Merging (default enabled)**
```python
# Compute gradient energy map
grad_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
energy_map = np.sqrt(grad_x**2 + grad_y**2) / 255.0

# Apply directional kernels (horizontal, vertical, diagonals)
responses = [cv2.filter2D(frame, -1, kernel) for kernel in directional_kernels]
enhanced = np.maximum.reduce(responses)

# Boost based on energy
boost_factor = 1.0 + momentum_boost * np.clip(energy_map, 0, 1)
result = enhanced * boost_factor
```

This step amplifies linear features (net strands) by detecting locally consistent directional energy and boosting those directions while suppressing isotropic noise.

**Option B: Gaussian Blur (fallback)**
```python
blurred = cv2.GaussianBlur(frame, (31, 31), 0)
```

#### Step 2: Canny Edge Detection

```python
edges = cv2.Canny(prepared_input, low=40, high=120)
```

Parameters tuned for sonar imagery contrast levels.

#### Step 3: Morphological Closing

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```

Connects nearby edge fragments.

#### Step 4: Edge Dilation

```python
edges_dilated = cv2.dilate(edges_closed, kernel, iterations=2)
```

Thickens edges for better contour extraction.

#### Momentum-based Directional Merging

**Implementation:** `utils/sonar_image_analysis.py::directional_momentum_merge()`

When enabled (`use_momentum_merging: true`), this preprocessing step enhances linear features before edge detection:

1. **Gradient Computation:**
   ```python
   grad_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
   grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
   energy_map = np.sqrt(grad_x**2 + grad_y**2) / 255.0
   ```

2. **Directional Responses:**
   Apply oriented kernels for horizontal, vertical, and diagonal directions:
   ```python
   h_kernel = [[0.1, 0.2, 0.4, 0.2, 0.1], ...]  # horizontal
   responses = [cv2.filter2D(frame, -1, kernel) for kernel in [h, v, d1, d2]]
   enhanced = np.maximum.reduce(responses)
   ```

3. **Energy-based Boosting:**
   ```python
   boost_factor = 1.0 + momentum_boost * np.clip(energy_map, 0, 1)
   result = enhanced * boost_factor
   ```

**Purpose:** Amplifies elongated structures (net strands) by detecting consistent directional energy and suppressing noise.

**Configuration:** `IMAGE_PROCESSING_CONFIG` in `utils/sonar_config.py`

### 5.3 Contour Scoring

**Objective:** Rank detected contours by likelihood of representing the net structure.

**Implementation:** `utils/sonar_image_analysis.py::select_best_contour()`

```python
contours, _ = cv2.findContours(edges_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
```

For each contour:

#### Geometric Features

Computed by `compute_contour_features()`:

```python
area = cv2.contourArea(cnt)
x, y, w, h = cv2.boundingRect(cnt)
aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = area / hull_area if hull_area > 0 else 0.0

rect_area = w * h
extent = area / rect_area if rect_area > 0 else 0.0

# Ellipse fit
if len(cnt) >= 5:
    (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
    ellipse_elongation = max(MA, ma) / (min(MA, ma) + 1e-6)
else:
    ellipse_elongation = aspect_ratio

# Straightness via line fit
pts = cnt.reshape(-1, 2)
vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
# Compute average distance to fitted line
straightness = max(0.1, 1.0 - (avg_distance / max(w, h) * 0.1))
```

#### Composite Score

Computed by `score_contour()`:

```python
base_score = (
    aspect_ratio * w_aspect +
    ellipse_elongation * w_ellipse +
    (1 - solidity) * w_solidity +
    extent * w_extent +
    min(aspect_ratio / 10.0, 0.5) * w_perimeter
)

straightness_boost = 0.5 + 1.5 * straightness
score = area * base_score * straightness_boost
```

**Weights:** Configured in `utils/sonar_config.py::ELONGATION_CONFIG`

```python
ELONGATION_CONFIG = {
    'aspect_ratio_weight': 0.4,
    'ellipse_elongation_weight': 0.7,
    'solidity_weight': 0.1,
    'extent_weight': 0.0,
    'perimeter_weight': 0.0,
}
```

#### AOI Boost

If a previous Area of Interest (AOI) exists, contours overlapping the expanded AOI receive a score multiplier:

```python
if rects_overlap(contour_rect, expanded_aoi):
    score *= aoi_boost_factor  # 1000.0 by default
```

#### Selection

```python
best_contour = max(scored_contours, key=lambda x: x['score'])
```

### 5.4 Tracking with Area of Interest (AOI)

To improve temporal consistency:

1. **Expand previous bounding box** by a margin (e.g., 10 pixels)
2. **Boost scores** for contours within this AOI
3. **Smooth ellipse parameters** using exponential moving average:

```python
# Smoothing factor α ∈ [0, 1]
cx_smooth = α * cx_new + (1 - α) * cx_prev
cy_smooth = α * cy_new + (1 - α) * cy_prev
```

4. **Limit movement** to prevent jumps:

```python
max_movement = 10.0  # pixels
distance = np.sqrt((cx_new - cx_prev)² + (cy_new - cy_prev)²)
if distance > max_movement:
    cx_new = cx_prev + (cx_new - cx_prev) * max_movement / distance
    cy_new = cy_prev + (cy_new - cy_prev) * max_movement / distance
```

**Configuration:** `utils/sonar_config.py::TRACKING_CONFIG`

---

## 6. Distance Measurement & Pixel-to-Meter Conversion

### 6.1 Red Line Definition

The "red line" represents the **major axis** of the fitted ellipse, which approximates the net's vertical extent.

```python
# Ellipse fit returns
(cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)

# Major axis endpoints
dx = (MA / 2) * np.cos(np.radians(angle))
dy = (MA / 2) * np.sin(np.radians(angle))

p1 = (cx - dx, cy - dy)  # Top endpoint
p2 = (cx + dx, cy + dy)  # Bottom endpoint

# Distance in pixels
distance_px = np.sqrt((p2[0] - p1[0])² + (p2[1] - p1[1])²)
```

### 6.2 Pixel-to-Meter Conversion

**Source:** NPZ extent metadata

```python
# Load from NPZ
with np.load(npz_file) as data:
    extent = data['extent']  # (x_min, x_max, y_min, y_max) in meters
    cones = data['cones']    # Shape (T, H, W)

T, H, W = cones.shape
x_min, x_max, y_min, y_max = extent

# Compute scale factors
meters_per_pixel_x = (x_max - x_min) / W
meters_per_pixel_y = (y_max - y_min) / H
meters_per_pixel_avg = (meters_per_pixel_x + meters_per_pixel_y) / 2
```

**Distance in meters:**

```python
distance_m = distance_px * meters_per_pixel_avg
```

**Why accurate?** The extent is computed during rasterization based on the sonar's configured range and FOV, ensuring geometric consistency.

### 6.3 Time Series Analysis

**Implementation:** `utils/sonar_image_analysis.py::analyze_red_line_distance_over_time()`

For each frame:
1. Detect and score contours
2. Fit ellipse to best contour
3. Compute major axis distance in pixels
4. Convert to meters using NPZ metadata
5. Store timestamp, pixel distance, meter distance, and ellipse parameters

**Output:** Pandas DataFrame

```python
{
    'frame': int,
    'timestamp': pd.Timestamp,
    'distance_px': float,
    'distance_m': float,
    'ellipse_cx': float,
    'ellipse_cy': float,
    'ellipse_MA': float,
    'ellipse_ma': float,
    'ellipse_angle': float,
    ...
}
```

---

## 7. DVL Data Integration

### 7.1 DVL Sensor Types

**Nucleus1000 DVL** provides multiple data streams:

- **Bottomtrack:** Velocity relative to seafloor
- **INS:** Inertial Navigation System (position, orientation)
- **IMU:** Accelerometer and gyroscope data
- **Altimeter:** Distance to seafloor
- **Magnetometer:** Heading reference
- **Watertrack:** Velocity relative to water column

Additionally:
- **Navigation:** ROV position and net distance estimates
- **Guidance:** Control system net distance measurements

### 7.2 Data Loading

**Implementation:** `utils/net_distance_analysis.py::load_all_distance_data_for_bag()`

For a given bag ID:

1. **Load navigation data**
   ```python
   nav_file = f"navigation_plane_approximation__{bag_id}_data.csv"
   nav_data = pd.read_csv(nav_file, usecols=['ts_oslo', 'NetDistance', 'Altitude'])
   nav_data['timestamp'] = pd.to_datetime(nav_data['ts_oslo']).dt.tz_convert('UTC')
   ```

2. **Load guidance data**
   ```python
   guidance_file = f"guidance__{bag_id}_data.csv"
   guidance_data = pd.read_csv(guidance_file)
   # Extract distance columns
   distance_cols = [col for col in guidance_data.columns if 'distance' in col.lower()]
   ```

3. **Load DVL sensors**
   ```python
   dvl_alt_file = f"nucleus1000dvl_altimeter__{bag_id}_data.csv"
   dvl_alt = pd.read_csv(dvl_alt_file)
   dvl_alt['timestamp'] = pd.to_datetime(dvl_alt['ts_utc'])
   ```

**Returns:**
- `raw_data`: Dictionary of DataFrames for each source
- `distance_measurements`: Structured dict with measurement metadata

### 7.3 Timestamp Synchronization

All timestamps are normalized to UTC:

```python
# Sonar timestamps from NPZ
sonar_ts = pd.DatetimeIndex(npz_data['ts'], tz='UTC')

# DVL timestamps from CSV
dvl_ts = pd.to_datetime(dvl_data['ts_utc']).dt.tz_localize('UTC')

# Merge on nearest timestamp
merged = pd.merge_asof(
    sonar_df.sort_values('timestamp'),
    dvl_df.sort_values('timestamp'),
    on='timestamp',
    direction='nearest',
    tolerance=pd.Timedelta('1s')  # Max time difference
)
```

---

## 8. Synchronized Video Generation

### 8.1 Video Components

Generated videos include:

**Left pane:** Camera footage (if available)
**Right pane:** Sonar cone display
**Overlays:**
- Net distance (from sonar image analysis)
- Net distance (from DVL navigation)
- Net pitch angle
- Timestamp
- Frame number

### 8.2 Generation Process

**Implementation:** `utils/sonar_and_foto_generation.py::export_optimized_sonar_video()`

#### Step 1: Load Data Sources

```python
# Load sonar NPZ
cones, ts, extent, meta = load_cone_run_npz(npz_file)

# Load camera frames (if available)
if VIDEO_SEQ_DIR:
    frames = sorted(VIDEO_SEQ_DIR.glob("*.png"))
    frame_index = load_timestamp_index(VIDEO_SEQ_DIR / "index.csv")

# Load DVL distance data
nav_data = load_navigation_data(bag_id)

# Load sonar distance analysis results
sonar_distances = load_analysis_results(analysis_csv)
```

#### Step 2: Temporal Alignment

```python
for i in range(START_IDX, END_IDX, STRIDE):
    # Get sonar timestamp
    sonar_ts = ts[i]
    
    # Find nearest camera frame
    if VIDEO_SEQ_DIR:
        camera_frame = find_nearest_frame(sonar_ts, frame_index)
    
    # Find nearest DVL measurement
    dvl_distance = interpolate_dvl(sonar_ts, nav_data)
    
    # Get sonar analysis result
    sonar_distance = sonar_distances.iloc[i]['distance_m']
```

#### Step 3: Composite Frame Creation

```python
# Create output frame
H, W = 720, 1280
output = np.zeros((H, W, 3), dtype=np.uint8)

if camera_frame is not None:
    # Left half: camera
    camera_resized = cv2.resize(camera_frame, (W//2, H))
    output[:, :W//2] = camera_resized

# Right half: sonar cone
cone_img = cones[i]  # Shape (cone_h, cone_w)
cone_rgb = cv2.applyColorMap((cone_img * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
cone_resized = cv2.resize(cone_rgb, (W//2, H))
output[:, W//2:] = cone_resized

# Add overlays
add_text_overlay(output, f"DVL Distance: {dvl_distance:.2f} m", (10, 30))
add_text_overlay(output, f"Sonar Distance: {sonar_distance:.2f} m", (10, 60))
add_text_overlay(output, f"Timestamp: {sonar_ts}", (10, H-30))
```

#### Step 4: Video Encoding

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

for frame in frames:
    out.write(frame)

out.release()
```

**Configuration:** `utils/sonar_config.py::VIDEO_CONFIG`

```python
VIDEO_CONFIG = {
    'fps': 15,
    'show_all_contours': True,
    'show_ellipse': True,
    'show_bounding_box': True,
    'text_scale': 0.6,
}
```

### 8.3 Known Issues

**Legacy Notebook (05):**
- **Issue:** Net pitch angle is inverted (sign error)
- **Impact:** Pitch overlay displays incorrect sign
- **Workaround:** Use notebook 06 which has corrected pitch calculation
- **Technical Detail:** Pitch angle derived from ellipse `angle` parameter; coordinate system convention was inconsistent

**Fix in notebook 06:**
```python
# Correct pitch calculation
pitch_deg = 90 - ellipse_angle  # Adjust for coordinate system
if pitch_deg > 90:
    pitch_deg -= 180  # Wrap to [-90, 90]
```

---

## 9. Configuration Parameters

All processing parameters are centralized in `utils/sonar_config.py` for consistency across notebooks and modules.

### 9.1 Sonar Visualization Defaults

```python
SONAR_VIS_DEFAULTS = {
    # Geometry
    "fov_deg": 120.0,
    "range_min_m": 0.0,
    "range_max_m": 20.0,
    "display_range_max_m": 5.0,
    
    # Enhancement
    "enh_scale": "db",           # 'linear' or 'db'
    "enh_tvg": "amplitude",      # 'amplitude', 'power', or 'none'
    "enh_alpha_db_per_m": 0.0,  # Absorption coefficient
    "enh_p_low": 1.0,           # Lower percentile clip
    "enh_p_high": 99.5,         # Upper percentile clip
    "enh_gamma": 0.9,           # Gamma correction
    
    # Cone view
    "img_w": 900,
    "img_h": 700,
    "bg_color": "black",
    "n_spokes": 5,              # Guide lines
}
```

### 9.2 Image Processing Config

```python
IMAGE_PROCESSING_CONFIG = {
    # Preprocessing
    'blur_kernel_size': (31, 31),
    'canny_low_threshold': 40,
    'canny_high_threshold': 120,
    'morph_close_kernel': 5,
    'edge_dilation_iterations': 2,
    
    # Contour filtering
    'min_contour_area': 100,
    'extreme_angle_threshold_deg': 20.0,
    
    # Advanced (momentum-based merging)
    'use_momentum_merging': True,
    'momentum_search_radius': 3,
    'momentum_threshold': 0.1,
    'momentum_decay': 0.9,
    'momentum_boost': 5.0,
}
```

### 9.3 Tracking Config

```python
TRACKING_CONFIG = {
    'aoi_boost_factor': 1000.0,          # Score multiplier for AOI
    'aoi_expansion_pixels': 10,          # AOI margin
    'ellipse_smoothing_alpha': 0.2,      # EMA smoothing
    'ellipse_max_movement_pixels': 10.0, # Movement constraint
}
```

### 9.4 File Paths

```python
# Default storage locations
EXPORTS_DIR_DEFAULT = "/Volumes/LaCie/SOLAQUA/exports"

EXPORTS_SUBDIRS = {
    'by_bag': 'by_bag',
    'videos': 'videos',
    'frames': 'frames',
    'camera_info': 'camera_info',
    'outputs': 'outputs',
    'index': '',  # Index files in root
}
```

**Usage in code:**
```python
from utils.sonar_config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS

exports_root = Path(EXPORTS_DIR_DEFAULT)
by_bag_folder = exports_root / EXPORTS_SUBDIRS['by_bag']
```

---

## 10. Code Reference Map

### 10.1 Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `sonar_utils.py` | Enhancement, rasterization | `enhance_intensity()`, `cone_raster_like_display_cell()` |
| `sonar_visualization.py` | Sonoptix display | `SonarVisualizer` class |
| `ping360_visualization.py` | Ping360 display | `Ping360Visualizer` class |
| `sonar_image_analysis.py` | Net detection, distance | `detect_and_score_contours()`, `analyze_red_line_distance_over_time()` |
| `net_distance_analysis.py` | DVL data loading | `load_all_distance_data_for_bag()` |
| `nucleus1000dvl_analysis.py` | DVL sensor analysis | `Nucleus1000DVLAnalyzer` class |
| `sonar_and_foto_generation.py` | Video creation | `export_optimized_sonar_video()` |
| `dataset_export_utils.py` | Bag export | `export_all_topics()`, `export_video_as_mp4()` |
| `sonar_config.py` | Configuration | All default parameter dicts |

### 10.2 Workflow Mapping

| Task | Primary Module(s) | Notebook |
|------|-------------------|----------|
| Extract ROS bags | `dataset_export_utils` | Run `solaqua_export.py` |
| Visualize Sonoptix | `sonar_visualization`, `sonar_utils` | `01_Sonar_Visualizer.ipynb` |
| Visualize Ping360 | `ping360_visualization` | `02_Ping360_Visualizer.ipynb` |
| DVL analysis | `nucleus1000dvl_analysis` | `03_rov_navigation_analysis.ipynb` |
| Sonar-DVL comparison | `net_distance_analysis`, `sonar_image_analysis` | `04_synchronized_sonar_net_distance_analysis.ipynb` |
| Net distance video | `sonar_and_foto_generation` | `06_image_analysis.ipynb` |

### 10.3 Data Format Reference

| File Type | Location | Contents |
|-----------|----------|----------|
| CSV/Parquet | `exports/by_bag/` | Per-topic sensor data tables |
| NPZ | `exports/outputs/` | Preprocessed cone image arrays + metadata |
| MP4 | `exports/videos/` | Generated visualization videos |
| PNG | `exports/frames/` | Individual camera frames with timestamp index |
| YAML | `exports/camera_info/` | Camera calibration parameters |

---

## 11. Advanced Topics

### 11.1 Custom Enhancement Pipelines

To create a custom enhancement chain:

```python
from utils import sonar_utils

# Load raw frame
df = sonar_utils.load_df('path/to/sonar.csv')
M_raw = sonar_utils.get_sonoptix_frame(df, frame_idx)

# Apply custom pipeline
M_custom = sonar_utils.enhance_intensity(
    M_raw,
    range_min_m=0.0,
    range_max_m=15.0,
    scale='db',
    tvg='power',  # Use power TVG instead of amplitude
    alpha_db_per_m=0.05,  # Higher absorption
    p_low=5.0,  # More aggressive clipping
    p_high=95.0,
    gamma=0.7
)

# Rasterize with custom extent
cone = sonar_utils.cone_raster_like_display_cell(
    M_custom,
    fov_deg=130,  # Wider FOV
    range_max_m=15.0,
    img_w=1200,
    img_h=900
)
```

### 11.2 Custom Contour Scoring

To implement domain-specific contour scoring:

```python
from utils import sonar_image_analysis as sia

def custom_scorer(contour_info):
    """
    Score contours based on custom criteria.
    
    Args:
        contour_info: dict with keys:
            - 'contour': cv2 contour
            - 'area': float
            - 'aspect_ratio': float
            - 'ellipse_elongation': float
            - ...
    
    Returns:
        score: float (higher is better)
    """
    # Example: Prefer contours in upper half of image
    cy = contour_info['ellipse_cy']
    img_height = 700
    position_score = 1.0 - (cy / img_height)
    
    # Combine with standard elongation score
    elongation_score = contour_info['ellipse_elongation']
    
    return elongation_score * position_score

# Use in analysis
sia.IMAGE_PROCESSING_CONFIG['custom_scorer'] = custom_scorer
```

### 11.3 Multi-Sonar Fusion

For systems with multiple sonar units:

1. Process each sonar independently
2. Transform to common coordinate frame using known mounting positions
3. Merge detections using weighted fusion:

```python
def fuse_detections(det1, det2, conf1, conf2):
    """Weighted fusion of two distance estimates."""
    w1 = conf1 / (conf1 + conf2)
    w2 = conf2 / (conf1 + conf2)
    return w1 * det1 + w2 * det2
```

---

## 12. Performance Considerations

### 12.1 Processing Speed

Typical performance on modern hardware:

| Operation | Speed | Bottleneck |
|-----------|-------|------------|
| CSV export | ~100 msgs/sec | Disk I/O |
| Enhancement | ~50 frames/sec | NumPy ops |
| Rasterization | ~30 frames/sec | Interpolation |
| Contour detection | ~20 frames/sec | OpenCV ops |
| Video encoding | ~25 fps | Codec overhead |

### 12.2 Optimization Tips

1. **Use Parquet instead of CSV**
   - 3-5x faster loading
   - 50-70% smaller file size

2. **Batch processing**
   - Process multiple frames in parallel
   - Use `multiprocessing` for CPU-bound tasks

3. **NPZ caching**
   - Precompute and cache cone rasterizations
   - Reuse for multiple analysis passes

4. **GPU acceleration**
   - OpenCV supports CUDA for morphological ops
   - NumPy operations can use CuPy drop-in replacement

### 12.3 Memory Management

For large datasets:

```python
# Process in chunks
chunk_size = 100
for start in range(0, len(df), chunk_size):
    end = min(start + chunk_size, len(df))
    chunk_df = df.iloc[start:end]
    process_chunk(chunk_df)
    # Memory released automatically
```

---

## 13. Troubleshooting

### Common Issues

#### 1. "No such file or directory"
**Cause:** Incorrect paths in `sonar_config.py`  
**Fix:** Verify `EXPORTS_DIR_DEFAULT` points to your exports location

#### 2. "Could not construct Sonoptix frame"
**Cause:** Missing `image` or `data` columns in CSV  
**Fix:** Re-export using latest `solaqua_export.py`

#### 3. "Invalid load key" (UnpicklingError)
**Cause:** Trying to load macOS resource fork files (`._filename.npz`)  
**Fix:** Updated code filters these automatically; ensure using latest version

#### 4. "No navigation data found"
**Cause:** Bag lacks `nucleus1000dvl_*` files  
**Fix:** Use bags from 2024-08-22 series with full DVL sensors

#### 5. Inverted pitch angle in videos
**Cause:** Using legacy notebook 05  
**Fix:** Switch to notebook 06 with corrected pitch calculation

---

## 14. References

### Academic Background

1. **Sonar Signal Processing:**
   - X. Lurton, *An Introduction to Underwater Acoustics*, 2nd ed., Springer, 2010
   - R. J. Urick, *Principles of Underwater Sound*, 3rd ed., McGraw-Hill, 1983

2. **TVG and Enhancement:**
   - A. D. Waite, *Sonar for Practising Engineers*, 3rd ed., Wiley, 2002
   - D. H. Johnson and D. E. Dudgeon, *Array Signal Processing*, Prentice Hall, 1993

3. **Computer Vision:**
   - R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, 2022
   - G. Bradski and A. Kaehler, *Learning OpenCV*, O'Reilly, 2008

### Software Dependencies

- [rosbags](https://gitlab.com/ternaris/rosbags) - ROS bag reading
- [OpenCV](https://opencv.org/) - Computer vision operations
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Plotly](https://plotly.com/python/) - Interactive visualization

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| M | Sonar intensity matrix (polar) |
| H | Number of range bins |
| W | Number of beam angles |
| r | Range distance (meters) |
| θ | Beam angle (radians or degrees) |
| T | Number of time steps |
| TVG | Time-Varying Gain |
| α | Absorption coefficient (dB/m) |
| γ | Gamma correction exponent |

---

## Appendix B: Glossary

**AOI (Area of Interest):** Expanded bounding box from previous frame used to prioritize contours

**Cone View:** Cartesian rasterization of polar sonar data in wedge shape

**DVL (Doppler Velocity Log):** Acoustic sensor measuring velocity relative to seafloor or water

**Ellipse Fitting:** Least-squares fit of ellipse to contour points

**NPZ:** NumPy compressed archive format (`.npz` files)

**Red Line:** Major axis of fitted ellipse representing net extent

**TVG (Time-Varying Gain):** Compensation for geometric spreading loss in sonar

---

**Document Version:** 1.0.0  
**Last Updated:** October 2025  
**Maintainer:** Eirik Varnes
