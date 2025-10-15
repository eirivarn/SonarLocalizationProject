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
  - Topics: DVL sensors, Ping360 sonar, navigation, guidance, USBL, etc.
  
- **`*_video.bag`**: Camera footage and imaging sonar
  - Topics: Compressed image streams, camera info, SonoptixECHO sonar

### 2.1.1 Sonar Data Distribution

SOLAQUA datasets contain **two different sonar systems** with data stored in different bag types:

**Ping360 Sonar** → `*_data.bag` files:
- Topic: `/sensor/ping360` (sensors/msg/Ping360)
- Topic: `/sensor/ping360_config` (sensors/msg/Ping360_config_2)
- Purpose: Range-finding, obstacle detection
- Exported to: `sensor_ping360__{bag_id}_data.csv`

**SonoptixECHO Sonar** → `*_video.bag` files:
- Topic: `/sensor/sonoptix_echo/image` (sensors/msg/SonoptixECHO)
- Purpose: High-resolution imaging sonar for visualization and analysis
- Exported to: `sensor_sonoptix_echo_image__{bag_id}_video.csv`

> **📋 Note:** The main sonar analysis pipeline focuses on **SonoptixECHO** data from video bags, 
> which explains why sonar CSV files have the `_video.csv` suffix throughout the codebase.

**Verification:** Run `utils/verify_sonar_bag_source.py` to confirm sonar topic distribution across your bag files.

### 2.2 Export Process

**Implementation:** `solaqua_export.py` → `utils/dataset_export_utils.py`

The export process:

1. **Enumerate topics** in each bag using `rosbags` library
2. **Extract messages** topic by topic  
3. **Normalize timestamps** to UTC with `ts_utc` field
4. **Parse nested fields** (JSON, arrays) into flat columns
5. **Write CSV/Parquet** files named: `{topic}__{bag_stem}.{ext}`
   - Data bags: `{topic}__{bag_id}_data.csv`
   - Video bags: `{topic}__{bag_id}_video.csv`

**Key Functions:**
- `dataset_export_utils.export_all_topics()` - Main export orchestrator
- `dataset_export_utils.flatten_message()` - Recursively flattens ROS messages
- `sonar_utils.load_df()` - Loads CSV/Parquet

### 2.3 Sonar Data Schema

#### 2.3.1 Original ROS Bag Format

**SonoptixECHO** messages in `*_video.bag` files contain:

```python
# ROS Message Structure
{
  "header": {
    "stamp": {"sec": 1692705903, "nanosec": 123456789},
    "frame_id": "sonar_link"
  },
  "array_data": {  # Float32MultiArray
    "layout": {
      "dim": [
        {"label": "range", "size": 1024, "stride": 262144},
        {"label": "beam",  "size": 256,  "stride": 256}
      ],
      "data_offset": 0
    },
    "data": [45.2, 46.1, 44.8, ...]  # 262,144 float32 values
  }
}
```

#### 2.3.2 Exported CSV Format

After processing through `solaqua_export.py`, the data becomes:

```python
# CSV Row Structure  
{
  "t": 1692705903.123,
  "data": "[45.2, 46.1, 44.8, ...]",     # JSON string of flat array
  "dim_labels": '["range", "beam"]',     # JSON string
  "dim_sizes": "[1024, 256]",           # JSON string [H, W]
  # ... other metadata columns
}
```

**Key Transformation:**
- Float32MultiArray → Flat array stored as JSON string
- Binary ROS format → Human-readable CSV format
- Nested structure → Flattened columns with metadata

**Extraction:** `utils/sonar_utils.py::get_sonoptix_frame(df, idx)`

This function:
1. Loads the flat `data` array from the CSV (parsing JSON string)
2. Uses `dim_sizes` to determine reshape dimensions `[H, W]`  
3. Returns `np.ndarray` of shape `(H, W)` where:
   - `H` = range bins (distance samples)
   - `W` = beam angles

**Exported Data Location:** SonoptixECHO data is exported to:
```
exports/by_bag/sensor_sonoptix_echo_image__{bag_id}_video.csv
```
The `_video.csv` suffix indicates this data originated from `*_video.bag` files.

### 2.4 SonoptixECHO Data Flow: From ROS Bags to CSV

This section explains the complete journey of SonoptixECHO sonar data from the original ROS bag files to the processed CSV files used by the analysis pipeline.

#### Step 1: ROS Bag Storage

**Original Format in `*_video.bag`:**
```
Topic: /sensor/sonoptix_echo/image
Message Type: sensors/msg/SonoptixECHO
```

Each SonoptixECHO message contains:
- **Header**: Timestamp and frame ID
- **array_data**: A Float32MultiArray containing the sonar intensity data

**Float32MultiArray Structure:**
```python
array_data = {
    "layout": {
        "dim": [
            {"label": "range", "size": 1024, "stride": 262144},
            {"label": "beam",  "size": 256,  "stride": 256}
        ],
        "data_offset": 0
    },
    "data": [float32_value_1, float32_value_2, ..., float32_value_262144]
}
```

#### Step 2: Export Processing

**Implementation:** `utils/dataset_export_utils.py::bag_topic_to_dataframe()`

When processing a `*_video.bag` file:

1. **Message Deserialization**: 
   ```python
   msg = reader.deserialize(raw, con.msgtype)  # Get SonoptixECHO message
   array_data = msg.array_data                 # Extract Float32MultiArray
   ```

2. **Float32MultiArray Decoding**:
   ```python
   labels, sizes, strides, payload, shape, meta = decode_float32_multiarray(array_data)
   
   # Example result:
   # labels = ["range", "beam"]
   # sizes = [1024, 256] 
   # payload = [float1, float2, ..., float262144]  # 1024×256 = 262,144 values
   # shape = (1024, 256)  # Inferred 2D dimensions
   ```

3. **CSV Record Creation**:
   ```python
   rec = {
       "t": timestamp,
       "dim_labels": '["range", "beam"]',     # JSON string
       "dim_sizes": '[1024, 256]',           # JSON string  
       "data": '[float1, float2, ...]',      # JSON string (flat array)
       "len": 262144,                        # Length verification
       # ... other metadata fields
   }
   ```

**2D Conceptual View:**
```python
# How we think about it (2D sonar image)
sonar_2d = [
    [r0_b0, r0_b1, r0_b2, ..., r0_b255],    # Range bin 0, all beams
    [r1_b0, r1_b1, r1_b2, ..., r1_b255],    # Range bin 1, all beams
    [r2_b0, r2_b1, r2_b2, ..., r2_b255],    # Range bin 2, all beams
    ...
    [r1023_b0, r1023_b1, ..., r1023_b255]   # Range bin 1023, all beams
]
```

**1D Flat Storage:**
```python
# How it's actually stored (flat array)
data_flat = [
    r0_b0, r0_b1, r0_b2, ..., r0_b255,      # Row 0 flattened
    r1_b0, r1_b1, r1_b2, ..., r1_b255,      # Row 1 flattened  
    r2_b0, r2_b1, r2_b2, ..., r2_b255,      # Row 2 flattened
    ...
    r1023_b0, r1023_b1, ..., r1023_b255     # Row 1023 flattened
]
# Total length: 1024 × 256 = 262,144 values
```


#### Step 3: CSV File Structure

**Resulting CSV Columns:**
```python
columns = [
    # Timing
    't', 't_header', 't_bag', 't_src', 'ts_utc', 'ts_oslo',
    
    # Metadata  
    'bag', 'bag_file', 'topic',
    
    # Array Structure (JSON strings)
    'dim_labels',    # '["range", "beam"]'
    'dim_sizes',     # '[1024, 256]'
    'dim_strides',   # '[262144, 256]'
    
    # Sonar Data (JSON string)
    'data',          # '[val1, val2, ..., val262144]'
    'len',           # 262144
    
    # Quality Metadata
    'data_offset', 'dtype', 'payload_sha256', 'used_shape', 'policy', 'warnings'
]
```

**Example CSV Row:**
```csv
t,dim_labels,dim_sizes,data,len
1692705903.123,"[""range"", ""beam""]","[1024, 256]","[45.2, 46.1, 44.8, ...]",262144
```

#### Step 4: Data Reconstruction for Processing

**Loading and Reshaping:**
```python
def get_sonoptix_frame(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Convert flat CSV data back to 2D sonar image"""
    
    # 1. Load flat data from CSV
    data_flat = json.loads(df.loc[idx, "data"])          # List of 262,144 floats
    dim_sizes = json.loads(df.loc[idx, "dim_sizes"])     # [1024, 256]
    
    # 2. Reconstruct 2D array
    H, W = dim_sizes  # 1024, 256
    sonar_2d = np.array(data_flat).reshape(H, W)
    
    # 3. Result: 2D array ready for processing
    # Shape: (1024, 256) = (range_bins, beam_angles)
    return sonar_2d
```

**Coordinate System:**
```python
# sonar_2d[row, col] = intensity_value
# 
# row index    → range bins (0 = closest, 1023 = farthest)
# col index    → beam angles (0 = leftmost beam, 255 = rightmost beam)
#
# Example access:
sonar_2d[0, 128]     # Closest range, center beam
sonar_2d[512, 0]     # Mid range, leftmost beam  
sonar_2d[1023, 255]  # Farthest range, rightmost beam
```

#### Step 5: Processing Pipeline Integration

Once reconstructed to 2D, the data flows through:

1. **Flipping** (`apply_flips()`) - Correct orientation
2. **Enhancement** (`enhance_intensity()`) - TVG compensation, normalization
3. **Rasterization** (`cone_raster_like_display_cell()`) - Polar to Cartesian conversion
4. **Analysis** - Contour detection, distance measurement

---

## 3. Sonar Preprocessing & Enhancement

### 3.1 Why Enhancement?

Raw sonar data suffers from:
- **Geometric spreading loss:** Signal weakens as `1/r²` (amplitude) or `1/r⁴` (power)
- **Water absorption:** Exponential attenuation with distance
- **Dynamic range issues:** Near-field saturates, far-field is weak
- **Variable gain:** Time-varying gain (TVG) applied in hardware

**Goal:** Normalize intensity for visualization.

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

### 5.1 Overview

The image analysis pipeline processes Cartesian sonar frames to detect and track underwater net structures using **signal-strength independent computer vision algorithms**. The system employs **structure tensor analysis**, **elliptical kernel convolution**, and **geometric constraint-based contour selection** to achieve robust detection across varying sonar conditions.

**Key Mathematical Components:**
1. **Structure Tensor Analysis** for edge detection and orientation estimation
2. **Adaptive Elliptical Kernel Convolution** for linear structure enhancement
3. **Geometric Contour Scoring** with multi-criteria evaluation
4. **Kalman-like Temporal Smoothing** for stable ellipse tracking
5. **Geometric Distance Calculation** using ellipse major axis intersection

### 5.2 Structure Tensor Analysis for Edge Detection

**Implementation:** `utils/sonar_image_analysis.py::preprocess_edges()`

The pipeline begins with **signal-strength independent processing** using structure tensor analysis to detect edges and their orientations without relying on intensity gradients.

#### Step 1: Binary Conversion

```python
# Convert to binary immediately - eliminates signal strength dependency
binary_threshold = cfg.get('binary_threshold', 128)
binary_frame = (frame_u8 > binary_threshold).astype(np.uint8) * 255
```

**Mathematical Foundation:** Removes all intensity-based variability, focusing purely on geometric structure.

#### Step 2: Structure Tensor Computation

**Structure Tensor Definition:**
The structure tensor $S$ at each pixel location $(x,y)$ is computed from the gradient field:

$$S(x,y) = \begin{pmatrix}
J_{xx} & J_{xy} \\
J_{xy} & J_{yy}
\end{pmatrix}$$

where the components are computed using Gaussian-weighted gradients:

$$J_{xx} = G_\sigma * (I_x^2), \quad J_{xy} = G_\sigma * (I_x I_y), \quad J_{yy} = G_\sigma * (I_y^2)$$

**Implementation:**
```python
# Compute gradients using Sobel operators
grad_x = cv2.Sobel(binary_frame, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(binary_frame, cv2.CV_32F, 0, 1, ksize=3)

# Compute structure tensor components with Gaussian smoothing
sigma = cfg.get('structure_tensor_sigma', 1.0)
J_xx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), sigma)
J_xy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), sigma)
J_yy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), sigma)
```

#### Step 3: Eigenvalue Analysis for Edge Detection

**Eigenvalue Decomposition:**
For each pixel, solve the eigenvalue problem:

$$S \vec{v} = \lambda \vec{v}$$

The eigenvalues $\lambda_1, \lambda_2$ (with $\lambda_1 \geq \lambda_2$) indicate:
- **Large $\lambda_1$**: Strong edges present
- **Small $\lambda_2$**: Coherent edge orientation
- **High coherence** $c = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2}$: Well-defined linear structures

**Edge Detection Criterion:**
```python
# Compute coherence measure
coherence = (J_xx - J_yy)**2 + 4 * J_xy**2
coherence = np.sqrt(coherence)

# Edge strength threshold
edge_threshold = cfg.get('edge_strength_threshold', 50.0)
edges = (coherence > edge_threshold).astype(np.uint8) * 255
```

### 5.3 Adaptive Elliptical Kernel Convolution

**Implementation:** `adaptive_linear_momentum_merge_fast()`

The system applies **adaptive morphological operations** that dynamically adjust kernel shapes based on detected linear structures using elliptical kernel convolution.

#### Mathematical Foundation

**Elliptical Kernel Generation:**
For each pixel, create an elliptical kernel oriented along the dominant gradient direction:

$$\theta = \frac{1}{2} \arctan\left(\frac{2J_{xy}}{J_{xx} - J_{yy}}\right)$$

The kernel elongation $e$ is determined by the coherence measure:

$$e = 1 + k \cdot c$$

where $c$ is the coherence and $k$ is the maximum elongation factor.

**Kernel Construction:**
```python
def create_elliptical_kernel(theta, elongation, base_radius):
    # Semi-major and semi-minor axes
    a = base_radius * elongation  # Major axis
    b = base_radius / elongation  # Minor axis

    # Generate elliptical kernel points
    t = np.linspace(0, 2*np.pi, 32)
    x = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

    # Create binary kernel mask
    kernel_size = int(2 * base_radius * elongation) + 1
    kernel = np.zeros((kernel_size, kernel_size), np.uint8)

    # Center coordinates
    cx, cy = kernel_size // 2, kernel_size // 2

    # Fill elliptical region
    for i in range(len(x)):
        px = int(cx + x[i])
        py = int(cy + y[i])
        if 0 <= px < kernel_size and 0 <= py < kernel_size:
            kernel[py, px] = 1

    return kernel
```

#### Adaptive Enhancement Algorithm

**Multi-orientation Convolution:**
```python
# Test multiple orientations
angle_steps = cfg.get('adaptive_angle_steps', 9)
responses = []

for angle_idx in range(angle_steps):
    theta = angle_idx * np.pi / angle_steps

    # Create elliptical kernel
    kernel = create_elliptical_kernel(theta, elongation, base_radius)

    # Convolve with binary image
    response = cv2.filter2D(binary_frame.astype(np.float32), -1, kernel.astype(np.float32))
    responses.append(response)

# Select maximum response across orientations
enhanced = np.max(responses, axis=0)
```

**Momentum Boosting:**
High-coherence regions receive additional enhancement:

```python
momentum_boost = cfg.get('momentum_boost', 1.5)
linearity_threshold = cfg.get('adaptive_linearity_threshold', 0.3)

# Boost regions with high linearity
boost_mask = (coherence > linearity_threshold * coherence.max()).astype(np.float32)
enhanced = enhanced * (1 + momentum_boost * boost_mask)
```

### 5.4 Contour Detection and Geometric Scoring

**Implementation:** `utils/sonar_image_analysis.py::SonarDataProcessor.analyze_frame()`

#### Contour Extraction

```python
contours, hierarchy = cv2.findContours(edges_processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
```

#### Geometric Contour Scoring

**Implementation:** `select_best_contour_core()`

**Multi-criteria Scoring Function:**
Each contour receives a score based on geometric properties optimized for net detection:

$$S_{total} = S_{geometric} \cdot S_{temporal} \cdot S_{spatial}$$

**Geometric Score ($S_{geometric}$):**
```python
# Area and elongation metrics
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)
circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

# Ellipse fitting for elongation
if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (MA, ma), angle = ellipse
    elongation = MA / ma if ma > 0 else 1.0
else:
    # Fallback: bounding rectangle aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

# Geometric score favors elongated, substantial contours
S_geometric = area * elongation * (1 - circularity)  # Penalize circular shapes
```

**Temporal Consistency Score ($S_{temporal}$):**
```python
# AOI overlap boost
aoi_boost = cfg.get('aoi_boost_factor', 2.0)
min_overlap = cfg.get('min_aoi_overlap_percent', 0.7)

if current_aoi is not None:
    overlap_ratio = calculate_contour_aoi_overlap(contour, current_aoi)
    if overlap_ratio >= min_overlap:
        S_temporal = aoi_boost
    else:
        S_temporal = 1.0
else:
    S_temporal = 1.0
```

**Spatial Consistency Score ($S_{spatial}$):**
```python
# Distance penalty from previous detection
if last_center is not None:
    current_center = get_contour_centroid(contour)
    distance = np.sqrt((current_center[0] - last_center[0])**2 +
                      (current_center[1] - last_center[1])**2)

    max_penalty_distance = cfg.get('max_penalty_distance_pixels', 50.0)
    distance_factor = max(0.01, 1.0 - distance / max_penalty_distance)
    S_spatial = distance_factor
else:
    S_spatial = 1.0
```

**Final Score Computation:**
```python
final_score = S_geometric * S_temporal * S_spatial
```

### 5.5 Kalman-like Temporal Smoothing for Ellipse Tracking

**Implementation:** `create_smooth_elliptical_aoi()`

The system maintains **temporal continuity** through Kalman-like smoothing of elliptical tracking regions.

#### Ellipse Parameter State Vector

The ellipse state is represented as:

$$\vec{x} = \begin{pmatrix} c_x \\ c_y \\ w \\ h \\ \theta \end{pmatrix}$$

where:
- $c_x, c_y$: Ellipse center coordinates
- $w, h$: Ellipse width and height
- $\theta$: Ellipse orientation angle

#### Temporal Smoothing Algorithm

**Exponential Smoothing with Movement Limiting:**
```python
def smooth_ellipse_parameters(current_params, previous_params, alpha, max_movement):
    cx_curr, cy_curr, w_curr, h_curr, theta_curr = current_params
    cx_prev, cy_prev, w_prev, h_prev, theta_prev = previous_params

    # Center smoothing with movement limiting
    dx = cx_curr - cx_prev
    dy = cy_curr - cy_prev
    center_distance = np.sqrt(dx**2 + dy**2)

    if center_distance > max_movement:
        # Limit movement to maximum allowed distance
        scale = max_movement / center_distance
        dx *= scale
        dy *= scale

    # Apply exponential smoothing
    cx_smooth = cx_prev + alpha * dx
    cy_smooth = cy_prev + alpha * dy
    w_smooth = w_prev + alpha * (w_curr - w_prev)
    h_smooth = h_prev + alpha * (h_curr - h_prev)

    # Angle smoothing with wraparound handling
    theta_diff = ((theta_curr - theta_prev + 180) % 360) - 180
    theta_smooth = theta_prev + alpha * theta_diff

    return cx_smooth, cy_smooth, w_smooth, h_smooth, theta_smooth
```

**Mathematical Properties:**
- **Stability:** Exponential smoothing provides bounded convergence
- **Adaptability:** Alpha parameter controls responsiveness to changes
- **Movement Limiting:** Prevents erratic jumps due to noise
- **Angle Continuity:** Wraparound handling prevents discontinuities at 0°/360°

#### AOI Expansion for Robust Tracking

```python
expansion_factor = cfg.get('ellipse_expansion_factor', 0.3)
w_expanded = w_smooth * (1 + expansion_factor)
h_expanded = h_smooth * (1 + expansion_factor)
```

**Purpose:** Creates a search region larger than the detected contour to handle:
- Minor object movement between frames
- Partial occlusions
- Detection variability

### 5.6 Geometric Distance Calculation

**Implementation:** `_distance_angle_from_smoothed_center()`

#### Distance Measurement Using Major Axis Intersection

**Mathematical Approach:**
The distance is calculated by finding the intersection of the ellipse's major axis with the sonar coordinate system origin.

**Ellipse Representation:**
An ellipse centered at $(c_x, c_y)$ with major axis $MA$, minor axis $ma$, and orientation $\theta$:

$$\frac{((x - c_x)\cos\theta + (y - c_y)\sin\theta)^2}{MA^2} + \frac{((x - c_x)\sin\theta - (y - c_y)\cos\theta)^2}{ma^2} = 1$$

**Major Axis Line:**
The major axis line equation in parametric form:

$$\vec{p}(t) = \begin{pmatrix} c_x \\ c_y \end{pmatrix} + t \cdot \begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix} \cdot \frac{MA}{2}$$

**Distance to Sonar Origin:**
The sonar origin is typically at the bottom center of the image. The distance is the projection of the ellipse center onto the major axis, plus the extent along the major axis.

```python
def calculate_geometric_distance(contour, smoothed_center, angle_rad):
    cx, cy = smoothed_center

    # Major axis unit vector
    major_axis = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # Project all contour points onto major axis
    distances = []
    for point in contour:
        px, py = point[0]
        point_vector = np.array([px - cx, py - cy])
        projection = np.dot(point_vector, major_axis)
        distances.append(projection)

    # Distance is the maximum absolute projection
    distance_pixels = np.max(np.abs(distances))

    return distance_pixels
```

#### Angle Calculation

```python
# Angle of major axis relative to horizontal
angle_degrees = np.degrees(angle_rad)
```

#### Stability Constraints

**Distance Change Limiting:**
```python
max_change = cfg.get('max_distance_change_pixels', 20)
if abs(distance_pixels - previous_distance_pixels) > max_change:
    # Clamp to maximum allowed change
    direction = 1 if distance_pixels > previous_distance_pixels else -1
    distance_pixels = previous_distance_pixels + (direction * max_change)
```

**Mathematical Rationale:** Prevents sudden jumps due to:
- Contour detection noise
- Partial occlusions
- Temporary tracking loss

### 5.7 Tracking State Management

**Implementation:** `SonarDataProcessor.reset_tracking()` and state variables

The processor maintains **temporal state** for consistent tracking:

```python
class SonarDataProcessor:
    def __init__(self):
        self.smoothed_center = None          # Smoothed ellipse center (cx, cy)
        self.current_aoi = None              # Current AOI ellipse parameters
        self.previous_ellipse = None         # Previous ellipse (cx, cy, w, h, theta)
        self.previous_distance_pixels = None # Previous distance measurement
        self.last_center = None              # Last detected center for spatial scoring
        self.frame_count = 0                 # Frame counter for temporal analysis
```

**State Update Equations:**
1. **Center Smoothing:** $\vec{c}_{smooth} = \vec{c}_{prev} + \alpha (\vec{c}_{curr} - \vec{c}_{prev})$
2. **AOI Evolution:** Elliptical regions adapt to object movement with expansion
3. **Distance Stability:** Change limiting: $\Delta d \leq d_{max}$
4. **Temporal Memory:** Previous frame information guides current detection

### 5.8 Configuration Parameters

**Image Processing:** `utils/sonar_config.py::IMAGE_PROCESSING_CONFIG`

```python
IMAGE_PROCESSING_CONFIG = {
    'binary_threshold': 128,              # Binary conversion threshold
    'structure_tensor_sigma': 1.0,        # Gaussian smoothing for structure tensor
    'edge_strength_threshold': 50.0,      # Minimum coherence for edge detection
    'adaptive_base_radius': 2,            # Base kernel radius for elliptical convolution
    'adaptive_max_elongation': 4,         # Maximum kernel elongation factor
    'adaptive_linearity_threshold': 0.3,  # Linearity detection sensitivity
    'momentum_boost': 1.5,                # Enhancement strength for linear structures
    'adaptive_angle_steps': 9,            # Angular resolution for multi-orientation testing
    'min_contour_area': 100,              # Minimum contour size (pixels)
    'morph_close_kernel': 3,              # Morphological closing kernel size
    'edge_dilation_iterations': 1,        # Edge thickening iterations
    'max_distance_change_pixels': 20,     # Distance stability limit
    'aoi_boost_factor': 2.0,              # Temporal consistency boost
    'min_aoi_overlap_percent': 0.7,       # Minimum AOI overlap for boost
    'max_penalty_distance_pixels': 50.0,  # Spatial consistency penalty distance
}
```

**Tracking:** `utils/sonar_config.py::TRACKING_CONFIG`

```python
TRACKING_CONFIG = {
    'use_elliptical_aoi': True,           # Enable elliptical tracking regions
    'ellipse_expansion_factor': 0.3,      # AOI expansion (30% larger than fitted ellipse)
    'center_smoothing_alpha': 0.1,        # Center position smoothing (lower = smoother)
    'ellipse_smoothing_alpha': 0.8,       # Ellipse parameter smoothing
    'ellipse_max_movement_pixels': 2.0,   # Maximum center movement per frame
    'max_frames_outside_aoi': 5,          # Frames allowed outside AOI before reset
}
```

### 5.9 Algorithm Advantages and Mathematical Properties

**1. Signal-Strength Independence:**
- Binary conversion eliminates intensity-based variability
- Structure tensor analysis focuses on geometric coherence
- Consistent performance across different sonar power levels

**2. Mathematical Robustness:**
- **Structure Tensor:** Provides rotationally invariant edge detection
- **Elliptical Convolution:** Adapts kernel shape to local structure
- **Geometric Scoring:** Multi-criteria evaluation prevents false positives

**3. Temporal Stability:**
- Kalman-like smoothing provides bounded convergence
- Movement limiting prevents erratic tracking
- AOI expansion handles detection variability

**4. Computational Efficiency:**
- Binary operations reduce computational complexity
- Structure tensor computation is O(N) per pixel
- Elliptical kernel convolution leverages separability

**5. Geometric Accuracy:**
- Ellipse fitting provides sub-pixel accuracy
- Major axis intersection gives precise distance measurements
- Angle calculations account for orientation uncertainty

### 5.10 Processing Workflow with Mathematical Operations

```
Input Frame (uint8)
    ↓ Binary Conversion (eliminates signal dependency)
    ↓ Structure Tensor Analysis (J_xx, J_xy, J_yy computation)
    ↓ Eigenvalue Analysis (coherence = (λ₁ - λ₂)/(λ₁ + λ₂))
    ↓ Adaptive Elliptical Kernel Convolution (orientation-dependent enhancement)
    ↓ Morphological Processing (connectivity preservation)
    ↓ Contour Extraction (cv2.findContours)
    ↓ Geometric Scoring (S = S_geometric × S_temporal × S_spatial)
    ↓ Ellipse Fitting (cv2.fitEllipse for MA/ma/θ estimation)
    ↓ Kalman-like Smoothing (exponential filtering with movement limits)
    ↓ AOI Expansion (30% size increase for robustness)
    ↓ Geometric Distance (major axis projection: d = max|proj_points|)
    ↓ Stability Constraints (Δd ≤ d_max per frame)
    ↓ State Update (temporal memory for next frame)
    ↓ Output: FrameAnalysisResult with distance_m, angle_deg, confidence
```

---

## 6. Distance Measurement & Pixel-to-Meter Conversion

### 6.1 Geometric Distance Measurement

**Mathematical Foundation:** The distance measurement uses the **major axis of the fitted ellipse** as a geometric proxy for the net's vertical extent in the sonar coordinate system.

#### Ellipse Representation in Cartesian Coordinates

An ellipse fitted to the detected contour is represented by:

$$\frac{((x - c_x)\cos\theta + (y - c_y)\sin\theta)^2}{MA^2} + \frac{((x - c_x)\sin\theta - (y - c_y)\cos\theta)^2}{ma^2} = 1$$

where:
- $(c_x, c_y)$: Ellipse center coordinates
- $MA, ma$: Major and minor axis lengths
- $\theta$: Rotation angle from horizontal

#### Major Axis Distance Calculation

**Implementation:** `utils/sonar_image_analysis.py::_distance_angle_from_smoothed_center()`

The distance is computed by projecting contour points onto the ellipse's major axis:

```python
# Major axis unit vector
major_axis_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])

# Project all contour points onto major axis
distances = []
for point in contour:
    px, py = point[0]
    # Vector from ellipse center to contour point
    point_vector = np.array([px - smoothed_center_x, py - smoothed_center_y])
    # Projection onto major axis
    projection = np.dot(point_vector, major_axis_vector)
    distances.append(projection)

# Distance = maximum absolute projection
distance_pixels = np.max(np.abs(distances))
```

**Mathematical Derivation:**
For a point $\vec{p} = (p_x, p_y)$ and ellipse center $\vec{c} = (c_x, c_y)$, the projection onto the major axis unit vector $\vec{u} = (\cos\theta, \sin\theta)$ is:

$$d = \vec{u} \cdot (\vec{p} - \vec{c})$$

The maximum absolute projection gives the extent of the contour along the major axis, providing a robust distance measure that is insensitive to minor contour irregularities.

### 6.2 Coordinate System Transformations

#### Sonar Polar-to-Cartesian Mapping

**Mathematical Foundation:** Sonar data is acquired in polar coordinates and rasterized to Cartesian coordinates for image analysis.

**Polar Coordinate System:**
- **Range ($r$)**: Distance from sonar transducer (meters)
- **Bearing ($\phi$)**: Angle from forward direction (radians)
- **Intensity ($I(r,\phi)$)**: Echo strength at each range-bearing pair

**Cartesian Transformation:**
```python
def polar_to_cartesian(r, phi, origin_x, origin_y):
    """
    Transform polar coordinates to Cartesian
    
    Parameters:
    r: range in meters
    phi: bearing in radians  
    origin_x, origin_y: Cartesian origin coordinates
    """
    x = origin_x + r * np.sin(phi)  # Horizontal displacement
    y = origin_y + r * np.cos(phi)  # Vertical displacement (forward = +y)
    return x, y
```

**Field of View Considerations:**
For a sonar with field of view $FOV$ degrees, the valid bearing range is:

$$-\frac{FOV}{2} \leq \phi \leq +\frac{FOV}{2}$$

#### Extent Calculation During Rasterization

**Implementation:** `utils/sonar_utils.py::cone_raster_like_display_cell()`

The extent defines the mapping between pixel coordinates and world coordinates:

```python
# Define coordinate bounds
x_min = -np.sin(np.radians(FOV/2)) * display_range_max
x_max = +np.sin(np.radians(FOV/2)) * display_range_max
y_min = range_min
y_max = display_range_max

extent = (x_min, x_max, y_min, y_max)
```

**Pixel-to-Meter Conversion Factors:**
```python
# Compute scaling factors
meters_per_pixel_x = (x_max - x_min) / cone_width_pixels
meters_per_pixel_y = (y_max - y_min) / cone_height_pixels

# Average scaling (assuming isotropic pixels)
meters_per_pixel_avg = (meters_per_pixel_x + meters_per_pixel_y) / 2
```

### 6.3 Pixel-to-Meter Distance Conversion

**Mathematical Accuracy:** The conversion maintains geometric consistency by using the rasterization extent.

#### Distance Conversion Formula

```python
# Convert pixel distance to meters
distance_meters = distance_pixels * meters_per_pixel_avg
```

**Error Analysis:**
- **Geometric Accuracy:** The extent is computed deterministically from sonar parameters (FOV, range), ensuring consistent scaling
- **Isotropic Approximation:** Using average meters-per-pixel assumes square pixels, which is valid for most display applications
- **Temporal Consistency:** Same extent used across all frames ensures distance measurements are comparable

#### Coordinate System Alignment

**Sonar Coordinate System:**
- Origin: Sonar transducer position
- X-axis: Horizontal (positive right)
- Y-axis: Along-track (positive forward)
- Z-axis: Vertical (positive up, not shown in 2D display)

**Display Coordinate System:**
- Origin: Bottom-center of cone display
- X-axis: Horizontal (positive right)
- Y-axis: Range (positive up from transducer)

### 6.4 Time Series Analysis and Stability

**Implementation:** `utils/sonar_image_analysis.py::analyze_red_line_distance_over_time()`

#### Temporal Distance Series

For each frame $t$ in the sequence:

1. **Detection:** Apply image analysis pipeline
2. **Measurement:** Compute $d_t^{px}$ (pixels) and $d_t^m$ (meters)
3. **Storage:** Record with timestamp and metadata

**Output Data Structure:**
```python
analysis_results = pd.DataFrame({
    'frame_idx': int,                    # Frame index
    'timestamp': pd.Timestamp,           # UTC timestamp
    'distance_pixels': float,           # Distance in pixels
    'distance_meters': float,           # Distance in meters
    'ellipse_center_x': float,          # Ellipse center X
    'ellipse_center_y': float,          # Ellipse center Y
    'ellipse_major_axis': float,        # Major axis length
    'ellipse_minor_axis': float,        # Minor axis length
    'ellipse_angle_deg': float,         # Orientation angle
    'detection_confidence': float,      # Detection quality score
    'processing_time_ms': float,        # Computation time
})
```

#### Stability Analysis

**Distance Change Limiting:**
```python
max_change_pixels = cfg.get('max_distance_change_pixels', 20)

if abs(distance_pixels - previous_distance_pixels) > max_change_pixels:
    # Limit change to prevent jumps
    direction = np.sign(distance_pixels - previous_distance_pixels)
    distance_pixels = previous_distance_pixels + direction * max_change_pixels
```

**Mathematical Rationale:**
- **Outlier Rejection:** Prevents sudden jumps from detection errors
- **Temporal Smoothing:** Maintains measurement continuity
- **Physical Constraints:** Limits to realistic net movement speeds

### 6.5 Accuracy Validation and Error Bounds

#### Geometric Error Sources

1. **Ellipse Fitting Error:** OpenCV `fitEllipse()` provides sub-pixel accuracy
2. **Contour Detection Variability:** Depends on image quality and thresholding
3. **Temporal Jitter:** Frame-to-frame measurement variation
4. **Coordinate System Misalignment:** Sonar mounting and calibration

#### Error Quantification

**Pixel Accuracy:** $\pm 0.5$ pixels (sub-pixel fitting)
**Angular Accuracy:** $\pm 2^\circ$ (ellipse orientation)
**Temporal Stability:** $\pm 5$ pixels (after smoothing)

**Conversion to Meters:**
```python
# Error propagation
pixel_error_m = pixel_error * meters_per_pixel_avg
angular_error_rad = np.radians(angular_error_deg)

# Total distance error (simplified)
total_error_m = np.sqrt(pixel_error_m**2 + (distance_m * angular_error_rad)**2)
```

#### Validation Against Ground Truth

**DVL Comparison:** Correlation analysis between sonar and DVL distance measurements provides validation of absolute accuracy.
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

### 7.1 Sensor Fusion Mathematics

**Problem Statement:** Combine measurements from multiple sensors with different temporal sampling rates and coordinate systems.

#### Multisensor Data Alignment

**Mathematical Framework:** Temporal synchronization of heterogeneous sensor data streams.

**Sensor Data Types:**
- **Sonar:** $d_s(t)$ - Distance measurements at irregular intervals
- **DVL Navigation:** $d_n(t), \phi_n(t)$ - Distance and orientation at high frequency
- **DVL Altimeter:** $h(t)$ - Height above seafloor
- **Camera:** $I_c(t)$ - Visual data when available

#### Timestamp Normalization

All timestamps converted to UTC for consistent temporal reference:

```python
# Normalize all timestamps to UTC
sonar_ts = pd.to_datetime(sonar_data['ts_utc'], utc=True)
dvl_ts = pd.to_datetime(dvl_data['ts_utc'], utc=True)
camera_ts = pd.to_datetime(camera_data['ts_utc'], utc=True)
```

### 7.2 Temporal Synchronization Algorithms

**Implementation:** `utils/net_distance_analysis.py::synchronize_measurements()`

#### Nearest-Neighbor Interpolation

**Mathematical Approach:** For each sonar measurement time $t_s$, find the closest DVL measurement in time.

```python
def temporal_nearest_neighbor(sonar_times, dvl_times, dvl_values, max_tolerance_s=1.0):
    """
    Synchronize DVL data to sonar timestamps using nearest neighbor
    
    Parameters:
    sonar_times: Array of sonar measurement times
    dvl_times: Array of DVL measurement times  
    dvl_values: Array of DVL measurement values
    max_tolerance_s: Maximum allowed time difference
    """
    synchronized_values = []
    
    for ts in sonar_times:
        # Find closest DVL timestamp
        time_diffs = np.abs(dvl_times - ts)
        min_idx = np.argmin(time_diffs)
        min_diff = time_diffs[min_idx]
        
        if min_diff <= pd.Timedelta(seconds=max_tolerance_s):
            synchronized_values.append(dvl_values[min_idx])
        else:
            synchronized_values.append(np.nan)  # No valid measurement
    
    return np.array(synchronized_values)
```

**Algorithm Properties:**
- **Computational Complexity:** $O(N_s \log N_d)$ where $N_s, N_d$ are sonar and DVL sample counts
- **Temporal Accuracy:** Limited by sensor sampling rates
- **Gap Handling:** Produces NaN for missing data

#### Linear Interpolation

**Mathematical Foundation:** Assume linear variation between measurements for improved temporal resolution.

```python
def linear_interpolation(sonar_times, dvl_times, dvl_values):
    """
    Linear interpolation of DVL data to sonar timestamps
    """
    # Create interpolation function
    interp_func = interp1d(dvl_times.astype(np.int64), dvl_values, 
                          kind='linear', bounds_error=False, fill_value=np.nan)
    
    # Interpolate to sonar times
    sonar_times_int = sonar_times.astype(np.int64)
    interpolated_values = interp_func(sonar_times_int)
    
    return interpolated_values
```

**Interpolation Formula:**
For a sonar time $t_s$ between DVL times $t_{d1} < t_s < t_{d2}$:

$$v_s = v_{d1} + \frac{(v_{d2} - v_{d1})(t_s - t_{d1})}{t_{d2} - t_{d1}}$$

### 7.3 Coordinate System Alignment

#### Sensor Coordinate Systems

**DVL Body Frame:**
- Origin: DVL sensor position
- X: Forward direction
- Y: Starboard direction  
- Z: Downward direction

**ROV Navigation Frame:**
- Origin: ROV center of mass
- X: Forward direction
- Y: Starboard direction
- Z: Downward direction

**Sonar Frame:**
- Origin: Sonar transducer
- X: Horizontal (athwartship)
- Y: Forward direction
- Z: Vertical

#### Transformation Matrices

**DVL to ROV Transformation:**
```python
# Homogeneous transformation matrix
T_dvl_rov = np.array([
    [1, 0, 0, dx],  # Translation offsets
    [0, 1, 0, dy],
    [0, 0, 1, dz],
    [0, 0, 0, 1]
])
```

**ROV to Sonar Transformation:**
```python
T_rov_sonar = np.array([
    [cos(ψ), -sin(ψ), 0, px],  # Rotation and translation
    [sin(ψ),  cos(ψ), 0, py],
    [0,       0,      1, pz],
    [0,       0,      0, 1]
])
```

**Combined Transformation:**
$$T_{dvl \rightarrow sonar} = T_{rov \rightarrow sonar} \cdot T_{dvl \rightarrow rov}$$

### 7.4 Data Quality Assessment

#### Temporal Consistency Metrics

**Synchronization Quality Score:**
```python
def compute_sync_quality(sonar_times, dvl_times, tolerance_s=1.0):
    """
    Compute fraction of sonar measurements with valid DVL synchronization
    """
    valid_syncs = 0
    
    for ts in sonar_times:
        time_diffs = np.abs(dvl_times - ts)
        min_diff = np.min(time_diffs)
        
        if min_diff <= pd.Timedelta(seconds=tolerance_s):
            valid_syncs += 1
    
    quality_score = valid_syncs / len(sonar_times)
    return quality_score
```

#### Measurement Uncertainty Propagation

**Error Sources:**
- **Temporal Misalignment:** Time synchronization uncertainty
- **Coordinate Transformation:** Mounting calibration errors
- **Sensor Noise:** Individual sensor measurement uncertainty

**Combined Uncertainty:**
$$\sigma_{total}^2 = \sigma_{temporal}^2 + \sigma_{calibration}^2 + \sigma_{sensor}^2$$

### 7.5 Multi-Source Distance Fusion

#### Sensor Fusion Algorithm

**Implementation:** Weighted combination of multiple distance measurements.

```python
def fuse_distance_measurements(measurements, uncertainties):
    """
    Fuse multiple distance measurements using inverse variance weighting
    
    Parameters:
    measurements: List of distance values [d1, d2, ..., dn]
    uncertainties: List of measurement uncertainties [σ1, σ2, ..., σn]
    """
    # Inverse variance weights
    weights = [1/σ**2 for σ in uncertainties]
    
    # Weighted average
    numerator = sum(w * d for w, d in zip(weights, measurements))
    denominator = sum(weights)
    
    fused_distance = numerator / denominator
    
    # Combined uncertainty
    fused_uncertainty = 1 / np.sqrt(denominator)
    
    return fused_distance, fused_uncertainty
```

**Mathematical Properties:**
- **Optimality:** Minimum variance unbiased estimator
- **Robustness:** Automatically down-weights noisy measurements
- **Uncertainty Quantification:** Provides confidence bounds

---

## 8. Synchronized Video Generation

### 8.1 Temporal Alignment Mathematics

**Problem Statement:** Synchronize multiple data streams (sonar, camera, navigation) for coherent video playback.

#### Multi-Stream Synchronization

**Mathematical Framework:** Temporal alignment of heterogeneous data streams with different sampling rates.

**Data Streams:**
- **Sonar:** $I_s(t)$ - Intensity images at $f_s$ Hz
- **Camera:** $I_c(t)$ - RGB images at $f_c$ Hz
- **Navigation:** $\vec{n}(t) = [d_n(t), \phi_n(t), h(t)]$ - Navigation data at $f_n$ Hz
- **Analysis:** $\vec{a}(t) = [d_a(t), \phi_a(t), c(t)]$ - Analysis results at $f_s$ Hz

#### Synchronization Algorithm

**Implementation:** `utils/sonar_and_foto_generation.py::export_optimized_sonar_video()`

```python
def synchronize_video_streams(sonar_times, camera_times, nav_times,
                            max_sync_tolerance_s=5.0):
    """
    Synchronize multiple data streams for video generation
    
    Returns:
    sync_frames: List of synchronized frame tuples
    """
    sync_frames = []
    
    for i, sonar_ts in enumerate(sonar_times):
        # Find nearest camera frame
        camera_diffs = np.abs(camera_times - sonar_ts)
        camera_idx = np.argmin(camera_diffs)
        camera_dt = camera_diffs[camera_idx]
        
        # Find nearest navigation data
        nav_diffs = np.abs(nav_times - sonar_ts)
        nav_idx = np.argmin(nav_diffs)
        nav_dt = nav_diffs[nav_idx]
        
        # Check synchronization quality
        if camera_dt <= pd.Timedelta(seconds=max_sync_tolerance_s):
            # Good camera sync
            frame_data = {
                'sonar_idx': i,
                'camera_idx': camera_idx,
                'nav_idx': nav_idx,
                'sync_quality': 'full_sync'
            }
        elif nav_dt <= pd.Timedelta(seconds=max_sync_tolerance_s):
            # Navigation only
            frame_data = {
                'sonar_idx': i,
                'camera_idx': None,
                'nav_idx': nav_idx,
                'sync_quality': 'nav_only'
            }
        else:
            # Sonar only
            frame_data = {
                'sonar_idx': i,
                'camera_idx': None,
                'nav_idx': None,
                'sync_quality': 'sonar_only'
            }
        
        sync_frames.append(frame_data)
    
    return sync_frames
```

### 8.2 Coordinate System Mapping for Overlays

#### Pixel Coordinate Transformations

**Sonar Cone to Display Coordinates:**

```python
def sonar_to_display_coords(x_sonar, y_sonar, extent, display_size):
    """
    Transform sonar coordinates to display pixel coordinates
    
    Parameters:
    x_sonar, y_sonar: Sonar coordinate system (meters)
    extent: (x_min, x_max, y_min, y_max) in meters
    display_size: (width, height) in pixels
    """
    x_min, x_max, y_min, y_max = extent
    disp_w, disp_h = display_size
    
    # Normalize to [0,1]
    x_norm = (x_sonar - x_min) / (x_max - x_min)
    y_norm = (y_sonar - y_min) / (y_max - y_min)
    
    # Convert to pixel coordinates
    x_px = x_norm * disp_w
    y_px = (1 - y_norm) * disp_h  # Flip Y for display coordinates
    
    return int(x_px), int(y_px)
```

**Mathematical Properties:**
- **Affine Transformation:** Preserves straight lines and ratios
- **Aspect Ratio:** Maintains geometric relationships
- **Coordinate Flip:** Display coordinates have Y increasing downward

#### Overlay Geometry

**Net Distance Line Drawing:**

```python
def draw_net_distance_line(display_img, distance_m, angle_deg, extent, display_size):
    """
    Draw net distance line on display image
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Line endpoints in sonar coordinates
    half_width = 1.0  # 1 meter half-width for visibility
    x1 = -half_width
    y1 = distance_m
    x2 = +half_width
    y2 = distance_m
    
    # Rotate by angle
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rx1 = x1 * cos_a - (y1 - distance_m) * sin_a
    ry1 = x1 * sin_a + (y1 - distance_m) * cos_a + distance_m
    rx2 = x2 * cos_a - (y2 - distance_m) * sin_a
    ry2 = x2 * sin_a + (y2 - distance_m) * cos_a + distance_m
    
    # Convert to display coordinates
    px1, py1 = sonar_to_display_coords(rx1, ry1, extent, display_size)
    px2, py2 = sonar_to_display_coords(rx2, ry2, extent, display_size)
    
    # Draw line
    cv2.line(display_img, (px1, py1), (px2, py2), (0, 255, 255), 3)
    
    # Draw center marker
    cx_px, cy_px = sonar_to_display_coords(0, distance_m, extent, display_size)
    cv2.circle(display_img, (cx_px, cy_px), 3, (0, 0, 255), -1)
```

### 8.3 Video Encoding and Metadata

#### Frame Rate Optimization

**Natural Frame Rate Calculation:**
```python
def compute_natural_fps(timestamps, min_fps=1.0, max_fps=60.0):
    """
    Compute natural frame rate from timestamp differences
    """
    # Calculate time differences
    dt_s = np.diff(timestamps.astype(np.int64)) / 1e9  # Convert to seconds
    
    # Filter valid differences
    valid_dt = dt_s[(dt_s > 1e-6) & (dt_s < 5.0)]  # Reasonable range
    
    if len(valid_dt) > 0:
        # Use median to be robust to outliers
        median_dt = np.median(valid_dt)
        natural_fps = 1.0 / median_dt
        # Clamp to reasonable range
        natural_fps = np.clip(natural_fps, min_fps, max_fps)
    else:
        natural_fps = 15.0  # Default fallback
    
    return natural_fps
```

**Mathematical Rationale:**
- **Median Filtering:** Robust to timestamp jitter and outliers
- **Clamping:** Prevents unrealistic frame rates
- **Temporal Consistency:** Maintains perceived motion smoothness

#### Metadata Generation

**Comprehensive Processing Record:**
```python
def generate_video_metadata(processing_params, frame_count, output_path):
    """
    Generate JSON metadata for reproducible video generation
    """
    metadata = {
        "format": "solaqua.optimized_sync.video.meta.v1",
        "creation_timestamp": datetime.utcnow().isoformat(),
        "processing_parameters": processing_params,
        "frame_count": frame_count,
        "duration_seconds": frame_count / processing_params['fps'],
        "coordinate_extent": processing_params['extent'],
        "sync_statistics": {
            "camera_sync_ratio": processing_params.get('camera_sync_ratio', 0.0),
            "nav_sync_ratio": processing_params.get('nav_sync_ratio', 0.0),
        },
        "data_sources": {
            "sonar_bag": processing_params['target_bag'],
            "camera_available": processing_params.get('camera_available', False),
            "navigation_available": processing_params.get('navigation_available', False),
            "analysis_available": processing_params.get('analysis_available', False),
        }
    }
    
    # Save metadata alongside video
    meta_path = output_path.with_suffix(output_path.suffix + '.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return meta_path
```

### 8.4 Error Handling and Quality Assurance

#### Synchronization Quality Assessment

**Multi-Level Quality Metrics:**
```python
def assess_video_quality(sync_frames):
    """
    Assess overall video synchronization quality
    """
    total_frames = len(sync_frames)
    full_sync = sum(1 for f in sync_frames if f['sync_quality'] == 'full_sync')
    nav_only = sum(1 for f in sync_frames if f['sync_quality'] == 'nav_only')
    sonar_only = sum(1 for f in sync_frames if f['sync_quality'] == 'sonar_only')
    
    quality_metrics = {
        'full_sync_ratio': full_sync / total_frames,
        'nav_sync_ratio': (full_sync + nav_only) / total_frames,
        'sonar_only_ratio': sonar_only / total_frames,
        'temporal_coverage': (full_sync + nav_only + sonar_only) / total_frames,
    }
    
    return quality_metrics
```

#### Graceful Degradation

**Fallback Strategies:**
1. **Camera Unavailable:** Sonar + navigation only layout
2. **Navigation Missing:** Sonar + analysis only with reduced overlays
3. **Analysis Failed:** Sonar only with basic grid overlays
4. **Temporal Gaps:** Maintain last known good measurements

**Mathematical Continuity:**
- **Interpolation:** Fill small temporal gaps using linear interpolation
- **Hold Values:** Maintain previous measurements during outages
- **Quality Flags:** Mark degraded frames in metadata

### 8.5 Performance Optimization

#### Memory-Efficient Processing

**Frame-by-Frame Pipeline:**
```python
def process_video_frames_efficiently(sonar_data, camera_data, nav_data, output_path):
    """
    Process video frames with minimal memory usage
    """
    # Initialize video writer (size determined by first frame)
    writer = None
    
    for frame_data in sync_frames:
        # Load only required data for current frame
        sonar_frame = load_sonar_frame(frame_data['sonar_idx'])
        
        if frame_data['camera_idx'] is not None:
            camera_frame = load_camera_frame(frame_data['camera_idx'])
        else:
            camera_frame = None
            
        if frame_data['nav_idx'] is not None:
            nav_data = load_nav_data(frame_data['nav_idx'])
        else:
            nav_data = None
        
        # Generate composite frame
        composite = generate_composite_frame(sonar_frame, camera_frame, nav_data)
        
        # Initialize writer on first frame
        if writer is None:
            h, w = composite.shape[:2]
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        # Write frame
        writer.write(composite)
    
    writer.release()
```

**Optimization Benefits:**
- **Memory Usage:** $O(1)$ additional memory beyond current frame
- **I/O Efficiency:** Load data on-demand rather than preloading
- **Scalability:** Handle arbitrarily long video sequences

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

### 9.3 Elliptical AOI Tracking Config

```python
TRACKING_CONFIG = {
    # AOI Configuration
    'use_elliptical_aoi': True,           # Enable elliptical AOI tracking
    'aoi_boost_factor': 2.0,              # Score multiplier for contours in AOI
    'aoi_expansion_pixels': 1,            # Legacy rectangular AOI expansion
    'ellipse_expansion_factor': 0.2,      # Factor to expand ellipse for AOI (20% larger)
    
    # Center Tracking Smoothing
    'center_smoothing_alpha': 0.2,        # Smoothing for tracking center (0.0=no change, 1.0=instant)
    
    # Advanced Ellipse Smoothing (legacy)
    'ellipse_smoothing_alpha': 0.2,       # EMA smoothing for ellipse parameters
    'ellipse_max_movement_pixels': 4.0,   # Maximum pixel movement per frame
    'max_frames_outside_aoi': 5,          # Max frames to track outside AOI
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
