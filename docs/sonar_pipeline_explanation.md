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
11. [End-to-End Data Flow (Bag → Distance Series)](#11-end-to-end-data-flow-bag--distance-series)
12. [Proposed Single-Call Pipeline](#12-proposed-single-call-pipeline)

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

> **Note:** The main sonar analysis pipeline focuses on **SonoptixECHO** data from video bags, 
> which explains why sonar CSV files have the `_video.csv` suffix throughout the codebase.

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
    "layout": {932265
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

Cone rasters from the previous stage are inspected by `utils/sonar_analysis.py::analyze_npz_sequence()`. The routine merges `IMAGE_PROCESSING_CONFIG` and `TRACKING_CONFIG`, instantiates a `NetTracker`, and iterates the requested frame indices. Each loop produces a `FrameAnalysisResult` entry that captures detection measurements and tracker status.

### 5.1 Per-frame pipeline (`analyze_npz_sequence`)

For every frame the analyzer performs:

1. **Grayscale preparation** – `to_uint8_gray()` rescales the floating-point cone into an 8-bit image (0–255) so OpenCV morphology works deterministically.
2. **Binary projection** – threshold each pixel (`binary_threshold`) to isolate strong reflectors; the result is a 0/255 mask that is intensity-invariant.
3. **Edge post-processing** – call `preprocess_edges(binary, config)` which now includes an advanced momentum merge path; on failure the raw binary mask acts as the edge map (see §5.2).
4. **Tracking update** – pass the processed edges to `tracker.find_and_update()` which returns the best contour (or `None`) and updates internal state (see §5.3–§5.4).
5. **Distance extraction** – `tracker.calculate_distance()` converts the tracked ellipse into a perpendicular (red-line) pixel distance plus heading angle (§5.5).
6. **Meter conversion** – scale the pixel distance by `(extent[3] - extent[2]) / image_height` and add `extent[2]` to recover the absolute range in meters.
7. **Result logging** – append a dict with frame index, timestamp, `distance_pixels`, `distance_meters`, `angle_degrees`, `detection_success`, `tracker.get_status()`, and contour area (0 when absent).

Progress is printed every 50 frames; optional CSV export mirrors the in-memory DataFrame.

### 5.2 Binary & morphological enhancement (`preprocess_edges`)

`utils.image_enhancement.preprocess_edges()` consumes the 8-bit cone frame and outputs a tuple `(raw_edges, enhanced_edges)`:

- **Binary projection** – re-applies the global `binary_threshold` to guarantee the input is strictly 0/255 regardless of caller state.
- **Adaptive momentum merge (advanced mode)** – when `use_advanced_momentum_merging=True`, the function:
  - Runs structure-tensor analysis on a downscaled copy to obtain orientation and linearity maps.
  - Keeps only the strongest angle bins (coverage and linearity driven) and builds ROI-specific elliptical kernels aligned with each bin.
  - Convolves those ROIs, blends the responses using linearity weights, and combines them with a Gaussian baseline (momentum boost + soft clipping) to emphasize elongated net strands while suppressing noise.
- **Basic enhancement fallback** – if the advanced flag is disabled:
  - Uses an elliptical dilation (`basic_dilation_kernel_size`, `basic_dilation_iterations`) to grow the binary mask, or optionally a Gaussian blur when dilation is disabled.
  - Produces a smoothed mask that still preserves structural continuity.
- **Laplacian edge extraction** – both the original binary frame and the enhanced mask are convolved with a fixed 3×3 Laplacian kernel; outputs are binarized to 0/255 to obtain `raw_edges` and `enhanced_edges`.
- **Morphological cleanup** – applies closing (`morph_close_kernel`) and dilation (`edge_dilation_iterations`) to `enhanced_edges`, bridging gaps and thickening lines so contours remain connected during tracking.
- **Error handling** – any exception bubbles up to `analyze_npz_sequence`, which then falls back to using the plain binary mask as the edge image, guaranteeing continuity in the detection loop.

The analyzer typically discards `raw_edges` and relies on the post-processed `enhanced_edges`, but returning both permits debugging and visualization of pre/post enhancement effects.

### 5.3 NetTracker architecture

`NetTracker` (defined in `utils/sonar_tracking.py`) maintains smoothed ellipse parameters (`center`, `size`, `angle`) and a loss counter. Its core responsibilities are:

- Restricting search to plausible regions.
- Selecting the most likely net contour.
- Smoothing ellipse parameters across time.
- Producing stabilized distance/angle outputs (`calculate_distance`).
- Resetting when detections are lost for too many frames.

### 5.4 Masked contour search

`find_and_update(edges, shape)` begins by building an optional search mask through `_get_search_mask()`:

- Expands the last-known ellipse using `ellipse_expansion_factor`, inflating further when `frames_lost` grows (controlled by `aoi_decay_factor`).
- Optionally merges an oriented corridor (`_make_corridor_mask`) aligned with the major axis, whose width/length are scaled by `corridor_band_k` and `corridor_length_factor`.

The mask is `bitwise_and`-ed with the edge image to suppress distant clutter before `cv2.findContours` (retrieval mode `RETR_LIST`). `_find_best_contour()` filters candidates:

- Dynamically lowers `min_contour_area` when already tracking.
- Scores by area and proximity to the smoothed center (encouraging temporal continuity).
- Retains the contour with the highest score.

If no contour survives, `frames_lost` increments and `_reset()` triggers once it exceeds `max_frames_without_detection`.

### 5.5 Ellipse fitting and smoothing

When a contour is accepted (`len ≥ 5`):

1. `cv2.fitEllipse` yields `(center, size, angle)`.
2. Exponential smoothing applies separate alphas (`center_smoothing_alpha`, `ellipse_size_smoothing_alpha`, `ellipse_orientation_smoothing_alpha`).
3. Center deltas are clamped by `ellipse_max_movement_pixels`.
4. Angle smoothing handles ±180° wrap-around by normalizing differences into ±90° before blending.

These operations keep the tracker stable despite noisy detections.

### 5.6 Distance and angle estimation

`calculate_distance(width, height)` derives measurement outputs only when an ellipse is available:

- Treats the fitted angle as the major axis; the “red line” used operationally is the perpendicular (`angle + 90°`).
- Solves for the intersection between this red line and the image vertical centerline; the resulting y-coordinate (clipped to frame bounds) is the pixel distance.
- Caps per-frame distance jumps via `max_distance_change_pixels` and stores the last value for future smoothing.
- Returns the stabilized pixel distance and the red-line angle (degrees). If state is missing, it falls back to the last distance and `None` for angle.

### 5.7 Result packaging

The analysis loop attaches additional metadata:

- Converts pixels to meters using `(extent[3] - extent[2]) / image_height`.
- Records contour area (0 when absent), detection success, and `tracker.get_status()` which reports `"TRACKED"`, `"SEARCHING (lost_count/limit)"`, or `"LOST"`.
- Optionally writes a CSV summary when `save_outputs=True`.

This combination of binary preprocessing, masked contour search, smoothed ellipse tracking, and geometric distance projection enables robust detection of net-like structures across sonar frames.

---

## 6. Distance Measurement & Pixel-to-Meter Conversion

### 6.1 Overview

Accurate distance measurement is critical for sonar applications. This section details the methods used to convert pixel measurements from the sonar images into real-world meters.

### 6.2 Measurement Principles

Sonar images provide distance information in pixels, which must be calibrated to physical units. The conversion relies on:

- **NPZ-provided spatial extent:** Each NPZ file contains an `extent` tuple defining the spatial bounds in meters: `(x_min, x_max, y_min, y_max)`.
- **Image dimensions:** The pixel dimensions of the sonar image (height and width).
- **Calibration equations** that translate pixel coordinates to meters, considering the sonar's field of view and range settings.

### 6.3 Pixel-to-Meter Conversion Formula

Given a pixel coordinate `(px, py)` in the sonar image, the conversion to meters `(x_m, y_m)` is performed as:

```python
x_m = extent[0] + (px / image_width) * (extent[1] - extent[0])
y_m = extent[2] + (py / image_height) * (extent[3] - extent[2])
```

Where:
- `extent[0]`, `extent[1]` are the minimum and maximum x-coordinates in meters.
- `extent[2]`, `extent[3]` are the minimum and maximum y-coordinates in meters.
- `image_width`, `image_height` are the dimensions of the sonar image in pixels.

### 6.4 Distance Smoothing and Stabilization

To mitigate noise and sudden jumps in distance measurements:

- A moving average or exponential smoothing is applied to the distance time series.
- The smoothing parameters can be configured based on the desired responsiveness and stability.

### 6.5 Implementation

**Key Functions:**
- `utils/distance_measurement.py::convert_pixel_to_meter()`
- `utils/distance_measurement.py::smooth_distance_series()`

---

## 7. DVL Data Integration

### 7.1 Overview

The Doppler Velocity Log (DVL) provides high-accuracy velocity and distance measurements that complement the sonar data. Integrating DVL data enhances the robustness and reliability of the distance estimates.

### 7.2 DVL Data Processing

DVL data is processed to extract:
- **Bottom track velocity:** The velocity of the sensor over the seafloor.
- **Water track velocity:** The velocity of the water current.

For distance measurements, the bottom track data is primarily used.

### 7.3 Integration with Sonar Data

The integration process involves:
1. Time synchronization: Aligning DVL data timestamps with the sonar data timestamps.
2. Spatial transformation: Converting DVL measurements from the body frame of the ROV to the world frame.
3. Data fusion: Combining DVL and sonar measurements to produce a unified distance estimate.

### 7.4 Implementation

**Key Functions:**
- `utils/dvl_integration.py::process_dvl_data()`
- `utils/dvl_integration.py::transform_to_world_frame()`
- `utils/dvl_integration.py::fuse_dvl_sonar_data()`

---

## 8. Synchronized Video Generation

### 8.1 Overview

Synchronized video generation creates a visual output that overlays the sonar distance measurements and DVL data onto the camera footage. This provides an intuitive understanding of the underwater scene and the detected objects.

### 8.2 Video Processing Pipeline

The video generation pipeline includes:

1. **Input synchronization:** Ensuring all input data (camera, sonar, DVL) are time-aligned.
2. **Data overlay:** Projecting the sonar and DVL data onto the video frames.
3. **Output encoding:** Compressing and saving the annotated video to disk.

### 8.3 Implementation

**Key Functions:**
- `utils/video_generation.py::synchronize_inputs()`
- `utils/video_generation.py::overlay_sonar_dvl_data()`
- `utils/video_generation.py::encode_video_output()`

---

## 9. Configuration Parameters

### 9.1 Overview

Configuration parameters control the behavior and settings of the SOLAQUA processing pipeline. They are defined in YAML files and can be overridden by command-line arguments or environment variables.

### 9.2 Parameter Categories

- **Input/Output Settings:** File paths, formats, and naming conventions.
- **Sonar Processing:** TVG settings, enhancement parameters, rasterization settings.
- **Image Analysis:** Net detection thresholds, contour filtering criteria.
- **Distance Measurement:** Calibration settings, smoothing parameters.
- **DVL Integration:** Transformation parameters, fusion settings.
- **Video Generation:** Encoding settings, output resolution.

### 9.3 Accessing Parameters

Parameters are accessed in the code via:

```python
from utils.config import Config

# Get a parameter
param_value = Config.get('section.subsection.parameter_name', default_value)
```

---

## 10. Code Reference Map

### 10.1 Overview

This section provides a high-level map of the SOLAQUA codebase, linking the documented pipeline stages to the corresponding implementation modules and functions.

### 10.2 Reference Table

| Pipeline Stage                     | Description                                      | Key Modules/Functions                               |
|-----------------------------------|--------------------------------------------------|-----------------------------------------------------|
| Data Extraction                   | Extracts data from ROS bags                     | `solaqua_export.py`, `utils/dataset_export_utils.py` |
| Sonar Enhancement                 | Applies TVG and other enhancements               | `utils/sonar_utils.py::enhance_intensity()`       |
| Polar-to-Cartesian Rasterization | Converts polar sonar data to Cartesian images   | `utils/sonar_utils.py::cone_raster_like_display_cell()` |
| Image Analysis & Net Detection    | Detects nets using computer vision              | `utils/sonar_tracking.py::NetTracker`              |
| Distance Measurement              | Converts pixel measurements to meters            | `utils/distance_measurement.py`                    |
| DVL Data Integration              | Integrates DVL data with sonar measurements     | `utils/dvl_integration.py`                         |
| Video Generation                  | Generates annotated video outputs                | `utils/video_generation.py`                        |

---

## 11. End-to-End Data Flow (Bag → Distance Series)

1. **Acquisition & Export (`scripts/solaqua_export.py`)**
   - `SOLAQUACompleteExporter.export_csv_data()` reads `*_data.bag` and `*_video.bag` files, flattening ROS topics into `exports/by_bag/*.csv` (telemetry, sonar, nav, DVL).  
   - `export_all_video_bags_to_mp4/png` (invoked via `export_videos` / `export_frames`) produces camera references.  
   - `create_cone_npz()` wraps `utils.sonar_utils.save_cone_run_npz()` to rasterize SonoptixECHO frames into `exports/outputs/{bag}_cones.npz`, embedding `extent` metadata needed later.

2. **Sonar Processing (`utils/sonar_analysis.py`)**
   - `analyze_npz_sequence()` consumes the NPZ cones plus `IMAGE_PROCESSING_CONFIG` / `TRACKING_CONFIG`, runs the tracker pipeline described in §§3–5, and emits a per-frame DataFrame with `distance_pixels`, `distance_meters`, and `angle_degrees`.  
   - Optional CSV export (`save_outputs=True`) writes `{bag}_analysis.csv` into the batch folder (e.g., `basic_full_batch/`).

3. **Fusion & Comparison (`utils/net_analysis.py`, `utils/comparison_analysis.py`)**
   - `prepare_three_system_comparison()` loads the sonar analysis CSV, DVL navigation CSV, and FFT pose CSV, time-aligns them, adds XY estimates, and saves `{bag}_raw_comparison.csv`.  
   - `load_and_prepare_data()` + downstream routines (`detect_stable_segments`, `compute_distance_pitch_statistics`, etc.) perform the statistical evaluation used in `06_full_net_analysis.ipynb`.

---

## 12. Proposed Single-Call Pipeline

The following helper coordinates the full workflow for a single bag ID—exporting required assets (if missing), generating cone NPZ files, running the sonar tracker, fusing DVL/FFT data, and returning net-relative distance/orientation series.

```python
def compute_net_distance_and_orientation(
    bag_id: str,
    data_dir: Path,
    exports_dir: Path,
    *,
    frame_count: int = 3000,
    frame_step: int = 1,
    tolerance_s: float = 0.5,
) -> pd.DataFrame:
    """Return a synchronized time series with sonar/DVL/FFT distances & orientations."""
    # 1) Ensure exports exist
    exporter = SOLAQUACompleteExporter(data_dir, exports_dir)
    exporter.export_csv_data()
    exporter.create_cone_npz(max_bags=None)  # no-op if NPZ already present

    # 2) Run sonar analysis (frames → distance_meters / angle_degrees)
    npz_files = iau.get_available_npz_files()
    target_idx = next(i for i, p in enumerate(npz_files) if bag_id in p.name)
    sonar_df = iau.analyze_npz_sequence(
        npz_file_index=target_idx,
        frame_start=1,
        frame_count=frame_count,
        frame_step=frame_step,
        save_outputs=True,
    )

    # 3) Build three-system comparison & return synchronized dataset
    df_sonar, _ = load_sonar_analysis_results(bag_id)
    df_nav, _ = load_navigation_dataset(bag_id)
    df_fft, _ = load_fft_dataset(bag_id, fft_root=Path("/Volumes/LaCie/SOLAQUA/relative_fft_pose"))

    comparison_df, _, _ = prepare_three_system_comparison(
        bag_id,
        df_sonar,
        df_nav,
        df_fft,
        tolerance_seconds=tolerance_s,
    )
    return ensure_xy_columns(comparison_df)
```

**Characteristics:**
- **Input:** `bag_id` plus data/export roots.
- **Side effects:** Reuses the existing export/analysis machinery; skips steps automatically when artifacts already exist.
- **Output:** A synchronized DataFrame containing `sonar_distance_m`, `nav_distance_m`, `fft_distance_m`, `sonar_pitch_deg`, `fft_pitch_deg`, `nav_pitch_deg`, and XY columns—exactly what downstream notebooks consume for statistics, plots, or real-time monitoring.

This orchestration wraps the entire pipeline into a single call suitable for CLI tools, dashboards, or automated QA jobs.

---

## Appendix A: Mathematical Foundations

### A.1 TVG Compensation

Time-Varying Gain (TVG) compensation corrects for signal attenuation in sonar data. The compensation is based on the inverse square law for geometric spreading and an exponential model for water absorption.

**Amplitude TVG:**
```math
TVG(r) = (r / r0)²
```

**Power TVG:**
```math
TVG(r) = (r / r0)⁴
```

Where `r` is the range, and `r0` is a reference distance.

### A.2 Ellipse Fitting

Ellipse fitting is used to model the shape of detected nets. The fitting is performed using the least squares method to minimize the algebraic distance between the data points and the ellipse.

**Ellipse Equation:**
```math
Ax² + Bxy + Cy² + Dx + Ey + F = 0
```

The parameters `(A, B, C, D, E, F)` are estimated from the contour points using singular value decomposition (SVD).

### A.3 Contour Detection

Contours are detected using the Suzuki and Abe algorithm, which is an improved version of the Moore-Neighbor tracing algorithm. It is efficient in detecting contours in binary images and is robust to noise.

**Key Steps:**
1. Find the most top-left point as the starting point.
2. Trace the contour using 8-connected neighbors.
3. Approximate the contour using the Douglas-Peucker algorithm to reduce the number of points.

### A.4 Coordinate Transformations

Transformations between different coordinate systems (e.g., body frame to world frame) are performed using rigid body transformation equations:

```math
\begin{bmatrix}
x' \\
y' \\
z'
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta) & -\sin(\theta) & 0 \\
\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
+
\begin{bmatrix}
t_x \\
t_y \\
t_z
\end{bmatrix}
```

Where `(x, y, z)` are the original coordinates, `(x', y', z')` are the transformed coordinates, `θ` is the rotation angle, and `(t_x, t_y, t_z)` are the translation parameters.

---

