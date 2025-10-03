# SOLAQUA: Complete Technical Pipeline Analysis

This document describes the complete SOLAQUA sonar data processing pipeline, from raw ROS bag extraction to the interactive distance comparison between sonar-based image analysis and DVL navigation measurements. It includes mathematical methods, algorithmic choices, and exact code references where the implementation lives in the repository.

---

## 1. Overview

Purpose: provide a reproducible, mathematically precise description of how raw sonar data becomes a distance measurement that can be compared to DVL (navigation) data. This covers:

- Bag file extraction to CSV/NPZ
- Sonar enhancement and TVG compensation
- Polar → Cartesian rasterization (cone display)
- Image preprocessing, edge detection and contour scoring
- Ellipse fitting and net-distance estimation
- Pixel → meter conversion and synchronization with navigation data
- Relevant code references

Files referenced below are located in the repository root and `utils/`:

- `solaqua_export.py` — bag export and NPZ creation
- `utils/dataset_export_utils.py` — helpers for bag extraction
- `utils/sonar_utils.py` — enhancement, rasterization, coordinate transforms
- `utils/sonar_image_analysis.py` — contour detection, distance measurement, plotting
- `utils/net_distance_analysis.py` — DVL/navigation data loading and aggregation
- `utils/sonar_config.py` — centralized configuration

---

## 2. Data extraction from ROS bag files

### 2.1 Bag file types

- Data bags: telemetry and sensor topics (sonar, navigation, DVL, USBL, etc.)
- Video bags: camera footage (exported to PNG/MP4 sequences)

### 2.2 CSV / Parquet export

Implementation: `solaqua_export.py` delegates to functions in `utils/dataset_export_utils.py` which:

- enumerate topics in `.bag` files
- export per-topic CSV (or Parquet) tables
- normalize timestamps to one canonical tz (UTC) and include `ts_utc`/`timestamp` fields
- parse JSON-encoded fields and flatten nested structures

Why: tabular exports make downstream processing deterministic and easy to debug. Storing Parquet (when available) is faster to reload.

### 2.3 Sonoptix / Sonar data schema

Sonoptix-style sonar rows hold either an `image` (nested lists) or `data` with `dim_*` metadata.

Key points:

- data are polar: rows ≈ range bins, cols ≈ beam angles
- each row has a timestamp and optional metadata (FOV, rmin, rmax, beam count)
- the repository helper `get_sonoptix_frame()` in `utils/sonar_utils.py` reads a single sonar image from a CSV/Parquet row and returns a 2D numpy array

Reference: `utils/sonar_utils.py::get_sonoptix_frame()`

---

## 3. Sonar preprocessing & enhancement

Implementation: `utils/sonar_utils.py::enhance_intensity()`

### 3.1 Goals

- Compensate for geometric spreading and time-varying gain (TVG)
- Optionally apply absorption correction (alpha dB/m)
- Convert to a display-friendly domain (log / decibel) while being robust to zeros
- Normalize via percentile clipping and gamma correction for stable contrast

### 3.2 Steps and equations

1. Range grid: create `ranges = linspace(rmin, rmax, H)` where `H` is range bins.

2. TVG compensation (amplitude or power):

   - amplitude -> scale ∝ r^2
   - power -> scale ∝ r^4

   Implementation multiplies each range row by the chosen TVG gain.

3. Absorption correction (optional): multiply by an exponential factor:

   gain_abs(r) = exp(alpha_db_per_m * r / 20)  (converts dB/m into amplitude gain factor)

4. Logarithmic scaling to decibels (zero-aware):

   I_db = 20 * log10(I_compensated + eps_log)

5. Robust percentile normalization: compute p_low and p_high percentiles of I_db, clamp, scale to [0,1], then apply gamma.

Notes: The code uses configurable parameters: `scale`, `tvg`, `alpha_db_per_m`, `eps_log`, `p_low`, `p_high`, `gamma`.

Reference: `utils/sonar_utils.py::enhance_intensity()`

---

## 4. Polar → Cartesian rasterization (cone display)

Purpose: produce a regular Cartesian image (cone display) with forward as +Y and starboard as +X. This image is what the image-processing pipeline operates on.

Implementation: `utils/sonar_utils.py::rasterize_cone()` and `cone_raster_like_display_cell()`

### 4.1 Coordinate mapping

Given a sonar matrix Z of shape (H_range, W_beams) with FOV (degrees), r_min and r_max:

1. Define grid extent: half = deg2rad(FOV/2)
2. y ∈ [y_min, y_max], x ∈ [x_min, x_max] where x_max = sin(half) * y_max and x_min = -x_max
3. For every pixel (x, y) in the target grid compute polar coords:

   theta = atan2(x, y)   # angle measured from +Y axis, starboard positive
   r = sqrt(x^2 + y^2)

4. Map (r, theta) back to sonar array indices:

   rowf = (r - r_min) / (r_max - r_min) * (H - 1)
   colf = (theta + half) / (2*half) * (W - 1)

5. Round/clamp and sample: rows = round(rowf), cols = round(colf). The code uses nearest sampling (rint) for simplicity.

6. Mask out points outside [r_min, r_max] or outside the angular sector.

Output: `cone` image plus `extent = (x_min, x_max, y_min, y_max)` in meters.

Notes: Bilinear interpolation would produce smoother resampling but nearest sampling is simpler and faster. The `extent` enables pixel→meter conversion.

Reference: `utils/sonar_utils.py::rasterize_cone()` and `cone_raster_like_display_cell()`

---

## 5. NPZ creation & why NPZ is used

Implementation: `solaqua_export.py` writes preprocessed frames to NPZ files (see function that creates cone NPZs).

Why NPZ:

- Stores `cones` as a compact binary array `shape = (T, H, W)` for fast random access
- Stores `timestamps` (tz-aware), `extent` and `meta` for reproducible processing
- Avoids repeatedly re-running enhancement and rasterization when analyzing multiple algorithms

Typical NPZ contents:

```json
{
  'cones': np.array([T, H, W]),
  'timestamps': np.array([T]),
  'extent': (x_min, x_max, y_min, y_max),
  'meta': {...}
}
```

Reference: `utils/sonar_image_analysis.py::load_cone_run_npz()` which loads exactly this layout.

---

## 6. Image processing pipeline (contour detection)

Implementation: `utils/sonar_image_analysis.py` (functions: `preprocess_edges()`, `directional_momentum_merge()`, `select_best_contour()`, `get_red_line_distance_and_angle()`)

### 6.1 Preprocessing

Steps applied to a grayscale cone image (`u8`):

1. Directional enhancement (momentum merge) OR Gaussian blur
   - Momentum merge computes Sobel gradients, forms an energy map, applies directional kernels (horizontal, vertical, two diagonals), takes the max response and boosts where energy is high.
   - Purpose: boost linear features (edges) that correspond to elongated nets.

2. Canny edge detection (configurable low/high thresholds)

3. Morphological close to fill gaps and dilation to thicken edges

These steps produce `edges_proc` used for `cv2.findContours()`.

Reference: `utils/sonar_image_analysis.py::preprocess_edges()` and `directional_momentum_merge()`

### 6.2 Contour detection

Contours are found with `cv2.findContours(edges_proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)`.

Each contour is characterized by a set of features (computed in `extract_contour_features()`):

- `area` (cv2.contourArea)
- bounding rect `(x,y,w,h)` and `aspect_ratio = max(w,h)/min(w,h)`
- `solidity = area / area_of_convex_hull`
- `extent = area / (w*h)`
- `ellipse_elongation` via `cv2.fitEllipse()` when available
- `straightness` using `cv2.fitLine()` and mean distance to the fitted line

Reference: `utils/sonar_image_analysis.py::select_best_contour()` and internal feature computations.

### 6.3 Contour scoring

Contours are scored using a weighted linear combination of features. The score formula (pseudocode):

```
comp = (aspect_ratio * w_ar) + (ellipse_elongation * w_el) + (1 - solidity) * w_sol + extent * w_ext + min(aspect_ratio/10, .5) * w_perim
comp *= (0.5 + 1.5 * straightness)
score = area * comp
```

Where weights live in `ELONGATION_CONFIG` in `utils/sonar_config.py`.

The highest-scoring contour is selected as the putative net.

Reference: `utils/sonar_image_analysis.py::score_contour()` and `ELONGATION_CONFIG` in `utils/sonar_config.py`.

---

## 7. Distance estimation from contour

Implementation: `utils/sonar_image_analysis.py::_distance_angle_from_contour()` and `get_red_line_distance_and_angle()`

### 7.1 Ellipse fitting and major axis intersection

1. Fit an ellipse to the selected contour using `cv2.fitEllipse(contour)` which returns `(cx, cy), (minor_axis, major_axis), angle`

2. Define the major axis direction as `angle + 90°` (OpenCV's `angle` is oriented relative to the ellipse's major/minor convention in image coordinates).

3. Parametric line for the major axis:

   (x(t), y(t)) = (cx, cy) + t * (cos(ang_r), sin(ang_r))

   where `ang_r = radians(angle + 90)`

4. To measure the distance straight ahead, intersect the major axis line with the center vertical line `x = image_width / 2` (the center beam corresponding to robot heading). Solve for `t`:

   t = (center_x - cx) / cos(ang_r)   (if cos(ang_r) ≠ 0)

   intersect_y = cy + t * sin(ang_r)

   distance_pixels = intersect_y  # Y is forward direction in the cone image

5. Fallback: if `cos(ang_r)` ≈ 0 (major axis nearly vertical), the code uses `cy` (ellipse center) as the distance.

6. Angle returned is `red_line_angle = (angle + 90) % 360`.

Notes and assumptions:

- This method assumes that the cone image's vertical coordinate corresponds directly to forward distance (y). That holds because the cone rasterization maps range → image row.
- No additional geometric projection is applied here because the rasterization already encoded the geometry (range per row). The final pixel→meter transform is applied using the `extent` metadata.

Reference: `utils/sonar_image_analysis.py::_distance_angle_from_contour()`

---

## 8. Pixel → meter conversion

Pixel-to-meter mapping is derived from the cone `extent` returned by `rasterize_cone()`/NPZ metadata.

Given `extent = (x_min, x_max, y_min, y_max)` and cone image `H, W`:

- px2m_x = (x_max - x_min) / W
- px2m_y = (y_max - y_min) / H

Distance in meters corresponding to `distance_pixels` (vertical pixel coordinate) is:

```
distance_m = y_min + distance_pixels * px2m_y
```

In the notebooks and interactive plotting code, a simplified conversion is sometimes used:

```
ppm = sonar_image_size / sonar_coverage_m
distance_m = distance_pixels / ppm
```

where `ppm` is pixels-per-meter (px/m) and `sonar_coverage_m` is the total displayed range in meters. This is algebraically equivalent to using `px2m_y` if `sonar_image_size` matches the image height and `sonar_coverage_m` matches `y_max - y_min`.

Reference: conversion code in `utils/sonar_image_analysis.py::analyze_red_line_distance_over_time()` and `interactive_distance_comparison()`

---

## 9. DVL & navigation data processing

Implementation: `utils/net_distance_analysis.py::load_all_distance_data_for_bag()`

### 9.1 Navigation/DVL fields used

- `NetDistance` — final DVL or navigation-derived distance to the net (meters)
- `NetPitch` — pitch angle (radians) toward the net
- `timestamp` / `ts_utc` — tz-aware timestamps used for synchronization

The files are CSVs exported from bag topics. The loader normalizes timestamps and computes auxiliary columns such as `relative_time` (seconds since start).

Reference: `utils/net_distance_analysis.py`

---

## 10. Synchronization & comparison

Implementation: `utils/sonar_image_analysis.py::interactive_distance_comparison()`

### 10.1 Temporal alignment

Sonar frame timestamps (from NPZ) and navigation timestamps (from CSV) are not always equally sampled. The interactive comparator uses a simple, robust approach:

1. Convert navigation timestamps to `relative_time = (ts - ts.min()).total_seconds()`.
2. Sonar frames may not have explicit timestamps for every frame in some NPZs; the notebook code computes `synthetic_time` by stretching the sonar frame index range to match the DVL duration:

   sonar['synthetic_time'] = (frame_index / (N_frames - 1)) * dvl_duration

   This maps sonar frame indices linearly onto the DVL timeline. If precise timestamps exist in the NPZ, the code uses them.

3. For direct pointwise comparisons, the code either interpolates one series onto the other's timebase or plots them together using shared axes and unified hover.

Notes: Linear stretching is a pragmatic choice when precise per-frame timestamps are missing or unreliable; if NPZ contains precise timestamps those are preferred.

### 10.2 Statistical comparisons

Metrics computed and displayed:

- mean, std, min, max for both sensors
- histograms (density) to compare distributions
- smoothed traces for sonar: moving average, Savitzky–Golay, Gaussian
- optionally compute correlation and RMSE (user code can compute this)

Reference: `utils/sonar_image_analysis.py::interactive_distance_comparison()` and plotting functions.

---

## 11. Error handling & debugging

Mechanisms implemented:

- `save_error_frames` option in `analyze_red_line_distance_over_time()` saves frames that produced None distance (useful for manual inspection)
- AOI (area-of-interest) expansion and tracking: `TRACKING_CONFIG['aoi_expansion_pixels']` is used to prefer contours inside a previously tracked region (keeps identity across frames)
- `select_best_contour()` returns scoring stats, letting the developer inspect why a contour was chosen

Reference: `utils/sonar_image_analysis.py::analyze_red_line_distance_over_time()` and `process_frame_for_video()`

---

## 12. Mathematical summary (equations)

- Polar→Cartesian: x = r * sin(θ), y = r * cos(θ)
- Range index → range: r(row) = r_min + row / (H - 1) * (r_max - r_min)
- Angle index → θ(col) = -FOV/2 + col / (W - 1) * FOV
- Pixel→meter (vertical): px2m_y = (y_max - y_min) / H
- Major axis parametric line: (x,y) = (cx,cy) + t*(cos(ang_r), sin(ang_r))
- Intersection with center line x=cx0: t = (cx0 - cx) / cos(ang_r), intersect_y = cy + t*sin(ang_r)
- TVG amplitude comp: gain(r) ∝ r^2 (amplitude) or r^4 (power)
- Decibel transform: 20 * log10(I + eps)

---

## 13. Common pitfalls & mitigations

- Mismatch between video and analysis pipelines: ensure both use same AOI, preprocessing and contour selection—this repository intentionally uses the same selection logic in `get_red_line_distance_and_angle()` and `process_frame_for_video()`.
- Vertical-axis conventions: cone rasterization uses +Y forward; ensure mapping when plotting or converting is consistent.
- Pixel→meter mismatch when the display size used for plotting differs from NPZ image size—prefer to derive `px2m` from `extent` and the NPZ's `H`/`W`.
- Near-vertical major axis: fallback to ellipse center is configurable via `TRACKING_CONFIG` thresholds.

---

## 14. Key code locations (quick map)

- Bag export and NPZ creation: `solaqua_export.py`
- Sonar enhancement, rasterization: `utils/sonar_utils.py` — functions: `enhance_intensity`, `rasterize_cone`, `cone_raster_like_display_cell`, `get_sonoptix_frame`
- Image preprocessing & contour scoring: `utils/sonar_image_analysis.py` — functions: `preprocess_edges`, `directional_momentum_merge`, `select_best_contour`, `score_contour`, `_distance_angle_from_contour`, `get_red_line_distance_and_angle`, `analyze_red_line_distance_over_time`, `interactive_distance_comparison`
- DVL and navigation loaders: `utils/net_distance_analysis.py` — function: `load_all_distance_data_for_bag`
- Configuration: `utils/sonar_config.py` (image & tracking parameters)

---

## 15. Suggestions for further improvements (research-oriented)

1. Replace nearest sampling in `rasterize_cone()` with bilinear interpolation for smoother coordinate transforms.
2. Use per-frame timestamps from NPZ (if present) instead of synthetic stretching when synchronizing.
3. Use RANSAC line/ellipse fits to be more robust to outliers in contour points.
4. Compute and log RMSE and cross-correlation between sonar and DVL after interpolation onto the same timebase.
5. Consider a learning-based contour score (small classifier) trained on labeled net vs non-net contours to improve selection in noisy scenes.

---

## 16. Final notes

This repository already implements a careful, reproducible pipeline suitable for specialization work: raw data extraction, robust enhancement, geometry-aware rasterization, and a principled computer-vision pipeline for detecting elongated net-like contours and estimating their forward distance. The interactive comparison tools make it straightforward to validate sonar-based estimates against independent navigation sensors.

If you want, I can:

- generate a short reproducible experiment notebook that runs one bag end-to-end and prints numeric RMSE vs DVL
- add explicit RMSE/correlation computations in `interactive_distance_comparison()` and return them in `comparison_stats`
- add a short `README` explaining how to regenerate NPZs and run the notebook

---

Copyright (c) 2025 Eirik Varnes — licensed under MIT (see `LICENSE`)
