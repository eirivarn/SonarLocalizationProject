# SOLAQUA Data Processing

Tools and notebooks for working with the SOLAQUA aquaculture dataset:

- Export sonar and auxiliary sensor data from ROS `.bag` files
- Export camera frames and MP4 sequences
- Visualize sonar pings (raw + enhanced)
- Synchronize sonar with video and generate combined videos

---

## Dependencies

- Python 3.10+
- [`rosbags`](https://gitlab.com/ternaris/rosbags) (read ROS 1/2 bags without ROS)
- `numpy`, `pandas`, `matplotlib`
- `opencv-python`
- `pyyaml`
- `pyarrow` (Parquet IO; optional but recommended)
- (Optional) `scipy`, `jupyterlab`

### Create & activate a virtual environment

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows PowerShell
# .\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

### `requirements.txt`

```txt
rosbags
numpy
pandas
matplotlib
opencv-python
pyyaml
pyarrow
# optional
scipy
jupyterlab
```

> Tip: If you plan to use Parquet (`.parquet`) exports/loads, keep `pyarrow` installed.

---

## Dataset layout

Place the SOLAQUA `.bag` files in the **`data/`** folder:

```
data/
  2024-08-20_13-39-34_data.bag
  2024-08-20_13-39-34_video.bag
  ...
```

Exports (CSV/Parquet/PNG/MP4) are written to **`exports/`** by the notebooks.

### `.gitignore`

If you haven’t already, ignore large generated artifacts:

```gitignore
# data and outputs are big; don’t commit
/data/
/exports/
```

---

## Workflow

### 1) Data Export — `data_export.ipynb`
Reads `*_data.bag` and (optionally) sonar topics from `*_video.bag`, then writes per-topic tables under `exports/by_bag/`.

**Highlights**
- Each sonar message becomes one row; 2D arrays are stored in an `image` column (JSON of lists)
- Per-row timestamp preserved as `t` (seconds) and `ts_utc` (tz-aware)
- Produces `exports/index_data_topics.csv` as an overview

### 2) Video Export — `video_export.ipynb`
Exports camera topics from `*_video.bag` to either MP4 or PNG sequences under `exports/videos/` and `exports/frames/`.

**Highlights**
- MP4: optional target FPS or median Δt estimate
- PNG: per-frame timestamp indexing (`index.csv`) for later alignment

### 3) Sonar + Video — `sonar_video.ipynb`
Loads exported sonar and video frames, aligns by timestamp, and renders synchronized views.

**Views**
- **Left:** RGB video frame
- **Right:** Sonar cone (raw or enhanced), apex at the bottom, spokes only (no arcs)

**Output**
- Side-by-side MP4 named by first sonar frame time (Europe/Oslo):
  `YYYYMMDD_HHMMSS_micro+TZ_sonar_video.mp4`

---

## Notes & tuning

- **Ranges & angles:** If available, read device metadata (range resolution / sample period & sound speed / per-beam angles). Otherwise the viewer uses `RANGE_MIN_M`, `RANGE_MAX_M`, and `FOV_DEG` as linear fallbacks.
- **Enhancement:** Optional log scale (dB), TVG (amplitude/power), absorption compensation, robust percentile clipping, and gamma. Toggle with `USE_ENHANCED` in the notebook.
- **Flips:** Use `FLIP_RANGE`, `FLIP_BEAMS`, and `CONE_FLIP_VERTICAL` to correct orientation.
- **Timestamps:** Stored in UTC; display is converted to Europe/Oslo in overlays and filenames.

---

## Getting started quickly

1. Place `.bag` files in `data/`
2. `pip install -r requirements.txt`
3. Open notebooks:
   - `data_export.ipynb` → run all
   - `video_export.ipynb` → export MP4 or PNG (creates `exports/frames/...`)
   - `sonar_video.ipynb` → align & render side-by-side video

That’s it — you should see outputs appear under `exports/`.

