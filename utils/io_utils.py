# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

"""
I/O and Data Loading Utilities

Core functions for loading and parsing sonar data, video indices, and timestamps.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd


# --------------------------- I/O + parsing helpers ---------------------------

def load_df(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet into a DataFrame."""
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def parse_json_cell(v):
    """Parse a JSON-encoded cell if it's a string, else pass through."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v

def load_png_rgb(path: Path) -> np.ndarray:
    """Load a PNG (or any OpenCV-readable image) as RGB numpy array."""
    bgr = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



def ensure_ts_col(dfs: pd.DataFrame, *, src_col: str = "t") -> pd.DataFrame:
    """
    Ensure a tz-aware UTC timestamp column 'ts_utc' exists.
    If missing, derive from 't' seconds epoch.
    """
    dfs = dfs.copy()
    if "ts_utc" not in dfs.columns:
        if src_col not in dfs.columns:
            raise RuntimeError("Sonar DataFrame missing both 'ts_utc' and epoch seconds column.")
        dfs["ts_utc"] = pd.to_datetime(dfs[src_col], unit="s", utc=True)
    else:
        dfs["ts_utc"] = pd.to_datetime(dfs["ts_utc"], utc=True, errors="coerce")
    return dfs


def read_video_index(seq_dir: Path) -> pd.DataFrame:
    """
    Return DataFrame with at least ['file','ts_utc'] from a frames folder.
    Uses index.csv if present; otherwise attempts to parse timestamps from filenames
    like YYYYmmdd_HHMMSS_micro+ZZZZ.png.
    """
    seq_dir = Path(seq_dir)
    idx_path = seq_dir / "index.csv"
    if idx_path.exists():
        dfv = pd.read_csv(idx_path)
        if "ts_utc" in dfv.columns:
            dfv["ts_utc"] = pd.to_datetime(dfv["ts_utc"], utc=True)
        elif "t_use" in dfv.columns:  # seconds epoch
            dfv["ts_utc"] = pd.to_datetime(dfv["t_use"], unit="s", utc=True)
        else:
            raise RuntimeError("index.csv missing ts_utc/t_use columns.")
        if "file" not in dfv.columns:
            dfv["file"] = [p.name for p in seq_dir.glob("*.png")]
        return dfv[["file", "ts_utc"]].dropna().sort_values("ts_utc").reset_index(drop=True)

    # Fallback: parse from filenames
    rows = []
    rx = re.compile(r"(\d{8})_(\d{6})_(\d{6})([+\-]\d{4})")
    for p in sorted(seq_dir.glob("*.png")):
        m = rx.search(p.stem)
        if not m:
            continue
        ymd, hms, micro, tz = m.groups()
        dt = datetime.strptime(ymd + hms + micro + tz, "%Y%m%d%H%M%S%f%z")
        rows.append({"file": p.name, "ts_utc": dt.astimezone(timezone.utc)})
    if not rows:
        raise FileNotFoundError("No index.csv and could not parse timestamps from filenames.")
    return pd.DataFrame(rows).sort_values("ts_utc").reset_index(drop=True)


def normalize_ts_utc(ts) -> pd.Timestamp:
    """Ensure a pandas Timestamp is tz-aware UTC."""
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def nearest_video_index(
    dfv: pd.DataFrame,
    ts_target,
    *,
    tolerance: Optional[pd.Timedelta] = None
) -> Tuple[int, Optional[pd.Timedelta]]:
    """
    Find nearest frame index in dfv (expects 'ts_utc' tz-aware). Returns (idx, |dt|).
    If tolerance is given and nearest exceeds it, returns the index but you may treat
    it as a soft failure by checking the returned dt.
    """
    dfv = dfv.copy()
    dfv["ts_utc"] = pd.to_datetime(dfv["ts_utc"], utc=True, errors="coerce")
    dfv = dfv.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    if dfv.empty:
        raise RuntimeError("Video index has no valid timestamps.")

    ts_target = normalize_ts_utc(ts_target)
    idx_near = pd.Index(dfv["ts_utc"]).get_indexer([ts_target], method="nearest")[0]
    if idx_near < 0:
        raise RuntimeError("Could not find a nearest video frame (indexer returned -1).")
    dt_best = abs(dfv.loc[idx_near, "ts_utc"] - ts_target)
    if tolerance is not None and dt_best > tolerance:
        # not raising—caller can decide; we still return the nearest + dt.
        pass
    return int(idx_near), pd.Timedelta(dt_best)


def get_available_npz_files(npz_dir: str | None = None) -> list[Path]:
    """Get list of available NPZ files."""
    from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    npz_dir = Path(npz_dir) if npz_dir is not None else Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('outputs', 'outputs')
    if not npz_dir.exists():
        return []
    return [f for f in npz_dir.glob("*_cones.npz") if not f.name.startswith('._')]


# --------------------------- Public API --------------------------------------

__all__ = [
    "load_df",
    "parse_json_cell",
    "load_png_rgb",
    "ensure_ts_col",
    "read_video_index",
    "normalize_ts_utc",
    "nearest_video_index",
    "get_available_npz_files",
]


