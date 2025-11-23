#!/usr/bin/env python3
"""Compute net distance/orientation for a bag.

Basic example (uses ./bags and ./outputs by default):
    python scripts/compute_net_distance.py --bag-id 2024-08-22_14-06-43

Full-volume example:
    python scripts/compute_net_distance.py \
        --bag-id 2024-08-22_14-06-43 \
        --bag-data-dir /Volumes/LaCie/SOLAQUA/raw_data \
        --exports-dir /Volumes/LaCie/SOLAQUA/exports
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAGS_DIR = PROJECT_ROOT / "bags"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
_EXPORTS_SUBDIRS: dict[str, str] = {}
_BAG_DATA_DIR: Path = DEFAULT_BAGS_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute net distance/orientation for a single bag and emit a compact CSV."
    )
    parser.add_argument("--bag-id", required=True, help="Bag stem, e.g. 2024-08-20_17-02-00")
    parser.add_argument(
        "--bag-data-dir",
        type=Path,
        default=DEFAULT_BAGS_DIR,
        help=f"Directory containing raw/CSV sonar data (default: {DEFAULT_BAGS_DIR})",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help=f"Root of outputs/NPZ/cache data (default: {DEFAULT_OUTPUTS_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Destination CSV (default: <exports>/{bag}_net_distance.csv)",
    )
    parser.add_argument("--frame-count", type=int, default=3000, help="Frames to analyze (default: 3000)")
    parser.add_argument("--frame-step", type=int, default=1, help="Analyze every Nth frame (default: 1)")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached analysis CSV and rerun sonar tracking.",
    )
    parser.add_argument(
        "--persist-artifacts",
        action="store_true",
        help="Store/reuse CSV and NPZ intermediates (default is in-memory streaming).",
    )
    return parser.parse_args()


def main() -> int:
    start_time = time.perf_counter()
    args = _parse_args()
    exports_dir = args.exports_dir.resolve()
    bag_data_dir = args.bag_data_dir.resolve()
    bag_data_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SOLAQUA_EXPORTS_DIR"] = str(exports_dir)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        from utils.sonar_analysis import get_available_npz_files  # noqa: F401
        import utils.sonar_analysis as iau  # noqa: WPS433
        import utils.sonar_utils as sonar_utils  # noqa: WPS433
        from utils.config import EXPORTS_SUBDIRS as CONFIG_EXPORTS_SUBDIRS  # noqa: WPS433
        from utils.config import TRACKING_CONFIG as _TRACKING_CONFIG  # noqa: WPS433

    global _EXPORTS_SUBDIRS, _BAG_DATA_DIR
    _EXPORTS_SUBDIRS = CONFIG_EXPORTS_SUBDIRS
    _BAG_DATA_DIR = bag_data_dir

    for key, value in {
        "corridor_both_directions": False,
        "corridor_widen": 0.0,
        "use_corridor_splitting": False,
    }.items():
        _TRACKING_CONFIG.setdefault(key, value)

    bag_id = args.bag_id
    use_stream = not args.persist_artifacts
    temp_artifacts: list[Path] = []

    analysis_dir = (
        Path(tempfile.mkdtemp(prefix="solaqua_analysis_"))
        if use_stream
        else exports_dir / _EXPORTS_SUBDIRS.get("basic_full_batch", "basic_full_batch")
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if use_stream:
        temp_artifacts.append(analysis_dir)

    cached_analysis = analysis_dir / f"{bag_id}_analysis.csv"
    steps = [
        "Data extraction (sonar CSV / cone NPZ)",
        "Sonar analysis (distance/orientation)",
        "Distance CSV emission",
    ]

    class StepProgress:
        def __init__(self, total_steps: int, label: str):
            self.total = max(1, total_steps)
            self.label = label
            self.current = 0

        def advance(self, message: str) -> None:
            self.current += 1
            ratio = min(1.0, self.current / self.total)
            bar_len = 20
            filled = int(bar_len * ratio)
            bar = "=" * filled + "." * (bar_len - filled)
            pct = int(ratio * 100)
            print(f"  {self.label} [{bar}] {pct:3d}% {message}")

    def start_step(idx: int, desc: str, substeps: int) -> StepProgress:
        pct = int((idx - 1) / len(steps) * 100)
        print(f"\nStep {idx}/{len(steps)} [{pct}%]: {desc}")
        return StepProgress(substeps, f"Step {idx}")

    step1 = start_step(1, steps[0], substeps=3)
    cache_hit = cached_analysis.exists() and not args.force_recompute and not use_stream
    if cache_hit:
        sonar_df = pd.read_csv(cached_analysis)
        step1.advance(
            f"Loaded cached analysis CSV → {cached_analysis.name} "
            f"({len(sonar_df)} rows, mode=persisted)"
        )
    else:
        step1.advance(
            f"{'Streaming raw sonar' if use_stream else 'Preparing cached artifacts'} "
            f"(mode={'stream' if use_stream else 'persisted'}, bag={bag_id})"
        )

    npz_path, extract_stats = _ensure_cone_npz(
         bag_id,
         exports_dir,
         sonar_utils,
         bag_data_dir,
         use_stream=use_stream,
         temp_artifacts=temp_artifacts,
     )
    step1.advance(
        "Cone NPZ ready → "
        f"{npz_path.name} | frames={extract_stats['frames']} | "
        f"source={extract_stats['source']} | mode={extract_stats['mode']}"
    )

    if cache_hit:
        pass
    else:
        sonar_df = _run_sonar_analysis(
            bag_id,
            npz_path,
            analysis_dir,
            args.frame_count,
            args.frame_step,
            iau,
        )
        if not use_stream:
            cached_analysis = analysis_dir / f"{bag_id}_analysis.csv"
            sonar_df.to_csv(cached_analysis, index=False)
    step1.advance("Analysis dataframe ready")

    step2 = start_step(2, steps[1], substeps=2)
    step2.advance("Sonar analysis in memory")
    output_path = (
        args.output_csv
        if args.output_csv
        else exports_dir / f"{bag_id}_net_distance.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    step2.advance(f"Output path prepared: {output_path.name}")

    step3 = start_step(3, steps[2], substeps=2)
    reduced = _build_distance_table(sonar_df)
    step3.advance(f"Reduced to {len(reduced)} timestamped rows")
    reduced.to_csv(output_path, index=False)
    step3.advance("CSV written to disk")

    elapsed = time.perf_counter() - start_time
    data_start = reduced["timestamp"].min() if not reduced.empty else None
    data_end = reduced["timestamp"].max() if not reduced.empty else None
    data_span = (data_end - data_start).total_seconds() if data_start and data_end else 0.0
    print(f"✓ Net distance CSV written to {output_path}")
    if data_start and data_end:
        print(f"Data range: {data_start} → {data_end} ({data_span:.1f}s span)")
        print(f"Realtime ratio (data_span / runtime): {data_span / elapsed:.2f}×")
    print(f"Runtime: {elapsed:.2f}s")
    print("✓ All steps complete (100%)")

    if use_stream:
        for artifact in temp_artifacts:
            try:
                if artifact.is_dir():
                    shutil.rmtree(artifact, ignore_errors=True)
                else:
                    artifact.unlink(missing_ok=True)
            except Exception:
                pass
    return 0


def _exports_subdir(exports_dir: Path, key: str, default: str) -> Path:
    sub = Path(_EXPORTS_SUBDIRS.get(key, default))
    if not sub or str(sub) in {".", ""}:
        return exports_dir
    candidate = (exports_dir / sub).resolve()
    if candidate == exports_dir.resolve():
        return exports_dir
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _ensure_cone_npz(
    bag_id: str,
    exports_dir: Path,
    sonar_utils_module,
    bag_data_dir: Path,
    *,
    use_stream: bool,
    temp_artifacts: list[Path],
) -> Tuple[Path, dict]:
    source_desc = ""
    if not use_stream:
        outputs_dir = _exports_subdir(exports_dir, "outputs", "outputs")
        npz_path = outputs_dir / f"{bag_id}_cones.npz"
        if npz_path.exists():
            return npz_path, {
                "mode": "persisted",
                "frames": None,
                "source": "cached NPZ",
            }
        sonar_csv = _ensure_sonar_csv(bag_id, exports_dir, bag_data_dir)
        sonar_df = sonar_utils_module.load_df(sonar_csv)
        source_desc = sonar_csv.name
    else:
        outputs_dir = _exports_subdir(exports_dir, "outputs", "outputs")
        sonar_df, raw_meta = _collect_sonar_dataframe_from_bag(bag_id, bag_data_dir)
        source_desc = Path(raw_meta["bag_path"]).name
        fd, tmp_path = tempfile.mkstemp(prefix=f"{bag_id}_stream_", suffix=".npz", dir=outputs_dir)
        os.close(fd)
        npz_path = Path(tmp_path)
        temp_artifacts.append(npz_path)

    if sonar_df.empty:
        raise ValueError("Sonar dataset is empty; cannot build cones.")

    cone_params = {
        "fov_deg": 120.0,
        "rmin": 0.0,
        "rmax": 20.0,
        "y_zoom": 10.0,
        "grid": sonar_utils_module.ConeGridSpec(img_w=900, img_h=700),
        "flip_range": False,
        "flip_beams": True,
        "enhanced": True,
        "enhance_kwargs": {
            "scale": "db",
            "tvg": "amplitude",
            "p_low": 1.0,
            "p_high": 99.5,
            "gamma": 0.9,
        },
    }
    sonar_utils_module.save_cone_run_npz(
        sonar_df,
        npz_path,
        progress=False,
        **cone_params,
    )
    print(f"Created cones NPZ: {npz_path}")
    return npz_path, {
        "mode": "stream" if use_stream else "persisted",
        "frames": len(sonar_df),
        "source": source_desc or ("raw bag" if use_stream else "sonar CSV"),
    }


def _ensure_sonar_csv(bag_id: str, exports_dir: Path, bag_data_dir: Path) -> Optional[Path]:
    existing = _find_sonoptix_csv(bag_id, exports_dir, bag_data_dir)
    if existing:
        return existing
    by_bag_dir = _exports_subdir(exports_dir, "by_bag", "by_bag")
    output_csv = by_bag_dir / f"sensor_sonoptix_echo_image__{bag_id}_video.csv"
    df, _ = _collect_sonar_dataframe_from_bag(bag_id, bag_data_dir)
    df.to_csv(output_csv, index=False)
    print(f"\nSonar CSV created: {output_csv} ({len(df)} frames)")
    return output_csv if output_csv.exists() else None


def _collect_sonar_dataframe_from_bag(bag_id: str, bag_data_dir: Path) -> Tuple[pd.DataFrame, dict]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise ImportError("rosbags is required to extract sonar data from raw bags.") from exc

    bag_path = _find_raw_bag_path(bag_id, bag_data_dir)
    if bag_path is None:
        raise FileNotFoundError(f"No raw bag found for '{bag_id}' under {bag_data_dir}")

    print(f"Extracting Sonoptix sonar directly from raw bag: {bag_path}")
    rows = []
    with AnyReader([bag_path]) as reader:
        sonar_conns = [c for c in reader.connections if "sonoptix" in c.topic.lower()]
        if not sonar_conns:
            raise RuntimeError(f"No Sonoptix topics found in bag {bag_path}")

        for conn in sonar_conns:
            for idx, (connection, timestamp, rawdata) in enumerate(reader.messages([conn]), start=1):
                msg = reader.deserialize(rawdata, connection.msgtype)
                sec = msg.header.stamp.sec
                nsec = msg.header.stamp.nanosec
                ts = pd.Timestamp(sec, unit="s", tz="UTC") + pd.to_timedelta(nsec, unit="ns")

                dims = msg.array_data.layout.dim
                dim_labels = [d.label or f"dim_{i}" for i, d in enumerate(dims)] or ["range", "beam"]
                dim_sizes = [int(d.size) for d in dims]
                dim_strides = [int(d.stride) for d in dims]
                data_vals = [float(v) for v in msg.array_data.data]

                rows.append(
                    {
                        "bag": bag_id,
                        "topic": conn.topic,
                        "ts_utc": ts.isoformat(),
                        "t": sec + nsec / 1e9,
                        "dim_labels": dim_labels,
                        "dim_sizes": dim_sizes,
                        "dim_strides": dim_strides,
                        "data": data_vals,
                        "len": len(data_vals),
                    }
                )
                if idx % 100 == 0:
                    print(f"  Extracted {idx} sonar frames...", end="\r")
    if not rows:
        raise RuntimeError(f"No sonar frames extracted from {bag_path}")

    df = pd.DataFrame(rows)
    df["dim_labels"] = df["dim_labels"].apply(json.dumps)
    df["dim_sizes"] = df["dim_sizes"].apply(json.dumps)
    df["dim_strides"] = df["dim_strides"].apply(json.dumps)
    df["data"] = df["data"].apply(json.dumps)
    meta = {
        "bag_path": str(bag_path),
        "frames": len(df),
        "topics": [c.topic for c in sonar_conns],
    }
    print(f"\nCollected {len(df)} sonar frames from raw bag")
    return df, meta


def _run_sonar_analysis(
    bag_id: str,
    npz_path: Path,
    analysis_dir: Path,
    frame_count: int,
    frame_step: int,
    iau_module,
) -> pd.DataFrame:
    available_npz = iau_module.get_available_npz_files()
    try:
        bag_idx = next(
            idx for idx, path in enumerate(available_npz)
            if path.resolve() == npz_path.resolve() or bag_id in path.name
        )
    except StopIteration as exc:
        raise FileNotFoundError(
            f"{npz_path} not registered in available NPZ files. "
            "Ensure SOLAQUA_EXPORTS_DIR points to the correct exports folder."
        ) from exc

    print(f"Running sonar analysis for {bag_id} (index {bag_idx})…")
    analysis_df = iau_module.analyze_npz_sequence(
        npz_file_index=bag_idx,
        frame_start=1,
        frame_count=frame_count,
        frame_step=frame_step,
        save_outputs=False,
    )
    if analysis_df is None or analysis_df.empty:
        raise ValueError(f"Sonar analysis returned no data for bag {bag_id}")
    return analysis_df


def _find_raw_bag_path(bag_id: str, bag_data_dir: Path) -> Optional[Path]:
    patterns = [f"{bag_id}_video.bag", f"{bag_id}.bag", f"{bag_id}_data.bag"]
    for pattern in patterns:
        hits = list(bag_data_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _find_sonoptix_csv(bag_id: str, exports_dir: Path, bag_data_dir: Path) -> Optional[Path]:
    search_roots = [
        _exports_subdir(exports_dir, "by_bag", "by_bag"),
        bag_data_dir,
    ]
    candidates = [
        f"sensor_sonoptix_echo_image__{bag_id}_video.csv",
        f"sensor_sonoptix_echo_image__{bag_id}.csv",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in candidates:
            candidate_path = root / pattern
            if candidate_path.exists():
                return candidate_path
    return None


def _first_available_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    return next((col for col in candidates if col in df.columns), None)


def _build_distance_table(sonar_df: pd.DataFrame) -> pd.DataFrame:
    timestamp_col = _first_available_column(sonar_df, ("timestamp", "ts", "ts_utc"))
    if timestamp_col is None:
        raise KeyError("No timestamp column found in sonar analysis DataFrame.")

    distance_col = _first_available_column(sonar_df, ("distance_meters", "net_distance_m"))
    if distance_col is None:
        raise KeyError("No distance column found in sonar analysis DataFrame.")

    orientation_col = _first_available_column(sonar_df, ("angle_degrees", "net_orientation_deg"))

    timestamps = pd.to_datetime(sonar_df[timestamp_col], utc=True, errors="coerce")
    distances = pd.to_numeric(sonar_df[distance_col], errors="coerce")
    orientations = (
        pd.to_numeric(sonar_df[orientation_col], errors="coerce")
        if orientation_col
        else pd.Series([float("nan")] * len(sonar_df))
    )

    reduced = pd.DataFrame(
        {
            "timestamp": timestamps,
            "net_distance_m": distances,
            "net_orientation_deg": orientations,
        }
    ).dropna(subset=["timestamp", "net_distance_m"])
    return reduced.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
