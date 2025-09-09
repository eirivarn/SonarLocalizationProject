from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from rosbags.highlevel import AnyReader

# ------------------------------ Data index loading ------------------------------

def load_data_index(out_dir: Path | str = "exports") -> pd.DataFrame:
    """
    Load the index created by save_all_topics_from_data_bags().
    Looks for exports/index_data_topics.csv (or .parquet).
    """
    out_dir = Path(out_dir)
    csv = out_dir / "index_data_topics.csv"
    pq  = out_dir / "index_data_topics.parquet"
    if csv.exists():
        return pd.read_csv(csv)
    if pq.exists():
        return pd.read_parquet(pq)
    raise FileNotFoundError(
        f"Could not find index_data_topics.csv or .parquet under {out_dir}. "
        "Run save_all_topics_from_data_bags() first."
    )

# ------------------------------ CSV summaries ------------------------------

def _summarize_csv_time(file_path: str | Path) -> Dict[str, Any]:
    """
    Fast summary of one exported CSV:
      - rows (count)
      - t_min, t_max (seconds)
      - duration_s
      - approx_rate_hz = rows / duration (if duration>0)
    Only reads 't' column when present (chunked).
    """
    file_path = Path(file_path)
    rows_total = 0
    t_min = None
    t_max = None
    try:
        for chunk in pd.read_csv(file_path, usecols=["t"], chunksize=200_000):
            rows_total += len(chunk)
            cmin = float(chunk["t"].min())
            cmax = float(chunk["t"].max())
            t_min = cmin if t_min is None else min(t_min, cmin)
            t_max = cmax if t_max is None else max(t_max, cmax)
    except ValueError:
        for chunk in pd.read_csv(file_path, chunksize=200_000):
            rows_total += len(chunk)

    duration = (t_max - t_min) if (t_min is not None and t_max is not None) else None
    rate = (rows_total / duration) if (duration and duration > 0) else None
    return {
        "rows": rows_total,
        "t_min": t_min,
        "t_max": t_max,
        "duration_s": duration,
        "approx_rate_hz": rate,
    }

# ------------------------------ Convenience listings ------------------------------

def list_exported_bag_stems(out_dir: Path | str = "exports",
                            bag_suffix: str = "_data") -> List[str]:
    """Return bag stems present in the export index, optionally filtered by suffix."""
    idx = load_data_index(out_dir)
    if bag_suffix:
        idx = idx[idx["bag"].str.endswith(bag_suffix)]
    return sorted(idx["bag"].unique().tolist())

def _normalize_time_str(time_str: str) -> str:
    """Accept '13-55-34' or '13:55:34' (or '13:55') and normalize to '13-55-34'."""
    s = time_str.strip().replace(":", "-")
    parts = s.split("-")
    if len(parts) == 2:  # HH-MM -> add seconds
        parts.append("00")
    if len(parts) != 3:
        raise ValueError(f"Time '{time_str}' must be HH:MM[:SS] or HH-MM-SS")
    hh, mm, ss = parts
    if len(hh) == 1: hh = f"0{hh}"
    if len(mm) == 1: mm = f"0{mm}"
    if len(ss) == 1: ss = f"0{ss}"
    return f"{hh}-{mm}-{ss}"

# ------------------------------ Overviews ------------------------------

def overview_by_bag(bag_stem: str, out_dir: Path | str = "exports") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two tables for a specific bag stem, e.g. '2024-08-20_13-55-34_data':
      files_df  – one row per exported CSV file for that bag:
          [bag, topic, out_file, rows, t_min, t_max, duration_s, approx_rate_hz]
      topics_df – aggregate per topic (for this bag):
          [topic, rows, duration_s, approx_rate_hz]
    """
    out_dir = Path(out_dir)
    idx = load_data_index(out_dir)

    sel = idx[idx["bag"] == bag_stem]
    if sel.empty:
        sel = idx[idx["bag"].str.contains(bag_stem)]
    if sel.empty:
        return pd.DataFrame(), pd.DataFrame()

    recs = []
    for _, r in sel.iterrows():
        s = _summarize_csv_time(r["out_file"])
        recs.append({
            "bag": r["bag"],
            "topic": r["topic"],
            "out_file": r["out_file"],
            **s,
        })
    files_df = pd.DataFrame(recs).sort_values(["topic"]).reset_index(drop=True)
    topics_df = files_df[["topic", "rows", "duration_s", "approx_rate_hz"]].copy()
    return files_df, topics_df

def overview_by_datetime(date_str: str,
                         time_str: str,
                         out_dir: Path | str = "exports",
                         bag_suffix: str = "_data") -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: given date='YYYY-MM-DD' and time='HH:MM[:SS]' or 'HH-MM-SS',
    build the bag stem 'YYYY-MM-DD_HH-MM-SS{bag_suffix}' and call overview_by_bag.
    Returns (bag_stem, files_df, topics_df).
    """
    time_norm = _normalize_time_str(time_str)
    bag_stem = f"{date_str}_{time_norm}{bag_suffix}"
    files_df, topics_df = overview_by_bag(bag_stem, out_dir=out_dir)
    return bag_stem, files_df, topics_df

# ------------------------------ Topic tools ------------------------------

def topics_in_bag_df(
    bagpath: Union[str, Path],
    with_counts: bool = False,
) -> pd.DataFrame:
    """
    List all topics in a .bag file.
    If with_counts=True, also count messages per topic (may take time on large bags).
    Returns columns: [topic, msgtype, count]
    """
    bagpath = Path(bagpath)
    recs: List[dict] = []

    from rosbags.highlevel import AnyReader
    with AnyReader([bagpath]) as reader:
        topic_types = {}
        for c in reader.connections:
            topic_types.setdefault(c.topic, set()).add(c.msgtype)

        counts = {}
        if with_counts:
            for c in reader.connections:
                n = 0
                for _ in reader.messages(connections=[c]):
                    n += 1
                counts[c.topic] = counts.get(c.topic, 0) + n

        for topic, types in sorted(topic_types.items()):
            msgtype = ", ".join(sorted(types))
            recs.append({
                "topic": topic,
                "msgtype": msgtype,
                "count": (counts.get(topic) if with_counts else None),
            })

    df = pd.DataFrame(recs)
    return df[["topic", "msgtype", "count"]] if with_counts else df[["topic", "msgtype"]]

def topics_overview_dir(
    data_dir: Union[str, Path],
    recursive: bool = False,
    suffix_filter: Optional[str] = None,   # e.g., "_data", "_video"
    with_counts: bool = False,
) -> pd.DataFrame:
    """
    Scan a folder for .bag files and list topics per bag.
    Returns: [bag_file, bag_stem, topic, msgtype, count?]
    """
    data_dir = Path(data_dir)
    bags = (data_dir.rglob("*.bag") if recursive else data_dir.glob("*.bag"))
    paths = []
    for p in bags:
        if suffix_filter and not p.stem.endswith(suffix_filter):
            continue
        paths.append(p)
    paths = sorted(paths)

    rows = []
    for p in paths:
        df = topics_in_bag_df(p, with_counts=with_counts)
        df.insert(0, "bag_file", p.name)
        df.insert(1, "bag_stem", p.stem)
        rows.append(df)

    if not rows:
        cols = ["bag_file", "bag_stem", "topic", "msgtype"]
        if with_counts:
            cols.append("count")
        return pd.DataFrame(columns=cols)

    out = pd.concat(rows, ignore_index=True)
    cols = ["bag_file", "bag_stem", "topic", "msgtype"]
    if with_counts:
        cols.append("count")
    return out[cols]
