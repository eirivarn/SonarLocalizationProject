from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict
import re
from .core import decode_float32_multiarray, base_msgtype

import pandas as pd
from rosbags.highlevel import AnyReader

from .core import (
    sanitize_topic,
    find_data_bags,
    find_video_bags,
    register_custom_msgs_from_dataset_root,
    stamp_seconds,
    flatten_msg,
    ensure_types_from_reader,
    decode_float32_multiarray,   # <-- keep helper in core.py
    base_msgtype,                # <-- need this
)

# ------------------------------ Topic listing ------------------------------

def list_topics_in_bag(bagpath: Union[str, Path]) -> List[Tuple[str, str]]:
    bagpath = Path(bagpath)
    with AnyReader([bagpath]) as reader:
        return sorted({(c.topic, c.msgtype) for c in reader.connections})

# ------------------------------ Topic -> DataFrame ------------------------------

def bag_topic_to_dataframe(
    bagpath: Union[str, Path],
    topic: str,
    arrays_as: str = "json",
) -> pd.DataFrame:
    """
    Read a single topic from a bag into a DataFrame with:
      columns: t, t0, t_rel, bag, bag_file, topic, [flattened fields...]
    """
    bagpath = Path(bagpath)
    rows: List[Dict] = []

    with AnyReader([bagpath]) as reader:
        _ = ensure_types_from_reader(reader)
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return pd.DataFrame()

        for con, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)

            # ---- SonoptixECHO: Float32MultiArray -> 2D 'image' (or flat 'data') ----
            if base_msgtype(con.msgtype) == "SonoptixECHO" and hasattr(msg, "array_data"):
                labels, sizes, strides, payload, shape = decode_float32_multiarray(msg.array_data)
                rec = {
                    "t": stamp_seconds(msg, t_ns),
                    "bag": bagpath.stem,
                    "bag_file": bagpath.name,
                    "topic": topic,
                    "dim_labels": json.dumps(labels, ensure_ascii=False),
                    "dim_sizes":  json.dumps(sizes, ensure_ascii=False),
                    "dim_strides": json.dumps(strides, ensure_ascii=False),
                }
                if shape is not None:   # (H, W) detected -> store as image
                    H, W = shape
                    rec["rows"] = int(H)
                    rec["cols"] = int(W)
                    rec["image"] = json.dumps(payload, ensure_ascii=False)  # 2D list
                else:                    # fallback: flat vector
                    rec["data"] = json.dumps(payload, ensure_ascii=False)
                    rec["len"]  = len(payload)
                rows.append(rec)
                continue

            # ---- Ping360-like: bearing + bins/intensities array ----
            if ("ping360" in topic.lower()) or (base_msgtype(con.msgtype) in ("Ping360", "Ping")):
                # Try common field names
                bearing = None
                for cand in ("bearing", "angle", "azimuth", "theta", "heading"):
                    if hasattr(msg, cand):
                        try:
                            bearing = float(getattr(msg, cand))
                        except Exception:
                            bearing = getattr(msg, cand)
                        break
                bins = None
                for cand in ("bins", "ranges", "samples", "intensities", "echo", "intensity"):
                    if hasattr(msg, cand):
                        v = getattr(msg, cand)
                        if isinstance(v, (list, tuple)):
                            bins = [float(x) for x in v]
                        break
                if (bearing is not None) or (bins is not None):
                    rec = {
                        "t": stamp_seconds(msg, t_ns),
                        "bag": bagpath.stem,
                        "bag_file": bagpath.name,
                        "topic": topic,
                    }
                    if bearing is not None:
                        rec["bearing_deg"] = bearing
                    if bins is not None:
                        rec["bins"] = json.dumps(bins, ensure_ascii=False)
                        rec["n_bins"] = len(bins)
                    rows.append(rec)
                    continue

            # ---- Generic fallback for everything else ----
            rec = {
                "t": stamp_seconds(msg, t_ns),
                "bag": bagpath.stem,
                "bag_file": bagpath.name,
                "topic": topic,
            }
            flatten_msg(msg, rec, arrays_as=arrays_as)
            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    df["t0"] = df["t"].iloc[0]
    df["t_rel"] = df["t"] - df["t0"]
    return df

# ------------------------------ Save helpers ------------------------------

def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    file_format: str = "csv",
) -> None:
    """Save DataFrame to CSV or Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

# ------------------------------ Batch export ------------------------------

def save_all_topics_from_data_bags(
    data_dir: Union[str, Path],
    out_dir: Union[str, Path] = "exports",
    file_format: str = "csv",
    arrays_as: str = "json",
    recursive: bool = False,
    exclude_msgtypes: Optional[Iterable[str]] = None,
    include_msgtypes: Optional[Iterable[str]] = None,
    *,
    include_video_sonar: bool = True,
    sonar_topic_patterns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Export ALL topics from every *_data.bag under data_dir to per-bag files.
    Optionally ALSO pull sonar topics from *_video.bag (but still skip RGB camera frames).

    Outputs to: <out_dir>/by_bag/<topic_sanitized>__<bag-stem>.(csv|parquet)
    Returns index DataFrame with columns: [bag, bag_file, topic, msgtypes, rows, out_file]
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    by_bag_dir = out_dir / "by_bag"
    by_bag_dir.mkdir(parents=True, exist_ok=True)

    # camera frames excluded by default
    default_exclude = {
        "sensor_msgs/Image",
        "sensor_msgs/msg/Image",
        "sensor_msgs/CompressedImage",
        "sensor_msgs/msg/CompressedImage",
    }
    if exclude_msgtypes is None and include_msgtypes is None:
        exclude_msgtypes = default_exclude

    sonar_whitelist = {
        "sensors/msg/SonoptixECHO",
        "sensors/msg/Ping360",
        "sensors/msg/Ping",
        # add other sonar msg types here
    }
    if include_msgtypes is not None:
        include_msgtypes = set(include_msgtypes) | sonar_whitelist

    if sonar_topic_patterns is None:
        # Typical sonar topic name fragments
        sonar_topic_patterns = [
            r"/sonoptix", r"/echo", r"/sonar", r"/ping360", r"/ping\b", r"/mbes",
        ]
    sonar_topic_re = re.compile("|".join(sonar_topic_patterns), re.IGNORECASE)

    registered = register_custom_msgs_from_dataset_root(data_dir)
    if registered:
        print(f"Registered {registered} custom ROS message types.")

    data_bags = find_data_bags(data_dir, recursive=recursive)
    if not data_bags:
        raise FileNotFoundError(f"No *_data.bag files found under {data_dir} (recursive={recursive}).")

    # Optionally pull sonar topics from *_video.bag
    video_bags: List[Path] = find_video_bags(data_dir, recursive=recursive) if include_video_sonar else []

    index_rows: List[Dict] = []

    def _export_from_bag(bag: Path, allow_only_sonar_from_video: bool = False) -> None:
        with AnyReader([bag]) as reader:
            _ = ensure_types_from_reader(reader)
            connections = sorted(reader.connections, key=lambda c: (c.topic, c.msgtype))
            topic_types: Dict[str, set] = {}
            for c in connections:
                topic_types.setdefault(c.topic, set()).add(c.msgtype)

            for topic, types in topic_types.items():
                # If this is a video bag and we only want sonar, filter by topic name
                if allow_only_sonar_from_video and not sonar_topic_re.search(topic):
                    continue

                if include_msgtypes:
                    if not any(t in include_msgtypes for t in types):
                        # if we are in video+sonar mode, allow image-based sonar too by name
                        if allow_only_sonar_from_video and sonar_topic_re.search(topic):
                            pass
                        else:
                            continue
                elif exclude_msgtypes:
                    # skip if ALL msgtypes are excluded (pure camera frames)
                    if all(t in exclude_msgtypes for t in types):
                        # but allow if topic name matches sonar (for image-based sonar stored as Image)
                        if allow_only_sonar_from_video and sonar_topic_re.search(topic):
                            pass
                        else:
                            continue

                df_topic = bag_topic_to_dataframe(bag, topic, arrays_as=arrays_as)
                if df_topic.empty:
                    continue

                fname = f"{sanitize_topic(topic)}__{bag.stem}.{ 'parquet' if file_format=='parquet' else 'csv' }"
                out_path = by_bag_dir / fname
                save_dataframe(df_topic, out_path, file_format=file_format)

                index_rows.append({
                    "bag": bag.stem,
                    "bag_file": bag.name,
                    "topic": topic,
                    "msgtypes": sorted(types),
                    "rows": int(len(df_topic)),
                    "out_file": str(out_path),
                })
                print(f"Wrote {out_path} ({len(df_topic)} rows).")

    # 1) Export from *_data.bag (everything except excluded camera frames)
    for bag in data_bags:
        _export_from_bag(bag, allow_only_sonar_from_video=False)

    # 2) Optionally sweep *_video.bag but keep ONLY sonar-like topics by name
    for vbag in video_bags:
        _export_from_bag(vbag, allow_only_sonar_from_video=True)

    index_df = pd.DataFrame(index_rows).sort_values(["bag", "topic"]).reset_index(drop=True)
    idx_path = out_dir / ("index_data_topics.parquet" if file_format == "parquet" else "index_data_topics.csv")
    save_dataframe(index_df, idx_path, file_format=file_format)
    print(f"\nIndex written to {idx_path} with {len(index_df)} entries.")
    return index_df

