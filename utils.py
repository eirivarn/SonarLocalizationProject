# --- SOLAQUA data.bag export helpers (paste in a Jupyter cell) ---

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Tuple, Union
import json
import base64

import numpy as np
import pandas as pd

from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg, register_types


# ============================== Utility ===================================

def sanitize_topic(topic: str) -> str:
    """Make a ROS topic safe for filenames."""
    return (topic or "").strip("/").replace("/", "_") or "root"

def base_msgtype(s: str) -> str:
    """Return the final segment of a ROS msgtype path."""
    return s.rsplit("/", 1)[-1] if s else s

def find_data_bags(data_dir: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Return a sorted list of paths to *_data.bag in data_dir."""
    data_dir = Path(data_dir)
    if recursive:
        return sorted(p for p in data_dir.rglob("*.bag") if p.name.endswith("_data.bag"))
    return sorted(p for p in data_dir.glob("*.bag") if p.name.endswith("_data.bag"))

def register_custom_msgs_from_dataset_root(root: Union[str, Path]) -> int:
    """
    Auto-register custom .msg types if present under: root/msg, root/msgs, root/custom_msgs.
    Returns number of message types registered.
    """
    root = Path(root)
    count = 0
    for cand in (root / "msg", root / "msgs", root / "custom_msgs"):
        if cand.exists():
            additions: Dict[str, str] = {}
            for p in cand.rglob("*.msg"):
                pkg = p.parent.parent.name  # expects <pkg>/msg/*.msg
                additions.update(get_types_from_msg(p.read_text(encoding="utf-8"), f"{pkg}/msg"))
            if additions:
                register_types(additions)
                count = len(additions)
            break
    return count

def _stamp_seconds(msg, fallback_t_ns: Optional[int] = None) -> Optional[float]:
    """Extract a floating timestamp in seconds (ROS2 -> ROS1 -> bag time)."""
    try:
        return float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    except Exception:
        pass
    try:
        return float(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
    except Exception:
        pass
    return float(fallback_t_ns * 1e-9) if fallback_t_ns is not None else None

def _coerce_for_table(val, arrays_as: str = "json"):
    """
    Convert values into table-friendly forms.
    - bytes -> base64 string
    - numpy arrays -> list/json depending on arrays_as
    - lists/tuples/dicts -> JSON string if arrays_as == 'json', else Python repr
    """
    if isinstance(val, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(val)).decode("ascii")

    if isinstance(val, np.ndarray):
        if arrays_as == "json":
            return json.dumps(val.tolist(), ensure_ascii=False)
        return repr(val.tolist())

    if isinstance(val, (list, tuple, dict)):
        if arrays_as == "json":
            try:
                return json.dumps(val, ensure_ascii=False)
            except TypeError:
                # Fallback if elements aren't JSON-serializable
                return repr(val)
        return repr(val)

    return val

def _flatten_msg(obj, rec: dict, prefix: str = "", arrays_as: str = "json") -> None:
    """
    Flatten a rosbags message into a flat dict:
    - Skips header (used for timestamps separately).
    - Recurses into slot-based fields.
    - Lists of slot-objects are expanded with indices.
    - Lists of scalars are stored as JSON/repr (configurable).
    """
    if hasattr(obj, "__slots__"):
        for name in obj.__slots__:
            if name == "header":
                continue
            val = getattr(obj, name)
            if hasattr(val, "__slots__"):
                _flatten_msg(val, rec, prefix + name + ".", arrays_as)
            elif isinstance(val, (list, tuple)) and val and hasattr(val[0], "__slots__"):
                for i, v in enumerate(val):
                    _flatten_msg(v, rec, prefix + f"{name}[{i}].", arrays_as)
            else:
                rec[prefix + name] = _coerce_for_table(val, arrays_as=arrays_as)
    else:
        rec[prefix.rstrip(".")] = _coerce_for_table(obj, arrays_as=arrays_as)


# ============================== Core API ===================================

def list_topics_in_bag(bagpath: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Return list of (topic, msgtype) for a bag.
    """
    bagpath = Path(bagpath)
    with AnyReader([bagpath]) as reader:
        return sorted({(c.topic, c.msgtype) for c in reader.connections})

def bag_topic_to_dataframe(
    bagpath: Union[str, Path],
    topic: str,
    arrays_as: str = "json",
) -> pd.DataFrame:
    """
    Read a single topic from a bag into a DataFrame with:
    columns: t, t0, t_rel, bag, bag_file, topic, [flattened fields...]

    arrays_as: 'json' (default) stores lists/arrays as JSON strings; 'repr' stores Python repr.
    """
    bagpath = Path(bagpath)
    rows = []
    with AnyReader([bagpath]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return pd.DataFrame()

        for con, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)
            rec = {
                "t": _stamp_seconds(msg, t_ns),
                "bag": bagpath.stem,
                "bag_file": bagpath.name,
                "topic": topic,
            }
            _flatten_msg(msg, rec, arrays_as=arrays_as)
            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    df["t0"] = df["t"].iloc[0]
    df["t_rel"] = df["t"] - df["t0"]
    return df

def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    file_format: str = "csv",
) -> None:
    """
    Save DataFrame to CSV or Parquet.
    file_format: 'csv' (default) or 'parquet'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def save_all_topics_from_data_bags(
    data_dir: Union[str, Path],
    out_dir: Union[str, Path] = "exports",
    file_format: str = "csv",
    arrays_as: str = "json",
    recursive: bool = False,
    exclude_msgtypes: Optional[Iterable[str]] = None,
    include_msgtypes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Export ALL topics from every *_data.bag under data_dir to per-bag files.

    Output files are written to:
        <out_dir>/by_bag/<topic_sanitized>__<bag-stem>.<ext>

    Returns an index DataFrame with columns:
        [bag, bag_file, topic, msgtype, rows, out_file]

    Parameters
    ----------
    data_dir : folder with bags (expects *:_data.bag)
    out_dir  : output root folder (default 'exports')
    file_format : 'csv' or 'parquet'
    arrays_as : for list/array fields -> 'json' or 'repr'
    recursive : search subfolders
    exclude_msgtypes : optional iterable of msgtype names to skip
                       (e.g., {'sensor_msgs/Image', 'sensor_msgs/msg/CompressedImage'})
    include_msgtypes : optional iterable to restrict saving to these msgtypes
                       (takes precedence over exclude if provided)
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    by_bag_dir = out_dir / "by_bag"
    by_bag_dir.mkdir(parents=True, exist_ok=True)

    # Reasonable defaults: skip image payloads if they appear in data.bag
    default_exclude = {
        "sensor_msgs/Image",
        "sensor_msgs/msg/Image",
        "sensor_msgs/CompressedImage",
        "sensor_msgs/msg/CompressedImage",
    }
    if exclude_msgtypes is None and include_msgtypes is None:
        exclude_msgtypes = default_exclude

    # Register custom msgs if shipped with dataset
    registered = register_custom_msgs_from_dataset_root(data_dir)
    if registered:
        print(f"Registered {registered} custom ROS message types.")

    bags = find_data_bags(data_dir, recursive=recursive)
    if not bags:
        raise FileNotFoundError(f"No *_data.bag files found under {data_dir} (recursive={recursive}).")

    index_rows = []

    for bag in bags:
        with AnyReader([bag]) as reader:
            connections = sorted(reader.connections, key=lambda c: (c.topic, c.msgtype))

            # build a per-topic list of msgtypes (rarely >1 per topic, but handle anyway)
            topic_types = {}
            for c in connections:
                topic_types.setdefault(c.topic, set()).add(c.msgtype)

            for topic, types in topic_types.items():
                # Filter by message type
                # (If include_msgtypes is set, only allow those; else skip excluded)
                if include_msgtypes:
                    if not any(t in include_msgtypes for t in types):
                        continue
                elif exclude_msgtypes:
                    if any(t in exclude_msgtypes for t in types):
                        continue

                # Extract DF for this topic
                df_topic = bag_topic_to_dataframe(bag, topic, arrays_as=arrays_as)
                if df_topic.empty:
                    continue

                # File name
                fname = f"{sanitize_topic(topic)}__{bag.stem}.{ 'parquet' if file_format=='parquet' else 'csv' }"
                out_path = by_bag_dir / fname

                # Save
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

    index_df = pd.DataFrame(index_rows).sort_values(["bag", "topic"]).reset_index(drop=True)
    # Also store an index for convenience
    idx_path = out_dir / ("index_data_topics.parquet" if file_format == "parquet" else "index_data_topics.csv")
    save_dataframe(index_df, idx_path, file_format=file_format)
    print(f"\nIndex written to {idx_path} with {len(index_df)} entries.")
    return index_df


# ============================== VIDEO EXPORT HELPERS ===============================

from typing import Iterable, List, Optional, Tuple, Dict, Union
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import yaml
from rosbags.highlevel import AnyReader

# If these exist above in your utils.py already, you can remove these duplicates:
def sanitize_topic(topic: str) -> str:
    return (topic or "").strip("/").replace("/", "_") or "root"

def base_msgtype(s: str) -> str:
    return s.rsplit("/", 1)[-1] if s else s

def _stamp_seconds(msg, fallback_t_ns: Optional[int] = None) -> Optional[float]:
    try:
        return float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
    except Exception:
        pass
    try:
        return float(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
    except Exception:
        pass
    return float(fallback_t_ns * 1e-9) if fallback_t_ns is not None else None

def find_video_bags(data_dir: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Return sorted list of paths to *_video.bag in data_dir."""
    data_dir = Path(data_dir)
    if recursive:
        return sorted(p for p in data_dir.rglob("*.bag") if p.name.endswith("_video.bag"))
    return sorted(p for p in data_dir.glob("*.bag") if p.name.endswith("_video.bag"))

def list_camera_topics_in_bag(bagpath: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Return [(topic, msgtype)] for topics that look like camera images
    (sensor_msgs/Image or sensor_msgs/CompressedImage).
    """
    bagpath = Path(bagpath)
    cams = []
    with AnyReader([bagpath]) as reader:
        for c in reader.connections:
            mt = base_msgtype(c.msgtype)
            if mt in ("Image", "CompressedImage"):
                cams.append((c.topic, c.msgtype))
    return sorted(set(cams))

def _decode_compressed_to_bgr(msg) -> Optional[np.ndarray]:
    # Force 3-channel BGR to keep writer happy
    arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _decode_raw_to_bgr(msg) -> Optional[np.ndarray]:
    enc = getattr(msg, "encoding", "").lower()
    h = getattr(msg, "height", None)
    w = getattr(msg, "width", None)
    if h is None or w is None:
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("bgr8",):
        return buf.reshape(h, w, 3)
    if enc in ("rgb8",):
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if enc in ("mono8",):
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    # Add more encodings if your camera uses them
    return None

def _to_native(obj):
    """Recursively convert NumPy scalars/arrays and containers to plain Python types."""
    import numpy as _np
    if isinstance(obj, _np.generic):     # np.float64, np.int64, etc.
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    return obj

def export_camera_info_for_bag(
    bagpath: Union[str, Path],
    out_dir: Union[str, Path] = "exports/camera_info",
    one_per_topic: bool = True,
) -> pd.DataFrame:
    """
    Save first CameraInfo message per camera-info topic to YAML.
    Returns a DataFrame with: [bag, topic, out_file].
    (This version coerces NumPy types to native Python for YAML.)
    """
    import yaml
    bagpath = Path(bagpath)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with AnyReader([bagpath]) as reader:
        infos = [c for c in reader.connections if base_msgtype(c.msgtype) == "CameraInfo"]
        for con in infos:
            it = reader.messages(connections=[con])
            try:
                _, t_ns, raw = next(it)
            except StopIteration:
                continue
            msg = reader.deserialize(raw, con.msgtype)

            # Some bags expose lower/upper case names; handle both.
            D = getattr(msg, "d", getattr(msg, "D", []))
            K = getattr(msg, "k", getattr(msg, "K", []))
            R = getattr(msg, "r", getattr(msg, "R", []))
            P = getattr(msg, "p", getattr(msg, "P", []))

            roi = getattr(msg, "roi", None)
            roi_dict = {
                "x_offset": int(getattr(roi, "x_offset", 0)) if roi is not None else 0,
                "y_offset": int(getattr(roi, "y_offset", 0)) if roi is not None else 0,
                "height":   int(getattr(roi, "height",   0)) if roi is not None else 0,
                "width":    int(getattr(roi, "width",    0)) if roi is not None else 0,
                "do_rectify": bool(getattr(roi, "do_rectify", False)) if roi is not None else False,
            }

            payload = {
                "height": int(getattr(msg, "height", 0)),
                "width":  int(getattr(msg, "width",  0)),
                "distortion_model": getattr(msg, "distortion_model", ""),
                "D": D, "K": K, "R": R, "P": P,
                "binning_x": int(getattr(msg, "binning_x", 0)),
                "binning_y": int(getattr(msg, "binning_y", 0)),
                "roi": roi_dict,
            }

            # Deep-convert to native Python types so YAML can serialize it
            payload = _to_native(payload)

            fname = f"{bagpath.stem}__{sanitize_topic(con.topic)}.yaml"
            out_path = out_dir / fname
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            rows.append({"bag": bagpath.stem, "topic": con.topic, "out_file": str(out_path)})

            if one_per_topic:
                # Only export the first sample per topic
                continue

    return pd.DataFrame(rows)

def export_camera_topic_to_mp4(
    bagpath: Union[str, Path],
    topic: str,
    out_path: Union[str, Path],
    codec: str = "mp4v",
    target_fps: Optional[float] = None,
    probe_frames: int = 120,
    min_frames_for_fps: int = 5,
    resize_to_first: bool = True,
) -> Dict[str, Union[str, int, float]]:
    """
    Export a single camera topic to MP4.
    - Estimate FPS from median dt of first `probe_frames` frames unless target_fps provided.
    - Keeps one AnyReader context (prevents 'Rosbag is not open').
    """
    bagpath = Path(bagpath)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_written = 0
    fps = target_fps or 15.0
    writer = None
    W = H = None

    def _ensure_writer(shape):
        nonlocal writer, W, H
        if writer is not None:
            return
        H, W = shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        w = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H), True)
        if not w.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_path}")
        writer = w

    with AnyReader([bagpath]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"Topic {topic!r} not found in {bagpath.name}")
        con = conns[0]
        mt = base_msgtype(con.msgtype)

        # Probe for fps if not given
        if target_fps is None:
            stamps = []
            gen = reader.messages(connections=[con])
            for i, (_, t_ns, raw) in enumerate(gen):
                msg = reader.deserialize(raw, con.msgtype)
                t = _stamp_seconds(msg, t_ns)
                if t is not None:
                    stamps.append(t)
                if len(stamps) >= probe_frames:
                    break
            if len(stamps) >= min_frames_for_fps + 1:
                dts = np.diff(stamps)
                dts = dts[(dts > 1e-6) & (dts < 5.0)]
                if len(dts):
                    fps = float(np.clip(1.0 / np.median(dts), 1.0, 120.0))

        # Now stream + write
        gen2 = reader.messages(connections=[con])
        for _, t_ns, raw in gen2:
            msg = reader.deserialize(raw, con.msgtype)
            if mt == "CompressedImage":
                img = _decode_compressed_to_bgr(msg)
            else:
                img = _decode_raw_to_bgr(msg)
            if img is None:
                continue
            if writer is None:
                _ensure_writer(img.shape)
            if resize_to_first and (img.shape[0] != H or img.shape[1] != W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            writer.write(img)
            frames_written += 1

    if writer is not None:
        writer.release()

    return {
        "bag": bagpath.stem,
        "topic": topic,
        "out_file": str(out_path),
        "frames": frames_written,
        "fps": float(fps),
        "codec": codec,
    }

def export_camera_topic_to_png_sequence(
    bagpath: Union[str, Path],
    topic: str,
    out_dir: Union[str, Path],
    stride: int = 1,
    limit: Optional[int] = None,
    zero_pad: int = 6,
    resize_to: Optional[Tuple[int, int]] = None,  # (W, H)
) -> Dict[str, Union[str, int]]:
    """
    Export frames of a camera topic as PNGs:
        out_dir/<bag-stem>__<topic_sanitized>/frame_000001.png
    """
    bagpath = Path(bagpath)
    topic_tag = sanitize_topic(topic)
    seq_dir = Path(out_dir) / f"{bagpath.stem}__{topic_tag}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    frames_written = 0
    with AnyReader([bagpath]) as reader:
        cons = [c for c in reader.connections if c.topic == topic]
        if not cons:
            raise ValueError(f"Topic {topic!r} not found in {bagpath.name}")
        con = cons[0]
        mt = base_msgtype(con.msgtype)
        idx = 0
        for _, t_ns, raw in reader.messages(connections=[con]):
            if limit is not None and frames_written >= limit:
                break
            if idx % stride != 0:
                idx += 1
                continue
            msg = reader.deserialize(raw, con.msgtype)
            img = _decode_compressed_to_bgr(msg) if mt == "CompressedImage" else _decode_raw_to_bgr(msg)
            idx += 1
            if img is None:
                continue
            if resize_to:
                W, H = resize_to
                if img.shape[1] != W or img.shape[0] != H:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            fname = seq_dir / f"frame_{frames_written:0{zero_pad}d}.png"
            cv2.imwrite(str(fname), img)
            frames_written += 1

    return {
        "bag": bagpath.stem,
        "topic": topic,
        "out_dir": str(seq_dir),
        "frames": frames_written,
    }

def export_all_video_bags_to_mp4(
    data_dir: Union[str, Path],
    out_dir: Union[str, Path] = "exports",
    recursive: bool = False,
    topics: Optional[Iterable[str]] = None,   # if None -> auto-discover all camera topics per bag
    codec: str = "mp4v",
    target_fps: Optional[float] = None,
    probe_frames: int = 120,
    min_frames_for_fps: int = 5,
    overwrite: bool = False,
    save_camera_info_yaml: bool = True,
) -> pd.DataFrame:
    """
    Export all camera topics from each *_video.bag to MP4.
    Returns a DataFrame index: [bag, topic, out_file, frames, fps, codec]
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    vid_dir = out_dir / "videos"
    info_dir = out_dir / "camera_info"
    vid_dir.mkdir(parents=True, exist_ok=True)
    if save_camera_info_yaml:
        info_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    bags = find_video_bags(data_dir, recursive=recursive)
    if not bags:
        raise FileNotFoundError(f"No *_video.bag files found under {data_dir} (recursive={recursive}).")

    for bag in bags:
        # optional: save camera info yaml (first samples)
        if save_camera_info_yaml:
            try:
                df_info = export_camera_info_for_bag(bag, out_dir=info_dir)
                for _, r in df_info.iterrows():
                    pass  # just to ensure it runs; df can be inspected later
            except Exception as e:
                print(f"(warn) CameraInfo export failed for {bag.name}: {e}")

        # discover topics if not provided
        if topics is None:
            bag_topics = [t for (t, mt) in list_camera_topics_in_bag(bag)]
        else:
            bag_topics = list(topics)

        for topic in bag_topics:
            mp4_path = vid_dir / f"{bag.stem}__{sanitize_topic(topic)}.mp4"
            if mp4_path.exists() and not overwrite:
                # quick stat if file exists already
                index_rows.append({
                    "bag": bag.stem, "topic": topic, "out_file": str(mp4_path),
                    "frames": np.nan, "fps": target_fps if target_fps else np.nan, "codec": codec, "skipped": True,
                })
                print(f"(skip) exists: {mp4_path}")
                continue
            try:
                res = export_camera_topic_to_mp4(
                    bag, topic, mp4_path,
                    codec=codec,
                    target_fps=target_fps,
                    probe_frames=probe_frames,
                    min_frames_for_fps=min_frames_for_fps,
                )
                res["skipped"] = False
                index_rows.append(res)
                print(f"Wrote {res['out_file']} ({res['frames']} frames @ ~{res['fps']:.2f} FPS).")
            except Exception as e:
                print(f"(error) {bag.name}:{topic}: {e}")

    idx = pd.DataFrame(index_rows)
    idx_path = out_dir / "index_video_mp4.csv"
    idx.to_csv(idx_path, index=False)
    print(f"\nIndex written: {idx_path} ({len(idx)} rows).")
    return idx

def export_all_video_bags_to_png(
    data_dir: Union[str, Path],
    out_dir: Union[str, Path] = "exports",
    recursive: bool = False,
    topics: Optional[Iterable[str]] = None,
    stride: int = 1,
    limit: Optional[int] = None,
    resize_to: Optional[Tuple[int, int]] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Export all camera topics from each *_video.bag to PNG sequences.
    Returns a DataFrame index: [bag, topic, out_dir, frames]
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    png_root = out_dir / "frames"
    png_root.mkdir(parents=True, exist_ok=True)

    index_rows = []
    bags = find_video_bags(data_dir, recursive=recursive)
    if not bags:
        raise FileNotFoundError(f"No *_video.bag files found under {data_dir} (recursive={recursive}).")

    for bag in bags:
        if topics is None:
            bag_topics = [t for (t, mt) in list_camera_topics_in_bag(bag)]
        else:
            bag_topics = list(topics)

        for topic in bag_topics:
            seq_dir = png_root / f"{bag.stem}__{sanitize_topic(topic)}"
            if seq_dir.exists() and not overwrite:
                index_rows.append({
                    "bag": bag.stem, "topic": topic, "out_dir": str(seq_dir), "frames": np.nan, "skipped": True,
                })
                print(f"(skip) exists: {seq_dir}")
                continue
            try:
                res = export_camera_topic_to_png_sequence(
                    bag, topic, png_root, stride=stride, limit=limit, resize_to=resize_to,
                )
                res["skipped"] = False
                index_rows.append(res)
                print(f"Wrote {res['frames']} PNGs to {res['out_dir']}.")
            except Exception as e:
                print(f"(error) {bag.name}:{topic}: {e}")

    idx = pd.DataFrame(index_rows)
    idx_path = out_dir / "index_video_png.csv"
    idx.to_csv(idx_path, index=False)
    print(f"\nIndex written: {idx_path} ({len(idx)} rows).")
    return idx


# ============================== OVERVIEW BY BAG (DATE+TIME) ==============================

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd

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
        # 't' missing? fall back to counting rows
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

def list_exported_bag_stems(out_dir: Path | str = "exports",
                            bag_suffix: str = "_data") -> List[str]:
    """
    Return all unique bag stems present in the export index, optionally filtered by suffix
    (e.g., '_data' to show only data-bag exports).
    """
    idx = load_data_index(out_dir)
    if bag_suffix:
        idx = idx[idx["bag"].str.endswith(bag_suffix)]
    stems = sorted(idx["bag"].unique().tolist())
    return stems

def _normalize_time_str(time_str: str) -> str:
    """
    Accept '13-55-34' or '13:55:34' (or '13:55') and normalize to '13-55-34'.
    """
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
        # try contains as a convenience (e.g., if user passed date only)
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


# ============================== TOPIC LISTING / COUNTS ==============================

from pathlib import Path
from typing import List, Tuple, Optional, Union
import pandas as pd
from rosbags.highlevel import AnyReader

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

    with AnyReader([bagpath]) as reader:
        # gather unique (topic, msgtype)
        topic_types = {}
        for c in reader.connections:
            topic_types.setdefault(c.topic, set()).add(c.msgtype)

        counts = {}
        if with_counts:
            # count messages per connection (sum per topic)
            for c in reader.connections:
                n = 0
                for _ in reader.messages(connections=[c]):
                    n += 1
                counts[c.topic] = counts.get(c.topic, 0) + n

        for topic, types in sorted(topic_types.items()):
            # If a topic somehow has multiple msgtypes, join them (rare)
            msgtype = ", ".join(sorted(types))
            recs.append({
                "topic": topic,
                "msgtype": msgtype,
                "count": (counts.get(topic) if with_counts else None),
            })

    df = pd.DataFrame(recs)
    if not with_counts:
        return df[["topic", "msgtype"]]
    return df[["topic", "msgtype", "count"]]


def topics_overview_dir(
    data_dir: Union[str, Path],
    recursive: bool = False,
    suffix_filter: Optional[str] = None,   # e.g. "_data" or "_video" or None for all
    with_counts: bool = False,
) -> pd.DataFrame:
    """
    Scan a folder for .bag files and list topics per bag.

    Returns: [bag_file, bag_stem, topic, msgtype, count]
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
        return pd.DataFrame(columns=["bag_file","bag_stem","topic","msgtype"] + (["count"] if with_counts else []))
    out = pd.concat(rows, ignore_index=True)
    cols = ["bag_file","bag_stem","topic","msgtype"]
    if with_counts:
        cols.append("count")
    return out[cols]

