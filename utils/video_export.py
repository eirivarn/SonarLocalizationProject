from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import cv2
import yaml
from rosbags.highlevel import AnyReader

from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None


from .core import (
    sanitize_topic, base_msgtype, find_video_bags, stamp_seconds, to_native,
    ensure_types_from_reader,   # <-- add this
)


# ------------------------------ Topic discovery ------------------------------

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

# ------------------------------ Decoders ------------------------------

def _decode_compressed_to_bgr(msg) -> Optional[np.ndarray]:
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
    return None

# ------------------------------ CameraInfo export ------------------------------

def export_camera_info_for_bag(
    bagpath: Union[str, Path],
    out_dir: Union[str, Path] = "exports/camera_info",
    one_per_topic: bool = True,
) -> pd.DataFrame:
    """
    Save first CameraInfo message per camera-info topic to YAML.
    Returns a DataFrame with: [bag, topic, out_file].
    """
    bagpath = Path(bagpath)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with AnyReader([bagpath]) as reader:
        _ = ensure_types_from_reader(reader)
        infos = [c for c in reader.connections if base_msgtype(c.msgtype) == "CameraInfo"]
        for con in infos:
            it = reader.messages(connections=[con])
            try:
                _, t_ns, raw = next(it)
            except StopIteration:
                continue
            msg = reader.deserialize(raw, con.msgtype)

            # Handle lower/upper case names
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

            payload = to_native(payload)

            fname = f"{bagpath.stem}__{sanitize_topic(con.topic)}.yaml"
            out_path = out_dir / fname
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            rows.append({"bag": bagpath.stem, "topic": con.topic, "out_file": str(out_path)})

            if one_per_topic:
                continue

    return pd.DataFrame(rows)

# ------------------------------ Video exports ------------------------------

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
    - Keeps one AnyReader context.
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
        _ = ensure_types_from_reader(reader)
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
                t = stamp_seconds(msg, t_ns)
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
    zero_pad: int = 6,                 # kept for optional ordinal suffix if collisions occur
    resize_to: Optional[Tuple[int, int]] = None,  # (W, H)
    *,
    timestamp_tz: str = "Europe/Oslo", # filename timezone
    timestamp_fmt: str = "%Y%m%d_%H%M%S_%f%z",    # filename format (safe)
    timestamp_source: str = "auto",    # "auto"|"header"|"bag"
    write_index_csv: bool = True,      # also write an index.csv with timestamps
) -> Dict[str, Union[str, int]]:
    """
    Export frames of a camera topic as PNGs, one file per frame with timestamped filenames:

        exports/frames/<bag-stem>__<topic_sanitized>/<YYYYmmdd_HHMMSS_micro+TZ>.png

    - Timestamp choice:
        * "auto": header stamp if present, else bag receive time
        * "header": force Header.stamp (may be None)
        * "bag": force bag receive time
    - Writes index.csv with per-frame t_header/t_bag and the chosen ts.
    """
    bagpath = Path(bagpath)
    topic_tag = sanitize_topic(topic)
    seq_dir = Path(out_dir) / f"{bagpath.stem}__{topic_tag}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # index rows (optional)
    index_rows: List[Dict] = []

    frames_written = 0
    with AnyReader([bagpath]) as reader:
        _ = ensure_types_from_reader(reader)
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

            # choose timestamp
            t_hdr, t_bag, t_auto = _extract_times(msg, t_ns)
            if timestamp_source == "header":
                ts = t_hdr
            elif timestamp_source == "bag":
                ts = t_bag
            else:
                ts = t_auto

            # filename (timestamped)
            ts_str = _format_ts(ts, tz_name=timestamp_tz, fmt=timestamp_fmt)

            # avoid collisions if two frames format to the same name
            base_name = ts_str
            fname = seq_dir / f"{base_name}.png"
            bump = 1
            while fname.exists():
                # append an ordinal suffix
                fname = seq_dir / f"{base_name}__{bump:0{zero_pad}d}.png"
                bump += 1

            # optional resize
            if resize_to:
                W, H = resize_to
                if img.shape[1] != W or img.shape[0] != H:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            # write the image
            cv2.imwrite(str(fname), img)
            frames_written += 1

            # index row
            if write_index_csv:
                dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc) if ts is not None else None
                if ZoneInfo is not None and ts is not None:
                    try:
                        dt_loc = dt_utc.astimezone(ZoneInfo(timestamp_tz))
                    except Exception:
                        dt_loc = dt_utc
                else:
                    dt_loc = dt_utc
                index_rows.append({
                    "bag": bagpath.stem,
                    "topic": topic,
                    "file": str(fname.relative_to(seq_dir)),
                    "t_use": ts,
                    "t_header": t_hdr,
                    "t_bag": t_bag,
                    "ts_utc": dt_utc.isoformat() if dt_utc else None,
                    "ts_local": dt_loc.isoformat() if dt_loc else None,
                })

    # write index.csv next to the frames
    if write_index_csv and index_rows:
        idx_path = seq_dir / "index.csv"
        pd.DataFrame(index_rows).to_csv(idx_path, index=False)
        print(f"Wrote {frames_written} PNGs to {seq_dir} with index: {idx_path}")
    else:
        print(f"Wrote {frames_written} PNGs to {seq_dir}.")

    return {
        "bag": bagpath.stem,
        "topic": topic,
        "out_dir": str(seq_dir),
        "frames": frames_written,
    }

# ------------------------------ Batch runners ------------------------------

def export_all_video_bags_to_mp4(
    data_dir: Union[str, Path],
    out_dir: Union[str, Path] = "exports",
    recursive: bool = False,
    topics: Optional[Iterable[str]] = None,
    codec: str = "mp4v",
    target_fps: Optional[float] = None,
    probe_frames: int = 120,
    min_frames_for_fps: int = 5,
    overwrite: bool = False,
    save_camera_info_yaml: bool = True,
) -> pd.DataFrame:
    """
    Export all camera topics from each *_video.bag to MP4.
    Returns DataFrame: [bag, topic, out_file, frames, fps, codec, skipped]
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
        if save_camera_info_yaml:
            try:
                df_info = export_camera_info_for_bag(bag, out_dir=info_dir)
                _ = df_info  # just to ensure it runs
            except Exception as e:
                print(f"(warn) CameraInfo export failed for {bag.name}: {e}")

        bag_topics = [t for (t, _mt) in list_camera_topics_in_bag(bag)] if topics is None else list(topics)

        for topic in bag_topics:
            mp4_path = vid_dir / f"{bag.stem}__{sanitize_topic(topic)}.mp4"
            if mp4_path.exists() and not overwrite:
                index_rows.append({
                    "bag": bag.stem, "topic": topic, "out_file": str(mp4_path),
                    "frames": np.nan, "fps": target_fps if target_fps else np.nan,
                    "codec": codec, "skipped": True,
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
    Returns DataFrame: [bag, topic, out_dir, frames, skipped]
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
        bag_topics = [t for (t, _mt) in list_camera_topics_in_bag(bag)] if topics is None else list(topics)

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

def _extract_times(msg, t_ns):
    """Return (t_header, t_bag, t_use) in seconds (float)."""
    # bag receive time
    t_bag = float(t_ns) * 1e-9 if t_ns is not None else None

    # header time (ROS1/ROS2 compatible)
    t_hdr = None
    for p in (("header","stamp","sec","nanosec"), ("header","stamp","secs","nsecs")):
        try:
            sec = getattr(getattr(getattr(msg, p[0]), p[1]), p[2])
            nsc = getattr(getattr(getattr(msg, p[0]), p[1]), p[3])
            t_hdr = float(sec) + float(nsc) * 1e-9
            break
        except Exception:
            pass

    t_use = t_hdr if (t_hdr is not None and t_hdr > 0) else t_bag
    return t_hdr, t_bag, t_use

def _format_ts(ts_sec, tz_name="Europe/Oslo", fmt="%Y%m%d_%H%M%S_%f%z"):
    """
    Format a POSIX timestamp (seconds) into a filename-safe string.
    Default tz: Europe/Oslo. Fallback to UTC if tz data is unavailable.
    """
    if ts_sec is None:
        return "unknown_ts"
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_name)
            dt_loc = dt_utc.astimezone(tz)
        except Exception:
            dt_loc = dt_utc  # fallback to UTC
    else:
        dt_loc = dt_utc     # fallback to UTC
    s = dt_loc.strftime(fmt)  # no ":" so it's filesystem-safe
    return s