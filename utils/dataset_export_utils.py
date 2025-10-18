# Copyright (c) 2025 Eirik Varnes
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterable, Any
import base64
import json
import numpy as np
import pandas as pd
import cv2
import yaml
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.highlevel import AnyReader
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def sanitize_topic(topic: str) -> str:
    return (topic or "").strip("/").replace("/", "_") or "root"

def base_msgtype(s: str) -> str:
    return s.rsplit("/", 1)[-1] if s else s

def find_data_bags(data_dir: Union[str, Path], recursive: bool = False) -> List[Path]:
    data_dir = Path(data_dir)
    bags = data_dir.rglob("*.bag") if recursive else data_dir.glob("*.bag")
    return sorted(p for p in bags if not p.name.startswith(('.', '._')) and p.name.endswith("_data.bag"))

def find_video_bags(data_dir: Union[str, Path], recursive: bool = False) -> List[Path]:
    data_dir = Path(data_dir)
    bags = data_dir.rglob("*.bag") if recursive else data_dir.glob("*.bag")
    return sorted(p for p in bags if not p.name.startswith(('.', '._')) and p.name.endswith("_video.bag"))

def register_custom_msgs_from_dataset_root(root: Union[str, Path]) -> int:
    root = Path(root)
    for cand in (root / "msg", root / "msgs", root / "custom_msgs"):
        if cand.exists():
            additions: Dict[str, str] = {}
            for p in cand.rglob("*.msg"):
                pkg = p.parent.parent.name
                additions.update(get_types_from_msg(p.read_text(encoding="utf-8"), f"{pkg}/msg"))
            if additions:
                register_types(additions)
                return len(additions)
    return 0

def stamp_seconds(msg, fallback_t_ns: Optional[int] = None) -> Optional[float]:
    for attr_path in [("header", "stamp", "sec", "nanosec"), ("header", "stamp", "secs", "nsecs")]:
        try:
            obj = msg
            for attr in attr_path[:-2]:
                obj = getattr(obj, attr)
            stamp = getattr(obj, attr_path[-2])
            sec, nanos = getattr(stamp, attr_path[-2]), getattr(stamp, attr_path[-1])
            return float(sec + nanos * 1e-9)
        except:
            continue
    return float(fallback_t_ns * 1e-9) if fallback_t_ns is not None else None

def coerce_for_table(val, arrays_as: str = "json"):
    if isinstance(val, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(val)).decode("ascii")
    if isinstance(val, np.ndarray):
        return json.dumps(val.tolist(), ensure_ascii=False) if arrays_as == "json" else repr(val.tolist())
    if isinstance(val, (list, tuple, dict)):
        if arrays_as == "json":
            try:
                return json.dumps(val, ensure_ascii=False)
            except TypeError:
                return repr(val)
        return repr(val)
    return val

def to_native(obj: Any):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    return obj

def flatten_msg(obj, rec: dict, prefix: str = "", arrays_as: str = "json") -> None:
    if hasattr(obj, "__slots__"):
        for name in obj.__slots__:
            if name == "header":
                continue
            val = getattr(obj, name)
            if hasattr(val, "__slots__"):
                flatten_msg(val, rec, prefix + name + ".", arrays_as)
            elif isinstance(val, (list, tuple)) and val and hasattr(val[0], "__slots__"):
                for i, v in enumerate(val):
                    flatten_msg(v, rec, prefix + f"{name}[{i}].", arrays_as)
            else:
                rec[prefix + name] = coerce_for_table(val, arrays_as=arrays_as)
    elif hasattr(obj, "__dict__"):
        for name, val in obj.__dict__.items():
            if name != "header":
                flatten_msg(val, rec, prefix + f"{name}.", arrays_as)
    else:
        rec[prefix.rstrip(".") or "payload"] = coerce_for_table(obj, arrays_as=arrays_as)

def ensure_types_from_reader(reader) -> int:
    additions = {}
    for c in reader.connections:
        try:
            pkg_prefix = c.msgtype.rsplit("/", 1)[0]
            if getattr(c, "msgdef", None):
                additions.update(get_types_from_msg(c.msgdef, pkg_prefix))
        except:
            pass
    if additions:
        register_types(additions)
        return len(additions)
    return 0

def _safe_int(x, default=0, max_digits=20):
    try:
        if isinstance(x, str) and len(x) > max_digits:
            return default
        return int(x)
    except:
        return default

def decode_float32_multiarray(ma, *, allow_heuristics: bool = False, allow_trailing_channel1: bool = True) -> Tuple[List[str], List[int], List[int], List[float] | List[List[float]], Optional[Tuple[int, int]], Dict[str, Any]]:
    dims = list(getattr(ma.layout, "dim", []) or [])
    labels = [str(getattr(d, "label", "") or "") for d in dims]
    sizes = [_safe_int(getattr(d, "size", 0)) for d in dims]
    strides = [_safe_int(getattr(d, "stride", 0)) for d in dims]
    data_off = _safe_int(getattr(ma.layout, "data_offset", 0))

    data = np.asarray(getattr(ma, "data", []), dtype=np.float32)
    if data_off > 0:
        data = data[data_off:]

    meta = {
        "data_offset": data_off, "dtype": "float32", "len_data": int(data.size),
        "policy": "flat", "used_shape": None, "warnings": [],
        "payload_sha256": hashlib.sha256(data.tobytes(order="C")).hexdigest()
    }

    if not sizes:
        return labels, sizes, strides, data.tolist(), None, meta

    prod = np.prod([max(s, 1) for s in sizes])
    
    # Handle H×W×1 tolerance
    if allow_trailing_channel1 and len(sizes) >= 3 and sizes[-1] == 1 and data.size == (prod // sizes[-1]):
        shape_tuple = tuple(sizes[:-1])
        meta["policy"] = "reshape_squeezed_c1"
    elif data.size != prod:
        meta["warnings"].append(f"Size mismatch: len(data)={data.size} vs ∏sizes={prod}. Returning flat.")
        return labels, sizes, strides, data.tolist(), None, meta
    else:
        shape_tuple = tuple(sizes)

    try:
        arr = data.reshape(shape_tuple, order="C")
    except Exception as e:
        meta["warnings"].append(f"reshape({shape_tuple}) failed: {e}. Returning flat.")
        return labels, sizes, strides, data.tolist(), None, meta

    meta["used_shape"] = arr.shape
    if arr.ndim == 2:
        meta["policy"] = "reshape_exact"
        return labels, sizes, strides, arr.tolist(), (arr.shape[0], arr.shape[1]), meta

    # Squeeze 3D H×W×1 to 2D
    if arr.ndim == 3 and allow_trailing_channel1 and arr.shape[-1] == 1:
        arr2 = arr[..., 0]
        meta["policy"] = "reshape_squeeze_c1"
        meta["used_shape"] = arr2.shape
        return labels, sizes, strides, arr2.tolist(), (arr2.shape[0], arr2.shape[1]), meta

    # Heuristic detection
    if allow_heuristics and arr.ndim >= 2:
        lab_l = [l.lower() for l in labels]
        hi = wi = None
        for k in reversed(range(len(lab_l))):
            if any(key in lab_l[k] for key in {"height","rows","beams"}): hi = k
            if any(key in lab_l[k] for key in {"width","cols","bins","range","samples"}): wi = k
        if hi is not None and wi is not None and hi != wi:
            Hs, Ws = sizes[hi], sizes[wi]
            if Hs > 0 and Ws > 0 and Hs * Ws == data.size:
                arr2 = data.reshape((Hs, Ws), order="C")
                meta["policy"] = "heuristic_hw_from_labels"
                meta["used_shape"] = (Hs, Ws)
                return labels, sizes, strides, arr2.tolist(), (Hs, Ws), meta

    # Return N-D as nested lists
    meta["policy"] = "reshape_nd"
    return labels, sizes, strides, arr.tolist(), tuple(arr.shape), meta

def list_topics_in_bag(bagpath: Union[str, Path]) -> List[Tuple[str, str]]:
    with AnyReader([Path(bagpath)]) as reader:
        return sorted({(c.topic, c.msgtype) for c in reader.connections})

def _extract_times_common(msg, t_ns):
    t_bag = float(t_ns) * 1e-9 if t_ns is not None else None
    t_hdr = stamp_seconds(msg)
    return (t_hdr, t_bag, t_hdr if t_hdr and t_hdr > 0 else t_bag, "header" if t_hdr and t_hdr > 0 else "bag")

def bag_topic_to_dataframe(bagpath: Union[str, Path], topic: str, arrays_as: str = "json") -> pd.DataFrame:
    bagpath = Path(bagpath)
    rows: List[Dict] = []

    with AnyReader([bagpath]) as reader:
        ensure_types_from_reader(reader)
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            return pd.DataFrame()

        for con, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, con.msgtype)
            t_hdr, t_bag, t_use, t_src = _extract_times_common(msg, t_ns)

            # SonoptixECHO handling
            if base_msgtype(con.msgtype) == "SonoptixECHO" and hasattr(msg, "array_data"):
                labels, sizes, strides, payload, shape, meta = decode_float32_multiarray(msg.array_data)
                rec = {
                    "t": t_use, "t_header": t_hdr, "t_bag": t_bag, "t_src": t_src,
                    "bag": bagpath.stem, "bag_file": bagpath.name, "topic": topic,
                    "dim_labels": json.dumps(labels, ensure_ascii=False),
                    "dim_sizes": json.dumps(sizes, ensure_ascii=False),
                    "dim_strides": json.dumps(strides, ensure_ascii=False),
                }
                if meta:
                    rec.update({
                        "data_offset": meta.get("data_offset"),
                        "dtype": meta.get("dtype"),
                        "len_data": meta.get("len_data"),
                        "payload_sha256": meta.get("payload_sha256"),
                        "used_shape": json.dumps(list(meta.get("used_shape") or [])),
                        "policy": meta.get("policy"),
                        "warnings": json.dumps(meta.get("warnings", []), ensure_ascii=False)
                    })
                if shape is not None:
                    H, W = shape
                    rec.update({"rows": int(H), "cols": int(W), "image": json.dumps(payload, ensure_ascii=False)})
                else:
                    rec.update({"data": json.dumps(payload, ensure_ascii=False), "len": len(payload)})
                rows.append(rec)
                continue

            # Ping360 handling
            if ("ping360" in topic.lower()) or (base_msgtype(con.msgtype) in ("Ping360", "Ping")):
                bearing = bins = None
                for attr in ("bearing", "angle", "azimuth", "theta", "heading"):
                    if hasattr(msg, attr):
                        bearing = float(getattr(msg, attr))
                        break
                for attr in ("bins", "ranges", "samples", "intensities", "echo", "intensity"):
                    if hasattr(msg, attr):
                        v = getattr(msg, attr)
                        if isinstance(v, (list, tuple)):
                            bins = [float(x) for x in v]
                        break
                if bearing is not None or bins is not None:
                    rec = {
                        "t": t_use, "t_header": t_hdr, "t_bag": t_bag, "t_src": t_src,
                        "bag": bagpath.stem, "bag_file": bagpath.name, "topic": topic,
                    }
                    if bearing is not None: rec["bearing_deg"] = bearing
                    if bins is not None: rec.update({"bins": json.dumps(bins, ensure_ascii=False), "n_bins": len(bins)})
                    rows.append(rec)
                    continue

            # Generic fallback
            rec = {
                "t": t_use, "t_header": t_hdr, "t_bag": t_bag, "t_src": t_src,
                "bag": bagpath.stem, "bag_file": bagpath.name, "topic": topic,
            }
            flatten_msg(msg, rec, arrays_as=arrays_as)
            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    df["t0"] = df["t"].iloc[0]
    df["t_rel"] = df["t"] - df["t0"]
    df["ts_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    try:
        df["ts_oslo"] = df["ts_utc"].dt.tz_convert("Europe/Oslo")
    except:
        df["ts_oslo"] = df["ts_utc"]
    return df

def save_dataframe(df: pd.DataFrame, path: Union[str, Path], file_format: str = "csv") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def save_all_topics_from_data_bags(data_dir: Union[str, Path], out_dir: Union[str, Path] = None, file_format: str = "csv", arrays_as: str = "json", recursive: bool = False, exclude_msgtypes: Optional[Iterable[str]] = None, include_msgtypes: Optional[Iterable[str]] = None, *, include_video_sonar: bool = True, sonar_topic_patterns: Optional[List[str]] = None) -> pd.DataFrame:
    data_dir = Path(data_dir)
    from utils.config import EXPORTS_DIR_DEFAULT
    out_dir = Path(out_dir or EXPORTS_DIR_DEFAULT)
    by_bag_dir = out_dir / "by_bag"
    by_bag_dir.mkdir(parents=True, exist_ok=True)

    sonar_whitelist = {"sensors/msg/SonoptixECHO", "sensors/msg/Ping360", "sensors/msg/Ping"}
    if include_msgtypes is not None:
        include_msgtypes = set(include_msgtypes) | sonar_whitelist

    if sonar_topic_patterns is None:
        sonar_topic_patterns = [r"/sonoptix", r"/echo", r"/sonar", r"/ping360", r"/ping\b", r"/mbes"]
    import re
    sonar_topic_re = re.compile("|".join(sonar_topic_patterns), re.IGNORECASE)

    registered = register_custom_msgs_from_dataset_root(data_dir)
    if registered:
        print(f"Registered {registered} custom ROS message types.")

    data_bags = find_data_bags(data_dir, recursive=recursive)
    if not data_bags:
        raise FileNotFoundError(f"No *_data.bag files found under {data_dir} (recursive={recursive}).")
    video_bags = find_video_bags(data_dir, recursive=recursive) if include_video_sonar else []

    index_rows = []

    def _export_from_bag(bag: Path, sonar_only: bool = False) -> None:
        with AnyReader([bag]) as reader:
            ensure_types_from_reader(reader)
            topic_types = {}
            for c in reader.connections:
                topic_types.setdefault(c.topic, set()).add(c.msgtype)

            for topic, types in topic_types.items():
                if sonar_only and not sonar_topic_re.search(topic):
                    continue

                should_include = True
                if include_msgtypes and not any(t in include_msgtypes for t in types):
                    should_include = sonar_only and sonar_topic_re.search(topic)
                elif exclude_msgtypes and all(t in exclude_msgtypes for t in types):
                    should_include = sonar_only and sonar_topic_re.search(topic)

                if not should_include:
                    continue

                df_topic = bag_topic_to_dataframe(bag, topic, arrays_as=arrays_as)
                if df_topic.empty:
                    continue

                ext = "parquet" if file_format == "parquet" else "csv"
                fname = f"{sanitize_topic(topic)}__{bag.stem}.{ext}"
                out_path = by_bag_dir / fname
                save_dataframe(df_topic, out_path, file_format=file_format)

                index_rows.append({
                    "bag": bag.stem, "bag_file": bag.name, "topic": topic,
                    "msgtypes": sorted(types), "rows": int(len(df_topic)), "out_file": str(out_path),
                })
                print(f"Wrote {out_path} ({len(df_topic)} rows).")

    for bag in data_bags:
        _export_from_bag(bag, sonar_only=False)
    for vbag in video_bags:
        _export_from_bag(vbag, sonar_only=True)

    index_df = pd.DataFrame(index_rows).sort_values(["bag", "topic"]).reset_index(drop=True)
    ext = "parquet" if file_format == "parquet" else "csv"
    idx_path = out_dir / f"index_data_topics.{ext}"
    save_dataframe(index_df, idx_path, file_format=file_format)
    print(f"\nIndex written to {idx_path} with {len(index_df)} entries.")
    return index_df

# Camera/Video functions
def list_camera_topics_in_bag(bagpath: Union[str, Path]) -> List[Tuple[str, str]]:
    with AnyReader([Path(bagpath)]) as reader:
        return sorted({(c.topic, c.msgtype) for c in reader.connections if base_msgtype(c.msgtype) in ("Image", "CompressedImage")})

def _decode_compressed_to_bgr(msg) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

def _decode_raw_to_bgr(msg) -> Optional[np.ndarray]:
    enc = getattr(msg, "encoding", "").lower()
    h, w = getattr(msg, "height", None), getattr(msg, "width", None)
    if h is None or w is None:
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "bgr8":
        return buf.reshape(h, w, 3)
    if enc == "rgb8":
        return cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    return None

def export_camera_info_for_bag(bagpath: Union[str, Path], out_dir: Union[str, Path] = None, one_per_topic: bool = True) -> pd.DataFrame:
    bagpath = Path(bagpath)
    from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    out_dir = Path(out_dir or Path(EXPORTS_DIR_DEFAULT) / EXPORTS_SUBDIRS.get('camera_info', 'camera_info'))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with AnyReader([bagpath]) as reader:
        ensure_types_from_reader(reader)
        for con in [c for c in reader.connections if base_msgtype(c.msgtype) == "CameraInfo"]:
            try:
                _, t_ns, raw = next(reader.messages(connections=[con]))
                msg = reader.deserialize(raw, con.msgtype)

                roi = getattr(msg, "roi", None)
                payload = {
                    "height": int(getattr(msg, "height", 0)),
                    "width": int(getattr(msg, "width", 0)),
                    "distortion_model": getattr(msg, "distortion_model", ""),
                    "D": getattr(msg, "d", getattr(msg, "D", [])),
                    "K": getattr(msg, "k", getattr(msg, "K", [])),
                    "R": getattr(msg, "r", getattr(msg, "R", [])),
                    "P": getattr(msg, "p", getattr(msg, "P", [])),
                    "binning_x": int(getattr(msg, "binning_x", 0)),
                    "binning_y": int(getattr(msg, "binning_y", 0)),
                    "roi": {
                        "x_offset": int(getattr(roi, "x_offset", 0)) if roi else 0,
                        "y_offset": int(getattr(roi, "y_offset", 0)) if roi else 0,
                        "height": int(getattr(roi, "height", 0)) if roi else 0,
                        "width": int(getattr(roi, "width", 0)) if roi else 0,
                        "do_rectify": bool(getattr(roi, "do_rectify", False)) if roi else False,
                    },
                }

                fname = f"{bagpath.stem}__{sanitize_topic(con.topic)}.yaml"
                out_path = out_dir / fname
                with open(out_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(to_native(payload), f, sort_keys=False)
                rows.append({"bag": bagpath.stem, "topic": con.topic, "out_file": str(out_path)})
            except StopIteration:
                continue
            if one_per_topic:
                break
    return pd.DataFrame(rows)

def export_camera_topic_to_mp4(bagpath: Union[str, Path], topic: str, out_path: Union[str, Path], codec: str = "mp4v", target_fps: Optional[float] = None, probe_frames: int = 120, min_frames_for_fps: int = 5, resize_to_first: bool = True) -> Dict[str, Union[str, int, float]]:
    bagpath, out_path = Path(bagpath), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_written = fps = 0
    writer = W = H = None

    with AnyReader([bagpath]) as reader:
        ensure_types_from_reader(reader)
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise ValueError(f"Topic {topic!r} not found in {bagpath.name}")
        con = conns[0]
        mt = base_msgtype(con.msgtype)

        # Auto-detect FPS if not provided
        if target_fps is None:
            stamps = []
            for i, (_, t_ns, raw) in enumerate(reader.messages(connections=[con])):
                if i >= probe_frames: break
                msg = reader.deserialize(raw, con.msgtype)
                t = stamp_seconds(msg, t_ns)
                if t: stamps.append(t)
            if len(stamps) >= min_frames_for_fps + 1:
                dts = np.diff(stamps)
                dts = dts[(dts > 1e-6) & (dts < 5.0)]
                fps = float(np.clip(1.0 / np.median(dts), 1.0, 120.0)) if len(dts) else 15.0
            else:
                fps = 15.0
        else:
            fps = target_fps

        for _, t_ns, raw in reader.messages(connections=[con]):
            msg = reader.deserialize(raw, con.msgtype)
            img = _decode_compressed_to_bgr(msg) if mt == "CompressedImage" else _decode_raw_to_bgr(msg)
            if img is None: continue

            if writer is None:
                H, W = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H), True)
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open VideoWriter for {out_path}")

            if resize_to_first and (img.shape[0] != H or img.shape[1] != W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            writer.write(img)
            frames_written += 1

    if writer: writer.release()
    return {"bag": bagpath.stem, "topic": topic, "out_file": str(out_path), "frames": frames_written, "fps": float(fps), "codec": codec}

def _format_ts(ts_sec, tz_name="Europe/Oslo", fmt="%Y%m%d_%H%M%S_%f%z"):
    if ts_sec is None: return "unknown_ts"
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    if ZoneInfo:
        try: dt_loc = dt_utc.astimezone(ZoneInfo(tz_name))
        except: dt_loc = dt_utc
    else: dt_loc = dt_utc
    return dt_loc.strftime(fmt)

def export_camera_topic_to_png_sequence(bagpath: Union[str, Path], topic: str, out_dir: Union[str, Path], stride: int = 1, limit: Optional[int] = None, zero_pad: int = 6, resize_to: Optional[Tuple[int, int]] = None, *, timestamp_tz: str = "Europe/Oslo", timestamp_fmt: str = "%Y%m%d_%H%M%S_%f%z", timestamp_source: str = "auto", write_index_csv: bool = True) -> Dict[str, Union[str, int]]:
    bagpath = Path(bagpath)
    seq_dir = Path(out_dir) / f"{bagpath.stem}__{sanitize_topic(topic)}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    index_rows, frames_written, idx = [], 0, 0
    with AnyReader([bagpath]) as reader:
        ensure_types_from_reader(reader)
        cons = [c for c in reader.connections if c.topic == topic]
        if not cons:
            raise ValueError(f"Topic {topic!r} not found in {bagpath.name}")
        con, mt = cons[0], base_msgtype(cons[0].msgtype)

        for _, t_ns, raw in reader.messages(connections=[con]):
            if limit and frames_written >= limit: break
            if idx % stride != 0:
                idx += 1
                continue

            msg = reader.deserialize(raw, con.msgtype)
            img = _decode_compressed_to_bgr(msg) if mt == "CompressedImage" else _decode_raw_to_bgr(msg)
            idx += 1
            if img is None: continue

            t_hdr, t_bag, t_auto, _ = _extract_times_common(msg, t_ns)
            ts = {"header": t_hdr, "bag": t_bag}.get(timestamp_source, t_auto)

            base_name = _format_ts(ts, tz_name=timestamp_tz, fmt=timestamp_fmt)
            fname, bump = seq_dir / f"{base_name}.png", 1
            while fname.exists():
                fname = seq_dir / f"{base_name}__{bump:0{zero_pad}d}.png"
                bump += 1

            if resize_to and (img.shape[1] != resize_to[0] or img.shape[0] != resize_to[1]):
                img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(fname), img)
            frames_written += 1

            if write_index_csv:
                dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
                dt_loc = dt_utc.astimezone(ZoneInfo(timestamp_tz)) if ZoneInfo and dt_utc else dt_utc
                index_rows.append({
                    "bag": bagpath.stem, "topic": topic, "file": str(fname.relative_to(seq_dir)),
                    "t_use": ts, "t_header": t_hdr, "t_bag": t_bag,
                    "ts_utc": dt_utc.isoformat() if dt_utc else None,
                    "ts_local": dt_loc.isoformat() if dt_loc else None,
                })

    if write_index_csv and index_rows:
        pd.DataFrame(index_rows).to_csv(seq_dir / "index.csv", index=False)
        print(f"Wrote {frames_written} PNGs to {seq_dir} with index.")
    else:
        print(f"Wrote {frames_written} PNGs to {seq_dir}.")

    return {"bag": bagpath.stem, "topic": topic, "out_dir": str(seq_dir), "frames": frames_written}

def export_all_video_bags_to_mp4(data_dir: Union[str, Path], out_dir: Union[str, Path] = None, recursive: bool = False, topics: Optional[Iterable[str]] = None, codec: str = "mp4v", target_fps: Optional[float] = None, probe_frames: int = 120, min_frames_for_fps: int = 5, overwrite: bool = False, save_camera_info_yaml: bool = True) -> pd.DataFrame:
    data_dir = Path(data_dir)
    from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    out_dir = Path(out_dir or EXPORTS_DIR_DEFAULT)
    vid_dir = out_dir / EXPORTS_SUBDIRS.get('videos', 'videos')
    info_dir = out_dir / EXPORTS_SUBDIRS.get('camera_info', 'camera_info')
    vid_dir.mkdir(parents=True, exist_ok=True)
    if save_camera_info_yaml: info_dir.mkdir(parents=True, exist_ok=True)

    index_rows, bags = [], find_video_bags(data_dir, recursive=recursive)
    if not bags:
        raise FileNotFoundError(f"No *_video.bag files found under {data_dir} (recursive={recursive}).")

    for bag in bags:
        if save_camera_info_yaml:
            try: export_camera_info_for_bag(bag, out_dir=info_dir)
            except Exception as e: print(f"(warn) CameraInfo export failed for {bag.name}: {e}")

        if topics is None:
            # Default: only compressed image topics, exclude ted cameras
            all_topics = [t for (t, _) in list_camera_topics_in_bag(bag)]
            bag_topics = [t for t in all_topics if 'compressed_image' in t and 'ted' not in t]
        else:
            bag_topics = list(topics)
            
        for topic in bag_topics:
            mp4_path = vid_dir / f"{bag.stem}__{sanitize_topic(topic)}.mp4"
            if mp4_path.exists() and not overwrite:
                index_rows.append({"bag": bag.stem, "topic": topic, "out_file": str(mp4_path), "frames": np.nan, "fps": target_fps or np.nan, "codec": codec, "skipped": True})
                print(f"(skip) exists: {mp4_path}")
                continue
            try:
                res = export_camera_topic_to_mp4(bag, topic, mp4_path, codec=codec, target_fps=target_fps, probe_frames=probe_frames, min_frames_for_fps=min_frames_for_fps)
                res["skipped"] = False
                index_rows.append(res)
                print(f"Wrote {res['out_file']} ({res['frames']} frames @ ~{res['fps']:.2f} FPS).")
            except Exception as e:
                print(f"(error) {bag.name}:{topic}: {e}")

    idx = pd.DataFrame(index_rows)
    idx_path = out_dir / EXPORTS_SUBDIRS.get('index', '') / "index_video_mp4.csv"
    idx.to_csv(idx_path, index=False)
    print(f"\nIndex written: {idx_path} ({len(idx)} rows).")
    return idx

def export_all_video_bags_to_png(data_dir: Union[str, Path], out_dir: Union[str, Path] = None, recursive: bool = False, topics: Optional[Iterable[str]] = None, stride: int = 1, limit: Optional[int] = None, resize_to: Optional[Tuple[int, int]] = None, overwrite: bool = False) -> pd.DataFrame:
    data_dir = Path(data_dir)
    from utils.config import EXPORTS_DIR_DEFAULT, EXPORTS_SUBDIRS
    out_dir = Path(out_dir or EXPORTS_DIR_DEFAULT)
    png_root = out_dir / EXPORTS_SUBDIRS.get('frames', 'frames')
    png_root.mkdir(parents=True, exist_ok=True)

    index_rows, bags = [], find_video_bags(data_dir, recursive=recursive)
    if not bags:
        raise FileNotFoundError(f"No *_video.bag files found under {data_dir} (recursive={recursive}).")

    for bag in bags:
        if topics is None:
            # Default: only compressed image topics, exclude ted cameras
            all_topics = [t for (t, _) in list_camera_topics_in_bag(bag)]
            bag_topics = [t for t in all_topics if 'compressed_image' in t and 'ted' not in t]
        else:
            bag_topics = list(topics)
            
        for topic in bag_topics:
            seq_dir = png_root / f"{bag.stem}__{sanitize_topic(topic)}"
            if seq_dir.exists() and not overwrite:
                index_rows.append({"bag": bag.stem, "topic": topic, "out_dir": str(seq_dir), "frames": np.nan, "skipped": True})
                print(f"(skip) exists: {seq_dir}")
                continue
            try:
                res = export_camera_topic_to_png_sequence(bag, topic, png_root, stride=stride, limit=limit, resize_to=resize_to)
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

__all__ = ["sanitize_topic","base_msgtype","find_data_bags","find_video_bags","register_custom_msgs_from_dataset_root","stamp_seconds","coerce_for_table","to_native","flatten_msg","ensure_types_from_reader","decode_float32_multiarray","list_topics_in_bag","bag_topic_to_dataframe","save_dataframe","save_all_topics_from_data_bags","list_camera_topics_in_bag","export_camera_info_for_bag","export_camera_topic_to_mp4","export_camera_topic_to_png_sequence","export_all_video_bags_to_mp4","export_all_video_bags_to_png"]
