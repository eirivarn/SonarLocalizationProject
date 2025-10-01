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
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# ------------------------------ Paths & Names ------------------------------

def sanitize_topic(topic: str) -> str:
    """Make a ROS topic safe for filenames."""
    return (topic or "").strip("/").replace("/", "_") or "root"

def base_msgtype(s: str) -> str:
    """Return the final segment of a ROS msgtype path."""
    return s.rsplit("/", 1)[-1] if s else s

def find_data_bags(data_dir: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Sorted list of paths to *_data.bag in data_dir."""
    data_dir = Path(data_dir)
    bags = data_dir.rglob("*.bag") if recursive else data_dir.glob("*.bag")
    return sorted(p for p in bags if p.name.endswith("_data.bag"))

def find_video_bags(data_dir: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Sorted list of paths to *_video.bag in data_dir."""
    data_dir = Path(data_dir)
    bags = data_dir.rglob("*.bag") if recursive else data_dir.glob("*.bag")
    return sorted(p for p in bags if p.name.endswith("_video.bag"))


# ------------------------------ Msg Registry ------------------------------

def register_custom_msgs_from_dataset_root(root: Union[str, Path]) -> int:
    """
    Auto-register custom .msg types if present under: root/msg, root/msgs, root/custom_msgs.
    Returns count of message types registered.
    """
    root = Path(root)
    count = 0
    for cand in (root / "msg", root / "msgs", root / "custom_msgs"):
        if not cand.exists():
            continue
        additions: Dict[str, str] = {}
        for p in cand.rglob("*.msg"):
            # expects <pkg>/msg/*.msg
            pkg = p.parent.parent.name
            additions.update(get_types_from_msg(p.read_text(encoding="utf-8"), f"{pkg}/msg"))
        if additions:
            register_types(additions)
            count = len(additions)
        break
    return count


# ------------------------------ Timestamps ------------------------------

def stamp_seconds(msg, fallback_t_ns: Optional[int] = None) -> Optional[float]:
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


# ------------------------------ Coercions ------------------------------

def coerce_for_table(val, arrays_as: str = "json"):
    """
    Convert values into table-friendly forms.
    - bytes -> base64 string
    - numpy arrays -> list/json depending on arrays_as
    - lists/tuples/dicts -> JSON string if arrays_as == 'json', else repr
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
                return repr(val)
        return repr(val)

    return val

def to_native(obj: Any):
    """Recursively convert NumPy scalars/arrays and containers to plain Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    return obj


# ------------------------------ Flattening ------------------------------

def flatten_msg(obj, rec: dict, prefix: str = "", arrays_as: str = "json") -> None:
    # ROS-style typed objects
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
        return

    # Dataclass / normal Python object
    if hasattr(obj, "__dict__"):
        for name, val in obj.__dict__.items():
            if name == "header":
                continue
            flatten_msg(val, rec, prefix + f"{name}.", arrays_as)
        return

    # Final fallback: do NOT use empty key; give it a name
    key = (prefix.rstrip(".") or "payload")
    rec[key] = coerce_for_table(obj, arrays_as=arrays_as)


# ------------------------------ Msg Registry from Bag ------------------------------
def ensure_types_from_reader(reader) -> int:
    """
    Ensure all message definitions present in this bag are registered.
    Returns how many types were registered.
    """
    additions = {}
    for c in reader.connections:
        try:
            # c.msgtype like 'sensors/msg/SonoptixECHO'
            pkg_prefix = c.msgtype.rsplit("/", 1)[0]  # 'sensors/msg'
            if getattr(c, "msgdef", None):
                additions.update(get_types_from_msg(c.msgdef, pkg_prefix))
        except Exception:
            # Some connections may not expose msgdef; ignore safely.
            pass
    if additions:
        register_types(additions)
        return len(additions)
    return 0


# --- Float32MultiArray helpers (Sonoptix etc.) ---------------------------------

def _sha256_floats32(data: np.ndarray) -> str:
    b = np.asarray(data, dtype=np.float32).tobytes(order="C")
    return hashlib.sha256(b).hexdigest()

def _safe_int(x, default=0, max_digits=20):
    """Safely convert to int; guard against giant strings / bad values."""
    try:
        if isinstance(x, str) and len(x) > max_digits:
            return default
        return int(x)
    except Exception:
        return default

def decode_float32_multiarray(
    ma,
    *,
    allow_heuristics: bool = False,          # default SAFE: no guessing
    allow_trailing_channel1: bool = True,    # common H×W×1 → H×W
) -> Tuple[
    List[str], List[int], List[int],
    List[float] | List[List[float]],
    Optional[Tuple[int, int]],
    Dict[str, Any]
]:
    """
    Lossless, auditable decoder for Float32MultiArray-like messages.

    Returns:
      labels, sizes, strides, payload(list or 2D list), shape(H,W)|None, meta(dict)
    Meta contains:
      data_offset, dtype, payload_sha256, len_data, used_shape, policy, warnings[list]
    """
    # ----- layout -----
    dims = list(getattr(ma.layout, "dim", []) or [])
    labels  = [str(getattr(d, "label", "") or "") for d in dims]
    sizes   = [_safe_int(getattr(d, "size", 0),   default=0) for d in dims]
    strides = [_safe_int(getattr(d, "stride", 0), default=0) for d in dims]
    data_off = _safe_int(getattr(ma.layout, "data_offset", 0), default=0)

    # ----- flat data (apply offset only) -----
    raw = getattr(ma, "data", [])
    data = np.asarray(raw, dtype=np.float32)
    if data_off > 0:
        data = data[data_off:]

    meta: Dict[str, Any] = {
        "data_offset": data_off,
        "dtype": "float32",
        "len_data": int(data.size),
        "policy": "flat",          # updated below if we reshape
        "used_shape": None,
        "warnings": [],
    }
    meta["payload_sha256"] = _sha256_floats32(data)

    # No dims → return flat as-is
    if not sizes:
        return labels, sizes, strides, data.tolist(), None, meta

    # Compute dimension product conservatively
    prod = 1
    for s in sizes:
        prod *= max(s, 1)

    # STRICT: only reshape when product matches
    if data.size != prod:
        # Special tolerance: H×W×1 declared but data is H×W (publisher may drop the channel array)
        if allow_trailing_channel1 and len(sizes) >= 3 and sizes[-1] == 1 and data.size == (prod // max(sizes[-1], 1)):
            # treat as if true product is H×W
            shape_tuple = tuple(sizes[:-1])
        else:
            meta["warnings"].append(
                f"Size mismatch: len(data)={data.size} vs ∏sizes={prod}. Returning flat."
            )
            return labels, sizes, strides, data.tolist(), None, meta
    else:
        shape_tuple = tuple(sizes)

    # Attempt reshape with the computed shape (row-major)
    try:
        arr = np.asarray(data, dtype=np.float32).reshape(shape_tuple, order="C")
    except Exception as e:
        meta["warnings"].append(f"reshape({shape_tuple}) failed: {e}. Returning flat.")
        return labels, sizes, strides, data.tolist(), None, meta

    # If 2D already → done
    if arr.ndim == 2:
        meta["policy"] = "reshape_exact" if tuple(sizes) == shape_tuple else "reshape_squeezed_c1"
        meta["used_shape"] = arr.shape
        if not np.array_equal(arr.ravel(order="C"), data):
            meta["warnings"].append("Round-trip mismatch after 2D reshape; falling back to flat.")
            return labels, sizes, strides, data.tolist(), None, meta
        return labels, sizes, strides, arr.tolist(), (arr.shape[0], arr.shape[1]), meta

    # If 3D H×W×1 and allowed → squeeze last channel
    if arr.ndim == 3 and allow_trailing_channel1 and arr.shape[-1] == 1:
        arr2 = arr[..., 0]
        meta["policy"] = "reshape_squeeze_c1"
        meta["used_shape"] = arr2.shape
        if not np.array_equal(arr2.ravel(order="C"), data.reshape(arr.shape, order="C")[..., 0].ravel(order="C")):
            meta["warnings"].append("Round-trip mismatch after squeeze; returning flat.")
            return labels, sizes, strides, data.tolist(), None, meta
        return labels, sizes, strides, arr2.tolist(), (arr2.shape[0], arr2.shape[1]), meta

    # Heuristic H×W detection (only if explicitly allowed)
    if allow_heuristics and arr.ndim >= 2:
        lab_l = [l.lower() for l in labels]
        def find_idx(keys):
            for k in reversed(range(len(lab_l))):
                if any(key in lab_l[k] for key in keys):
                    return k
            return None
        hi = find_idx({"height","rows","beams"})
        wi = find_idx({"width","cols","bins","range","samples"})
        if hi is not None and wi is not None and hi != wi:
            Hs, Ws = sizes[hi], sizes[wi]
            if Hs > 0 and Ws > 0 and Hs * Ws == data.size:
                arr2 = np.asarray(data, dtype=np.float32).reshape((Hs, Ws), order="C")
                meta["policy"] = "heuristic_hw_from_labels"
                meta["used_shape"] = (Hs, Ws)
                if not np.array_equal(arr2.ravel(order="C"), data):
                    meta["warnings"].append("Heuristic reshape round-trip mismatch; returning flat.")
                    return labels, sizes, strides, data.tolist(), None, meta
                return labels, sizes, strides, arr2.tolist(), (Hs, Ws), meta

    # Couldn’t confidently form 2D → return full-shape as nested lists (lossless)
    meta["policy"] = "reshape_nd"
    meta["used_shape"] = tuple(arr.shape)
    if not np.array_equal(np.asarray(arr, dtype=np.float32).ravel(order="C"), data):
        meta["warnings"].append("ND reshape round-trip mismatch; returning flat.")
        return labels, sizes, strides, data.tolist(), None, meta
    return labels, sizes, strides, arr.tolist(), tuple(arr.shape), meta


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
      columns: t, t0, t_rel, ts_utc, ts_oslo, t_header, t_bag, t_src, bag, bag_file, topic, [...]
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
            t_hdr, t_bag, t_use, t_src = _extract_times_common(msg, t_ns)

            # ---- SonoptixECHO: Float32MultiArray -> 2D 'image' (or flat 'data') ----
            if base_msgtype(con.msgtype) == "SonoptixECHO" and hasattr(msg, "array_data"):
                res = decode_float32_multiarray(msg.array_data)  # new returns 6, old returns 5
                if isinstance(res, tuple) and len(res) == 6:
                    labels, sizes, strides, payload, shape, meta = res
                else:
                    labels, sizes, strides, payload, shape = res
                    meta = {}

                rec = {
                    "t": t_use,                 # chosen time (header if present else bag)
                    "t_header": t_hdr,          # keep both for auditing
                    "t_bag": t_bag,
                    "t_src": t_src,             # "header" | "bag"
                    "bag": bagpath.stem,
                    "bag_file": bagpath.name,
                    "topic": topic,
                    "dim_labels": json.dumps(labels, ensure_ascii=False),
                    "dim_sizes":  json.dumps(sizes, ensure_ascii=False),
                    "dim_strides": json.dumps(strides, ensure_ascii=False),
                }

                # optional meta fields for auditability (present with new decoder)
                if meta:
                    rec["data_offset"]     = meta.get("data_offset")
                    rec["dtype"]           = meta.get("dtype")
                    rec["len_data"]        = meta.get("len_data")
                    rec["payload_sha256"]  = meta.get("payload_sha256")
                    rec["used_shape"]      = json.dumps(list(meta.get("used_shape") or []))
                    rec["policy"]          = meta.get("policy")
                    # store warnings as JSON so nothing is lost
                    rec["warnings"]        = json.dumps(meta.get("warnings", []), ensure_ascii=False)

                if shape is not None:
                    H, W = shape
                    rec["rows"] = int(H)
                    rec["cols"] = int(W)
                    rec["image"] = json.dumps(payload, ensure_ascii=False)
                else:
                    rec["data"] = json.dumps(payload, ensure_ascii=False)
                    rec["len"]  = len(payload)

                rows.append(rec)
                continue


            # ---- Ping360-like: bearing + bins/intensities array ----
            if ("ping360" in topic.lower()) or (base_msgtype(con.msgtype) in ("Ping360", "Ping")):
                bearing = None
                for cand in ("bearing", "angle", "azimuth", "theta", "heading"):
                    if hasattr(msg, cand):
                        try:    bearing = float(getattr(msg, cand))
                        except: bearing = getattr(msg, cand)
                        break
                bins = None
                for cand in ("bins", "ranges", "samples", "intensities", "echo", "intensity"):
                    if hasattr(msg, cand):
                        v = getattr(msg, cand)
                        if isinstance(v, (list, tuple)): bins = [float(x) for x in v]
                        break
                if (bearing is not None) or (bins is not None):
                    rec = {
                        "t": t_use, "t_header": t_hdr, "t_bag": t_bag, "t_src": t_src,
                        "bag": bagpath.stem, "bag_file": bagpath.name, "topic": topic,
                    }
                    if bearing is not None: rec["bearing_deg"] = bearing
                    if bins is not None:
                        rec["bins"] = json.dumps(bins, ensure_ascii=False)
                        rec["n_bins"] = len(bins)
                    rows.append(rec)
                    continue

            # ---- Generic fallback ----
            rec = {
                "t": t_use, "t_header": t_hdr, "t_bag": t_bag, "t_src": t_src,
                "bag": bagpath.stem, "bag_file": bagpath.name, "topic": topic,
            }
            flatten_msg(msg, rec, arrays_as=arrays_as)
            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    # relative time & readable timestamps
    df["t0"]    = df["t"].iloc[0]
    df["t_rel"] = df["t"] - df["t0"]
    df["ts_utc"]  = pd.to_datetime(df["t"], unit="s", utc=True)
    # Europe/Oslo (handles DST automatically if tz data is available)
    try:
        df["ts_oslo"] = df["ts_utc"].dt.tz_convert("Europe/Oslo")
    except Exception:
        df["ts_oslo"] = df["ts_utc"]  # fallback if tz database missing

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

    sonar_whitelist = {
        "/sensor/sonoptix_echo/image"
        "sensors/msg/SonoptixECHO",
        "sensors/msg/Ping360",
        "sensors/msg/Ping",
    }
    if include_msgtypes is not None:
        include_msgtypes = set(include_msgtypes) | sonar_whitelist

    if sonar_topic_patterns is None:
        sonar_topic_patterns = [
            r"/sonoptix", r"/echo", r"/sonar", r"/ping360", r"/ping\b", r"/mbes",
        ]
    import re
    sonar_topic_re = re.compile("|".join(sonar_topic_patterns), re.IGNORECASE)

    registered = register_custom_msgs_from_dataset_root(data_dir)
    if registered:
        print(f"Registered {registered} custom ROS message types.")

    data_bags = find_data_bags(data_dir, recursive=recursive)
    if not data_bags:
        raise FileNotFoundError(f"No *_data.bag files found under {data_dir} (recursive={recursive}).")

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
                if allow_only_sonar_from_video and not sonar_topic_re.search(topic):
                    continue

                if include_msgtypes:
                    if not any(t in include_msgtypes for t in types):
                        if allow_only_sonar_from_video and sonar_topic_re.search(topic):
                            pass
                        else:
                            continue
                elif exclude_msgtypes:
                    if all(t in exclude_msgtypes for t in types):
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

    for bag in data_bags:
        _export_from_bag(bag, allow_only_sonar_from_video=False)

    for vbag in video_bags:
        _export_from_bag(vbag, allow_only_sonar_from_video=True)

    index_df = pd.DataFrame(index_rows).sort_values(["bag", "topic"]).reset_index(drop=True)
    idx_path = out_dir / ("index_data_topics.parquet" if file_format == "parquet" else "index_data_topics.csv")
    save_dataframe(index_df, idx_path, file_format=file_format)
    print(f"\nIndex written to {idx_path} with {len(index_df)} entries.")
    return index_df


def _extract_times_common(msg, t_ns):
    """Return (t_header, t_bag, t_use, t_src) in seconds where t_src is 'header' or 'bag'."""
    t_bag = float(t_ns) * 1e-9 if t_ns is not None else None

    t_hdr = None
    for p in (("header","stamp","sec","nanosec"), ("header","stamp","secs","nsecs")):
        try:
            sec = getattr(getattr(getattr(msg, p[0]), p[1]), p[2])
            nsc = getattr(getattr(getattr(msg, p[0]), p[1]), p[3])
            t_hdr = float(sec) + float(nsc) * 1e-9
            break
        except Exception:
            pass

    if t_hdr is not None and t_hdr > 0:
        return t_hdr, t_bag, t_hdr, "header"
    return t_hdr, t_bag, t_bag, "bag"


# ------------------------------ Video / Camera helpers ------------------------------

def list_camera_topics_in_bag(bagpath: Union[str, Path]) -> List[Tuple[str, str]]:
    bagpath = Path(bagpath)
    cams = []
    with AnyReader([bagpath]) as reader:
        for c in reader.connections:
            mt = base_msgtype(c.msgtype)
            if mt in ("Image", "CompressedImage"):
                cams.append((c.topic, c.msgtype))
    return sorted(set(cams))

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

def export_camera_info_for_bag(
    bagpath: Union[str, Path],
    out_dir: Union[str, Path] = "exports/camera_info",
    one_per_topic: bool = True,
) -> pd.DataFrame:
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
    resize_to: Optional[Tuple[int, int]] = None,
    *,
    timestamp_tz: str = "Europe/Oslo",
    timestamp_fmt: str = "%Y%m%d_%H%M%S_%f%z",
    timestamp_source: str = "auto",
    write_index_csv: bool = True,
) -> Dict[str, Union[str, int]]:
    bagpath = Path(bagpath)
    topic_tag = sanitize_topic(topic)
    seq_dir = Path(out_dir) / f"{bagpath.stem}__{topic_tag}"
    seq_dir.mkdir(parents=True, exist_ok=True)

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

            t_hdr, t_bag, t_auto, _src = _extract_times_common(msg, t_ns)
            if timestamp_source == "header":
                ts = t_hdr
            elif timestamp_source == "bag":
                ts = t_bag
            else:
                ts = t_auto

            ts_str = _format_ts(ts, tz_name=timestamp_tz, fmt=timestamp_fmt)

            base_name = ts_str
            fname = seq_dir / f"{base_name}.png"
            bump = 1
            while fname.exists():
                fname = seq_dir / f"{base_name}__{bump:0{zero_pad}d}.png"
                bump += 1

            if resize_to:
                W, H = resize_to
                if img.shape[1] != W or img.shape[0] != H:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(fname), img)
            frames_written += 1

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
                _ = df_info
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


def _format_ts(ts_sec, tz_name="Europe/Oslo", fmt="%Y%m%d_%H%M%S_%f%z"):
    if ts_sec is None:
        return "unknown_ts"
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_name)
            dt_loc = dt_utc.astimezone(tz)
        except Exception:
            dt_loc = dt_utc
    else:
        dt_loc = dt_utc
    s = dt_loc.strftime(fmt)
    return s


__all__ = [
    # utilities
    "sanitize_topic","base_msgtype","find_data_bags","find_video_bags",
    "register_custom_msgs_from_dataset_root","stamp_seconds","coerce_for_table","to_native",
    "flatten_msg","ensure_types_from_reader","decode_float32_multiarray",
    # data exports
    "list_topics_in_bag","bag_topic_to_dataframe","save_dataframe","save_all_topics_from_data_bags",
    # video/camera
    "list_camera_topics_in_bag","export_camera_info_for_bag","export_camera_topic_to_mp4",
    "export_camera_topic_to_png_sequence","export_all_video_bags_to_mp4","export_all_video_bags_to_png",
]
