from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterable, Any
import base64
import json

import numpy as np

from rosbags.typesys import get_types_from_msg, register_types

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


# --- Float32MultiArray helpers (Sonoptix etc.) ---

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
