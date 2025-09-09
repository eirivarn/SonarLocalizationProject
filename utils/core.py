from __future__ import annotations
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

def _to_py_floats(seq):
    """Convert numpy scalars to built-in float for JSON safety."""
    return [float(x) for x in seq]

def decode_float32_multiarray(ma):
    """
    Strict ROS MultiArrayLayout decoder.
    Returns: labels, sizes, strides, payload (list or 2D list), shape or None
    - honors layout.data_offset
    - uses dims in outer-most -> inner-most order
    - prefers 2D (H,W) view when possible; keeps full shape otherwise
    """
    # Extract layout
    dims = list(getattr(ma.layout, "dim", []) or [])
    labels  = [str(getattr(d, "label", "") or "") for d in dims]
    sizes   = [int(getattr(d, "size", 0) or 0) for d in dims]
    strides = [int(getattr(d, "stride", 0) or 0) for d in dims]
    data_off = int(getattr(ma.layout, "data_offset", 0) or 0)

    # Flat data with offset applied
    data = [float(x) for x in getattr(ma, "data", [])]
    if data_off > 0:
        data = data[data_off:]

    # If no dims given → return flat vector (legacy/minimal publishers)
    if not sizes:
        return labels, sizes, strides, data, None

    # Validate size product (ignore trailing singleton channel=1)
    prod = 1
    for s in sizes:
        prod *= max(s, 1)
    if len(data) < prod:
        # publisher might be dropping a trailing 1-dim or misreporting; try to be lenient
        prod = len(data)

    # Derive a canonical shape (outer->inner)
    shape = tuple(sizes)
    # Prefer a 2D image view if recognizable:
    # H ∈ {"height","rows","beams"} and W ∈ {"width","cols","bins","range","samples"}
    lab_l = [l.lower() for l in labels]
    def find_idx(keys):
        for k in reversed(range(len(lab_l))):  # prefer last match (often width/bins last)
            if any(key in lab_l[k] for key in keys):
                return k
        return None
    hi = find_idx({"height","rows","beams"})
    wi = find_idx({"width","cols","bins","range","samples"})

    arr = np.asarray(data, dtype=float)

    try:
        arr = arr.reshape(shape, order='C')  # ROS layout uses standard row-major with given dims
    except Exception:
        # Fallback: if shape seems off, assume 2D by inferring H×W from product
        # Prefer last two dims if there are >=2 dims
        if len(sizes) >= 2:
            H, W = sizes[-2], sizes[-1]
            if H*W == len(data):
                arr = arr.reshape((H, W))
                labels2 = [labels[-2], labels[-1]]
                sizes2  = [H, W]
                strides2 = strides[-2:] if strides else []
                return labels2, sizes2, strides2, arr.tolist(), (H, W)
        # As a last resort, return flat
        return labels, sizes, strides, data, None

    # If we found plausible H/W labels, collapse other dims if they are singleton or "channel"
    if hi is not None and wi is not None and hi != wi:
        H, W = sizes[hi], sizes[wi]
        if H > 0 and W > 0 and H*W == arr.size:
            arr2 = arr.reshape((H, W))
            return labels, sizes, strides, arr2.tolist(), (H, W)

    # Common 3D case: H×W×C with C=1 or C small; favor returning H×W
    if arr.ndim == 3:
        H, W, C = arr.shape
        if C == 1:
            return labels, sizes, strides, arr[..., 0].tolist(), (H, W)
        # If dims are (C,H,W) or (H,C,W), try to detect H/W by label/size
        if hi is not None and wi is not None:
            Hs, Ws = sizes[hi], sizes[wi]
            if {Hs, Ws}.issubset(set(arr.shape)):
                # move axes so H,W are first two
                axes = list(range(arr.ndim))
                # map current axes to H and W positions
                h_ax = arr.shape.index(Hs)
                w_ax = arr.shape.index(Ws)
                axes.remove(h_ax); axes.remove(w_ax)
                new_order = [h_ax, w_ax] + axes
                arr2 = np.transpose(arr, new_order)
                # collapse the rest (e.g., channel) by taking mean or first; here we take first
                arr2 = arr2[..., 0]
                return labels, sizes, strides, arr2.tolist(), (Hs, Ws)

    # If nothing matched cleanly, return full shape (caller can still use it)
    return labels, sizes, strides, arr.tolist(), tuple(arr.shape)
