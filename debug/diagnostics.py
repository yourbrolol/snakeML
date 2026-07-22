import json


def collect_diagnostics(obj=None, *, include_data=False):
    """Collect a lightweight diagnostic snapshot for a library object."""
    if obj is None:
        return {"type": "NoneType"}

    payload = {
        "type": type(obj).__name__,
        "shape": getattr(obj, "shape", None),
        "ndim": getattr(obj, "ndim", None),
        "len": len(obj) if hasattr(obj, "__len__") else None,
    }

    if include_data and hasattr(obj, "data"):
        payload["data"] = obj.data
    if hasattr(obj, "dtype"):
        payload["dtype"] = obj.dtype
    return payload


def summarize_diagnostics(obj=None, *, include_data=False):
    """Serialize diagnostics to a compact JSON string."""
    return json.dumps(collect_diagnostics(obj, include_data=include_data), sort_keys=True)
