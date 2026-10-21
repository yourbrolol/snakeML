"""Lightweight diagnostic snapshots for library objects.

Produces plain JSON-serializable mappings describing a value's type,
shape and (optionally) backing data, so failures can be reported
without dragging arbitrary objects across error boundaries.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

__all__ = ["collect_diagnostics", "summarize_diagnostics"]


def collect_diagnostics(
    obj: Optional[Any] = None, *, include_data: bool = False
) -> Dict[str, Any]:
    """Collect a lightweight diagnostic snapshot for a library object.

    Args:
        obj: Any object exposing duck-typed ``shape``/``ndim``/``dtype``
            attributes; ``None`` yields a minimal ``{"type": "NoneType"}``.
        include_data: When ``True`` (and available) attach the object's
            raw ``data`` payload to the snapshot.

    Returns:
        A JSON-serializable mapping of the object's diagnostics.
    """
    if obj is None:
        return {"type": "NoneType"}

    payload: Dict[str, Any] = {
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


def summarize_diagnostics(
    obj: Optional[Any] = None, *, include_data: bool = False
) -> str:
    """Serialize diagnostics to a compact JSON string.

    Args:
        obj: See :func:`collect_diagnostics`.
        include_data: See :func:`collect_diagnostics`.

    Returns:
        A JSON string with keys sorted for stable comparison.
    """
    return json.dumps(
        collect_diagnostics(obj, include_data=include_data), sort_keys=True
    )
