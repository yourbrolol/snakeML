"""Message formatting helpers for the debug package.

Turns raw (message, level, context) triples into the single-line
``[PREFIX][LEVEL] [module] {key=value} message`` payloads emitted by
:class:`debug.logger.Logger`, plus small value "describers" used when
embedding runtime objects into messages.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Dict, Mapping, Optional, Union

from .config import get_config
from .context import current_context

__all__ = ["describe", "format_message"]


def format_message(
    message: Any,
    *,
    level: Union[str, int] = "info",
    module: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Format a log message with snakeML prefixes and optional context.

    Args:
        message: The payload to render; coerced with ``str``.
        level: Severity name or numeric severity; rendered uppercased.
        module: Emitting module name; inferred from the caller's frame
            when omitted.
        context: Extra key/value metadata; falls back to the active
            :func:`debug.context.current_context`.

    Returns:
        The fully formatted one-line payload.
    """
    config = get_config()
    prefix = config.get("prefix", "snakeML")
    level_name = str(level).upper()
    parts = [f"[{prefix}][{level_name}]"]

    if module is None:
        module = (
            inspect.currentframe().f_back.f_globals.get("__name__", "unknown")
            if inspect.currentframe() and inspect.currentframe().f_back
            else "unknown"
        )
    if config.get("include_module", True):
        parts.append(f"[{module}]")

    active_context = dict(context or current_context())
    if config.get("include_context", True) and active_context:
        compact_context = ", ".join(
            f"{key}={value}" for key, value in sorted(active_context.items())
        )
        parts.append(f"{{{compact_context}}}")

    parts.append(str(message))
    return " ".join(parts)


def describe(value: Any) -> str:
    """Produce a concise description for common library values.

    Args:
        value: Any runtime object (scalar, container, array-like...).

    Returns:
        A short, single-line representation such as ``list(len=3)`` or
        ``Array(shape=(2, 2), ndim=2)``.
    """
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return repr(value)
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"

    type_name = type(value).__name__
    shape = getattr(value, "shape", None)
    ndim = getattr(value, "ndim", None)
    if shape is not None or ndim is not None:
        shape_text = "unknown" if shape is None else shape
        return f"{type_name}(shape={shape_text}, ndim={ndim})"

    data = getattr(value, "data", None)
    if data is not None:
        return f"{type_name}(data={type(data).__name__})"
    return type_name
