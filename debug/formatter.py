import inspect
import sys

from .config import get_config
from .context import current_context


def format_message(message, *, level="info", module=None, context=None):
    """Format a log message with snakeML prefixes and optional context."""
    config = get_config()
    prefix = config.get("prefix", "snakeML")
    level_name = str(level).upper()
    parts = [f"[{prefix}][{level_name}]"]

    if module is None:
        module = inspect.currentframe().f_back.f_globals.get("__name__", "unknown") if inspect.currentframe() and inspect.currentframe().f_back else "unknown"
    if config.get("include_module", True):
        parts.append(f"[{module}]")

    active_context = dict(context or current_context())
    if config.get("include_context", True) and active_context:
        compact_context = ", ".join(f"{key}={value}" for key, value in sorted(active_context.items()))
        parts.append(f"{{{compact_context}}}")

    parts.append(str(message))
    return " ".join(parts)


def describe(value):
    """Produce a concise description for common library values."""
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
