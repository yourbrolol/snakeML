"""Runtime debugging utilities for snakeML.

Exposes exactly one small, orthogonal toolkit:

- :mod:`debug.config` - global verbosity/stream/file configuration.
- :mod:`debug.context` - ambient, context-local operation metadata.
- :mod:`debug.logger` - level-gated formatting and emission.
- :mod:`debug.errors` - the exception hierarchy above :class:`LibraryError`.
- :mod:`debug.validator` - type/shape validation raising those errors.
- :mod:`debug.diagnostics` - JSON snapshots of runtime objects.
"""

from .config import (
    configure,
    configure_from_env,
    get_config,
    is_enabled,
    reset_config,
    set_level,
)
from .context import clear_context, current_context, operation_context, update_context
from .diagnostics import collect_diagnostics, summarize_diagnostics
from .errors import (
    LibraryError,
    OperationError,
    ShapeError,
    TypeMismatchError,
    ValidationError,
)
from .formatter import describe, format_message
from .logger import Logger, get_logger, logger
from .validator import ensure_shape, ensure_type

__all__ = [
    "LibraryError",
    "Logger",
    "OperationError",
    "ShapeError",
    "TypeMismatchError",
    "ValidationError",
    "clear_context",
    "collect_diagnostics",
    "configure",
    "configure_from_env",
    "current_context",
    "describe",
    "ensure_shape",
    "ensure_type",
    "format_message",
    "get_config",
    "get_logger",
    "is_enabled",
    "logger",
    "operation_context",
    "reset_config",
    "set_level",
    "summarize_diagnostics",
    "update_context",
]
