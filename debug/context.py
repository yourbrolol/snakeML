"""Contextual debugging metadata via :mod:`contextvars`.

An ambient, context-local mapping (operation name, tensor shapes, ...)
that :func:`debug.formatter.format_message` automatically embeds into
every formatted line. Safe to use across threads and async tasks.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

#: The ambient, context-local metadata mapping.
_CONTEXT: "ContextVar[Dict[str, Any]]" = ContextVar(
    "snakeML_debug_context", default={}
)


@contextmanager
def operation_context(
    operation: Optional[str] = None, **metadata: Any
) -> Iterator[Dict[str, Any]]:
    """Temporarily attach operation metadata to the active debug context.

    Args:
        operation: Name of the operation being performed, stored under
            the ``"operation"`` key.
        **metadata: Additional key/value pairs merged over the previous
            context.

    Yields:
        The context mapping active for the duration of the ``with`` block.
    """
    previous = _CONTEXT.get()
    current = dict(previous)
    if operation is not None:
        current["operation"] = operation
    current.update(metadata)
    token = _CONTEXT.set(current)
    try:
        yield current
    finally:
        _CONTEXT.reset(token)


def current_context() -> Dict[str, Any]:
    """Return a copy of the active debug context metadata."""
    return dict(_CONTEXT.get())


def update_context(**metadata: Any) -> Dict[str, Any]:
    """Merge metadata into the active debug context.

    Args:
        **metadata: Key/value pairs merged over the current context.

    Returns:
        The new, post-update context mapping.
    """
    current = dict(current_context())
    current.update(metadata)
    _CONTEXT.set(current)
    return current


def clear_context() -> Dict[str, Any]:
    """Clear the active debug context.

    Returns:
        The now-empty context mapping.
    """
    _CONTEXT.set({})
    return {}
