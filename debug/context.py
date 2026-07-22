from contextlib import contextmanager
from contextvars import ContextVar

_CONTEXT = ContextVar("snakeML_debug_context", default={})


@contextmanager
def operation_context(operation=None, **metadata):
    """Temporarily attach operation metadata to the active debug context."""
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


def current_context():
    """Return a copy of the active debug context metadata."""
    return dict(_CONTEXT.get())


def update_context(**metadata):
    """Merge metadata into the active debug context."""
    current = dict(current_context())
    current.update(metadata)
    _CONTEXT.set(current)
    return current


def clear_context():
    """Clear the active debug context."""
    _CONTEXT.set({})
    return {}
