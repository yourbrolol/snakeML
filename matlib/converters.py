"""Conversion helpers between nested lists, buffers, and Arrays."""

from __future__ import annotations

from typing import Any

__all__ = ["is_array", "is_buffer", "to_array", "to_buffer", "to_list"]


def _array_type() -> type:
    from structs.array.array import Array

    return Array


def _buffer_type() -> type:
    from structs.array.buffer import Buffer

    return Buffer


def is_array(value: Any) -> bool:
    """Return whether *value* is an Array."""
    return isinstance(value, _array_type())


def is_buffer(value: Any) -> bool:
    """Return whether *value* is a Buffer."""
    return isinstance(value, _buffer_type())


def to_list(value: Any) -> Any:
    """Convert Arrays, buffers, and nested sequences into plain lists."""
    if is_array(value):
        return value.tolist()
    if is_buffer(value):
        return [to_list(item) for item in value.buffer]
    if isinstance(value, list):
        return [to_list(item) for item in value]
    if isinstance(value, tuple):
        return [to_list(item) for item in value]
    return value


def to_array(value: Any) -> Any:
    """Convert a list-like value into an Array, leaving Arrays unchanged."""
    if is_array(value):
        return value
    if is_buffer(value):
        value = [to_list(item) for item in value.buffer]
    return _array_type()(value)


def to_buffer(value: Any) -> Any:
    """Convert an Array or list-like value into a Buffer."""
    if is_buffer(value):
        return value
    return to_array(value).data
