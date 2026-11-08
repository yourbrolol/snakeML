"""Buffer and shape helpers for the Array bufferization roadmap.

These are standalone functions (no ``self``) that contain the core logic
formerly inside ``ArrayUtilsOps`` methods.  The mixin methods now delegate
to these functions so that the public API is unchanged.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from structs.dtypes import NestedList, Scalar, Shape
from .buffer import Buffer


def _to_buffer(values: Any) -> Buffer:
    """Convert an Array-like input into a flat ``Buffer``.

    Standalone version of ``ArrayUtilsOps._to_buffer``.
    """
    from .utils import ArrayUtilsOps as _Ops

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"cannot create Array from {type(values).__name__}")
    if isinstance(values, _Ops):
        if isinstance(values.data, Buffer):
            return values.data
        return _to_buffer(values.data)
    if isinstance(values, Buffer):
        return values
    if isinstance(values, range):
        return Buffer(list(values))
    if isinstance(values, (list, tuple)):
        out: List[Scalar] = []
        for v in values:
            out.extend(_to_buffer(v).buffer)
        return Buffer(out)
    if hasattr(values, "__iter__") and not isinstance(values, (str, bytes, bytearray, dict)):
        out = []
        for v in values:
            out.extend(_to_buffer(v).buffer)
        return Buffer(out)
    if isinstance(values, (bool, int, float, complex)):
        return Buffer([values])
    raise TypeError(f"cannot create Array from {type(values).__name__}")


def _get_shape(data: Any) -> Shape:
    """Recursively calculate the dimensions (shape) of nested list data.

    Standalone version of ``ArrayUtilsOps._get_shape``.
    """
    if not isinstance(data, (list, tuple)):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _get_shape(data[0])


def _restore_from_flat(flat: Sequence[Any], shape: Shape) -> NestedList:
    """Rebuild nested lists from a flat sequence and a target shape.

    Standalone version of ``ArrayUtilsOps._restore_from_flat``.
    """
    if not shape:
        return flat[0]
    size = shape[0]
    if len(shape) == 1:
        return list(flat[:size])
    chunk = len(flat) // size
    return [
        _restore_from_flat(flat[i * chunk : (i + 1) * chunk], shape[1:])
        for i in range(size)
    ]


def _buffer_from(values: Any) -> Tuple[Buffer, Shape]:
    """Convert an Array-like input into a flat ``Buffer`` and its ``Shape``.

    Standalone version of ``ArrayUtilsOps._buffer_from``.
    """
    from .utils import ArrayUtilsOps as _Ops

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"cannot create Array from {type(values).__name__}")
    if isinstance(values, _Ops):
        if isinstance(values.data, Buffer):
            return values.data, values.shape
        # Array with nested-list data – fall through to the generic walk
        values = values.data
    if isinstance(values, Buffer):
        if len(values) == 0:
            return values, (0,)
        return values, (len(values),)
    if not isinstance(values, (list, tuple, dict)):
        if hasattr(values, "__iter__"):
            values = list(values)
    buffer = _to_buffer(values)
    shape = _get_shape(values)
    return buffer, shape


def _strides(shape: Shape) -> Tuple[int, ...]:
    """Compute C-contiguous strides for a resolved shape.

    Standalone version of ``ArrayUtilsOps._strides``.
    """
    strides: List[int] = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)
