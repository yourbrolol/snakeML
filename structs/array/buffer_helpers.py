"""Buffer and shape helpers for the Array bufferization roadmap.

These are standalone functions (no ``self``) that contain the core logic
formerly inside ``ArrayUtilsOps`` methods.  The mixin methods now delegate
to these functions so that the public API is unchanged.
"""

from __future__ import annotations

import operator as _op
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union, cast

from structs.dtypes import DTypeName, IndexKey, NestedList, Scalar, Shape, Self
from .buffer import Buffer


def _to_buffer(values: Any) -> Buffer:
    """Convert an Array-like input into a flat ``Buffer``.

    Standalone version of ``ArrayUtilsOps._to_buffer``.
    """
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"cannot create Array from {type(values).__name__}")
    if isinstance(values, Self):
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
    if isinstance(values, Scalar):
        return Buffer([values])
    raise TypeError(f"cannot create Array from {type(values).__name__}")


def _get_shape(data: Any) -> Shape:
    """Recursively calculate the dimensions (shape) of nested list data.

    Standalone version of ``ArrayUtilsOps._get_shape``.
    """
    if not isinstance(data, list):
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
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"cannot create Array from {type(values).__name__}")
    if isinstance(values, Self):
        if isinstance(values.data, Buffer):
            return values.data, values.shape
        # Array with nested-list data – fall through to the generic walk
        values = values.data
    if isinstance(values, Buffer):
        if len(values) == 0:
            return values, (0,)
        return values, (len(values),)
    buffer = _to_buffer(values)
    shape = _get_shape(values) if not isinstance(values, Buffer) else ((0,) if len(values) == 0 else (len(values),))
    if isinstance(values, Self) and isinstance(values.data, Buffer):
        return values.data, values.shape
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


def _flatten(array: Any = None) -> Iterator[Any]:
    """Yield every element of nested list data in row-major order.

    Standalone version of ``ArrayUtilsOps._flatten``.
    """
    from .utils import ArrayUtilsOps as _Ops
    data = _Ops().data if array is None else array
    if not isinstance(data, list):
        yield data
    else:
        for item in data:
            yield from _flatten(item)


def _unwrap(value: Any) -> Any:
    """Strip Array wrappers recursively, exposing the raw nested lists.

    Standalone version of ``ArrayUtilsOps._unwrap``.
    """
    from .utils import ArrayUtilsOps as _Ops
    if isinstance(value, _Ops):
        return value.data
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def _wrap_result(result: Any) -> Any:
    """Wrap a raw nested-list result in an Array, passing scalars through.

    Standalone version of ``ArrayUtilsOps._wrap_result``.
    """
    from .utils import ArrayUtilsOps as _Ops
    if isinstance(result, list):
        return _Ops(result)
    return result


def _wrap(value: Any) -> Self:
    """Wrap a value in an Array unless it already is one.

    Standalone version of ``ArrayUtilsOps._wrap``.
    """
    from .utils import ArrayUtilsOps as _Ops
    if isinstance(value, _Ops):
        return value
    return _Ops(value)


def indices(shape: Optional[Shape] = None) -> Iterator[Tuple[int, ...]]:
    """Yield every multi-dimensional index of a shape.

    Moved to top-level for easier testing; delegates to structs.utils.indices.
    """
    from structs import utils as _struct_utils
    return _struct_utils.indices(shape)


def flatten(self) -> Self:
    """Return a new Array with copied flat data.

    .. deprecated: use ``ArrayUtilsOps.copy`` instead.
    """
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.copy(self)


def item(self, idx: Optional[int] = None) -> Any:
    """Return item at *idx*, or the first item if *idx* is ``None``."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.item(self, idx)


def dtype(self) -> DTypeName:
    """Return the data type of the array's elements."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.dtype(self)


def tolist(self) -> NestedList:
    """Return a full nested-list copy of the array."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.tolist(self)


def size(self) -> int:
    """Return the total number of elements."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.size(self)


def wraparray(value: Any) -> Any:
    """Wrap *value* in an Array, any heterogeneous type allowed."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.wraparray(value)


def concatenate(arrays: Sequence[Any], axis: int = 0) -> Any:
    """Concatenate *arrays* along *axis*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.concatenate(arrays, axis)


def stack(arrays: Sequence[Any], axis: int = 0) -> Any:
    """Stack *arrays* along *axis*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.stack(arrays, axis)


def split(arrays: Sequence[Any], axis: int = 0) -> List[Any]:
    """Split *arrays* along *axis*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.split(arrays, axis)


def unbind(self, axis: int = 0) -> List[Any]:
    """Unbind *self* along *axis*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.unbind(self, axis)


def vstack(self: Self, other: Any) -> Self:
    """Stack *other* vertically below *self*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.vstack(self, other)


def hstack(self: Self, other: Any) -> Self:
    """Stack *other* horizontally to the right of *self*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.hstack(self, other)


def array_split(arrays: Sequence[Any], axis: int = 0) -> List[Any]:
    """Return a list of sub-arrays."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.array_split(arrays, axis)


def chunk(self: Self, chunks: int, axis: int = 0) -> List[Self]:
    """Return *chunks* sub-arrays along *axis*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.chunk(self, chunks, axis)


def astype(self: Self, dtype: DTypeName) -> Self:
    """Cast the array to a different ``DTypeName``."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.astype(self, dtype)


def fill(self: Self, value: Any) -> Self:
    """Fill the array *value* in-place."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.fill(self, value)


def where(self: Self, condition: Any, x: Any, y: Any) -> Self:
    """Return an array chosen from *x* and *y* based on *condition*."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.where(self, condition, x, y)


def unique(self: Self) -> Self:
    """Return unique elements of the array."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.unique(self)


def sort(self: Self, axis: int = -1) -> Self:
    """Return a sorted copy of the array."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.sort(self, axis)


def argsort(self: Self, axis: int = -1) -> Self:
    """Return indices that would sort the array."""
    from .utils import ArrayUtilsOps as _Ops
    return _Ops.argsort(self, axis)