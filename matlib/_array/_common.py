"""Shared flat-buffer helpers for the Array-native kernels."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

Shape = Tuple[int, ...]


def is_array_like(value: Any) -> bool:
    """Return whether *value* exposes the Array ``shape``/``data`` contract."""
    return (
        not isinstance(value, (list, tuple))
        and hasattr(value, "shape")
        and hasattr(value, "data")
    )


def is_buffer(value: Any) -> bool:
    """Return whether *value* is a flat Buffer-like object."""
    return (
        not isinstance(value, (list, tuple))
        and hasattr(value, "buffer")
        and not hasattr(value, "shape")
    )


def nested_shape(value: Any) -> Shape:
    """Infer the shape of a nested list/tuple."""
    if not isinstance(value, (list, tuple)):
        return ()
    if len(value) == 0:
        return (0,)
    return (len(value),) + nested_shape(value[0])


def flatten_nested(value: Any) -> List[Any]:
    """Flatten a nested list/tuple in row-major order."""
    if not isinstance(value, (list, tuple)):
        return [value]
    values: List[Any] = []
    for item in value:
        values.extend(flatten_nested(item))
    return values


def shape_of(value: Any) -> Shape:
    """Return the logical shape of an Array, Buffer, nested sequence or scalar."""
    if is_array_like(value):
        return tuple(value.shape)
    if is_buffer(value):
        return (len(value),)
    if isinstance(value, (list, tuple)):
        return nested_shape(value)
    return ()


def flat_values(value: Any) -> Tuple[List[Any], Shape]:
    """Return ``(flat logical values, shape)`` for any supported input."""
    if is_array_like(value):
        if hasattr(value, "_flatten"):
            return list(value._flatten()), tuple(value.shape)
        data = value.data
        if is_buffer(data):
            return list(data.buffer), tuple(value.shape)
        return flatten_nested(data), tuple(value.shape)
    if is_buffer(value):
        return list(value.buffer), (len(value),)
    if isinstance(value, (list, tuple)):
        return flatten_nested(value), nested_shape(value)
    return [value], ()


def array_from(values: Any, shape: Shape) -> Any:
    """Build an ``Array`` around *values* with the requested logical shape."""
    from structs.array.array import Array
    from structs.array.buffer import Buffer

    return Array(Buffer(list(values)), raw=True, shape=tuple(shape))


def array_view(
    value: Any,
    shape: Shape,
    strides: Sequence[int],
    offset: Optional[int] = None,
) -> Any:
    """Return an Array view sharing *value*'s Buffer, or ``None``.

    Plain nested lists, Buffers, and scalars return ``None`` so callers can
    fall back to materialising a new contiguous array.
    """
    from structs.array.array import Array

    if not isinstance(value, Array):
        return None
    return value.__class__(
        value.data,
        raw=True,
        shape=tuple(shape),
        strides=tuple(strides),
        offset=value.offset if offset is None else offset,
    )


def permuted_view(value: Any, axes: Sequence[int]) -> Any:
    """Return a shape/stride-permuted view of an Array, or ``None``.

    The new array shares the original ``Buffer``; no element data is copied.
    Plain nested lists, Buffers, and scalars return ``None`` so callers can
    fall back to materialising a contiguous result.
    """
    from structs.array.array import Array

    if not isinstance(value, Array):
        return None
    old_shape = tuple(value.shape)
    old_strides = tuple(value.strides)
    return array_view(
        value,
        tuple(old_shape[axis] for axis in axes),
        tuple(old_strides[axis] for axis in axes),
    )


def sliced_view(value: Any, axis: int, start: int, stop: int) -> Any:
    """Return an Array slice view along *axis*, or ``None``."""
    from structs.array.array import Array

    if not isinstance(value, Array):
        return None
    axis %= len(value.shape)
    shape = list(value.shape)
    shape[axis] = max(0, stop - start)
    offset = value.offset + start * value.strides[axis]
    return array_view(value, tuple(shape), value.strides, offset)


def dropped_axis_view(value: Any, axis: int, position: int) -> Any:
    """Return an Array view with *axis* fixed at *position* and removed."""
    from structs.array.array import Array

    if not isinstance(value, Array):
        return None
    axis %= len(value.shape)
    shape = tuple(dim for index, dim in enumerate(value.shape) if index != axis)
    strides = tuple(stride for index, stride in enumerate(value.strides) if index != axis)
    offset = value.offset + position * value.strides[axis]
    return array_view(value, shape, strides, offset)


def as_array(value: Any) -> Any:
    """Return *value* as an ``Array``, converting Buffer/scalar/list inputs."""
    from structs.array.array import Array
    from structs.array.buffer import Buffer

    if isinstance(value, Array):
        return value
    if isinstance(value, Buffer):
        return Array(value, raw=True, shape=(len(value),))
    if is_buffer(value):
        return array_from(value.buffer, (len(value),))
    if isinstance(value, (list, tuple)):
        return Array(value)
    return array_from([value], ())


def size_of(shape: Shape) -> int:
    """Return the number of elements described by *shape*."""
    total = 1
    for dim in shape:
        total *= dim
    return total


def coords_from_flat(index: int, shape: Shape) -> List[int]:
    """Convert a C-order flat index into multi-dimensional coordinates."""
    coords = [0] * len(shape)
    remaining = index
    for dim in range(len(shape) - 1, -1, -1):
        coords[dim] = remaining % shape[dim]
        remaining //= shape[dim]
    return coords


def flat_from_coords(coords: List[int], shape: Shape) -> int:
    """Convert multi-dimensional coordinates into a C-order flat index."""
    index = 0
    for coord, dim in zip(coords, shape):
        index = index * dim + coord
    return index


def broadcast_shapes(a: Shape, b: Shape) -> Shape:
    """Return the broadcast shape of *a* and *b* using NumPy alignment."""
    ndim = max(len(a), len(b))
    result: List[int] = []
    for offset in range(1, ndim + 1):
        dim_a = a[-offset] if offset <= len(a) else 1
        dim_b = b[-offset] if offset <= len(b) else 1
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            result.append(max(dim_a, dim_b))
        else:
            raise ValueError("Shapes cannot be broadcast")
    return tuple(reversed(result))


def source_flat_index(coords: List[int], out_shape: Shape, src_shape: Shape) -> int:
    """Map broadcast-result coordinates back to a source flat index."""
    if not src_shape:
        return 0
    offset = len(out_shape) - len(src_shape)
    src_coords = []
    for axis, dim in enumerate(src_shape):
        coord = coords[offset + axis]
        src_coords.append(0 if dim == 1 else coord)
    return flat_from_coords(src_coords, src_shape)
