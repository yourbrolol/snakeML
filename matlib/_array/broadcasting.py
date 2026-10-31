"""Array-native broadcasting and element-wise binary kernels."""

from __future__ import annotations

from typing import Any, Callable, Tuple

from ._common import (
    array_from,
    as_array,
    broadcast_shapes,
    coords_from_flat,
    flat_values,
    is_array_like,
    is_buffer,
    size_of,
    source_flat_index,
)

__all__ = ["broadcast", "elementwise"]


def _is_1d(value: Any) -> bool:
    """Return whether *value* behaves like a one-dimensional sequence."""
    if is_array_like(value):
        return len(tuple(value.shape)) == 1
    if is_buffer(value):
        return True
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True
        return not isinstance(value[0], (list, tuple))
    return False


def broadcast(a: Any, b: Any) -> Tuple[Any, Any, str]:
    """Normalize operands for broadcasting and matrix-style alignment.

    1-D operands are reshaped towards their counterpart (``a`` becomes a row,
    ``b`` becomes a column vector); 0-d operands are promoted to ``(1, 1)``.
    The returned ``squeeze_type`` records how to project the result back.
    """
    a_is_1d = _is_1d(a)
    b_is_1d = _is_1d(b)

    a_array = as_array(a)
    b_array = as_array(b)

    if a_is_1d:
        a_array = a_array.unsqueeze(0)
    elif a_array.ndim == 0:
        a_array = a_array.reshape(1, 1)

    if b_is_1d:
        b_array = b_array.unsqueeze(1)
    elif b_array.ndim == 0:
        b_array = b_array.reshape(1, 1)

    if a_is_1d and b_is_1d:
        squeeze_type = "scalar"
    elif b_is_1d:
        squeeze_type = "column"
    elif a_is_1d:
        squeeze_type = "row"
    else:
        squeeze_type = "none"

    return a_array, b_array, squeeze_type


def elementwise(a: Any, b: Any, op: Callable[[Any, Any], Any]) -> Any:
    """Apply *op* element-wise over *a* and *b* with NumPy broadcasting.

    Raises:
        ValueError: If the operand shapes cannot be broadcast.
    """
    scalar_a = not (
        is_array_like(a) or is_buffer(a) or isinstance(a, (list, tuple))
    )
    scalar_b = not (
        is_array_like(b) or is_buffer(b) or isinstance(b, (list, tuple))
    )
    if scalar_a and scalar_b:
        return op(a, b)

    a_values, a_shape = flat_values(a)
    b_values, b_shape = flat_values(b)
    out_shape = broadcast_shapes(a_shape, b_shape)

    values = []
    for index in range(size_of(out_shape)):
        coords = coords_from_flat(index, out_shape)
        a_value = a_values[source_flat_index(coords, out_shape, a_shape)]
        b_value = b_values[source_flat_index(coords, out_shape, b_shape)]
        values.append(op(a_value, b_value))
    return array_from(values, out_shape)
