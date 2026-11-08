"""Array-native statistical reductions over flat Buffers."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from debug import get_logger
from debug.errors import ValidationError

from ._common import (
    array_from,
    coords_from_flat,
    flat_from_coords,
    flat_values,
    is_array_like,
    is_buffer,
    size_of,
)

logger = get_logger(__name__)

__all__ = [
    "argmax",
    "argmax_axis",
    "argmin",
    "argmin_axis",
    "count_nonzero",
    "count_nonzero_axis",
    "max",
    "max_axis",
    "mean",
    "mean_axis",
    "min",
    "min_axis",
    "norm",
    "norm_axis",
    "prod",
    "prod_axis",
    "std",
    "sum",
    "sum_axis",
    "variance",
    "variance_axis",
]


def _iter_values(values: Any) -> Iterable[Any]:
    """Yield scalars from an Array, Buffer, nested sequence or iterable."""
    if is_array_like(values) or is_buffer(values):
        flat, _ = flat_values(values)
        return iter(flat)
    if isinstance(values, (list, tuple)):
        flat, _ = flat_values(values)
        return iter(flat)
    if isinstance(values, (str, bytes, bytearray)):
        return iter(values)
    try:
        return iter(values)
    except TypeError:
        return iter([values])


def _sum_values(flat: Iterable[Any]) -> Any:
    total = 0
    for value in flat:
        total += value
    return total


def sum(flat: Iterable[Any]) -> Any:
    """Return the sum of all elements."""
    return _sum_values(_iter_values(flat))


def mean(flat: Iterable[Any]) -> float:
    """Return the mean of all elements.

    Raises:
        ValidationError: If *flat* is empty.
    """
    values = list(_iter_values(flat))
    if not values:
        logger.error("mean called on empty collection")
        raise ValidationError("mean of empty array")
    return _sum_values(values) / len(values)


def variance(flat: Iterable[Any]) -> float:
    """Return the population variance of all elements.

    Raises:
        ValidationError: If *flat* is empty.
    """
    values = list(_iter_values(flat))
    if not values:
        logger.error("variance called on empty collection")
        raise ValidationError("variance of empty array")
    average = mean(values)
    return _sum_values((value - average) ** 2 for value in values) / len(values)


def std(flat: Iterable[Any]) -> float:
    """Return the standard deviation of all elements.

    Raises:
        ValidationError: If *flat* is empty.
    """
    values = list(_iter_values(flat))
    if not values:
        logger.error("std called on empty collection")
        raise ValidationError("std of empty array")
    return math.sqrt(variance(values))


def max(flat: Iterable[Any]) -> Any:
    """Return the maximum value.

    Raises:
        ValidationError: If *flat* is empty.
    """
    values = list(_iter_values(flat))
    if not values:
        logger.error("max called on empty collection")
        raise ValidationError("max of empty array")
    result = values[0]
    for value in values[1:]:
        if value > result:
            result = value
    return result


def min(flat: Iterable[Any]) -> Any:
    """Return the minimum value.

    Raises:
        ValidationError: If *flat* is empty.
    """
    values = list(_iter_values(flat))
    if not values:
        logger.error("min called on empty collection")
        raise ValidationError("min of empty array")
    result = values[0]
    for value in values[1:]:
        if value < result:
            result = value
    return result


def prod(flat: Iterable[Any]) -> Any:
    """Return the product of all values."""
    result = 1
    for value in _iter_values(flat):
        result *= value
    return result


def argmax(flat: Iterable[Any]) -> int:
    """Return the index of the maximum value (first occurrence on ties)."""
    values = list(_iter_values(flat))
    if not values:
        raise ValidationError("argmax of empty array")
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def argmin(flat: Iterable[Any]) -> int:
    """Return the index of the minimum value (first occurrence on ties)."""
    values = list(_iter_values(flat))
    if not values:
        raise ValidationError("argmin of empty array")
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value < best_value:
            best_index = index
            best_value = value
    return best_index


def count_nonzero(flat: Iterable[Any]) -> int:
    """Return the number of non-zero entries."""
    return sum(1 for value in _iter_values(flat) if value != 0)


def norm(flat: Iterable[Any]) -> float:
    """Return the L2 norm of all elements."""
    return math.sqrt(sum(value * value for value in _iter_values(flat)))


def _reduce_axis(
    array: Any,
    axis: int,
    reduce_fn: Callable[[Iterable[Any]], Any],
    keepdims: bool = False,
) -> Any:
    """Apply *reduce_fn* along a single axis of an Array-like object."""
    values, shape = flat_values(array)
    ndim = len(shape)
    if ndim == 0:
        return reduce_fn(values)

    axis = axis % ndim
    if keepdims:
        out_shape = tuple(1 if i == axis else dim for i, dim in enumerate(shape))
    else:
        out_shape = tuple(dim for i, dim in enumerate(shape) if i != axis)

    if not out_shape:
        return reduce_fn(values)

    out_values: List[Any] = []
    for out_index in range(size_of(out_shape)):
        out_coords = coords_from_flat(out_index, out_shape)
        collected: List[Any] = []
        for position in range(shape[axis]):
            if keepdims:
                in_coords = list(out_coords)
                in_coords[axis] = position
            else:
                in_coords = list(out_coords)
                in_coords.insert(axis, position)
            collected.append(values[flat_from_coords(in_coords, shape)])
        out_values.append(reduce_fn(collected))

    return array_from(out_values, out_shape)


def sum_axis(
    array: Any,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Any:
    """Sum along a given axis or axes."""
    if axis is None:
        result = array
        for position in reversed(range(len(flat_values(array)[1]))):
            result = _reduce_axis(result, position, _sum_values, keepdims=keepdims)
        return result

    if isinstance(axis, tuple):
        result = array
        for position in sorted(axis, reverse=True):
            result = _reduce_axis(result, position, _sum_values, keepdims=keepdims)
        return result

    return _reduce_axis(array, axis, _sum_values, keepdims=keepdims)


def mean_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Mean along a given axis."""
    return _reduce_axis(array, axis, mean, keepdims=keepdims)


def variance_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Population variance along a given axis."""
    return _reduce_axis(array, axis, variance, keepdims=keepdims)


def max_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Maximum value along a given axis."""
    return _reduce_axis(array, axis, max, keepdims=keepdims)


def min_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Minimum value along a given axis."""
    return _reduce_axis(array, axis, min, keepdims=keepdims)


def prod_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Product along a given axis."""
    return _reduce_axis(array, axis, prod, keepdims=keepdims)


def argmax_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Argmax along a given axis."""
    return _reduce_axis(array, axis, argmax, keepdims=keepdims)


def argmin_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Argmin along a given axis."""
    return _reduce_axis(array, axis, argmin, keepdims=keepdims)


def count_nonzero_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """Count of non-zero values along a given axis."""
    return _reduce_axis(array, axis, count_nonzero, keepdims=keepdims)


def norm_axis(array: Any, axis: int, keepdims: bool = False) -> Any:
    """L2 norm along a given axis."""
    return _reduce_axis(array, axis, norm, keepdims=keepdims)
