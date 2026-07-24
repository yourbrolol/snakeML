import math

from debug import get_logger
from debug.errors import ValidationError
from structs.utils import zeroes, indices, set_nested

logger = get_logger(__name__)


def _sum_values(flat):
    """Accumulate and return the sum of flat elements."""
    total = 0
    for value in flat:
        total += value
    return total


# ---------------------------------------------------------------------------
# Global reductions (flatten first – backwards-compatible)
# ---------------------------------------------------------------------------

def sum(flat):
    """Returns the sum of all elements in a flat list."""
    return _sum_values(flat)


def mean(flat):
    """Returns the mean of all elements in a flat list."""
    flat = list(flat)
    if not flat:
        logger.error("mean called on empty collection")
        raise ValidationError("mean of empty array")
    return _sum_values(flat) / len(flat)


def variance(flat):
    """Returns the population variance of all elements in a flat list."""
    flat = list(flat)
    if not flat:
        logger.error("variance called on empty collection")
        raise ValidationError("variance of empty array")
    m = mean(flat)
    return _sum_values((x - m) ** 2 for x in flat) / len(flat)


def std(flat):
    """Returns the standard deviation of all elements in a flat list."""
    flat = list(flat)
    if not flat:
        logger.error("std called on empty collection")
        raise ValidationError("std of empty array")
    return math.sqrt(variance(flat))


def max(flat):
    """Returns the maximum value from a flat list."""
    flat = list(flat)
    if not flat:
        logger.error("max called on empty collection")
        raise ValidationError("max of empty array")
    result = flat[0]
    for v in flat[1:]:
        if v > result:
            result = v
    return result


# ---------------------------------------------------------------------------
# Axis-aware reductions
# ---------------------------------------------------------------------------

def _reduce_axis(array, axis, reduce_fn, keepdims=False):
    """Apply reduce_fn along a single axis of an Array-like object.

    Parameters
    ----------
    array : Array
        Input array with .shape, .ndim, and __getitem__ supporting tuple keys.
    axis : int
        The axis to reduce along.
    reduce_fn : callable
        A function that accepts a flat list and returns a scalar.
    keepdims : bool
        If True, the reduced axis is kept as a dimension of size 1.

    Returns
    -------
    list
        A nested list (same nesting depth as input if keepdims, else ndim-1).
    """
    ndim = array.ndim
    shape = array.shape
    axis = axis % ndim

    # Output shape: remove the contracted axis (or size-1 if keepdims)
    if keepdims:
        out_shape = tuple(1 if i == axis else s for i, s in enumerate(shape))
    else:
        out_shape = tuple(s for i, s in enumerate(shape) if i != axis)

    if not out_shape:
        # Reducing a 1-D array to a scalar
        flat = [array[(i,)] for i in range(shape[0])]
        return reduce_fn(flat)

    result = zeroes(out_shape)

    for out_idx in indices(out_shape):
        # Map output index back to input indices along the contracted axis
        if keepdims:
            # out_idx has a 1 at axis; we need to iterate the real axis
            in_base = list(out_idx)
            values = []
            for k in range(shape[axis]):
                in_idx = list(out_idx)
                in_idx[axis] = k
                values.append(array[tuple(in_idx)])
        else:
            # Insert a slice over the contracted axis to collect values
            in_base = list(out_idx)
            in_base.insert(axis, 0)   # placeholder
            values = []
            for k in range(shape[axis]):
                in_base[axis] = k
                values.append(array[tuple(in_base)])

        set_nested(result, out_idx, reduce_fn(values))

    return result


def sum_axis(array, axis, keepdims=False):
    """Sum along a given axis."""
    return _reduce_axis(array, axis, _sum_values, keepdims=keepdims)


def mean_axis(array, axis, keepdims=False):
    """Mean along a given axis."""
    return _reduce_axis(array, axis, mean, keepdims=keepdims)


def variance_axis(array, axis, keepdims=False):
    """Population variance along a given axis."""
    return _reduce_axis(array, axis, variance, keepdims=keepdims)


def max_axis(array, axis, keepdims=False):
    """Maximum value along a given axis."""
    return _reduce_axis(array, axis, max, keepdims=keepdims)
