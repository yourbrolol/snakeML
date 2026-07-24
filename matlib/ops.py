import math

from debug import get_logger
from debug.errors import ValidationError

logger = get_logger(__name__)


def _apply_unary(data, fn):
    """Recursively apply a scalar function to every element of a nested list."""
    if isinstance(data, list):
        return [_apply_unary(item, fn) for item in data]
    return fn(data)


def exp(array):
    """Element-wise natural exponential e^x.

    Parameters
    ----------
    array : Array
        Input array.

    Returns
    -------
    list
        Nested list with e^x applied to every element.
    """
    data = array.data if hasattr(array, "data") else array
    logger.debug("exp requested", shape=getattr(array, "shape", None))
    return _apply_unary(data, math.exp)


def log(array):
    """Element-wise natural logarithm ln(x).

    Parameters
    ----------
    array : Array
        Input array. All elements must be positive.

    Returns
    -------
    list
        Nested list with ln(x) applied to every element.
    """
    data = array.data if hasattr(array, "data") else array
    logger.debug("log requested", shape=getattr(array, "shape", None))

    def _safe_log(x):
        if x <= 0:
            logger.error("log of non-positive value", value=x)
            raise ValidationError(f"log requires positive values, got {x}")
        return math.log(x)

    return _apply_unary(data, _safe_log)


def sqrt(array):
    """Element-wise square root.

    Parameters
    ----------
    array : Array
        Input array. All elements must be non-negative.

    Returns
    -------
    list
        Nested list with sqrt(x) applied to every element.
    """
    data = array.data if hasattr(array, "data") else array
    logger.debug("sqrt requested", shape=getattr(array, "shape", None))

    def _safe_sqrt(x):
        if x < 0:
            logger.error("sqrt of negative value", value=x)
            raise ValidationError(f"sqrt requires non-negative values, got {x}")
        return math.sqrt(x)

    return _apply_unary(data, _safe_sqrt)


def abs(array):
    """Element-wise absolute value |x|.

    Parameters
    ----------
    array : Array
        Input array.

    Returns
    -------
    list
        Nested list with |x| applied to every element.
    """
    data = array.data if hasattr(array, "data") else array
    logger.debug("abs requested", shape=getattr(array, "shape", None))
    return _apply_unary(data, lambda x: x if x >= 0 else -x)
