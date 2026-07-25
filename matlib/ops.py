import math

from debug import get_logger
from debug.errors import ValidationError

logger = get_logger(__name__)


def _apply_unary(data, fn):
    """Recursively apply a scalar function to every element of a nested list."""
    if isinstance(data, list):
        return [_apply_unary(item, fn) for item in data]
    return fn(data)


def log2(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, lambda x: math.log2(x))


def log10(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, lambda x: math.log10(x))


def square(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, lambda x: x * x)


def sin(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.sin)


def cos(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.cos)


def tan(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.tan)


def asin(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.asin)


def acos(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.acos)


def atan(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.atan)


def sinh(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.sinh)


def cosh(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.cosh)


def tanh(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.tanh)


def floor(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.floor)


def ceil(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, math.ceil)


def round(array, ndigits=None):
    return _apply_unary(array.data if hasattr(array, "data") else array, lambda x: round(x, ndigits) if ndigits is not None else round(x))


def clip(array, min_value=None, max_value=None):
    def _clip_value(x):
        if min_value is not None:
            x = max(x, min_value)
        if max_value is not None:
            x = min(x, max_value)
        return x

    return _apply_unary(array.data if hasattr(array, "data") else array, _clip_value)


def sign(array):
    return _apply_unary(array.data if hasattr(array, "data") else array, lambda x: 1 if x > 0 else -1 if x < 0 else 0)


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
