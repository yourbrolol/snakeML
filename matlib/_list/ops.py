"""Element-wise unary math functions over array-like data.

Every function accepts an array (unwrapped via ``.data`` when present)
or a plain nested list and returns nested lists produced by recursively
applying a scalar :mod:`math` operation. Domain violations (``log``,
``sqrt``) raise :class:`debug.errors.ValidationError`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Union

from debug import get_logger
from debug.errors import ValidationError

logger = get_logger(__name__)

__all__ = [
    "acos",
    "asin",
    "atan",
    "abs",
    "ceil",
    "clip",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "log10",
    "log2",
    "round",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
]


def _apply_unary(data: Any, fn: Callable[[float], float]) -> Any:
    """Recursively apply a scalar function to every element of a nested list.

    Args:
        data: Nested list (or scalar) of numeric elements.
        fn: Unary callable applied to each scalar leaf.

    Returns:
        A nested list mirroring ``data``'s shape with ``fn`` applied.
    """
    if isinstance(data, list):
        return [_apply_unary(item, fn) for item in data]
    return fn(data)


def log2(array: Any) -> Any:
    """Element-wise logarithm base 2."""
    return _apply_unary(
        array.data if hasattr(array, "data") else array, lambda x: math.log2(x)
    )


def log10(array: Any) -> Any:
    """Element-wise logarithm base 10."""
    return _apply_unary(
        array.data if hasattr(array, "data") else array, lambda x: math.log10(x)
    )


def square(array: Any) -> Any:
    """Element-wise square ``x * x``."""
    return _apply_unary(
        array.data if hasattr(array, "data") else array, lambda x: x * x
    )


def sin(array: Any) -> Any:
    """Element-wise sine (radians)."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.sin)


def cos(array: Any) -> Any:
    """Element-wise cosine (radians)."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.cos)


def tan(array: Any) -> Any:
    """Element-wise tangent (radians)."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.tan)


def asin(array: Any) -> Any:
    """Element-wise arcsine."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.asin)


def acos(array: Any) -> Any:
    """Element-wise arccosine."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.acos)


def atan(array: Any) -> Any:
    """Element-wise arctangent."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.atan)


def sinh(array: Any) -> Any:
    """Element-wise hyperbolic sine."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.sinh)


def cosh(array: Any) -> Any:
    """Element-wise hyperbolic cosine."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.cosh)


def tanh(array: Any) -> Any:
    """Element-wise hyperbolic tangent."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.tanh)


def floor(array: Any) -> Any:
    """Element-wise floor."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.floor)


def ceil(array: Any) -> Any:
    """Element-wise ceiling."""
    return _apply_unary(array.data if hasattr(array, "data") else array, math.ceil)


def round(array: Any, ndigits: Optional[int] = None) -> Any:
    """Element-wise rounding to ``ndigits`` decimal places (half to even)."""
    return _apply_unary(
        array.data if hasattr(array, "data") else array,
        lambda x: round(x, ndigits) if ndigits is not None else round(x),
    )


def clip(
    array: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Any:
    """Element-wise clamping of values into ``[min_value, max_value]``.

    Args:
        array: Input array or nested list.
        min_value: Inclusive lower bound; unbounded when ``None``.
        max_value: Inclusive upper bound; unbounded when ``None``.

    Returns:
        A nested list with each element clamped to the requested bounds.
    """

    def _clip_value(x: float) -> float:
        if min_value is not None:
            x = max(x, min_value)
        if max_value is not None:
            x = min(x, max_value)
        return x

    return _apply_unary(array.data if hasattr(array, "data") else array, _clip_value)


def sign(array: Any) -> Any:
    """Element-wise sign: ``1``/``-1``/``0`` for positive/negative/zero inputs."""
    return _apply_unary(
        array.data if hasattr(array, "data") else array,
        lambda x: 1 if x > 0 else -1 if x < 0 else 0,
    )


def exp(array: Any) -> Any:
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


def log(array: Any) -> Any:
    """Element-wise natural logarithm ln(x).

    Parameters
    ----------
    array : Array
        Input array. All elements must be positive.

    Returns
    -------
    list
        Nested list with ln(x) applied to every element.

    Raises:
        ValidationError: If any element is zero or negative.
    """
    data = array.data if hasattr(array, "data") else array
    logger.debug("log requested", shape=getattr(array, "shape", None))

    def _safe_log(x: float) -> float:
        if x <= 0:
            logger.error("log of non-positive value", value=x)
            raise ValidationError(f"log requires positive values, got {x}")
        return math.log(x)

    return _apply_unary(data, _safe_log)


def sqrt(array: Any) -> Any:
    """Element-wise square root.

    Parameters
    ----------
    array : Array
        Input array. All elements must be non-negative.

    Returns
    -------
    list
        Nested list with sqrt(x) applied to every element.

    Raises:
        ValidationError: If any element is negative.
    """
    data = array.data if hasattr(array, "data") else array
    logger.debug("sqrt requested", shape=getattr(array, "shape", None))

    def _safe_sqrt(x: float) -> float:
        if x < 0:
            logger.error("sqrt of negative value", value=x)
            raise ValidationError(f"sqrt requires non-negative values, got {x}")
        return math.sqrt(x)

    return _apply_unary(data, _safe_sqrt)


def abs(array: Any) -> Any:
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
