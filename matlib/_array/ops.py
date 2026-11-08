"""Array-native element-wise unary kernels.

Every kernel reads the flat ``Buffer`` behind an Array, applies a scalar
function to each logical element and returns a new ``Array`` with the same
shape. Scalars are passed through unchanged.
"""

from __future__ import annotations

import builtins as _builtins
import math
from typing import Any, Callable, Optional

from debug import get_logger
from debug.errors import ValidationError

from ._common import array_from, flat_values, is_array_like, is_buffer

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


def _apply_unary(data: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply *fn* to every logical element of *data*.

    Arrays, Buffers and nested sequences produce a new ``Array``; a bare
    scalar returns ``fn(scalar)``.
    """
    values, shape = flat_values(data)
    if shape == () and not is_array_like(data) and not is_buffer(data):
        return fn(values[0])
    return array_from([fn(value) for value in values], shape)


def log2(array: Any) -> Any:
    """Element-wise logarithm base 2."""
    return _apply_unary(array, math.log2)


def log10(array: Any) -> Any:
    """Element-wise logarithm base 10."""
    return _apply_unary(array, math.log10)


def square(array: Any) -> Any:
    """Element-wise square ``x * x``."""
    return _apply_unary(array, lambda x: x * x)


def sin(array: Any) -> Any:
    """Element-wise sine (radians)."""
    return _apply_unary(array, math.sin)


def cos(array: Any) -> Any:
    """Element-wise cosine (radians)."""
    return _apply_unary(array, math.cos)


def tan(array: Any) -> Any:
    """Element-wise tangent (radians)."""
    return _apply_unary(array, math.tan)


def asin(array: Any) -> Any:
    """Element-wise arcsine."""
    return _apply_unary(array, math.asin)


def acos(array: Any) -> Any:
    """Element-wise arccosine."""
    return _apply_unary(array, math.acos)


def atan(array: Any) -> Any:
    """Element-wise arctangent."""
    return _apply_unary(array, math.atan)


def sinh(array: Any) -> Any:
    """Element-wise hyperbolic sine."""
    return _apply_unary(array, math.sinh)


def cosh(array: Any) -> Any:
    """Element-wise hyperbolic cosine."""
    return _apply_unary(array, math.cosh)


def tanh(array: Any) -> Any:
    """Element-wise hyperbolic tangent."""
    return _apply_unary(array, math.tanh)


def floor(array: Any) -> Any:
    """Element-wise floor."""
    return _apply_unary(array, math.floor)


def ceil(array: Any) -> Any:
    """Element-wise ceiling."""
    return _apply_unary(array, math.ceil)


def round(array: Any, ndigits: Optional[int] = None) -> Any:
    """Element-wise rounding to ``ndigits`` decimal places (half to even)."""
    return _apply_unary(
        array,
        lambda x: _builtins.round(x, ndigits) if ndigits is not None else _builtins.round(x),
    )


def clip(
    array: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Any:
    """Element-wise clamping of values into ``[min_value, max_value]``."""

    def _clip_value(x: Any) -> Any:
        if min_value is not None:
            x = max(x, min_value)
        if max_value is not None:
            x = min(x, max_value)
        return x

    return _apply_unary(array, _clip_value)


def sign(array: Any) -> Any:
    """Element-wise sign: ``1``/``-1``/``0`` for positive/negative/zero inputs."""
    return _apply_unary(
        array, lambda x: 1 if x > 0 else -1 if x < 0 else 0
    )


def exp(array: Any) -> Any:
    """Element-wise natural exponential e^x."""
    logger.debug("exp requested", shape=getattr(array, "shape", None))
    return _apply_unary(array, math.exp)


def log(array: Any) -> Any:
    """Element-wise natural logarithm ln(x).

    Raises:
        ValidationError: If any element is zero or negative.
    """
    logger.debug("log requested", shape=getattr(array, "shape", None))

    def _safe_log(x: Any) -> Any:
        if x <= 0:
            logger.error("log of non-positive value", value=x)
            raise ValidationError(f"log requires positive values, got {x}")
        return math.log(x)

    return _apply_unary(array, _safe_log)


def sqrt(array: Any) -> Any:
    """Element-wise square root.

    Raises:
        ValidationError: If any element is negative.
    """
    logger.debug("sqrt requested", shape=getattr(array, "shape", None))

    def _safe_sqrt(x: Any) -> Any:
        if x < 0:
            logger.error("sqrt of negative value", value=x)
            raise ValidationError(f"sqrt requires non-negative values, got {x}")
        return math.sqrt(x)

    return _apply_unary(array, _safe_sqrt)


def abs(array: Any) -> Any:
    """Element-wise absolute value |x|."""
    logger.debug("abs requested", shape=getattr(array, "shape", None))
    return _apply_unary(array, lambda x: x if x >= 0 else -x)
