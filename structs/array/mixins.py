"""Composite mixin collection for Array organization by concern."""

from __future__ import annotations

from .constructors import ArrayConstructors
from .ops import ArrayMathOps
from .shape import ArrayShapeOps
from .stats import ArrayStatsOps
from .utils import ArrayUtilsOps


class ArrayMixins(
    ArrayConstructors,
    ArrayUtilsOps,
    ArrayShapeOps,
    ArrayMathOps,
    ArrayStatsOps,
):
    """Composite mixin for the Array API split by concern."""

    pass


__all__ = ["ArrayMixins"]
