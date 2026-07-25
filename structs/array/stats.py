"""Statistical helpers for Array."""

from __future__ import annotations

import math as _math

from debug import get_logger
from matlib import ops as matlib_ops, stats as matlib_stats

logger = get_logger(__name__)


class ArrayStatsOps:
    """Mixin providing statistics-related array methods."""

    def sum(self, axis=None, keepdims=False):
        """Sum of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.sum(list(self._flatten()))
        return self._wrap_result(matlib_stats.sum_axis(self, axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        """Mean of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.mean(list(self._flatten()))
        return self._wrap_result(matlib_stats.mean_axis(self, axis, keepdims=keepdims))

    def variance(self, axis=None, keepdims=False):
        """Population variance of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.variance(list(self._flatten()))
        return self._wrap_result(matlib_stats.variance_axis(self, axis, keepdims=keepdims))

    def std(self, axis=None, keepdims=False):
        """Standard deviation of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.std(list(self._flatten()))
        return self._wrap_result(
            matlib_ops._apply_unary(
                matlib_stats.variance_axis(self, axis, keepdims=keepdims),
                _math.sqrt,
            )
        )

    def max(self, axis=None, keepdims=False):
        """Maximum value of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.max(list(self._flatten()))
        return self._wrap_result(matlib_stats.max_axis(self, axis, keepdims=keepdims))


__all__ = ["ArrayStatsOps"]
