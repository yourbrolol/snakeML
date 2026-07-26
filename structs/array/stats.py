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
            return matlib_stats.sum(list(self.flatten()))
        return self._wrap_result(matlib_stats.sum_axis(self, axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        """Mean of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.mean(list(self.flatten()))
        return self._wrap_result(matlib_stats.mean_axis(self, axis, keepdims=keepdims))

    def var(self, axis=None, keepdims=False):
        """Variance alias for compatibility with common array APIs."""
        return self.variance(axis=axis, keepdims=keepdims)

    def variance(self, axis=None, keepdims=False):
        """Population variance of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.variance(list(self.flatten()))
        return self._wrap_result(matlib_stats.variance_axis(self, axis, keepdims=keepdims))

    def std(self, axis=None, keepdims=False):
        """Standard deviation of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.std(list(self.flatten()))
        return self._wrap_result(
            matlib_ops._apply_unary(
                matlib_stats.variance_axis(self, axis, keepdims=keepdims),
                _math.sqrt,
            )
        )

    def max(self, axis=None, keepdims=False):
        """Maximum value of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.max(list(self.flatten()))
        return self._wrap_result(matlib_stats.max_axis(self, axis, keepdims=keepdims))

    def min(self, axis=None, keepdims=False):
        """Minimum value of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.min(list(self.flatten()))
        return self._wrap_result(matlib_stats.min_axis(self, axis, keepdims=keepdims))

    def prod(self, axis=None, keepdims=False):
        """Product of all elements, or along a given axis."""
        if axis is None:
            return matlib_stats.prod(list(self.flatten()))
        return self._wrap_result(matlib_stats.prod_axis(self, axis, keepdims=keepdims))

    def argmax(self, axis=None, keepdims=False):
        """Index of the maximum value."""
        if axis is None:
            return matlib_stats.argmax(list(self.flatten()))
        return self._wrap_result(matlib_stats.argmax_axis(self, axis, keepdims=keepdims))

    def argmin(self, axis=None, keepdims=False):
        """Index of the minimum value."""
        if axis is None:
            return matlib_stats.argmin(list(self.flatten()))
        return self._wrap_result(matlib_stats.argmin_axis(self, axis, keepdims=keepdims))

    def count_nonzero(self, axis=None, keepdims=False):
        """Count of non-zero values."""
        if axis is None:
            return matlib_stats.count_nonzero(list(self.flatten()))
        return self._wrap_result(matlib_stats.count_nonzero_axis(self, axis, keepdims=keepdims))

    def norm(self, axis=None, keepdims=False):
        """L2 norm of the array."""
        if axis is None:
            return matlib_stats.norm(list(self.flatten()))
        return self._wrap_result(matlib_stats.norm_axis(self, axis, keepdims=keepdims))


__all__ = ["ArrayStatsOps"]
