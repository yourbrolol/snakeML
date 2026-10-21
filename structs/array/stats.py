"""Statistical helpers for Array."""

from __future__ import annotations

import math as _math
from typing import Any, Optional, Union

from debug import get_logger
from matlib import ops as matlib_ops
from matlib import stats as matlib_stats
from structs.dtypes import Self

from .protocol import ArrayProtocol

logger = get_logger(__name__)

__all__ = ["ArrayStatsOps"]


class ArrayStatsOps(ArrayProtocol):
    """Mixin providing statistics-related array methods.

    Every method takes the same ``axis``/``keepdims`` pair:

    Args:
        axis: Axis to reduce over. ``None`` (default) flattens the array first
            and reduces over all elements.
        keepdims: When ``True`` the reduced axis is kept with length 1, so the
            result still broadcasts against the input shape.

    The return type therefore follows the axis: a plain ``int``/``float`` for a
    full reduction, or a wrapped ``Array`` with one fewer dimension when
    reducing over a single axis.
    """

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Sum of all elements, or along a given axis.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar for ``axis=None``, otherwise a new array with that
            axis reduced.

        Example:
            >>> Array([[1, 2], [3, 4]]).sum(axis=0)
            [4, 6]
        """
        if axis is None:
            return matlib_stats.sum(list(self.flatten()))
        return self._wrap_result(matlib_stats.sum_axis(self, axis, keepdims=keepdims))

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Arithmetic mean of all elements, or along a given axis.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar for ``axis=None``, otherwise a new array with that
            axis reduced. Empty reductions yield ``0`` rather than raising.

        Example:
            >>> Array([[1, 3], [3, 5]]).mean()
            3.0
        """
        if axis is None:
            return matlib_stats.mean(list(self.flatten()))
        return self._wrap_result(matlib_stats.mean_axis(self, axis, keepdims=keepdims))

    def var(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Variance, an alias of :meth:`variance`.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: The population variance, scalar or array as per :meth:`variance`.
        """
        return self.variance(axis=axis, keepdims=keepdims)

    def variance(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Population variance of all elements, or along a given axis.

        Uses the population definition (divides by ``N``, not ``N - 1``).

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar for ``axis=None``, otherwise a new array with that
            axis reduced. Always non-negative.

        Example:
            >>> Array([1, 3, 5]).variance()
            2.6666666666666665
        """
        if axis is None:
            return matlib_stats.variance(list(self.flatten()))
        return self._wrap_result(
            matlib_stats.variance_axis(self, axis, keepdims=keepdims)
        )

    def std(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Standard deviation of all elements, or along a given axis.

        Computed as the square root of the population :meth:`variance`.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar for ``axis=None``, otherwise a new array with that
            axis reduced. Always non-negative.

        Example:
            >>> Array([1, 3, 5]).std()
            1.632993161855452
        """
        if axis is None:
            return matlib_stats.std(list(self.flatten()))
        return self._wrap_result(
            matlib_ops._apply_unary(
                matlib_stats.variance_axis(self, axis, keepdims=keepdims),
                _math.sqrt,
            )
        )

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Largest value of all elements, or along a given axis.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar for ``axis=None``, otherwise a new array with that
            axis reduced.

        Raises:
            ValueError: If reducing an empty selection.
        """
        if axis is None:
            return matlib_stats.max(list(self.flatten()))
        return self._wrap_result(matlib_stats.max_axis(self, axis, keepdims=keepdims))

    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Smallest value of all elements, or along a given axis.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar for ``axis=None``, otherwise a new array with that
            axis reduced.

        Raises:
            ValueError: If reducing an empty selection.
        """
        if axis is None:
            return matlib_stats.min(list(self.flatten()))
        return self._wrap_result(matlib_stats.min_axis(self, axis, keepdims=keepdims))

    def prod(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Product of all elements, or along a given axis.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar for ``axis=None``, otherwise a new array with that
            axis reduced.

        Example:
            >>> Array([[1, 2], [3, 4]]).prod(axis=0)
            [3, 8]
        """
        if axis is None:
            return matlib_stats.prod(list(self.flatten()))
        return self._wrap_result(matlib_stats.prod_axis(self, axis, keepdims=keepdims))

    def argmax(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Position of the largest value.

        Args:
            axis: Axis to search along; ``None`` searches the flattened array.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: The first ``int`` index attaining the maximum for a full
            reduction, otherwise a new array of indices shaped like the
            reduced result.

        Raises:
            ValueError: If reducing an empty selection.

        Example:
            >>> Array([1, 9, 3]).argmax()
            1
        """
        if axis is None:
            return matlib_stats.argmax(list(self.flatten()))
        return self._wrap_result(
            matlib_stats.argmax_axis(self, axis, keepdims=keepdims)
        )

    def argmin(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Position of the smallest value.

        Args:
            axis: Axis to search along; ``None`` searches the flattened array.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: The first ``int`` index attaining the minimum for a full
            reduction, otherwise a new array of indices.

        Raises:
            ValueError: If reducing an empty selection.

        Example:
            >>> Array([4, 1, 9]).argmin()
            1
        """
        if axis is None:
            return matlib_stats.argmin(list(self.flatten()))
        return self._wrap_result(
            matlib_stats.argmin_axis(self, axis, keepdims=keepdims)
        )

    def count_nonzero(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """Number of elements that are not zero.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A scalar ``int`` for ``axis=None``, otherwise a new array of
            counts shaped like the reduced result.

        Example:
            >>> Array([[0, 1], [2, 0]]).count_nonzero()
            2
        """
        if axis is None:
            return matlib_stats.count_nonzero(list(self.flatten()))
        return self._wrap_result(
            matlib_stats.count_nonzero_axis(self, axis, keepdims=keepdims)
        )

    def norm(self, axis: Optional[int] = None, keepdims: bool = False) -> Any:
        """L2 (Euclidean) norm of all elements, or along a given axis.

        Args:
            axis: Axis to reduce over; ``None`` reduces every element.
            keepdims: Keep the reduced axis with length 1.

        Returns:
            Any: A non-negative ``float`` for ``axis=None``, otherwise a new
            array of per-axis norms.

        Example:
            >>> Array([3, 4]).norm()
            5.0
        """
        if axis is None:
            return matlib_stats.norm(list(self.flatten()))
        return self._wrap_result(matlib_stats.norm_axis(self, axis, keepdims=keepdims))


__all__ = ["ArrayStatsOps"]