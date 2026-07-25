"""Elementwise and unary math operations for Array."""

from __future__ import annotations

from debug import get_logger
from matlib import broadcasting, linalg, ops as matlib_ops

logger = get_logger(__name__)


class ArrayMathOps:
    """Mixin providing elementwise and unary math behavior."""

    def _elementwise(self, other, op):
        """Applies a math operator element-wise between self and another Array/scalar."""
        other_data = self._unwrap(other)
        return self.__class__(broadcasting.elementwise(self.data, other_data, op))

    def __add__(self, other):
        return self._elementwise(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._elementwise(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._elementwise(other, lambda x, y: y - x)

    def __mul__(self, other):
        return self._elementwise(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._elementwise(other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self._elementwise(other, lambda x, y: y / x)

    def __neg__(self):
        return -1 * self

    def __gt__(self, other):
        return self._elementwise(other, lambda x, y: x > y)

    def __lt__(self, other):
        return self._elementwise(other, lambda x, y: x < y)

    def __ge__(self, other):
        return self._elementwise(other, lambda x, y: x >= y)

    def __le__(self, other):
        return self._elementwise(other, lambda x, y: x <= y)

    def dot(self, other, axes=1):
        return self._wrap_result(linalg.tensordot(self, other, axes=axes))

    def matvec(self, other):
        return self._wrap_result(linalg.matvec(self, other))

    def matmul(self, other):
        return self._wrap_result(linalg.matmul(self, other))

    def outer(self, other):
        return self._wrap_result(linalg.outer(self, other))

    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, other):
        return self._elementwise(other, lambda x, y: x ** y)

    def exp(self):
        return self._wrap_result(matlib_ops.exp(self))

    def log(self):
        return self._wrap_result(matlib_ops.log(self))

    def sqrt(self):
        return self._wrap_result(matlib_ops.sqrt(self))

    def abs(self):
        return self._wrap_result(matlib_ops.abs(self))


__all__ = ["ArrayMathOps"]
