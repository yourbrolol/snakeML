"""Elementwise and unary math operations for Array."""

from __future__ import annotations

import math

from debug import get_logger
from matlib import broadcasting, linalg, ops as matlib_ops

logger = get_logger(__name__)


class ArrayMathOps:
    """Mixin providing elementwise and unary math behavior."""

    def _elementwise(self, other, op):
        """Applies a math operator element-wise between self and another Array/scalar."""
        other_data = self._unwrap(other)
        return self.__class__(broadcasting.elementwise(self.data, other_data, op))

    def _update_self(self, result):
        self.data = result.data if hasattr(result, "data") else result
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

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

    def __floordiv__(self, other):
        return self._elementwise(other, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        return self._elementwise(other, lambda x, y: y // x)

    def __mod__(self, other):
        return self._elementwise(other, lambda x, y: x % y)

    def __rmod__(self, other):
        return self._elementwise(other, lambda x, y: y % x)

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

    def __eq__(self, other):
        return self._elementwise(other, lambda x, y: x == y)

    def __ne__(self, other):
        return self._elementwise(other, lambda x, y: x != y)

    def __and__(self, other):
        return self._elementwise(other, lambda x, y: x and y)

    def __or__(self, other):
        return self._elementwise(other, lambda x, y: x or y)

    def __xor__(self, other):
        return self._elementwise(other, lambda x, y: x ^ y)

    def __abs__(self):
        return self._wrap_result(matlib_ops.abs(self))

    def __iadd__(self, other):
        return self._update_self(self._elementwise(other, lambda x, y: x + y))

    def __isub__(self, other):
        return self._update_self(self._elementwise(other, lambda x, y: x - y))

    def __imul__(self, other):
        return self._update_self(self._elementwise(other, lambda x, y: x * y))

    def __itruediv__(self, other):
        return self._update_self(self._elementwise(other, lambda x, y: x / y))

    def __ipow__(self, other):
        return self._update_self(self._elementwise(other, lambda x, y: x ** y))

    def __imatmul__(self, other):
        return self._update_self(self.matmul(other))

    def dot(self, other, axes=1):
        return self._wrap_result(linalg.tensordot(self, other, axes=axes))

    def inner(self, other):
        if self.ndim == 1 and getattr(other, "ndim", 0) == 1:
            return sum(x * y for x, y in zip(self, other))
        return self.dot(other, axes=1)

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

    def neg(self):
        return self._elementwise(0, lambda x, y: -x)

    def exp(self):
        return self._wrap_result(matlib_ops.exp(self))

    def log(self):
        return self._wrap_result(matlib_ops.log(self))

    def log2(self):
        return self._wrap_result(matlib_ops.log2(self))

    def log10(self):
        return self._wrap_result(matlib_ops.log10(self))

    def sqrt(self):
        return self._wrap_result(matlib_ops.sqrt(self))

    def square(self):
        return self._wrap_result(matlib_ops.square(self))

    def abs(self):
        return self._wrap_result(matlib_ops.abs(self))

    def sin(self):
        return self._wrap_result(matlib_ops.sin(self))

    def cos(self):
        return self._wrap_result(matlib_ops.cos(self))

    def tan(self):
        return self._wrap_result(matlib_ops.tan(self))

    def asin(self):
        return self._wrap_result(matlib_ops.asin(self))

    def acos(self):
        return self._wrap_result(matlib_ops.acos(self))

    def atan(self):
        return self._wrap_result(matlib_ops.atan(self))

    def atan2(self, other):
        return self._elementwise(other, lambda x, y: math.atan2(x, y))

    def sinh(self):
        return self._wrap_result(matlib_ops.sinh(self))

    def cosh(self):
        return self._wrap_result(matlib_ops.cosh(self))

    def tanh(self):
        return self._wrap_result(matlib_ops.tanh(self))

    def floor(self):
        return self._wrap_result(matlib_ops.floor(self))

    def ceil(self):
        return self._wrap_result(matlib_ops.ceil(self))

    def round(self, ndigits=None):
        return self._wrap_result(matlib_ops.round(self, ndigits))

    def clip(self, min_value=None, max_value=None):
        return self._wrap_result(matlib_ops.clip(self, min_value=min_value, max_value=max_value))

    def sign(self):
        return self._wrap_result(matlib_ops.sign(self))

    def trace(self):
        if self.ndim != 2:
            raise ValueError("trace requires a 2D array")
        return sum(self[i][i] for i in range(min(self.shape)))

    def diag(self):
        if self.ndim == 1:
            return self.__class__([[value if i == j else 0 for j in range(len(self))] for i, value in enumerate(self)])
        if self.ndim == 2:
            return self.__class__([self[i][i] for i in range(min(self.shape))])
        raise ValueError("diag only supports 1D or 2D arrays")

    def diagonal(self):
        if self.ndim != 2:
            raise ValueError("diagonal requires a 2D array")
        return self.__class__([self[i][i] for i in range(min(self.shape))])

    def inverse(self):
        if self.shape != (2, 2):
            raise ValueError("inverse is only implemented for 2x2 arrays")
        a, b = self[0][0], self[0][1]
        c, d = self[1][0], self[1][1]
        det = a * d - b * c
        if det == 0:
            raise ValueError("matrix is singular")
        return self.__class__([[d / det, -b / det], [-c / det, a / det]])

    def det(self):
        if self.shape != (2, 2):
            raise ValueError("det is only implemented for 2x2 arrays")
        return self[0][0] * self[1][1] - self[0][1] * self[1][0]

    def solve(self, other):
        if self.shape != (2, 2):
            raise ValueError("solve is only implemented for 2x2 arrays")
        a, b = self[0][0], self[0][1]
        c, d = self[1][0], self[1][1]
        det = a * d - b * c
        if det == 0:
            raise ValueError("matrix is singular")
        other_data = other.data if hasattr(other, "data") else other
        if isinstance(other_data, list) and not isinstance(other_data[0], list):
            return self.__class__([((d * other_data[0] - b * other_data[1]) / det), ((-c * other_data[0] + a * other_data[1]) / det)])
        raise ValueError("solve currently expects a 2-element vector")

    def pinv(self):
        return self.inverse()

    def cholesky(self):
        if self.shape != (2, 2):
            raise ValueError("cholesky is only implemented for 2x2 arrays")
        a, b = self[0][0], self[0][1]
        c = self[1][1]
        if a <= 0:
            raise ValueError("matrix must be positive-definite")
        l11 = math.sqrt(a)
        l21 = b / l11
        l22 = math.sqrt(c - l21 * l21)
        return self.__class__([[l11, 0], [l21, l22]])

    def cross(self, other):
        if self.ndim != 1 or getattr(other, "ndim", 0) != 1:
            raise ValueError("cross requires 1D vectors")
        a, b, c = self[0], self[1], self[2]
        d, e, f = other[0], other[1], other[2]
        return self.__class__([b * f - c * e, c * d - a * f, a * e - b * d])

    def svd(self):
        raise NotImplementedError("svd is not implemented yet")

    def qr(self):
        raise NotImplementedError("qr is not implemented yet")

    def eig(self):
        raise NotImplementedError("eig is not implemented yet")

    def eigh(self):
        raise NotImplementedError("eigh is not implemented yet")


__all__ = ["ArrayMathOps"]
