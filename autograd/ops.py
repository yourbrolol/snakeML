from __future__ import annotations

import math
from typing import Any, Optional

from structs import Array


def _unbroadcast(grad: Any, target: Array) -> Array:
    if not isinstance(grad, Array):
        return Array.full(target.shape, grad)
    while grad.ndim > target.ndim:
        grad = grad.sum(axis=0)
    while grad.ndim < target.ndim:
        grad = grad.unsqueeze(0)
    for axis, size in enumerate(target.shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _reshape_like(grad: Array, target: Array) -> Array:
    if grad.shape != target.shape:
        return grad.reshape(target.shape)
    return grad


def _diag_matrix(values: Array, shape: tuple) -> Array:
    flat = list(values.flatten())
    rows, cols = shape
    return Array(
        [
            [flat[i] if i == j and i < len(flat) else 0 for j in range(cols)]
            for i in range(rows)
        ]
    )


def _diag_mask(shape: tuple) -> Array:
    rows, cols = shape
    return Array([[1 if i == j else 0 for j in range(cols)] for i in range(rows)])


class ValueMathOps:
    data: Array
    grad: Array
    requires_grad: bool
    _backward: Any

    def _coerce(self, other: Any):
        if isinstance(other, ValueMathOps):
            return other
        cls: Any = self.__class__
        return cls(other if isinstance(other, Array) else Array(other))

    def _make(self, data: Any, children: tuple, op: str):
        cls: Any = self.__class__
        return cls(data, True, children, op)

    # ----------------------------------------------------------------- #
    # Arithmetic
    # ----------------------------------------------------------------- #

    def __add__(self, other):
        other = self._coerce(other)
        out = self._make(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data)
            other.grad += _unbroadcast(out.grad, other.data)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        out = self._make(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data)
            other.grad -= _unbroadcast(out.grad, other.data)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __mul__(self, other):
        other = self._coerce(other)
        out = self._make(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data)
            other.grad += _unbroadcast(self.data * out.grad, other.data)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        out = self._make(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data)
            other.grad -= _unbroadcast(
                out.grad * self.data / (other.data**2), other.data
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)

    def __pow__(self, other):
        other = self._coerce(other)
        out = self._make(self.data**other.data, (self, other), "**")

        def _backward():
            self.grad += _unbroadcast(
                other.data * (self.data ** (other.data - 1)) * out.grad, self.data
            )
            if other.requires_grad:
                other.grad += _unbroadcast(
                    out.data * self.data.log() * out.grad, other.data
                )

        out._backward = _backward
        return out

    def __rpow__(self, other):
        return self._coerce(other).__pow__(self)

    def __neg__(self):
        out = self._make(-self.data, (self,), "neg")

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __abs__(self):
        out = self._make(abs(self.data), (self,), "abs")

        def _backward():
            self.grad += out.grad * self.data.sign()

        out._backward = _backward
        return out

    def abs(self):
        return self.__abs__()

    # ----------------------------------------------------------------- #
    # Unary math
    # ----------------------------------------------------------------- #

    def _unary(self, name: str, backward, op: str):
        out = self._make(getattr(self.data, name)(), (self,), op)

        def _backward():
            self.grad += backward(out) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        return self._unary("exp", lambda out: out.data, "exp")

    def log(self):
        return self._unary("log", lambda out: 1.0 / self.data, "log")

    def log2(self):
        return self._unary(
            "log2", lambda out: 1.0 / (self.data * math.log(2)), "log2"
        )

    def log10(self):
        return self._unary(
            "log10", lambda out: 1.0 / (self.data * math.log(10)), "log10"
        )

    def sqrt(self):
        return self._unary("sqrt", lambda out: 1.0 / (2.0 * out.data), "sqrt")

    def square(self):
        out = self._make(self.data.square(), (self,), "square")

        def _backward():
            self.grad += 2.0 * self.data * out.grad

        out._backward = _backward
        return out

    def sin(self):
        return self._unary("sin", lambda out: self.data.cos(), "sin")

    def cos(self):
        return self._unary("cos", lambda out: -self.data.sin(), "cos")

    def tan(self):
        return self._unary("tan", lambda out: 1.0 + out.data**2, "tan")

    def asin(self):
        return self._unary(
            "asin", lambda out: 1.0 / (1.0 - self.data**2).sqrt(), "asin"
        )

    def acos(self):
        return self._unary(
            "acos", lambda out: -1.0 / (1.0 - self.data**2).sqrt(), "acos"
        )

    def atan(self):
        return self._unary("atan", lambda out: 1.0 / (1.0 + self.data**2), "atan")

    def atan2(self, other):
        other = self._coerce(other)
        out = self._make(self.data.atan2(other.data), (self, other), "atan2")

        def _backward():
            denom = self.data**2 + other.data**2
            self.grad += _unbroadcast(out.grad * other.data / denom, self.data)
            other.grad -= _unbroadcast(out.grad * self.data / denom, other.data)

        out._backward = _backward
        return out

    def sinh(self):
        return self._unary("sinh", lambda out: self.data.cosh(), "sinh")

    def cosh(self):
        return self._unary("cosh", lambda out: self.data.sinh(), "cosh")

    def tanh(self):
        out = self._make(self.data.tanh(), (self,), "tanh")

        def _backward():
            self.grad += (1.0 - out.data**2) * out.grad

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # Activations
    # ----------------------------------------------------------------- #

    def relu(self):
        out = self._make(self.data.clip(min_value=0), (self,), "relu")

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def sigmoid(self):
        out = self._make(1.0 / (1.0 + (-self.data).exp()), (self,), "sigmoid")

        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # Non-differentiable (zero gradient)
    # ----------------------------------------------------------------- #

    def floor(self):
        out = self._make(self.data.floor(), (self,), "floor")
        out._backward = lambda: None
        return out

    def ceil(self):
        out = self._make(self.data.ceil(), (self,), "ceil")
        out._backward = lambda: None
        return out

    def round(self, ndigits: Optional[int] = None):
        out = self._make(self.data.round(ndigits), (self,), "round")
        out._backward = lambda: None
        return out

    def __round__(self, ndigits: Optional[int] = None):
        return self.round(ndigits)

    def sign(self):
        out = self._make(self.data.sign(), (self,), "sign")
        out._backward = lambda: None
        return out

    def clip(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        if min_value is None and max_value is None:
            raise ValueError("clip requires at least one bound")
        data = self.data.clip(min_value=min_value, max_value=max_value)
        out = self._make(data, (self,), "clip")

        def _backward():
            self.grad += out.grad * (self.data == data)

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # Reductions
    # ----------------------------------------------------------------- #

    def sum(self, axis=None, keepdims=False):
        # TODO: Implement me!
        raise NotImplementedError

    def mean(self, axis=None, keepdims=False):
        # TODO: Implement me!
        raise NotImplementedError

    def max(self, axis=None, keepdims=False):
        # TODO: Implement me!
        raise NotImplementedError

    def argmax(self, axis=None, keepdims=False):
        # TODO: Implement me!
        raise NotImplementedError

    # ----------------------------------------------------------------- #
    # Shape
    # ----------------------------------------------------------------- #

    def reshape(self, *shape):
        # TODO: Implement me!
        raise NotImplementedError

    def transpose(self, *axes):
        # TODO: Implement me!
        raise NotImplementedError

    def squeeze(self, axis=None):
        # TODO: Implement me!
        raise NotImplementedError

    # ----------------------------------------------------------------- #
    # Linear algebra
    # ----------------------------------------------------------------- #

    def __matmul__(self, other):
        other = self._coerce(other)
        out = self._make(self.data.matmul(other.data), (self, other), "@")

        def _backward():
            grad = out.grad
            a, b = self.data, other.data
            a_ndim, b_ndim = a.ndim, b.ndim

            if a_ndim == 0 and b_ndim == 0:
                self.grad += grad * b
                other.grad += grad * a
            elif a_ndim == 0:
                self.grad += (grad * b).sum()
                other.grad += _unbroadcast(grad * a, b)
            elif b_ndim == 0:
                self.grad += _unbroadcast(grad * b, a)
                other.grad += (grad * a).sum()
            elif a_ndim == 1 and b_ndim == 1:
                self.grad += grad * b
                other.grad += grad * a
            elif a_ndim == 2 and b_ndim == 1:
                self.grad += grad.outer(b)
                other.grad += a.T.matmul(grad)
            elif a_ndim == 1 and b_ndim == 2:
                self.grad += b.matmul(grad)
                other.grad += a.outer(grad)
            elif a_ndim == 2 and b_ndim == 2:
                self.grad += grad.matmul(b.T)
                other.grad += a.T.matmul(grad)
            else:
                raise NotImplementedError(
                    f"matmul backward not implemented for ranks {a_ndim} and {b_ndim}"
                )

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return self._coerce(other).__matmul__(self)

    def matmul(self, other):
        return self.__matmul__(other)

    def matvec(self, other):
        return self.__matmul__(other)

    def dot(self, other, axes=1):
        if axes != 1:
            raise NotImplementedError("dot only supports axes=1")
        return self.__matmul__(other)

    def inner(self, other):
        return self.dot(other)

    def outer(self, other):
        other = self._coerce(other)
        out = self._make(self.data.outer(other.data), (self, other), "outer")

        def _backward():
            grad = out.grad
            self.grad += _reshape_like(
                grad.matmul(other.data.flatten()), self.data
            )
            other.grad += _reshape_like(
                self.data.flatten().matmul(grad), other.data
            )

        out._backward = _backward
        return out

    def trace(self):
        out = self._make(self.data.trace(), (self,), "trace")

        def _backward():
            self.grad += out.grad * _diag_mask(self.data.shape)

        out._backward = _backward
        return out

    def diag(self):
        out = self._make(self.data.diag(), (self,), "diag")
        source_ndim = self.data.ndim

        def _backward():
            if source_ndim == 1:
                self.grad += out.grad.diagonal()
            else:
                self.grad += _diag_matrix(out.grad, self.data.shape)

        out._backward = _backward
        return out

    def diagonal(self):
        out = self._make(self.data.diagonal(), (self,), "diagonal")

        def _backward():
            self.grad += _diag_matrix(out.grad, self.data.shape)

        out._backward = _backward
        return out

    def inverse(self):
        data = 1.0 / self.data if self.data.ndim == 0 else self.data.inverse()
        out = self._make(data, (self,), "inverse")

        def _backward():
            if self.data.ndim == 0:
                self.grad -= out.grad / (self.data**2)
            else:
                transposed = out.data.T
                self.grad -= transposed.matmul(out.grad).matmul(transposed)

        out._backward = _backward
        return out

    def pinv(self):
        return self.inverse()

    def det(self):
        data = self.data if self.data.ndim == 0 else self.data.det()
        out = self._make(data, (self,), "det")

        def _backward():
            if self.data.ndim == 0:
                self.grad += out.grad
            elif self.data.shape == (1, 1):
                self.grad += out.grad
            else:
                self.grad += out.grad * out.data * self.data.inverse().T

        out._backward = _backward
        return out

    def solve(self, other):
        other = self._coerce(other)
        a, b = self.data, other.data
        data = b / a if a.ndim == 0 else a.solve(b)
        out = self._make(data, (self, other), "solve")

        def _backward():
            grad = out.grad
            inverse = 1.0 / a if a.ndim == 0 else a.inverse()
            db = _unbroadcast(inverse.T.matmul(grad), b)
            da = -db.outer(out.data)
            self.grad += _reshape_like(da, a)
            other.grad += db

        out._backward = _backward
        return out

    def cross(self, other):
        other = self._coerce(other)
        out = self._make(self.data.cross(other.data), (self, other), "cross")

        def _backward():
            self.grad += other.data.cross(out.grad)
            other.grad += out.grad.cross(self.data)

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # Decompositions
    # ----------------------------------------------------------------- #

    def cholesky(self):
        # TODO: Implement me!
        raise NotImplementedError

    def svd(self):
        # TODO: Implement me!
        raise NotImplementedError

    def qr(self):
        # TODO: Implement me!
        raise NotImplementedError

    def eig(self):
        # TODO: Implement me!
        raise NotImplementedError
