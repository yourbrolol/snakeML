from __future__ import annotations

import math
from typing import Any, Optional


class ValueMathOps:
    data: Any
    grad: float
    requires_grad: bool
    _backward: Any

    def _coerce(self, other: Any):
        cls: Any = self.__class__
        return other if isinstance(other, ValueMathOps) else cls(other)

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
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._coerce(other)
        out = self._make(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __mul__(self, other):
        other = self._coerce(other)
        out = self._make(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._coerce(other)
        out = self._make(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= out.grad * self.data / (other.data**2)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)

    def __pow__(self, other):
        other = self._coerce(other)
        out = self._make(self.data**other.data, (self, other), "**")

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            if other.requires_grad:
                other.grad += out.data * math.log(self.data) * out.grad

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
            if self.data > 0:
                self.grad += out.grad
            elif self.data < 0:
                self.grad -= out.grad

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # Unary math
    # ----------------------------------------------------------------- #

    def _unary(self, data: Any, backward, op: str):
        out = self._make(data, (self,), op)

        def _backward():
            self.grad += backward(out) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        return self._unary(math.exp(self.data), lambda out: out.data, "exp")

    def log(self):
        return self._unary(math.log(self.data), lambda out: 1.0 / self.data, "log")

    def log2(self):
        return self._unary(
            math.log2(self.data), lambda out: 1.0 / (self.data * math.log(2)), "log2"
        )

    def log10(self):
        return self._unary(
            math.log10(self.data),
            lambda out: 1.0 / (self.data * math.log(10)),
            "log10",
        )

    def sqrt(self):
        return self._unary(
            math.sqrt(self.data), lambda out: 1.0 / (2.0 * out.data), "sqrt"
        )

    def square(self):
        return self._unary(self.data**2, lambda out: 2.0 * self.data, "square")

    def sin(self):
        return self._unary(math.sin(self.data), lambda out: math.cos(self.data), "sin")

    def cos(self):
        return self._unary(math.cos(self.data), lambda out: -math.sin(self.data), "cos")

    def tan(self):
        return self._unary(
            math.tan(self.data), lambda out: 1.0 + out.data**2, "tan"
        )

    def asin(self):
        return self._unary(
            math.asin(self.data),
            lambda out: 1.0 / math.sqrt(1.0 - self.data**2),
            "asin",
        )

    def acos(self):
        return self._unary(
            math.acos(self.data),
            lambda out: -1.0 / math.sqrt(1.0 - self.data**2),
            "acos",
        )

    def atan(self):
        return self._unary(
            math.atan(self.data), lambda out: 1.0 / (1.0 + self.data**2), "atan"
        )

    def sinh(self):
        return self._unary(
            math.sinh(self.data), lambda out: math.cosh(self.data), "sinh"
        )

    def cosh(self):
        return self._unary(
            math.cosh(self.data), lambda out: math.sinh(self.data), "cosh"
        )

    def tanh(self):
        out = self._make(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1.0 - out.data**2) * out.grad

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # Activations
    # ----------------------------------------------------------------- #

    def relu(self):
        out = self._make(max(0.0, self.data), (self,), "relu")

        def _backward():
            if self.data > 0:
                self.grad += out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        out = self._make(1.0 / (1.0 + math.exp(-self.data)), (self,), "sigmoid")

        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad

        out._backward = _backward
        return out

    # ----------------------------------------------------------------- #
    # Non-differentiable (zero gradient)
    # ----------------------------------------------------------------- #

    def floor(self):
        out = self._make(math.floor(self.data), (self,), "floor")
        out._backward = lambda: None
        return out

    def ceil(self):
        out = self._make(math.ceil(self.data), (self,), "ceil")
        out._backward = lambda: None
        return out

    def round(self, ndigits: Optional[int] = None):
        out = self._make(round(self.data, ndigits), (self,), "round")
        out._backward = lambda: None
        return out

    def __round__(self, ndigits: Optional[int] = None):
        return self.round(ndigits)

    def sign(self):
        out = self._make(
            (self.data > 0) - (self.data < 0), (self,), "sign"
        )
        out._backward = lambda: None
        return out

    def clip(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        if min_value is None and max_value is None:
            raise ValueError("clip requires at least one bound")
        data = self.data
        if min_value is not None and data < min_value:
            data = min_value
        if max_value is not None and data > max_value:
            data = max_value
        out = self._make(data, (self,), "clip")

        def _backward():
            if data == self.data:
                self.grad += out.grad

        out._backward = _backward
        return out
