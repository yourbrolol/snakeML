from numbers import Number
from operator import add, sub, mul


class Vector:
    def __init__(self, values=()):
        self.values = list(values)

    def __repr__(self):
        return f"Vector({self.values})"

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def __setitem__(self, i, value):
        self.values[i] = value
    
    def __gt__(self, other):
        return isinstance(other, int | float) and all(i > other for i in self.values)

    def __lt__(self, other):
        return isinstance(other, int | float) and all(i < other for i in self.values)

    def __eq__(self, other):
        return isinstance(other, Vector) and self.values == other.values

    def _check_size(self, other):
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length.")

    def _apply(self, other, op):
        if isinstance(other, Number):
            return Vector(op(x, other) for x in self)

        if isinstance(other, Vector):
            self._check_size(other)
            return Vector(op(x, y) for x, y in zip(self, other))

        return NotImplemented

    def _rapply(self, other, op):
        if isinstance(other, Number):
            return Vector(op(other, x) for x in self)
        return NotImplemented

    def _iapply(self, other, op):
        result = self._apply(other, op)
        if result is NotImplemented:
            return NotImplemented
        self.values = result.values
        return self

    # Addition
    def __add__(self, other):
        return self._apply(other, add)

    def __radd__(self, other):
        return self._rapply(other, add)

    def __iadd__(self, other):
        return self._iapply(other, add)

    # Subtraction
    def __sub__(self, other):
        return self._apply(other, sub)

    def __rsub__(self, other):
        return self._rapply(other, sub)

    def __isub__(self, other):
        return self._iapply(other, sub)

    # Multiplication (scalar or element-wise)
    def __mul__(self, other):
        return self._apply(other, mul)

    def __rmul__(self, other):
        return self._rapply(other, mul)

    def __imul__(self, other):
        return self._iapply(other, mul)

    # Division
    def __truediv__(self, other):
        return self._apply(other, lambda a, b: a / b)

    def __itruediv__(self, other):
        return self._iapply(other, lambda a, b: a / b)

    # Power
    def __pow__(self, other):
        return self._apply(other, lambda a, b: a ** b)

    # Dot product
    def __matmul__(self, other):
        self._check_size(other)
        return sum(x * y for x, y in zip(self, other))

    # Unary
    def __neg__(self):
        return Vector(-x for x in self)

    def __pos__(self):
        return Vector(+x for x in self)


