from numbers import Number
from operator import add, sub, mul
import math


class Array:
    def __init__(self, values):
        """Initializes the array with a nested or flat list/tuple."""
        self.data = self._to_list(values)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        
    def _to_list(self, values):
        """Converts inputs (tuples, generators) into standard lists."""
        if isinstance(values, (list, tuple)):
            return [self._to_list(v) for v in values]
        return values

    def _get_shape(self, data):
        """Recursively calculates the dimensions (shape) of the array."""
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        return (len(data),) + self._get_shape(data[0])

    def _flatten(self, data):
        """Helper to flatten the multi-dimensional list."""
        if not isinstance(data, list):
            yield data
        else:
            for item in data:
                yield from self._flatten(item)

    def flatten(self):
        """Returns a flat 1D Array version of the current array."""
        return Array(list(self._flatten(self.data)))

    # --- Element-wise Operations Helper ---
    def _elementwise(self, other, op):
        """Applies a math operator element-wise between self and another Array/scalar."""
        def _apply(a, b):
            if not isinstance(a, list) and not isinstance(b, list):
                return op(a, b)
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    raise ValueError("Operands could not be broadcast together due to shape mismatch.")
                return [_apply(item_a, item_b) for item_a, item_b in zip(a, b)]
            # Broadcasting scalar
            if isinstance(a, list):
                return [_apply(item_a, b) for item_a in a]
            if isinstance(b, list):
                return [_apply(a, item_b) for item_b in b]

        other_data = other.data if isinstance(other, Array) else other
        return Array(_apply(self.data, other_data))

    # --- Math Operators ---
    def __add__(self, other): return self._elementwise(other, lambda x, y: x + y)
    def __radd__(self, other): return self.__add__(other)
    
    def __sub__(self, other): return self._elementwise(other, lambda x, y: x - y)
    def __rsub__(self, other): return self._elementwise(other, lambda x, y: y - x)
    
    def __mul__(self, other): return self._elementwise(other, lambda x, y: x * y)
    def __rmul__(self, other): return self.__mul__(other)
    
    def __truediv__(self, other): return self._elementwise(other, lambda x, y: x / y)
    def __rtruediv__(self, other): return self._elementwise(other, lambda x, y: y / x)

    # --- Indexing and Slicing ---
    def __getitem__(self, index):
        """Allows 1D and multi-dimensional indexing/slicing."""
        if isinstance(index, tuple):
            result = self.data
            for idx in index:
                result = result[idx]
            return Array(result) if isinstance(result, list) else result
        
        result = self.data[index]
        return Array(result) if isinstance(result, list) else result

    # --- Statistics (Using standard math functions) ---
    def sum(self):
        """Returns the sum of all elements."""
        return sum(self._flatten(self.data))

    def mean(self):
        """Returns the mean of all elements."""
        flat = list(self._flatten(self.data))
        return sum(flat) / len(flat)

    def std(self):
        """Returns the standard deviation of all elements."""
        flat = list(self._flatten(self.data))
        m = self.mean()
        variance = sum((x - m) ** 2 for x in flat) / len(flat)
        return math.sqrt(variance)

    # --- Representation ---
    def __repr__(self):
        return f"Array({self.data})"


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


