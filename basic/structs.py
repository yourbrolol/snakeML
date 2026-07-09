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

    # --- Comparison Operators ---
    def __gt__(self, other):
        return self._elementwise(other, lambda x, y: x > y)

    def __lt__(self, other):
        return self._elementwise(other, lambda x, y: x < y)
    
    def __ge__(self, other):
        return self._elementwise(other, lambda x, y: x >= y)

    def __le__(self, other):
        return self._elementwise(other, lambda x, y: x <= y)


    # --- Dot Product / Matrix Multiplication ---
    def dot(self, other):
        """
        Computes the dot product of two arrays.
        Supports 1D/1D, 2D/1D, and 2D/2D operations.
        """
        if not isinstance(other, Array):
            raise TypeError("Dot product requires another Array instance.")

        # Case 1: 1D Array . 1D Array (Vector Dot Product -> returns scalar float/int)
        if self.ndim == 1 and other.ndim == 1:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} not aligned.")
            return sum(a * b for a, b in zip(self.data, other.data))

        # Case 2: 2D Array . 1D Array (Matrix-Vector Multiplication -> returns 1D Array)
        elif self.ndim == 2 and other.ndim == 1:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} not aligned.")
            result = []
            for row in self.data:
                result.append(sum(r * v for r, v in zip(row, other.data)))
            return Array(result)

        # Case 3: 2D Array . 2D Array (Matrix Multiplication -> returns 2D Array)
        elif self.ndim == 2 and other.ndim == 2:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} not aligned.")
            
            # Transpose 'other' to easily grab its columns as rows
            other_cols = list(zip(*other.data))
            
            result = []
            for row in self.data:
                new_row = []
                for col in other_cols:
                    new_row.append(sum(r * c for r, c in zip(row, col)))
                result.append(new_row)
            return Array(result)

        else:
            raise NotImplementedError("Dot product only implemented for up to 2 dimensions.")
    
    # Transposition

    @property
    def T(self):
        return self.transpose()
    
    def transpose(self):
        if self.ndim != 2:
            raise ValueError("transpose is only defined for 2D arrays")

        rows = len(self.data)
        cols = len(self.data[0])

        return Array([
            [self.data[r][c] for r in range(rows)]
            for c in range(cols)
        ])

    # Outer product

    def outer(self, other):
        if not isinstance(other, Array): raise NotImplementedError("Outer product only implemented for Array and Array.")
        if self.ndim != other.ndim: raise NotImplementedError("Outer product only implemented for same length Arrays.")
        return Array([[i*j for j in other] for i in self])

    # Operator overload for python's matrix multiplication operator (@)
    def __matmul__(self, other):
        return self.dot(other)
    
    def __pow__(self, other): 
        return self._elementwise(other, lambda x, y: x ** y)

    # --- Indexing and Slicing ---
    @staticmethod
    def _normalize_key(key):
        if not isinstance(key, tuple): return (key,)
        return key

    def __getitem__(self, _index):
        key = self._normalize_key(_index)
        """Allows 1D and multi-dimensional indexing/slicing."""
        result = self.data
        for idx in key: result = result[idx]
        return Array(result) if isinstance(result, list) else result
        
    def __setitem__(self, _key, value):
        key = self._normalize_key(_key)
        target = self.data
        for idx in key[:-1]: target = target[idx]
        target[key[-1]] = value

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

if __name__ == "__main__":
    u = Array([1,2,3])
    v = Array([4,5])
    print(u.outer(v))
