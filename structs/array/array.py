from . import (indexing)

from structs import (broadcasting, linalg, stats, utils)

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

    def _indices(self): utils.indices(self.shape)

    def _flatten(self, array=None):
        data = array or self.data
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
        other_data = other.data if isinstance(other, Array) else other
        return Array(broadcasting.elementwise(self.data, other_data, op))

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
    def __gt__(self, other): return self._elementwise(other, lambda x, y: x > y)

    def __lt__(self, other): return self._elementwise(other, lambda x, y: x < y)
    
    def __ge__(self, other): return self._elementwise(other, lambda x, y: x >= y)

    def __le__(self, other): return self._elementwise(other, lambda x, y: x <= y)

    # --- Lenght ---
    def __len__(self): return len(self.data)

    # --- Dot Product ---
    def dot(self, other): return Array(linalg.tensordot(self, other, axes=1))

    # --- Transposition ---
    @property
    def T(self): return Array(linalg.transpose(self))
    
    # --- Linear algebra ---
    def matvec(self, other): return Array(linalg.matvec(self, other))
    
    def matmul(self, other): return Array(linalg.matmul(self, other))

    def outer(self, other): return Array(linalg.outer(self, other))
    
    # --- Operator overload ---
    def __matmul__(self, other): return Array(linalg.matmul(self, other))
    
    def __pow__(self, other): return self._elementwise(other, lambda x, y: x ** y)

    # --- Indexing and Slicing --- 
    def __getitem__(self, key):
        result = indexing._getitem(self, key)
        return Array(result) if not isinstance(result, Array) else result

    def __setitem__(self, key, value): indexing._setitem(self, key, value)
    
    # --- Statistics ---
    def sum(self): return stats.sum(self._flatten())
    
    def mean(self): return stats.mean(self._flatten())

    def std(self): return stats.std(self._flatten())
    
    # --- Representation ---
    def __repr__(self):
        return f"Array({self.data})"

if __name__ == "__main__":
    a = Array([
        [1,2],
        [3,4]
    ])
    b = Array([
        [5,6],
    ])
    print((linalg.matvec(a,b)))
