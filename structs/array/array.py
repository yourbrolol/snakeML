from structs import (linalg, stats)

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
    def __gt__(self, other): return self._elementwise(other, lambda x, y: x > y)

    def __lt__(self, other): return self._elementwise(other, lambda x, y: x < y)
    
    def __ge__(self, other): return self._elementwise(other, lambda x, y: x >= y)

    def __le__(self, other): return self._elementwise(other, lambda x, y: x <= y)

    # --- Lenght ---
    def __len__(self): return len(self.data)

    # --- Dot Product / Matrix Multiplication ---
    def dot(self, other):
        """
        Computes the dot product of two arrays.
        Supports 1D/1D, 2D/1D, and 2D/2D operations.
        """
        if not isinstance(other, Array):
            raise TypeError("Dot product requires another Array instance.")

        # 1D Array . 1D Array -> returns scalar float/int
        if self.ndim == 1 and other.ndim == 1:
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} not aligned.")
            return sum(a * b for a, b in zip(self.data, other.data))

        # 2D Array . 1D Array -> returns 1D Array
        elif self.ndim == 2 and other.ndim == 1:
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} not aligned.")
            result = []
            for row in self.data:
                result.append(sum(r * v for r, v in zip(row, other.data)))
            return Array(result)

        # 2D Array . 2D Array -> returns 2D Array
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
    
    # --- Transposition ---
    @property
    def T(self): return Array(linalg.transpose(self))
    
    # --- Outer product ---
    def outer(self, other): return Array(linalg.outer(self, other))
    
    # --- Operator overload ---
    def __matmul__(self, other): return self.dot(other)
    
    def __pow__(self, other): return self._elementwise(other, lambda x, y: x ** y)

    # --- Indexing and Slicing ---
    def _normalize_key(self, _key):
        key = _key if isinstance(_key, tuple) else (_key,)
        
        if key.count(Ellipsis) > 1:
            raise IndexError("__getitem__ should at most contain one Ellipsis (...)!")
            
        cleaned = []
        for idx in key:
            if idx is Ellipsis:
                pad_count = self.ndim - (len(key) - 1)
                cleaned.extend([slice(None)] * pad_count)
            else:
                cleaned.append(idx)
                
        return tuple(cleaned)
    
    def _getitem(self, target, key):
        if not key:
            return target

        head, *tail = key

        if isinstance(head, int):
            return self._getitem(target[head], tail)

        if isinstance(head, slice):
            return [self._getitem(item, tail) for item in target[head]]

        raise TypeError(f"Unsupported index type: {type(head).__name__}")

    def __getitem__(self, _key):
        key = self._normalize_key(_key)
        result = self._getitem(self.data, key)
        return Array(result) if isinstance(result, list) else result
    
    def _setitem(self, key, value, target):
        if len(key) == 1:
            k = key[0]
            if isinstance(k, slice):
                target[k] = value if isinstance(value, list) else [value]
            else:
                target[k] = value
            return

        head, tail = key[0], key[1:]

        if isinstance(head, slice):
            children = target[head]
            for i, child in enumerate(children):
                sub_val = value[i] if isinstance(value, list) and i < len(value) else value
                self._setitem(tail, sub_val, child)
        else:
            self._setitem(tail, value, target[head])

    def __setitem__(self, _key, value):
        key = self._normalize_key(_key)
        parent = self.data
        self._setitem(key, value, parent)
    
    # --- Statistics ---
    def sum(self): return stats.sum(self._flatten())
    
    def mean(self): return stats.mean(self._flatten())

    def std(self): return stats.std(self._flatten())
    
    # --- Representation ---
    def __repr__(self):
        return f"Array({self.data})"

if __name__ == "__main__":
    arr = Array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])
    arr[0:2, 1] = 4
    print(arr[0:2, 1])
