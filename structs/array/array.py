from debug import get_logger
from debug.errors import ShapeError, ValidationError
from . import indexing

from structs import broadcasting, linalg, stats, utils


logger = get_logger(__name__)


class Array:
    def __init__(self, values):
        """Initializes the array with a nested or flat list/tuple."""
        self.data = self._to_list(values)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        logger.debug("Array created", shape=self.shape, ndim=self.ndim)

    @staticmethod
    def wraparray(value):
        """Return value as an Array, avoiding double-wrapping."""
        return value if isinstance(value, Array) else Array(value)

    def _to_list(self, values):
        """Converts inputs (tuples, generators) into standard lists."""
        if isinstance(values, Array):
            return values.data
        if isinstance(values, (list, tuple)):
            return [self._to_list(v) for v in values]
        if isinstance(values, range):
            return list(values)
        if hasattr(values, "__iter__") and not isinstance(values, (str, bytes, bytearray)):
            return [self._to_list(v) for v in values]
        return values

    def _get_shape(self, data):
        """Recursively calculates the dimensions (shape) of the array."""
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        return (len(data),) + self._get_shape(data[0])

    def reshape(self, *shape):
        """
        Returns a reshaped copy of the array.

        Examples:
            a.reshape(2, 3)
            a.reshape((2, 3))
            a.reshape(-1, 3)
        """

        # Allow tuple/list as the only argument
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        flat = list(self._flatten())
        total = len(flat)

        # Handle one inferred dimension (-1)
        if shape.count(-1) > 1:
            logger.error("invalid reshape shape has multiple inferred dimensions", shape=shape)
            raise ShapeError("Only one dimension may be -1.")

        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s

            if total % known != 0:
                logger.error("cannot infer reshape size", total=total, shape=shape)
                raise ShapeError(
                    f"Cannot reshape array of size {total} into shape {shape}"
                )

            inferred = total // known
            shape = tuple(inferred if s == -1 else s for s in shape)

        # Verify total size
        expected = 1
        for s in shape:
            if s < 0:
                logger.error("negative reshape dimension", shape=shape)
                raise ShapeError("Dimensions must be non-negative.")
            expected *= s

        if expected != total:
            logger.error("reshape total mismatch", total=total, shape=shape, expected=expected)
            raise ShapeError(
                f"Cannot reshape array of size {total} into shape {shape}"
            )

        # Build nested lists recursively
        flat_iter = iter(flat)

        def build(dim):
            if dim == len(shape):
                return next(flat_iter)
            return [build(dim + 1) for _ in range(shape[dim])]

        return Array(build(0))

    def _unwrap(self, value):
        if isinstance(value, Array):
            return value.data
        if isinstance(value, list):
            return [self._unwrap(item) for item in value]
        return value

    def _wrap_result(self, result):
        if isinstance(result, list):
            return Array(result)
        return result

    def _indices(self):
        return utils.indices(self.shape)

    def indices(self, shape=None):
        return utils.indices(self.shape if shape is None else shape)

    def _flatten(self, array=None):
        data = self.data if array is None else array
        """Helper to flatten the multi-dimensional list."""
        if not isinstance(data, list):
            yield data
        else:
            for item in data:
                yield from self._flatten(item)

    def flatten(self):
        """Returns a flat 1D Array version of the current array."""
        flat_array = Array(list(self._flatten(self.data)))
        logger.debug("flattened array", original_shape=self.shape, flattened_shape=flat_array.shape)
        return flat_array

    # --- Element-wise Operations Helper ---
    def _elementwise(self, other, op):
        """Applies a math operator element-wise between self and another Array/scalar."""
        other_data = self._unwrap(other)
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

    # --- Length ---
    def __len__(self): return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # --- Dot Product ---
    def dot(self, other, axes=1): return self._wrap_result(linalg.tensordot(self, other, axes=axes))

    # --- Transposition ---
    @property
    def T(self): return self._wrap_result(linalg.transpose(self))

    # --- Linear algebra ---
    def matvec(self, other): return self._wrap_result(linalg.matvec(self, other))
    def matmul(self, other): return self._wrap_result(linalg.matmul(self, other))
    def outer(self, other): return self._wrap_result(linalg.outer(self, other))

    # --- Operator overload ---
    def __matmul__(self, other): return self.matmul(other)
    def __pow__(self, other): return self._elementwise(other, lambda x, y: x ** y)

    # --- Indexing and Slicing ---
    def __getitem__(self, key):
        result = indexing._getitem(self, key)
        if isinstance(result, list):
            return Array(result)
        return result

    def __setitem__(self, key, value):
        indexing._setitem(self, key, value)

    # --- Statistics ---
    def sum(self): return stats.sum(list(self._flatten()))
    def mean(self): return stats.mean(list(self._flatten()))
    def std(self): return stats.std(list(self._flatten()))

    # --- Representation ---
    def __repr__(self):
        return f"Array({self.data})"


if __name__ == "__main__":
    a = Array([[1, 2], [3, 4]])
    b = Array([[5, 6], [7, 8]])
    logger.info("example matmul", result=linalg.matmul(a, b))
