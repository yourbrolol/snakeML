from debug import get_logger
from debug.errors import ShapeError, ValidationError
from . import indexing

from structs import utils
from matlib import broadcasting, linalg, stats, ops, manipulation


logger = get_logger(__name__)


class Array:
    def __init__(self, values):
        """Initializes the array with a nested or flat list/tuple."""
        self.data = self._to_list(values)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)

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
    def T(self):
        """Transpose: swaps last two axes (generalises to N-D via permute)."""
        if self.ndim == 2:
            return self._wrap_result(linalg.transpose(self))
        if self.ndim < 2:
            return self
        # General N-D: reverse all axes
        axes = list(range(self.ndim - 1, -1, -1))
        return self._wrap_result(linalg.permute(self, axes))

    def permute(self, *axes):
        """Return a view with axes permuted.

        Parameters
        ----------
        axes : int or sequence of int
            New axis ordering, e.g. ``a.permute(2, 0, 1)``.
        """
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return self._wrap_result(linalg.permute(self, axes))

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

    # --- Statistics (global, flattened) ---
    def sum(self, axis=None, keepdims=False):
        """Sum of all elements, or along a given axis."""
        if axis is None:
            return stats.sum(list(self._flatten()))
        return self._wrap_result(stats.sum_axis(self, axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        """Mean of all elements, or along a given axis."""
        if axis is None:
            return stats.mean(list(self._flatten()))
        return self._wrap_result(stats.mean_axis(self, axis, keepdims=keepdims))

    def variance(self, axis=None, keepdims=False):
        """Population variance of all elements, or along a given axis."""
        if axis is None:
            return stats.variance(list(self._flatten()))
        return self._wrap_result(stats.variance_axis(self, axis, keepdims=keepdims))

    def std(self, axis=None, keepdims=False):
        """Standard deviation of all elements, or along a given axis."""
        import math as _math
        if axis is None:
            return stats.std(list(self._flatten()))
        return self._wrap_result(
            # std = sqrt(variance) applied element-wise to the variance result
            ops._apply_unary(
                stats.variance_axis(self, axis, keepdims=keepdims),
                _math.sqrt,
            )
        )

    def max(self, axis=None, keepdims=False):
        """Maximum value of all elements, or along a given axis."""
        if axis is None:
            return stats.max(list(self._flatten()))
        return self._wrap_result(stats.max_axis(self, axis, keepdims=keepdims))

    # --- Element-wise unary math ---
    def exp(self):
        """Element-wise e^x."""
        return self._wrap_result(ops.exp(self))

    def log(self):
        """Element-wise natural logarithm ln(x)."""
        return self._wrap_result(ops.log(self))

    def sqrt(self):
        """Element-wise square root."""
        return self._wrap_result(ops.sqrt(self))

    def abs(self):
        """Element-wise absolute value."""
        return self._wrap_result(ops.abs(self))

    # --- Array manipulation (class-level convenience) ---
    @staticmethod
    def concatenate(arrays, axis=0):
        """Concatenate a list of Arrays along an existing axis."""
        return Array(manipulation.concatenate(arrays, axis=axis))

    @staticmethod
    def stack(arrays, axis=0):
        """Stack a list of Arrays along a new axis."""
        return Array(manipulation.stack(arrays, axis=axis))

    def split(self, indices_or_sections, axis=0):
        """Split this Array into sub-arrays along an axis.

        Returns a list of Array objects.
        """
        parts = manipulation.split(self, indices_or_sections, axis=axis)
        return [Array(p) for p in parts]

    def unbind(self, axis=0):
        """Remove a dimension by returning all slices along it.

        Each returned Array has one fewer dimension than ``self``.

        Parameters
        ----------
        axis : int, optional
            The axis to unbind along (default 0).

        Returns
        -------
        list of Array
            ``self.shape[axis]`` Arrays, each of shape
            ``self.shape[:axis] + self.shape[axis+1:]``.

        Examples
        --------
        ::

            # a has shape (3, 4)
            rows = a.unbind(axis=0)    # → list of 3 Arrays, each shape (4,)
            cols = a.unbind(axis=1)    # → list of 4 Arrays, each shape (3,)
        """
        parts = manipulation.unbind(self, axis=axis)
        return [Array(p) if isinstance(p, list) else p for p in parts]

    # --- Representation ---
    def __repr__(self):
        return f"Array({self.data})"


if __name__ == "__main__":
    a = Array([[1, 2], [3, 4]])
    b = Array([[5, 6], [7, 8]])
    logger.info("example matmul", result=linalg.matmul(a, b))
