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
    
    # --- Constructors ---
    @classmethod
    def full(cls, shape, fill_value=0):
        """Creates an Array of the given shape filled with a specified value."""
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for full", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")
        def build(dim):
            if dim == len(shape):
                return fill_value
            return [build(dim + 1) for _ in range(shape[dim])]
        return cls(build(0))
    
    @classmethod
    def zeros(cls, shape):
        """Creates an Array of the given shape filled with zeros."""
        return cls.full(shape, fill_value=0)
    
    @classmethod
    def ones(cls, shape):
        """Creates an Array of the given shape filled with ones."""
        return cls.full(shape, fill_value=1)
    
    @classmethod
    def empty(cls, shape):
        """Creates an Array of the given shape without initializing values."""
        return cls.full(shape, fill_value=None)
    
    @classmethod
    def randn(cls, shape):
        """Creates an Array of the given shape filled with random floats in [0, 1)."""
        import random
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for randn", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")
        def build(dim):
            if dim == len(shape):
                return random.random()
            return [build(dim + 1) for _ in range(shape[dim])]
        return cls(build(0))
    
    @classmethod
    def randint(cls, shape, low=0, high=10):
        """Creates an Array of the given shape filled with random integers."""
        import random
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for randint", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")
        def build(dim):
            if dim == len(shape):
                return random.randint(low, high - 1)
            return [build(dim + 1) for _ in range(shape[dim])]
        return cls(build(0))
    
    @classmethod
    def repeat(cls, value, shape):
        """Creates an Array by repeating a given value to fill the specified shape."""
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for repeat", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")
        def build(dim):
            if dim == len(shape):
                return value
            return [build(dim + 1) for _ in range(shape[dim])]
        return cls(build(0))
    
    # --- Utility Methods ---
    @classmethod
    def arange(cls, start, stop=None, step=1):
        """Creates a 1D Array with evenly spaced values within a given interval."""
        if stop is None:
            stop = start
            start = 0
        if step == 0:
            logger.error("step cannot be zero in arange", start=start, stop=stop)
            raise ValidationError("Step must be non-zero.")
        return cls(list(range(start, stop, step)))
    
    @classmethod
    def linspace(cls, start, stop, num=50):
        """Creates a 1D Array with evenly spaced values over a specified interval."""
        if num <= 0:
            logger.error("num must be positive in linspace", num=num)
            raise ValidationError("Number of samples must be positive.")
        if num == 1:
            return cls([start])
        step = (stop - start) / (num - 1)
        return cls([start + i * step for i in range(num)])
    
    @classmethod
    def tile(cls, value, reps):
        """Creates an Array by repeating a given value along specified dimensions."""
        if isinstance(reps, int):
            reps = (reps,)
        if not isinstance(reps, (tuple, list)):
            logger.error("invalid reps type for tile", reps=reps)
            raise ValidationError("Repetitions must be an int, tuple, or list.")
        def build(dim):
            if dim == len(reps):
                return value
            return [build(dim + 1) for _ in range(reps[dim])]
        return cls(build(0))

    @staticmethod
    def wraparray(value):
        """Return value as an Array, avoiding double-wrapping."""
        return value if isinstance(value, Array) else Array(value)
    
    # --- Array functions ---
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
    
    def flip(self, axis):
        """Flips the array along the specified axis."""
        if axis < 0:
            axis += self.ndim
        if axis < 0 or axis >= self.ndim:
            logger.error("invalid axis for flip", axis=axis, ndim=self.ndim)
            raise ShapeError(f"Axis {axis} is out of bounds for array of dimension {self.ndim}.")
        return self._wrap_result(manipulation.flip(self, axis))
    
    def copy(self):
        """Returns a deep copy of the array."""
        return Array(self.data)

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
    
    def squeeze(self, axis=None):
        """Removes axes of size 1 from the array."""
        if axis is None:
            new_shape = tuple(dim for dim in self.shape if dim != 1)
        else:
            if isinstance(axis, int):
                axis = (axis,)
            new_shape = list(self.shape)
            for ax in sorted(axis, reverse=True):
                if new_shape[ax] != 1:
                    logger.error("cannot squeeze axis with size > 1", axis=ax, size=new_shape[ax])
                    raise ShapeError(f"Cannot squeeze axis {ax} with size {new_shape[ax]}.")
                del new_shape[ax]
            new_shape = tuple(new_shape)

        return self.reshape(new_shape)
    
    def unsqueeze(self, axis):
        """Adds a new axis of size 1 at the specified position."""
        if isinstance(axis, int):
            axis = (axis,)
        new_shape = list(self.shape)
        for ax in sorted(axis):
            if ax < 0:
                ax += len(new_shape) + 1
            if ax < 0 or ax > len(new_shape):
                logger.error("invalid axis for unsqueeze", axis=ax, current_shape=self.shape)
                raise ShapeError(f"Axis {ax} is out of bounds for array of dimension {len(new_shape)}.")
            new_shape.insert(ax, 1)
        return self.reshape(tuple(new_shape))
    
    def swapaxes(self, axis1, axis2):
        """Swaps two axes of the array."""
        if axis1 < 0:
            axis1 += self.ndim
        if axis2 < 0:
            axis2 += self.ndim
        if axis1 < 0 or axis1 >= self.ndim or axis2 < 0 or axis2 >= self.ndim:
            logger.error("invalid axes for swapaxes", axis1=axis1, axis2=axis2, ndim=self.ndim)
            raise ShapeError(f"Axes {axis1} and {axis2} are out of bounds for array of dimension {self.ndim}.")
        new_axes = list(range(self.ndim))
        new_axes[axis1], new_axes[axis2] = new_axes[axis2], new_axes[axis1]
        return self.permute(*new_axes)

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

    # --- Unary ---
    def __neg__(self): return -1 * self

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
    
    @property
    def size(self):
        """Total number of elements in the array."""
        total = 1
        for dim in self.shape:
            total *= dim
        return total
    
    @property
    def strides(self):
        """Returns the strides of the array, which indicate how many elements to skip to move along each axis."""
        if self.ndim == 0:
            return ()
        strides = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        return tuple(strides)

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
