"""Generic utility helpers for Array."""

from __future__ import annotations

import copy
import math
import random
from bisect import bisect_left, bisect_right

from debug import get_logger
from matlib import manipulation
from structs import utils as struct_utils

logger = get_logger(__name__)


class ArrayUtilsOps:
    """Mixin for general utility methods and wrapping helpers."""

    def _to_list(self, values):
        """Converts inputs (tuples, generators) into standard lists."""
        if isinstance(values, self.__class__):
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

    def copy(self):
        """Returns a deep copy of the array."""
        return self.__class__(copy.deepcopy(self.data))

    def clone(self):
        """Alias for copy()."""
        return self.copy()

    def _unwrap(self, value):
        if isinstance(value, self.__class__):
            return value.data
        if isinstance(value, list):
            return [self._unwrap(item) for item in value]
        return value

    # Deprecated.
    def _wrap_result(self, result):
        if isinstance(result, list):
            return self.__class__(result)
        return result
    
    def _wrap(self, value):
        """Wraps a value in an Array if it's not already one."""
        if isinstance(value, self.__class__):
            return value
        return self.__class__(value)

    def indices(self, shape=None):
        return struct_utils.indices(self.shape if shape is None else shape)

    def _flatten(self, array=None):
        data = self.data if array is None else array
        if not isinstance(data, list):
            yield data
        else:
            for item in data:
                yield from self._flatten(item)

    def flatten(self):
        """Returns a flat 1D Array version of the current array."""
        flat_array = self.__class__(list(self._flatten(self.data)))
        logger.debug("flattened array", original_shape=self.shape, flattened_shape=flat_array.shape)
        return flat_array

    def item(self, idx=None):
        """Retrieve a single item from the array, ensuring the result is a scalar."""
        flat = self.flatten()
        if idx is None:
            if flat.size != 1:
                raise ValueError("item() requires an index for arrays with more than one element")
            idx = 0
        return flat[idx]

    @property
    def dtype(self):
        """Infer a simple dtype label for the array contents."""
        def infer(data):
            if isinstance(data, list):
                if not data:
                    return "int"
                inferred = infer(data[0])
                for item in data[1:]:
                    other = infer(item)
                    if other == "float" or inferred == "float":
                        inferred = "float"
                    elif other == "bool" or inferred == "bool":
                        inferred = "bool"
                return inferred
            if isinstance(data, bool):
                return "bool"
            if isinstance(data, float):
                return "float"
            if isinstance(data, int):
                return "int"
            return type(data).__name__

        return infer(self.data)

    def tolist(self):
        """Return the nested Python data structure."""
        return copy.deepcopy(self.data)

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

    @staticmethod
    def wraparray(value):
        """Return value as an Array, avoiding double-wrapping."""
        from .array import Array

        return value if isinstance(value, Array) else Array(value)

    @staticmethod
    def concatenate(arrays, axis=0):
        """Concatenate a list of Arrays along an existing axis."""
        from .array import Array

        return Array(manipulation.concatenate(arrays, axis=axis))

    @staticmethod
    def stack(arrays, axis=0):
        """Stack a list of Arrays along a new axis."""
        from .array import Array

        return Array(manipulation.stack(arrays, axis=axis))

    def split(self, indices_or_sections, axis=0):
        """Split this Array into sub-arrays along an axis."""
        parts = manipulation.split(self, indices_or_sections, axis=axis)
        return [self.__class__(p) for p in parts]

    def unbind(self, axis=0):
        """Remove a dimension by returning all slices along it."""
        parts = manipulation.unbind(self, axis=axis)
        return [self.__class__(p) if isinstance(p, list) else p for p in parts]

    def vstack(self, other):
        """Stack arrays vertically (concatenate along axis 0)."""
        return self.__class__(manipulation.concatenate([self, other], axis=0))

    def hstack(self, other):
        """Stack arrays horizontally (concatenate along axis 1)."""
        other_data = other.data if hasattr(other, "data") else other
        if isinstance(other_data, list) and len(other_data) == 1 and isinstance(other_data[0], list):
            if self.ndim == 2 and len(self.data) != 1:
                row = other_data[0]
                other_data = [row[:] for _ in range(len(self.data))]
        return self.__class__(manipulation.concatenate([self, self.__class__(other_data)], axis=1))

    def array_split(self, indices_or_sections, axis=0):
        """Split an array into multiple parts, allowing uneven sizes."""
        if isinstance(indices_or_sections, int):
            if indices_or_sections <= 0:
                raise ValueError("indices_or_sections must be positive")
            size = self.shape[axis % self.ndim]
            base, extra = divmod(size, indices_or_sections)
            start = 0
            parts = []
            for i in range(indices_or_sections):
                length = base + (1 if i < extra else 0)
                parts.append(self._slice_along_axis(self.data, axis, start, start + length))
                start += length
            return [self.__class__(part) for part in parts]
        return [self.__class__(self._slice_along_axis(self.data, axis, start, stop)) for start, stop in self._split_points(indices_or_sections, axis)]

    def chunk(self, chunks, axis=0):
        """Split the array into chunks of approximately the requested size."""
        if chunks <= 0:
            raise ValueError("chunks must be positive")
        return self.array_split(chunks, axis=axis)

    def astype(self, dtype):
        """Cast the array contents to the requested dtype."""
        def _cast(value):
            if isinstance(value, list):
                return [_cast(item) for item in value]
            if dtype == "float":
                return float(value)
            if dtype == "int":
                return int(value)
            if dtype == "bool":
                return bool(value)
            return value

        return self.__class__(_cast(self.data))

    def fill(self, value):
        """Fill the array with a scalar value and return self."""
        self.data = self._fill_value(self.data, value)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

    def where(self, condition, x, y):
        """Select values from x/y based on a boolean condition."""
        cond_data = condition.data if hasattr(condition, "data") else condition
        x_data = x.data if hasattr(x, "data") else x
        y_data = y.data if hasattr(y, "data") else y

        def _where(data, cond, x_val, y_val):
            if isinstance(data, list):
                return [_where(item, cond_item, x_val, y_val) for item, cond_item in zip(data, cond)]
            return x_val if cond else y_val

        return self.__class__(_where(self.data, cond_data, x_data, y_data))

    def take(self, indices):
        """Select values from a flat view of the array."""
        flat = list(self.flatten())
        return self.__class__([flat[i] for i in indices])

    def put(self, indices, values):
        """Assign values to flat positions in the array."""
        flat = list(self.flatten())
        for idx, value in zip(indices, values):
            flat[idx] = value
        self.data = self._restore_from_flat(flat, self.shape)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

    def gather(self, indices, axis=0):
        """Alias for take()."""
        return self.take(indices)

    def scatter(self, indices, values, axis=0):
        """Alias for put()."""
        return self.put(indices, values)

    def masked_fill(self, mask, value):
        """Replace values where the mask is True."""
        mask_data = mask.data if hasattr(mask, "data") else mask

        def _masked(data, mask_item):
            if isinstance(data, list):
                return [_masked(item, mask_child) for item, mask_child in zip(data, mask_item)]
            return value if mask_item else data

        return self.__class__(_masked(self.data, mask_data))

    def unique(self):
        """Return the unique values in the flattened array."""
        values = list(self.flatten())
        seen = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return self.__class__(seen)

    def sort(self, axis=-1):
        """Sort the flattened values and return a new array."""
        values = sorted(list(self.flatten()))
        return self.__class__(values)

    def argsort(self, axis=-1):
        """Return the indices that would sort the flattened values."""
        values = list(self.flatten())
        return self.__class__(sorted(range(len(values)), key=lambda idx: values[idx]))

    def searchsorted(self, value, side="left"):
        """Return the insertion index for a value in a sorted 1D array."""
        values = list(self.flatten())
        if side == "right":
            return bisect_right(values, value)
        return bisect_left(values, value)

    def shuffle(self):
        """Shuffle a flat view of the array in-place."""
        flat = list(self.flatten())
        random.shuffle(flat)
        self.data = self._restore_from_flat(flat, self.shape)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

    def permutation(self):
        """Return a shuffled copy of the flattened array."""
        flat = list(self.flatten())
        shuffled = list(flat)
        random.shuffle(shuffled)
        return self.__class__(shuffled)

    def choice(self, size=None):
        """Return a random choice from the flattened array."""
        flat = list(self.flatten())
        if size is None:
            return random.choice(flat)
        return self.__class__([random.choice(flat) for _ in range(size)])

    def __contains__(self, item):
        return item in list(self.flatten())

    def __str__(self):
        return repr(self)

    def __bool__(self):
        return bool(self.size and self.item(0))

    def _slice_along_axis(self, data, axis, start, stop):
        if axis == 0:
            return data[start:stop]
        if isinstance(data, list):
            return [self._slice_along_axis(item, axis - 1, start, stop) for item in data]
        return data

    def _split_points(self, indices_or_sections, axis):
        size = self.shape[axis % self.ndim]
        points = []
        if not isinstance(indices_or_sections, (list, tuple)):
            indices_or_sections = list(indices_or_sections)
        current = 0
        for point in indices_or_sections:
            if point < 0:
                point += size
            if point < current or point > size:
                raise ValueError("split points must be increasing and within bounds")
            points.append((current, point))
            current = point
        points.append((current, size))
        return points

    def _fill_value(self, data, value):
        if isinstance(data, list):
            return [self._fill_value(item, value) for item in data]
        return value

    def _restore_from_flat(self, flat, shape):
        if not shape:
            return flat[0]
        size = shape[0]
        if len(shape) == 1:
            return flat[:size]
        chunk = len(flat) // size
        return [self._restore_from_flat(flat[i * chunk:(i + 1) * chunk], shape[1:]) for i in range(size)]


__all__ = ["ArrayUtilsOps"]
