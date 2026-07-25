"""Generic utility helpers for Array."""

from __future__ import annotations

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
        return self.__class__(self.data)

    def _unwrap(self, value):
        if isinstance(value, self.__class__):
            return value.data
        if isinstance(value, list):
            return [self._unwrap(item) for item in value]
        return value

    def _wrap_result(self, result):
        if isinstance(result, list):
            return self.__class__(result)
        return result

    def _indices(self):
        return struct_utils.indices(self.shape)

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


__all__ = ["ArrayUtilsOps"]
