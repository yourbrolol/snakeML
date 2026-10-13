from debug import get_logger
from matlib import linalg

from . import indexing
from .mixins import ArrayMixins

logger = get_logger(__name__)


class Array(ArrayMixins):
    def __init__(self, values, raw=False, shape=None, strides=None, offset=0):
        """Initializes the array with a nested or flat list/tuple."""
        if raw:
            self.data = values
            self.shape = tuple(shape) if shape is not None else self._get_shape(values)
            self.strides = tuple(strides) if strides is not None else self._strides(self.shape)
        else:
            self.data = self._to_list(values)
            self.shape = self._get_shape(self.data)
            self.strides = self._strides(self.shape)
        self.offset = offset
        self.ndim = len(self.shape)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        result = indexing._getitem(self, key)
        if isinstance(result, list):
            return Array(result)
        return result

    def __setitem__(self, key, value):
        indexing._setitem(self, key, value)

    def _repr(self, data):
        if isinstance(data, list):
            return "[" + ", ".join(self._repr(item) for item in data) + "]"
        return repr(data)

    def __repr__(self):
        return self._repr(self.data)


if __name__ == "__main__":
    a = Array([[1, 2], [3, 4]])
    b = Array([[5, 6], [7, 8]])
    logger.info("example matmul", result=linalg.matmul(a, b))
