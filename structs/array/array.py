from debug import get_logger
from . import indexing
from .mixins import ArrayMixins
from matlib import linalg

logger = get_logger(__name__)


class Array(ArrayMixins):
    def __init__(self, values, raw=False, shape=None, strides=None, offset=None):
        """Initializes the array with a nested or flat list/tuple."""
        if not raw:
            self.data = self._flatten(values)
            self.shape = self._get_shape(values)
            self.strides = self._strides(values)
        else:
            self.data = values
            self.shape = shape
            self.strides = strides
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

    def _repr(self, dim, offset):
        parts = []
        for i in range(self.shape[self.ndim]):
            _offset = offset + i * self.strides[dim]
            parts.append(self._repr(dim+1, _offset))
        return "[" + ", ".join(parts) + "]"

    def __repr__(self):
        self._repr

if __name__ == "__main__":
    a = Array([[1, 2], [3, 4]])
    b = Array([[5, 6], [7, 8]])
    logger.info("example matmul", result=linalg.matmul(a, b))
