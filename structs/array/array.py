"""The :class:`Array` container: a dense, nested-list backed n-dimensional array."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, List, Optional, Union, cast

from debug import get_logger
from matlib import linalg
from structs.dtypes import IndexKey, NestedList, Shape, Strides

from . import indexing
from .mixins import ArrayMixins

logger = get_logger(__name__)

__all__ = ["Array"]


class Array(ArrayMixins):
    """A dense n-dimensional array backed by nested Python lists.

    Attributes are resolved eagerly at construction time and stay in sync
    whenever a method mutates ``data`` in place:

    Attributes:
        data: Nested row-major lists holding the elements. Depth equals
            ``ndim``; the innermost values are scalars.
        shape: One ``int`` per dimension, e.g. ``(2, 3)``. An empty tuple
            denotes a 0-d array.
        strides: Element offset between consecutive entries of each dimension,
            in the same layout as ``shape`` (C-contiguous).
        offset: Flat index of the first visible element. Always ``0`` today;
            reserved for future view support.
        ndim: Number of dimensions, i.e. ``len(shape)``.

    Example:
        >>> a = Array([[1, 2, 3], [4, 5, 6]])
        >>> a.shape, a.ndim, a.strides
        ((2, 3), 2, (3, 1))
    """

    data: NestedList
    shape: Shape
    strides: Strides
    offset: int
    ndim: int

    def __init__(
        self,
        values: Any,
        raw: bool = False,
        shape: Optional[Union[list, tuple]] = None,
        strides: Optional[Union[list, tuple]] = None,
        offset: int = 0,
    ) -> None:
        """Initializes the array with a nested or flat list/tuple.

        Args:
            values: A nested ``list``/``tuple``, a flat sequence, a ``range``,
                or another ``Array``. With ``raw=True`` this is stored verbatim.
            raw: When ``True``, skip normalisation and trust the supplied
                ``shape``/``strides`` (fast path for internal construction).
                When ``False`` the input is converted to plain nested lists and
                the shape is inferred.
            shape: Optional explicit shape. Required when ``raw=True`` for
                inputs whose nesting does not match the real layout, e.g.
                0-d or empty arrays. Ignored when ``raw=False``.
            strides: Optional explicit strides. Computed from ``shape`` when
                omitted. Only honoured when ``raw=True``.
            offset: Flat offset of the first element of a view. Defaults to
                ``0`` (the array owns all of its data).

        Raises:
            TypeError: If ``values`` is not iterable or is a ``str``/``bytes``.
        """
        if raw:
            self.data = values
            self.shape = tuple(shape) if shape is not None else self._get_shape(values)
            self.strides = (
                tuple(strides) if strides is not None else self._strides(self.shape)
            )
        else:
            self.data = self._to_list(values)
            self.shape = self._get_shape(self.data)
            self.strides = self._strides(self.shape)
        self.offset = offset
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        """Length of the outermost dimension.

        Returns:
            int: ``shape[0]``, or ``0`` for an empty array.

        Raises:
            TypeError: If called on a 0-d array, whose ``data`` is a bare scalar
                and therefore has no outer dimension.
        """
        return len(cast("List[Any]", self.data))

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the outermost dimension.

        Yields:
            Array | Any: One entry per index along axis 0. For ``ndim <= 1``
            these are scalars; otherwise each is an ``Array`` view.
        """
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key: IndexKey) -> Any:
        """Read elements using NumPy-style indexing.

        Args:
            key: An ``int``, ``slice``, boolean mask, integer list, or a tuple
                combining them. A single ``Ellipsis`` expands to the remaining
                dimensions and missing trailing entries default to full slices.

        Returns:
            Array | Any: An ``Array`` when the selection spans more than one
            element, otherwise the scalar itself.

        Raises:
            IndexError: On more than one ``Ellipsis``, or a boolean mask whose
                length differs from the dimension it indexes.
            TypeMismatchError: If an index entry is not an int, slice,
                boolean mask or integer list.

        Example:
            >>> Array([[1, 2], [3, 4]])[0, 1]
            2
        """
        result = indexing._getitem(self, key)
        if isinstance(result, list):
            return Array(result)
        return result

    def __setitem__(self, key: IndexKey, value: Any) -> None:
        """Write elements using NumPy-style indexing.

        Args:
            key: Same index grammar as ``__getitem__``.
            value: Scalar, or a nested list/``Array`` broadcast across the
                selected positions. ``Array`` inputs are unwrapped to their
                underlying lists first.

        Raises:
            IndexError: If a boolean mask length does not match its dimension.
            TypeMismatchError: If an index entry has an unsupported type.
        """
        indexing._setitem(self, key, value)

    def _repr(self, data: Any) -> str:
        """Render nested list data as a bracketed repr string.

        Args:
            data: Either a ``list`` (recursed into) or a scalar.

        Returns:
            str: A fully rendered representation of the whole subtree.
        """
        if isinstance(data, list):
            return "[" + ", ".join(self._repr(item) for item in data) + "]"
        return repr(data)

    def __repr__(self) -> str:
        """Unambiguous representation of the array contents.

        Returns:
            str: Nested bracket form of ``data``, e.g. ``"[[1, 2], [3, 4]]"``.
        """
        return self._repr(self.data)


if __name__ == "__main__":
    a = Array([[1, 2], [3, 4]])
    b = Array([[5, 6], [7, 8]])
    logger.info("example matmul", result=linalg.matmul(a, b))