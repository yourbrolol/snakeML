"""Structural contract shared by every Array mixin.

The behaviour of :class:`~structs.array.array.Array` is split across five
mixins, each of which calls helpers defined in the others. Declaring those
cross-mixin dependencies in one :class:`Protocol` keeps every mixin
independently type-checkable and documents exactly what each mixin expects the
composed class to provide.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Protocol, Sequence, Union

from structs.dtypes import Axis, IndexKey, NestedList, Self, Shape, Strides


class ArrayProtocol(Protocol):
    """The attributes and helpers every Array mixin may rely on.

    :class:`~structs.array.array.Array` satisfies this protocol by inheriting
    the mixins below; nothing else needs to be implemented.
    """

    #: Nested row-major element storage, one ``list`` level per dimension.
    data: NestedList
    #: One ``int`` per dimension, e.g. ``(2, 3)``.
    shape: Shape
    #: Element offset between consecutive entries of each dimension.
    strides: Strides
    #: Flat index of the first element; reserved for future views.
    offset: int
    #: Number of dimensions, equal to ``len(shape)``.
    ndim: int

    def __init__(
        self,
        values: Any,
        raw: bool = False,
        shape: Optional[Union[list, tuple]] = None,
        strides: Optional[Union[list, tuple]] = None,
        offset: int = 0,
    ) -> None:
        """Construct from nested or flat data, optionally skipping normalisation."""

    # -- helpers provided by ArrayUtilsOps ---------------------------------- #

    def _to_list(self, values: Any) -> NestedList:
        """Normalise tuples/ranges/iterables into plain nested lists."""

    def _get_shape(self, data: Any) -> Shape:
        """Recursively infer the shape of nested list data."""

    def _strides(self, shape: Shape) -> Strides:
        """Compute C-contiguous strides for a resolved shape."""

    def _unwrap(self, value: Any) -> Any:
        """Strip Array wrappers, recursively, exposing the raw nested lists."""

    def _wrap_result(self, result: Any) -> Any:
        """Wrap a raw nested-list result in ``self.__class__`` when needed."""

    def _flatten(self, array: Any = None) -> Iterator[Any]:
        """Yield every element in row-major order."""

    def flatten(self) -> Self:
        """Return a flat 1-D copy of the array."""

    # -- indexing ----------------------------------------------------------- #

    def __getitem__(self, key: IndexKey) -> Any:
        """Read elements with NumPy-style indexing."""

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the outermost dimension."""

    def __len__(self) -> int:
        """Length of the outermost dimension."""

    def __setitem__(self, key: IndexKey, value: Any) -> None:
        """Write elements with NumPy-style indexing."""

    # -- cross-mixin operations --------------------------------------------- #

    def matmul(self, other: Any) -> Self:
        """Matrix product with broadcasting over batch axes."""

    # -- metadata ----------------------------------------------------------- #

    @property
    def size(self) -> int:
        """Total number of elements across all dimensions."""

    @property
    def dtype(self) -> str:
        """Inferred dtype label for the array contents."""

    @property
    def T(self) -> Self:
        """Transpose, swapping the last two axes."""

    def reshape(self, *shape: Union[int, Sequence[int]]) -> Self:
        """Return a copy reshaped to the given dimensions."""

    def permute(self, *axes: Axis) -> Self:
        """Return a copy with the axes reordered."""


__all__ = ["ArrayProtocol"]