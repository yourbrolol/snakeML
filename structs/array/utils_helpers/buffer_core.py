"""Core Buffer-backed helpers for Array."""

from __future__ import annotations

import copy
from typing import Any, Iterator, List, Optional, Sequence

from debug import get_logger
from structs import utils as struct_utils
from structs.dtypes import DTypeName, NestedList, Self, Shape

from ..buffer import Buffer
from ..protocol import ArrayProtocol

logger = get_logger(__name__)

__all__ = ["ArrayBufferOps"]


class ArrayBufferOps(ArrayProtocol):
    """Core conversion, traversal, metadata, and wrapping helpers."""

    data: NestedList
    shape: Shape
    ndim: int

    def _from_buffer(self, buffer: Buffer, shape: Shape) -> Self:
        return self.__class__(buffer, raw=True, shape=shape)

    def _buffer_from(self, values: Any) -> tuple[Buffer, Shape]:
        """Convert an Array-like input into a flat ``Buffer`` and its ``Shape``.

        Returns a ``(buffer, shape)`` pair that ``Array.__init__`` can use
        in its non‑raw path.  Delegates to :func:`~structs.array.buffer_helpers._buffer_from`.

        Args:
            values: see ``_buffer_from`` docstring for full list.

        Returns:
            tuple[Buffer, Shape]: ``(buffer, shape)`` pair.

        Raises:
            TypeError: for unsupported input types.
        """
        from ..buffer_helpers import _buffer_from as _bf

        return _bf(values)

    def _to_buffer(self, values: Any) -> Buffer:
        """Convert an Array-like input into a flat Buffer.

        Delegates to :func:`~structs.array.buffer_helpers._to_buffer`.
        """
        from ..buffer_helpers import _to_buffer as _tb

        return _tb(values)

    def _to_list(self, values: Any) -> NestedList:
        """Convert inputs (tuples, generators, other arrays) into standard lists.

        Args:
            values: An ``Array`` (returns its ``data``), a ``list``/``tuple``
                (rebuilt recursively), a ``range`` (expanded directly), any
                other non-string iterable (rebuilt recursively), or a scalar.

        Returns:
            NestedList: Plain nested ``list`` structure, or the scalar itself
            when ``values`` is not iterable. ``str``/``bytes`` are treated as
            scalars so that text is never exploded into characters.
        """
        if isinstance(values, self.__class__):
            return values.data
        if isinstance(values, (list, tuple)):
            return [self._to_list(v) for v in values]
        if isinstance(values, range):
            return list(values)
        if hasattr(values, "__iter__") and not isinstance(
            values, (str, bytes, bytearray)
        ):
            return [self._to_list(v) for v in values]
        return values

    def _get_shape(self, data: Any) -> Shape:
        """Recursively calculate the dimensions (shape) of nested list data.

        Delegates to :func:`~structs.array.buffer_helpers._get_shape`.
        """
        from ..buffer_helpers import _get_shape as _gs

        return _gs(data)

    def _strides(self, shape: Shape) -> tuple[int, ...]:
        """Compute C-contiguous strides for a resolved shape.

        Args:
            shape: Fully resolved shape, one ``int`` per dimension.

        Returns:
            tuple[int, ...]: Element offset per dimension, e.g. ``(3, 1)`` for
            a ``(2, 3)`` shape.
        """
        strides: List[int] = [1] * len(shape)
        acc = 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = acc
            acc *= shape[i]
        return tuple(strides)

    def _is_contiguous(self) -> bool:
        """Return whether the visible elements form a contiguous Buffer span."""
        expected = 1
        for dim, stride in zip(reversed(self.shape), reversed(self.strides)):
            if dim > 1 and stride != expected:
                return False
            expected *= dim
        return True

    def _flat_physical_index(self, index: int) -> int:
        """Map a C-order logical flat index to its physical Buffer position.

        Negative indices count from the end. This respects ``shape``,
        ``strides`` and ``offset``, so it also works for future views.
        """
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(f"flat index {index} is out of bounds for size {self.size}")
        if self.ndim == 0:
            return self.offset
        physical = self.offset
        remaining = index
        for dim, stride in zip(reversed(self.shape), reversed(self.strides)):
            physical += (remaining % dim) * stride
            remaining //= dim
        return physical

    def _flat_physical_indices(self) -> Iterator[int]:
        """Yield physical Buffer positions for every visible element in C order."""
        if self.size == 0:
            return
        if self._is_contiguous():
            yield from range(self.offset, self.offset + self.size)
            return
        coordinates = [0] * self.ndim
        for _ in range(self.size):
            physical = self.offset
            for dim in range(self.ndim):
                physical += coordinates[dim] * self.strides[dim]
            yield physical
            for dim in range(self.ndim - 1, -1, -1):
                coordinates[dim] += 1
                if coordinates[dim] < self.shape[dim]:
                    break
                coordinates[dim] = 0

    @property
    def size(self) -> int:
        """Total number of elements in the array.

        Returns:
            int: The product of all dimensions, i.e. ``1`` for a 0-d array.
        """
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def dtype(self) -> DTypeName:
        """Infer a simple dtype label for the array contents.

        Returns:
            DTypeName: ``"int"``, ``"float"``, ``"bool"``, or the name of the
            element type when it is something else. Any ``float`` anywhere
            makes the whole array ``"float"``; otherwise any ``bool`` makes it
            ``"bool"``.

        Example:
            >>> Array([[1, 2], [3, 4]]).dtype
            'int'
        """

        def infer(value: Any) -> str:
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, float):
                return "float"
            if isinstance(value, int):
                return "int"
            return type(value).__name__

        values = iter(self._flatten())
        try:
            inferred = infer(next(values))
        except StopIteration:
            return "int"
        for value in values:
            other = infer(value)
            if other == "float" or inferred == "float":
                inferred = "float"
            elif other == "bool" or inferred == "bool":
                inferred = "bool"
        return inferred

    def copy(self) -> Self:
        """Return a deep copy of the array.

        Returns:
            Self: A new array of class ``type(self)`` whose nested lists are
            fully independent of the original.
        """
        return self.__class__(
            copy.deepcopy(self.data),
            raw=True,
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
        )

    def clone(self) -> Self:
        """Return a deep copy of the array; alias of :meth:`copy`.

        Returns:
            Self: A new array of class ``type(self)`` with copied contents.
        """
        return self.copy()

    def _flatten(self, array: Any = None) -> Iterator[Any]:
        """Yield every visible element in logical row-major order.

        Args:
            array: Container to walk. ``None`` (default) walks ``self.data``
                using ``shape``/``strides``/``offset``; a ``Buffer`` is walked
                verbatim; nested lists/sequences are walked recursively.

        Yields:
            Any: Each leaf scalar, depth-first and left to right.
        """
        if array is None:
            for physical in self._flat_physical_indices():
                yield self.data[physical]
            return
        if isinstance(array, Buffer):
            yield from array.buffer
            return
        if isinstance(array, (list, tuple)):
            for item in array:
                yield from self._flatten(item)
            return
        yield array

    def flatten(self) -> Self:
        """Return a flat 1D copy of the array.

        Returns:
            Self: A new 1-D array of class ``type(self)`` holding every
            element in C order. ``self`` is unchanged.
        """
        flat_array = self._from_buffer(Buffer(list(self._flatten())), (self.size,))
        logger.debug(
            "flattened array",
            original_shape=self.shape,
            flattened_shape=flat_array.shape,
        )
        return flat_array

    def item(self, idx: Optional[int] = None) -> Any:
        """Retrieve a single element as a scalar.

        Args:
            idx: Flat position in the flattened array. ``None`` (default)
                requires the array to hold exactly one element.

        Returns:
            Any: The scalar at that flat position.

        Raises:
            ValueError: If ``idx`` is ``None`` and the array holds more or
                fewer than one element.
            IndexError: If ``idx`` is out of range.
        """
        flat = self.flatten()
        if idx is None:
            if flat.size != 1:
                raise ValueError(
                    "item() requires an index for arrays with more than one element"
                )
            idx = 0
        return flat[idx]

    def tolist(self) -> NestedList:
        """Return the nested Python data structure.

        Returns:
            NestedList: A nested copy of the visible elements, detached from
            the array.
        """
        return self._unwrap(self)

    def indices(self, shape: Optional[Shape] = None) -> Iterator[tuple[int, ...]]:
        """Yield every multi-dimensional index of a shape.

        Args:
            shape: Shape to enumerate. ``None`` (default) uses ``self.shape``.

        Yields:
            tuple[int, ...]: One index tuple per element, in C order.

        Example:
            >>> list(Array.zeros((2,)).indices())
            [(0,), (1,)]
        """
        return struct_utils.indices(self.shape if shape is None else shape)

    def _unwrap(self, value: Any) -> Any:
        """Strip Array wrappers recursively, exposing raw nested lists.

        Args:
            value: An ``Array``, a nested sequence possibly containing
                ``Array`` objects, a ``Buffer``, or a scalar.

        Returns:
            Any: Nested lists for arrays/buffers (rebuilt recursively for
            nested sequences); scalars are returned unchanged.
        """
        from ..indexing import _buffer_to_nested

        if isinstance(value, self.__class__):
            if isinstance(value.data, Buffer):
                return _buffer_to_nested(
                    value.data, value.shape, value.strides, value.offset
                )
            return value.data
        if isinstance(value, Buffer):
            return list(value.buffer)
        if isinstance(value, (list, tuple)):
            return [self._unwrap(item) for item in value]
        return value

    # Deprecated.
    def _wrap_result(self, result: Any) -> Any:
        """Wrap a raw nested-list result in an Array, passing scalars through.

        Args:
            result: Usually a nested ``list`` produced by a matlib helper, or
                an already-wrapped value.

        Returns:
            Any: An ``Array`` when ``result`` is a ``list``, otherwise
            ``result`` unchanged.
        """
        if isinstance(result, list):
            return self.__class__(result)
        return result

    def _wrap(self, value: Any) -> Self:
        """Wrap a value in an Array unless it already is one.

        Args:
            value: Nested data or an existing ``Array``.

        Returns:
            Self: ``value`` itself if it is already an ``Array`` (avoiding
            double-wrapping), otherwise a new array of class ``type(self)``.
        """
        if isinstance(value, self.__class__):
            return value
        return self.__class__(value)

    @staticmethod
    def wraparray(value: Any) -> Any:
        """Return value as an Array, avoiding double-wrapping.

        Args:
            value: Nested data, or an already-constructed ``Array``.

        Returns:
            Any: ``value`` itself if it is already an ``Array``, otherwise a
            new ``Array`` wrapping it.
        """
        from ..array import Array

        return value if isinstance(value, Array) else Array(value)

    def _restore_from_flat(self, flat: Sequence[Any], shape: Shape) -> NestedList:
        """Rebuild nested lists from a flat sequence and a target shape.

        Delegates to :func:`~structs.array.buffer_helpers._restore_from_flat`.
        """
        from ..buffer_helpers import _restore_from_flat as _rff

        return _rff(flat, shape)
