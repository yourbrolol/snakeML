"""Generic utility helpers for Array."""

from __future__ import annotations

import copy
import random
from bisect import bisect_left, bisect_right
from typing import Any, Iterator, List, Optional, Sequence, Type, Union, cast

from debug import get_logger
from matlib import manipulation
from structs import utils as struct_utils
from structs.dtypes import (
    Axis,
    DTypeName,
    IndexList,
    NestedList,
    Scalar,
    SearchSide,
    Self,
    Shape,
)
from .buffer import Buffer

from .protocol import ArrayProtocol

logger = get_logger(__name__)

__all__ = ["ArrayUtilsOps"]


class ArrayUtilsOps(ArrayProtocol):
    """Mixin for general utility methods and wrapping helpers."""

    # Re-declared because this is the mixin that mutates them in place
    # (fill/put/shuffle), and assignment through an inherited Protocol
    # attribute cannot be type-checked.
    data: NestedList
    shape: Shape
    ndim: int

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

    def _to_buffer(self, values: Any) -> Buffer:
        """Convert an Array-like input into a flat Buffer.

        Delegates to :func:`~structs.array.buffer_helpers._to_buffer`.
        """
        from .buffer_helpers import _to_buffer as _tb
        return _tb(values)

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
        from .buffer_helpers import _buffer_from as _bf
        return _bf(values)

    def _get_shape(self, data: Any) -> Shape:
        """Recursively calculate the dimensions (shape) of nested list data.

        Delegates to :func:`~structs.array.buffer_helpers._get_shape`.
        """
        from .buffer_helpers import _get_shape as _gs
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

    def copy(self) -> Self:
        """Return a deep copy of the array.

        Returns:
            Self: A new array of class ``type(self)`` whose nested lists are
            fully independent of the original.
        """
        return self.__class__(copy.deepcopy(self.data))

    def clone(self) -> Self:
        """Return a deep copy of the array; alias of :meth:`copy`.

        Returns:
            Self: A new array of class ``type(self)`` with copied contents.
        """
        return self.copy()

    def _unwrap(self, value: Any) -> Any:
        """Strip Array wrappers recursively, exposing the raw nested lists.

        Args:
            value: An ``Array``, a nested sequence possibly containing
                ``Array`` objects, or a scalar.

        Returns:
            Any: ``Array.data`` for arrays (rebuilt recursively for nested
            sequences); anything else is returned unchanged.
        """
        if isinstance(value, self.__class__):
            return value.data
        if isinstance(value, list):
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

    def _flatten(self, array: Any = None) -> Iterator[Any]:
        """Yield every element of nested list data in row-major order.

        Args:
            array: Container to walk. ``None`` (default) starts at ``self.data``.

        Yields:
            Any: Each leaf scalar, depth-first and left to right.
        """
        data = self.data if array is None else array
        if not isinstance(data, list):
            yield data
        else:
            for item in data:
                yield from self._flatten(item)

    def flatten(self) -> Self:
        """Return a flat 1D copy of the array.

        Returns:
            Self: A new 1-D array of class ``type(self)`` holding every
            element in C order. ``self`` is unchanged.
        """
        flat_array = self.__class__(list(self._flatten(self.data)))
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

    @property
    def dtype(self) -> DTypeName:
        """Infer a simple dtype label for the array contents.

        Returns:
            DTypeName: ``"int"``, ``"float"``, ``"bool"``, or the name of the
            element type when it is something else. Only ``data[0]`` of each
            level is inspected, and any ``float`` anywhere makes the whole
            array ``"float"``.

        Example:
            >>> Array([[1, 2], [3, 4]]).dtype
            'int'
        """

        def infer(data: Any) -> str:
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

    def tolist(self) -> NestedList:
        """Return the nested Python data structure.

        Returns:
            NestedList: A deep copy of ``self.data``, detached from the array.
        """
        return copy.deepcopy(self.data)

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

    @staticmethod
    def wraparray(value: Any) -> Any:
        """Return value as an Array, avoiding double-wrapping.

        Args:
            value: Nested data, or an already-constructed ``Array``.

        Returns:
            Any: ``value`` itself if it is already an ``Array``, otherwise a
            new ``Array`` wrapping it.
        """
        from .array import Array

        return value if isinstance(value, Array) else Array(value)

    @staticmethod
    def concatenate(arrays: Sequence[Any], axis: int = 0) -> Any:
        """Concatenate several Arrays along an existing axis.

        Args:
            arrays: Non-empty sequence of arrays to join.
            axis: Axis along which to concatenate. Negative values count from
                the end.

        Returns:
            Any: A new ``Array`` whose shape matches the inputs except along
            ``axis``, where the lengths are summed.

        Raises:
            ShapeError: If the inputs disagree on every other dimension.
        """
        from .array import Array

        return Array(manipulation.concatenate(arrays, axis=axis))

    @staticmethod
    def stack(arrays: Sequence[Any], axis: int = 0) -> Any:
        """Stack several Arrays along a new axis.

        Args:
            arrays: Non-empty sequence of arrays, all of the same shape.
            axis: Position of the new axis. ``0`` (default) prepends it.

        Returns:
            Any: A new ``Array`` of rank ``ndim + 1`` holding the inputs
            stacked in order.

        Raises:
            ShapeError: If the inputs are not identically shaped.
        """
        from .array import Array

        return Array(manipulation.stack(arrays, axis=axis))

    def split(
        self, indices_or_sections: Union[int, Sequence[int]], axis: int = 0
    ) -> List[Any]:
        """Split this Array into sub-arrays along an axis.

        Args:
            indices_or_sections: Either a positive ``int`` giving the number of
                equal parts, or a sequence of ``n - 1`` split points.
            axis: Axis to split along; negative values count from the end.

        Returns:
            list[Any]: The resulting arrays in order.

        Raises:
            ValueError: If the split points are not increasing or out of bounds.
            ShapeError: If the resulting parts would not all be equal-sized.
        """
        parts = manipulation.split(self, indices_or_sections, axis=axis)
        return [self.__class__(p) for p in parts]

    def unbind(self, axis: int = 0) -> List[Any]:
        """Remove a dimension by returning every slice along it.

        Args:
            axis: Axis to remove; negative values count from the end.

        Returns:
            list[Any]: One array per index along ``axis``, each with that
            dimension dropped.
        """
        parts = manipulation.unbind(self, axis=axis)
        return [self.__class__(p) if isinstance(p, list) else p for p in parts]

    def vstack(self, other: Any) -> Self:
        """Stack arrays vertically (concatenate along axis 0).

        Args:
            other: Array (or nested data) with a matching column count.

        Returns:
            Self: A new array with the rows of both inputs, ``self`` first.

        Raises:
            ShapeError: If the column counts differ.
        """
        return self.__class__(manipulation.concatenate([self, other], axis=0))

    def hstack(self, other: Any) -> Self:
        """Stack arrays horizontally (concatenate along axis 1).

        A single-row ``other`` is broadcast across the rows of ``self``.

        Args:
            other: Array (or nested data) with a matching row count, or a
                single row to repeat for every row of ``self``.

        Returns:
            Self: A new array with the columns of both inputs.

        Raises:
            ShapeError: If the row counts differ and cannot be broadcast.
        """
        other_data = other.data if hasattr(other, "data") else other
        if (
            isinstance(other_data, list)
            and len(other_data) == 1
            and isinstance(other_data[0], list)
        ):
            if self.ndim == 2 and len(cast("List[Any]", self.data)) != 1:
                row = other_data[0]
                other_data = [row[:] for _ in range(len(cast("List[Any]", self.data)))]
        return self.__class__(
            manipulation.concatenate([self, self.__class__(other_data)], axis=1)
        )

    def array_split(
        self, indices_or_sections: Union[int, Sequence[int]], axis: int = 0
    ) -> List[Any]:
        """Split the array into multiple parts, allowing uneven sizes.

        Args:
            indices_or_sections: Either a positive ``int`` giving the number of
                parts, or a sequence of ``n - 1`` split points.
            axis: Axis to split along; negative values count from the end.

        Returns:
            list[Any]: The resulting arrays in order. When splitting into an
            ``int`` number of parts, the first parts carry one extra element.

        Raises:
            ValueError: If ``indices_or_sections`` is not positive, or if the
                split points are not increasing and within bounds.
        """
        if isinstance(indices_or_sections, int):
            if indices_or_sections <= 0:
                raise ValueError("indices_or_sections must be positive")
            size = self.shape[axis % self.ndim]
            base, extra = divmod(size, indices_or_sections)
            start = 0
            parts = []
            for i in range(indices_or_sections):
                length = base + (1 if i < extra else 0)
                parts.append(
                    self._slice_along_axis(self.data, axis, start, start + length)
                )
                start += length
            return [self.__class__(part) for part in parts]
        return [
            self.__class__(self._slice_along_axis(self.data, axis, start, stop))
            for start, stop in self._split_points(indices_or_sections, axis)
        ]

    def chunk(self, chunks: int, axis: int = 0) -> List[Any]:
        """Split the array into chunks of approximately the requested size.

        Args:
            chunks: Number of chunks to produce; must be positive.
            axis: Axis to split along; negative values count from the end.

        Returns:
            list[Any]: The chunks in order; see :meth:`array_split`.

        Raises:
            ValueError: If ``chunks`` is not positive.
        """
        if chunks <= 0:
            raise ValueError("chunks must be positive")
        return self.array_split(chunks, axis=axis)

    def astype(self, dtype: DTypeName) -> Self:
        """Cast the array contents to the requested dtype.

        Args:
            dtype: Target label. ``"float"``, ``"int"`` and ``"bool"`` convert
                every leaf; any other value leaves the data unchanged.

        Returns:
            Self: A new array of class ``type(self)`` holding the converted
            values.

        Raises:
            ValueError: If a leaf cannot be converted to the requested dtype.
        """

        def _cast(value: Any) -> Any:
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

    def fill(self, value: Any) -> Self:
        """Overwrite every element with a scalar, in place.

        Args:
            value: The scalar written to every position.

        Returns:
            Self: ``self``, with ``data``, ``shape`` and ``ndim`` refreshed.
        """
        self.data = self._fill_value(self.data, value)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

    def where(self, condition: Any, x: Any, y: Any) -> Self:
        """Select values from x/y based on a boolean condition.

        Args:
            condition: Array-like of the same shape as ``self``; truthy values
                select from ``x``.
            x: Values taken where ``condition`` is truthy. Arrays are unwrapped
                to their raw nested lists.
            y: Values taken where ``condition`` is falsy.

        Returns:
            Self: A new array of class ``type(self)`` with the same shape as
            ``self``.
        """
        cond_data = condition.data if hasattr(condition, "data") else condition
        x_data = x.data if hasattr(x, "data") else x
        y_data = y.data if hasattr(y, "data") else y

        def _where(data: Any, cond: Any, x_val: Any, y_val: Any) -> NestedList:
            if isinstance(data, list):
                return [
                    _where(item, cond_item, x_val, y_val)
                    for item, cond_item in zip(data, cond)
                ]
            return x_val if cond else y_val

        return self.__class__(_where(self.data, cond_data, x_data, y_data))

    def take(self, indices: IndexList) -> Self:
        """Select values from a flat view of the array.

        Args:
            indices: Flat positions to read. Negative positions count from the
                end, and repeats are allowed.

        Returns:
            Self: A new 1-D array of class ``type(self)`` holding the selected
            values in the given order.

        Raises:
            IndexError: If a position is out of range.

        Example:
            >>> Array([10, 20, 30]).take([2, 0])
            [30, 10]
        """
        flat = list(self.flatten())
        return self.__class__([flat[i] for i in indices])

    def put(self, indices: IndexList, values: Sequence[Any]) -> Self:
        """Assign values to flat positions, in place.

        Args:
            indices: Flat positions to write. Negative positions count from the
                end.
            values: Values to write; zipped against ``indices``, so extra
                values are ignored and extra indices are left untouched.

        Returns:
            Self: ``self``, with ``data`` rebuilt from the flat view and
            ``shape``/``ndim`` refreshed.

        Raises:
            IndexError: If a position is out of range.
        """
        flat = list(self.flatten())
        for idx, value in zip(indices, values):
            flat[idx] = value
        self.data = self._restore_from_flat(flat, self.shape)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

    def gather(self, indices: IndexList, axis: int = 0) -> Self:
        """Select flat positions; alias of :meth:`take`.

        Args:
            indices: Flat positions to read.
            axis: Accepted for API compatibility; currently ignored, since
                gathering always happens on the flattened array.

        Returns:
            Self: A new 1-D array holding the selected values.
        """
        return self.take(indices)

    def scatter(self, indices: IndexList, values: Sequence[Any], axis: int = 0) -> Self:
        """Assign flat positions; alias of :meth:`put`.

        Args:
            indices: Flat positions to write.
            values: Values to write, zipped against ``indices``.
            axis: Accepted for API compatibility; currently ignored, since
                scattering always happens on the flattened array.

        Returns:
            Self: ``self``, updated in place.
        """
        return self.put(indices, values)

    def masked_fill(self, mask: Any, value: Any) -> Self:
        """Replace elements wherever the mask is truthy.

        Args:
            mask: Array-like of the same shape as ``self``; truthy positions
                are overwritten.
            value: The replacement scalar.

        Returns:
            Self: A new array of class ``type(self)``; ``self`` is unchanged.

        Example:
            >>> Array([1, 2, 3]).masked_fill(Array([True, False, False]), 0)
            [0, 2, 3]
        """
        mask_data = mask.data if hasattr(mask, "data") else mask

        def _masked(data: Any, mask_item: Any) -> NestedList:
            if isinstance(data, list):
                return [
                    _masked(item, mask_child)
                    for item, mask_child in zip(data, mask_item)
                ]
            return value if mask_item else data

        return self.__class__(_masked(self.data, mask_data))

    def unique(self) -> Self:
        """Return the unique values of the flattened array.

        Returns:
            Self: A new 1-D array of class ``type(self)`` whose elements are
            distinct, ordered by first appearance.
        """
        values = list(self.flatten())
        seen: List[Any] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return self.__class__(seen)

    def sort(self, axis: int = -1) -> Self:
        """Sort the flattened values, ascending.

        Args:
            axis: Accepted for API compatibility; the flattened array is
                always sorted regardless of the value.

        Returns:
            Self: A new 1-D array of class ``type(self)`` in ascending order;
            ``self`` is unchanged.
        """
        values = sorted(list(self.flatten()))
        return self.__class__(values)

    def argsort(self, axis: int = -1) -> Self:
        """Return the indices that would sort the flattened values.

        Args:
            axis: Accepted for API compatibility; currently ignored.

        Returns:
            Self: A new 1-D array of flat positions ordered by ascending value.
        """
        values = list(self.flatten())
        return self.__class__(sorted(range(len(values)), key=lambda idx: values[idx]))

    def searchsorted(self, value: Any, side: SearchSide = "left") -> int:
        """Return the insertion index for a value in a sorted 1D array.

        Args:
            value: The value to locate.
            side: ``"left"`` (default) returns the first index where ``value``
                could be inserted keeping the sort stable, ``"right"`` returns
                the last one.

        Returns:
            int: The insertion position in ``range(size + 1)``.

        Raises:
            ValueError: If ``side`` is neither ``"left"`` nor ``"right"``.

        Example:
            >>> Array([1, 3, 5]).searchsorted(4)
            2
        """
        values = list(self.flatten())
        if side == "right":
            return bisect_right(values, value)
        return bisect_left(values, value)

    def shuffle(self) -> Self:
        """Shuffle a flat view of the array, in place.

        Returns:
            Self: ``self``, with the elements randomly permuted and the
            original shape restored.
        """
        flat = list(self.flatten())
        random.shuffle(flat)
        self.data = self._restore_from_flat(flat, self.shape)
        self.shape = self._get_shape(self.data)
        self.ndim = len(self.shape)
        return self

    def permutation(self) -> Self:
        """Return a randomly shuffled copy of the flattened array.

        Returns:
            Self: A new 1-D array of class ``type(self)`` with the elements in
            random order; ``self`` is unchanged.
        """
        flat = list(self.flatten())
        shuffled = list(flat)
        random.shuffle(shuffled)
        return self.__class__(shuffled)

    def choice(self, size: Optional[int] = None) -> Any:
        """Draw random elements from the flattened array.

        Args:
            size: Number of draws. ``None`` (default) draws a single element
                and returns it directly.

        Returns:
            Any: A scalar when ``size`` is ``None``, otherwise a new 1-D array
            of class ``type(self)`` with ``size`` independent draws, repeats
            allowed.

        Raises:
            IndexError: If the array is empty.
        """
        flat = list(self.flatten())
        if size is None:
            return random.choice(flat)
        return self.__class__([random.choice(flat) for _ in range(size)])

    def __contains__(self, item: Any) -> bool:
        """Test whether a value occurs anywhere in the array.

        Args:
            item: The value to look for, compared with ``==`` against every
                flattened element.

        Returns:
            bool: ``True`` if any element matches, else ``False``.
        """
        return item in list(self.flatten())

    def __str__(self) -> str:
        """Readable representation of the array contents.

        Returns:
            str: Same as :meth:`__repr__`.
        """
        return repr(self)

    def __bool__(self) -> bool:
        """Truthiness of the array.

        Returns:
            bool: ``True`` only when the array is non-empty *and* its first
            flattened element is truthy. Note that this is an element test,
            not a general notion of array truthiness.
        """
        return bool(self.size and self.item(0))

    def _slice_along_axis(self, data: Any, axis: int, start: int, stop: int) -> Any:
        """Slice nested list data along a single axis.

        Args:
            data: Nested lists to slice.
            axis: Remaining axis index at this recursion level; ``0`` slices
                the outermost list.
            start: Inclusive start position.
            stop: Exclusive stop position.

        Returns:
            Any: The sliced container, or ``data`` unchanged once ``axis``
            reaches the leaves.
        """
        if axis == 0:
            return data[start:stop]
        if isinstance(data, list):
            return [
                self._slice_along_axis(item, axis - 1, start, stop) for item in data
            ]
        return data

    def _split_points(
        self, indices_or_sections: Sequence[int], axis: int
    ) -> List[tuple[int, int]]:
        """Turn split points into explicit ``(start, stop)`` bounds.

        Args:
            indices_or_sections: ``n - 1`` split positions; negative values
                count from the end of the axis.
            axis: Axis being split; negative values count from the end.

        Returns:
            list[tuple[int, int]]: One bounds pair per resulting part, in order.

        Raises:
            ValueError: If a point is out of range or not strictly increasing.

        Example:
            >>> Array.zeros((6,))._split_points([2], axis=0)
            [(0, 2), (2, 6)]
        """
        size = self.shape[axis % self.ndim]
        points: List[tuple[int, int]] = []
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

    def _fill_value(self, data: Any, value: Any) -> NestedList:
        """Rebuild nested data with every leaf replaced by ``value``.

        Args:
            data: Nested lists to rebuild.
            value: The replacement scalar.

        Returns:
            NestedList: A new nested structure of the same shape.
        """
        if isinstance(data, list):
            return [self._fill_value(item, value) for item in data]
        return value

    def _restore_from_flat(self, flat: Sequence[Any], shape: Shape) -> NestedList:
        """Rebuild nested lists from a flat sequence and a target shape.

        Delegates to :func:`~structs.array.buffer_helpers._restore_from_flat`.
        """
        from .buffer_helpers import _restore_from_flat as _rff
        return _rff(flat, shape)


__all__ = ["ArrayUtilsOps"]

