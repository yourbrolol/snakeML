"""Elementwise and ordering transformations for Array."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, List

from structs.dtypes import DTypeName, NestedList, SearchSide, Self, Shape

from ..buffer import Buffer
from ..protocol import ArrayProtocol

__all__ = ["ArrayTransformOps"]


class ArrayTransformOps(ArrayProtocol):
    """Elementwise, fill, masking, and ordering operations."""

    # Re-declared because this mixin mutates Array data in place (fill).
    data: NestedList
    shape: Shape
    ndim: int

    def _op_flat_values(self, value: Any) -> List[Any]:
        """Return a flat value list for where/mask operands.

        Arrays and Buffers are walked through their logical flat view; plain
        nested sequences are converted via ``_to_buffer``; scalars become a
        one-element list so callers can broadcast them.
        """
        if isinstance(value, self.__class__):
            return list(value._flatten())
        if isinstance(value, Buffer):
            return list(value.buffer)
        if isinstance(value, (list, tuple)):
            return list(self._to_buffer(value).buffer)
        return [value]

    def _op_value_at(self, values: List[Any], index: int, label: str) -> Any:
        """Select a broadcast value for ``index`` or raise a size mismatch."""
        if len(values) == 1:
            return values[0]
        if len(values) == self.size:
            return values[index]
        raise ValueError(
            f"{label} has {len(values)} elements, expected 1 or {self.size}"
        )

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
            if dtype == "float":
                return float(value)
            if dtype == "int":
                return int(value)
            if dtype == "bool":
                return bool(value)
            return value

        values = [_cast(item) for item in self._flatten()]
        return self._from_buffer(Buffer(values), self.shape)

    def fill(self, value: Any) -> Self:
        """Overwrite every visible element with a scalar, in place.

        Args:
            value: The scalar written to every position.

        Returns:
            Self: ``self``, with its Buffer mutated in place.
        """
        if isinstance(self.data, Buffer):
            for physical in self._flat_physical_indices():
                self.data[physical] = value
            return self
        self.data = self._fill_value(self.data, value)
        self.shape = self._get_shape(self.data)
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
        conditions = self._op_flat_values(condition)
        x_values = self._op_flat_values(x)
        y_values = self._op_flat_values(y)

        values = []
        for index in range(self.size):
            cond = self._op_value_at(conditions, index, "condition")
            x_value = self._op_value_at(x_values, index, "x")
            y_value = self._op_value_at(y_values, index, "y")
            values.append(x_value if cond else y_value)
        return self._from_buffer(Buffer(values), self.shape)

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
        masks = self._op_flat_values(mask)
        values = []
        for index, original in enumerate(self._flatten()):
            keep = self._op_value_at(masks, index, "mask")
            values.append(value if keep else original)
        return self._from_buffer(Buffer(values), self.shape)

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
