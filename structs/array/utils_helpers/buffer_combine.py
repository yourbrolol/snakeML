"""Concatenation, stacking, and splitting operations for Array."""

from __future__ import annotations

from typing import Any, List, Sequence, Union

from debug.errors import ShapeError
from matlib._array import manipulation
from structs.dtypes import Self

from ..buffer import Buffer
from ..protocol import ArrayProtocol

__all__ = ["ArrayCombineOps"]


class ArrayCombineOps(ArrayProtocol):
    """Concatenate, stack, split, and axis-slicing operations."""

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
        from ..array import Array

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
        from ..array import Array

        return Array(manipulation.stack(arrays, axis=axis))

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
        other_array = (
            other if isinstance(other, self.__class__) else self.__class__(other)
        )

        if self.ndim == 1 and other_array.ndim == 1:
            values = list(self._flatten()) + list(other_array._flatten())
            return self._from_buffer(Buffer(values), (len(values),))

        if self.ndim != 2 or other_array.ndim != 2:
            raise ShapeError("hstack supports only 1-D and 2-D arrays")

        rows, cols = self.shape
        other_rows, other_cols = other_array.shape
        if other_rows not in (1, rows):
            raise ShapeError(
                f"cannot hstack arrays with {rows} and {other_rows} rows"
            )

        self_values = list(self._flatten())
        other_values = list(other_array._flatten())
        values = []
        for row in range(rows):
            values.extend(self_values[row * cols : (row + 1) * cols])
            if other_rows == 1:
                values.extend(other_values)
            else:
                values.extend(
                    other_values[row * other_cols : (row + 1) * other_cols]
                )
        return self._from_buffer(Buffer(values), (rows, cols + other_cols))

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
        if self.ndim == 0:
            raise ValueError("cannot split a 0-d array")

        axis = axis % self.ndim
        if isinstance(indices_or_sections, int):
            if indices_or_sections <= 0:
                raise ValueError("indices_or_sections must be positive")
            size = self.shape[axis]
            base, extra = divmod(size, indices_or_sections)
            bounds = []
            start = 0
            for i in range(indices_or_sections):
                length = base + (1 if i < extra else 0)
                bounds.append((start, start + length))
                start += length
        else:
            bounds = self._split_points(indices_or_sections, axis)

        parts = []
        for start, stop in bounds:
            key = [slice(None)] * self.ndim
            key[axis] = slice(start, stop)
            part = self[tuple(key)]
            parts.append(part.copy() if isinstance(part, self.__class__) else part)
        return parts

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
