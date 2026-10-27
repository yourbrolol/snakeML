"""Flat selection and random operations for Array."""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from structs.dtypes import IndexList, NestedList, Self, Shape

from ..buffer import Buffer
from ..protocol import ArrayProtocol

__all__ = ["ArraySelectOps"]


class ArraySelectOps(ArrayProtocol):
    """Flat take/put/gather/scatter and random sampling operations."""

    # Re-declared because this mixin mutates Array data in place (put/shuffle).
    data: NestedList
    shape: Shape
    ndim: int

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
        flat = list(self._flatten())
        values = []
        for index in indices:
            position = index + len(flat) if index < 0 else index
            if position < 0 or position >= len(flat):
                raise IndexError(
                    f"flat index {index} is out of bounds for size {len(flat)}"
                )
            values.append(flat[position])
        return self._from_buffer(Buffer(values), (len(values),))

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
        if not isinstance(self.data, Buffer):
            for idx, value in zip(indices, values):
                self[idx] = value
            return self
        for idx, value in zip(indices, values):
            physical = self._flat_physical_index(idx)
            self.data[physical] = value
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

    def shuffle(self) -> Self:
        """Shuffle a flat view of the array, in place.

        Returns:
            Self: ``self``, with the elements randomly permuted and the
            original shape restored.
        """
        if not isinstance(self.data, Buffer):
            flat = list(self._flatten())
            random.shuffle(flat)
            self.data = self.__class__(flat).data
            return self
        positions = list(self._flat_physical_indices())
        values = [self.data[position] for position in positions]
        random.shuffle(values)
        for position, value in zip(positions, values):
            self.data[position] = value
        return self

    def permutation(self) -> Self:
        """Return a randomly shuffled copy of the flattened array.

        Returns:
            Self: A new 1-D array of class ``type(self)`` with the elements in
            random order; ``self`` is unchanged.
        """
        shuffled = list(self._flatten())
        random.shuffle(shuffled)
        return self._from_buffer(Buffer(shuffled), (len(shuffled),))

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
