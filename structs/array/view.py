"""Layout descriptor for Buffer-backed arrays.

A ``View`` does not own data. It describes which positions of a flat
:class:`~structs.array.buffer.Buffer` are visible, and in what logical shape.
"""

from __future__ import annotations

from itertools import product
from typing import Iterator, Optional, Sequence

from structs.dtypes import Shape, Strides

from .buffer import Buffer

__all__ = ["View"]


class View:
    """A flat Buffer plus the shape/stride/offset metadata that exposes it.

    Attributes:
        buffer: Flat storage shared with one or more arrays.
        shape: One ``int`` per logical dimension.
        strides: Element offset between consecutive entries per dimension.
        offset: Flat position of the first logical element.
    """

    __slots__ = ("buffer", "shape", "strides", "offset")

    def __init__(
        self,
        buffer: Buffer,
        shape: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        offset: int = 0,
    ) -> None:
        if not isinstance(buffer, Buffer):
            raise TypeError("View requires a Buffer")
        self.buffer = buffer
        self.shape: Shape = (
            tuple(shape) if shape is not None else (len(buffer),)
        )
        self.strides: Strides = (
            tuple(strides)
            if strides is not None
            else self.contiguous_strides(self.shape)
        )
        self.offset = offset
        self._validate()

    @staticmethod
    def contiguous_strides(shape: Sequence[int]) -> Strides:
        """Return C-contiguous strides for *shape*."""
        strides = [1] * len(shape)
        acc = 1
        for axis in range(len(shape) - 1, -1, -1):
            strides[axis] = acc
            acc *= shape[axis]
        return tuple(strides)

    def _validate(self) -> None:
        if len(self.shape) != len(self.strides):
            raise ValueError(
                f"shape and strides must have equal length, got "
                f"{self.shape} and {self.strides}"
            )
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"shape dimensions must be non-negative, got {self.shape}")
        if self.offset < 0:
            raise ValueError(f"offset must be non-negative, got {self.offset}")

    @property
    def ndim(self) -> int:
        """Number of logical dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of visible elements."""
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def is_contiguous(self) -> bool:
        """Return whether visible elements occupy one C-ordered Buffer span."""
        expected = 1
        for dim, stride in zip(reversed(self.shape), reversed(self.strides)):
            if dim > 1 and stride != expected:
                return False
            expected *= dim
        return True

    def physical_index(self, index: Sequence[int]) -> int:
        """Map a logical index tuple to its physical Buffer position."""
        if len(index) != self.ndim:
            raise IndexError(
                f"expected {self.ndim} indices, got {len(index)}"
            )
        physical = self.offset
        for axis, position in enumerate(index):
            dim = self.shape[axis]
            resolved = position + dim if position < 0 else position
            if resolved < 0 or resolved >= dim:
                raise IndexError(
                    f"index {position} is out of bounds for axis {axis} "
                    f"with size {dim}"
                )
            physical += resolved * self.strides[axis]
        return physical

    def iter_physical(self) -> Iterator[int]:
        """Yield physical Buffer positions in logical C order."""
        if self.size == 0:
            return
        if self.is_contiguous():
            yield from range(self.offset, self.offset + self.size)
            return
        for index in self.iter_indices():
            yield self.physical_index(index)

    def iter_indices(self) -> Iterator[tuple[int, ...]]:
        """Yield every logical index tuple in C order."""
        if self.ndim == 0:
            yield ()
            return
        yield from product(*(range(dim) for dim in self.shape))

    def to_contiguous(self) -> Buffer:
        """Copy the visible elements into a new C-contiguous Buffer."""
        return Buffer([self.buffer[position] for position in self.iter_physical()])

    def __repr__(self) -> str:
        return (
            f"View(shape={self.shape!r}, strides={self.strides!r}, "
            f"offset={self.offset!r}, size={self.size!r})"
        )
