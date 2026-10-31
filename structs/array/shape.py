"""Shape-manipulation helpers for Array."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from debug import get_logger
from debug.errors import ShapeError
from matlib._array import linalg, manipulation
from structs.dtypes import AxisArg, Self, ShapeArg

from .buffer import Buffer
from .protocol import ArrayProtocol

logger = get_logger(__name__)

__all__ = ["ArrayShapeOps"]


def _as_dims(args: Sequence[ShapeArg]) -> tuple[int, ...]:
    """Flatten varargs dimension arguments into a concrete tuple of ``int``.

    Accepts both ``f(2, 3)`` and ``f((2, 3))`` spellings; a single sequence
    argument is treated as the whole dimension list.

    Args:
        args: Dimension arguments captured from ``*shape`` / ``*axes``.

    Returns:
        tuple[int, ...]: The dimensions as a plain tuple of ``int``.

    Raises:
        TypeError: If any entry is neither an ``int`` nor a nested sequence
            of ``int``.
    """
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        args = args[0]
    return tuple(int(dim) for dim in args)  # type: ignore[call-overload,arg-type]


def _shape_size(shape: tuple[int, ...]) -> int:
    """Return the product of ``shape``, i.e. its element count."""
    total = 1
    for dim in shape:
        total *= dim
    return total


def _coords_from_flat(index: int, shape: tuple[int, ...]) -> list[int]:
    """Convert a C-order flat index into multi-dimensional coordinates."""
    coords = [0] * len(shape)
    remaining = index
    for dim in range(len(shape) - 1, -1, -1):
        coords[dim] = remaining % shape[dim]
        remaining //= shape[dim]
    return coords


def _flat_from_coords(coords: Sequence[int], shape: tuple[int, ...]) -> int:
    """Convert multi-dimensional coordinates into a C-order flat index."""
    index = 0
    for coord, dim in zip(coords, shape):
        index = index * dim + coord
    return index


class ArrayShapeOps(ArrayProtocol):
    """Mixin providing reshape/squeeze/transpose-like operations."""

    def _array_view(
        self,
        shape: Sequence[int],
        strides: Sequence[int],
        offset: Optional[int] = None,
    ) -> Self:
        """Return a new Array sharing this Buffer with new layout metadata."""
        return self.__class__(
            self.data,
            raw=True,
            shape=tuple(shape),
            strides=tuple(strides),
            offset=self.offset if offset is None else offset,
        )

    def reshape(self, *shape: ShapeArg) -> Self:
        """Return a reshaped array, sharing storage when possible.

        Contiguous arrays return a view; non-contiguous arrays are copied.
        Exactly one dimension may be ``-1``, in which case it is inferred from
        the total element count. A single sequence argument is accepted as an
        alternative to varargs, and rows are filled in C order.

        Args:
            *shape: Target dimensions, either as separate ``int`` arguments
                (``a.reshape(2, 3)``), a single sequence
                (``a.reshape((2, 3))``), or with one ``-1`` placeholder
                (``a.reshape(-1, 3)``) to infer that axis.

        Returns:
            Self: A new array with the requested shape; ``self`` is unchanged.

        Raises:
            ShapeError: If more than one dimension is ``-1``, a dimension is
                negative, or the product of the dimensions does not equal the
                current element count.

        Example:
            >>> Array([[1, 2, 3], [4, 5, 6]]).reshape(3, 2)
            [[1, 2], [3, 4], [5, 6]]
        """
        dims = _as_dims(shape)
        total = self.size

        if dims.count(-1) > 1:
            logger.error(
                "invalid reshape shape has multiple inferred dimensions", shape=dims
            )
            raise ShapeError("Only one dimension may be -1.")

        if -1 in dims:
            known = 1
            for dim in dims:
                if dim != -1:
                    known *= dim
            if known == 0:
                if total != 0:
                    raise ShapeError(
                        f"Cannot reshape array of size {total} into shape {dims}"
                    )
                inferred = 0
            else:
                if total % known != 0:
                    logger.error("cannot infer reshape size", total=total, shape=dims)
                    raise ShapeError(
                        f"Cannot reshape array of size {total} into shape {dims}"
                    )
                inferred = total // known
            dims = tuple(inferred if dim == -1 else dim for dim in dims)

        if any(dim < 0 for dim in dims):
            logger.error("negative reshape dimension", shape=dims)
            raise ShapeError("Dimensions must be non-negative.")

        expected = _shape_size(dims)
        if expected != total:
            logger.error(
                "reshape total mismatch", total=total, shape=dims, expected=expected
            )
            raise ShapeError(f"Cannot reshape array of size {total} into shape {dims}")

        if self.view.is_contiguous():
            return self._array_view(dims, self._strides(dims))

        values = list(self._flatten())
        return self._from_buffer(Buffer(values), dims)

    def flip(self, axis: int) -> Self:
        """Reverse the order of elements along one axis.

        Args:
            axis: Axis to reverse. Negative values count from the end, e.g.
                ``-1`` is the last axis.

        Returns:
            Self: A new array with the chosen axis reversed.

        Raises:
            ShapeError: If ``axis`` falls outside ``[-ndim, ndim - 1]``.
        """
        if axis < 0:
            axis += self.ndim
        if axis < 0 or axis >= self.ndim:
            logger.error("invalid axis for flip", axis=axis, ndim=self.ndim)
            raise ShapeError(
                f"Axis {axis} is out of bounds for array of dimension {self.ndim}."
            )
        return self._wrap_result(manipulation.flip(self, axis))

    def squeeze(self, axis: Optional[AxisArg] = None) -> Self:
        """Remove axes of length one.

        Args:
            axis: Axis or axes to drop. ``None`` (default) drops every
                length-1 axis; a single ``int`` or a sequence selects specific
                axes, which must each have length 1.

        Returns:
            Self: A new array with the selected axes removed, rank reduced
            accordingly.

        Raises:
            ShapeError: If a requested axis does not have length 1, or an axis
                index is out of range.
        """
        if axis is None:
            dropped = {index for index, dim in enumerate(self.shape) if dim == 1}
        else:
            if isinstance(axis, int):
                axis = (axis,)
            dropped = set()
            for ax in axis:
                resolved = ax + self.ndim if ax < 0 else ax
                if resolved < 0 or resolved >= self.ndim:
                    logger.error(
                        "invalid axis for squeeze", axis=ax, ndim=self.ndim
                    )
                    raise ShapeError(
                        f"Axis {ax} is out of bounds for array of dimension {self.ndim}."
                    )
                if self.shape[resolved] != 1:
                    logger.error(
                        "cannot squeeze axis with size > 1",
                        axis=ax,
                        size=self.shape[resolved],
                    )
                    raise ShapeError(
                        f"Cannot squeeze axis {ax} with size {self.shape[resolved]}."
                    )
                dropped.add(resolved)

        keep = [index for index in range(self.ndim) if index not in dropped]
        return self._array_view(
            tuple(self.shape[index] for index in keep),
            tuple(self.strides[index] for index in keep),
        )

    def unsqueeze(self, axis: AxisArg) -> Self:
        """Insert new axes of length one.

        Args:
            axis: Position(s) at which to insert the new axes. Negative values
                count from the final position; valid range is
                ``[-ndim - 1, ndim]``.

        Returns:
            Self: A new array with ``len(axis)`` extra dimensions inserted.

        Raises:
            ShapeError: If an insertion position is out of range.
        """
        if isinstance(axis, int):
            axis = (axis,)
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        for ax in sorted(axis):
            if ax < 0:
                ax += len(new_shape) + 1
            if ax < 0 or ax > len(new_shape):
                logger.error(
                    "invalid axis for unsqueeze", axis=ax, current_shape=self.shape
                )
                raise ShapeError(
                    f"Axis {ax} is out of bounds for array of dimension {len(new_shape)}."
                )
            new_shape.insert(ax, 1)
            new_strides.insert(ax, 1)
        return self._array_view(tuple(new_shape), tuple(new_strides))

    def swapaxes(self, axis1: int, axis2: int) -> Self:
        """Exchange two axes.

        Args:
            axis1: First axis to swap; negative values count from the end.
            axis2: Second axis to swap.

        Returns:
            Self: A new array with the two axes exchanged.

        Raises:
            ShapeError: If either axis is out of range.
        """
        if axis1 < 0:
            axis1 += self.ndim
        if axis2 < 0:
            axis2 += self.ndim
        if axis1 < 0 or axis1 >= self.ndim or axis2 < 0 or axis2 >= self.ndim:
            logger.error(
                "invalid axes for swapaxes", axis1=axis1, axis2=axis2, ndim=self.ndim
            )
            raise ShapeError(
                f"Axes {axis1} and {axis2} are out of bounds for array of dimension {self.ndim}."
            )
        new_axes = list(range(self.ndim))
        new_axes[axis1], new_axes[axis2] = new_axes[axis2], new_axes[axis1]
        return self.permute(*new_axes)

    def moveaxis(self, pos1: int, pos2: int) -> Self:
        """Move one axis to a new position, shifting the others along.

        Args:
            pos1: Current position of the axis being moved; negative values
                count from the end.
            pos2: Position the axis should end up at, interpreted in the
                original ordering.

        Returns:
            Self: A new array with the axis relocated.

        Raises:
            ShapeError: If either position is out of range.
        """
        if pos1 < 0:
            pos1 += self.ndim
        if pos2 < 0:
            pos2 += self.ndim
        if pos1 < 0 or pos1 >= self.ndim or pos2 < 0 or pos2 >= self.ndim:
            logger.error(
                "invalid axes for moveaxis", pos1=pos1, pos2=pos2, ndim=self.ndim
            )
            raise ShapeError(
                f"Axes {pos1} and {pos2} are out of bounds for array of dimension {self.ndim}."
            )
        axes = list(range(self.ndim))
        axes.pop(pos1)
        axes.insert(pos2, pos1)
        return self.permute(*axes)

    @property
    def T(self) -> Self:
        """Transpose of the array.

        Returns:
            Self: For ``ndim == 2`` the matrix transpose; for higher ranks the
            full axis reversal. Arrays with ``ndim < 2`` are returned unchanged.
        """
        if self.ndim == 2:
            return self._wrap_result(linalg.transpose(self))
        if self.ndim < 2:
            return self
        axes = list(range(self.ndim - 1, -1, -1))
        return self._wrap_result(linalg.permute(self, axes))

    def permute(self, *axes: ShapeArg) -> Self:
        """Reorder the axes of the array.

        Args:
            *axes: New axis order, either as separate ``int`` arguments or as a
                single sequence. Must be a full permutation of
                ``range(ndim)``.

        Returns:
            Self: A new array whose ``i``-th axis is the original
            ``axes[i]``-th axis.

        Raises:
            ShapeError: If the permutation is invalid or not a permutation.
        """
        return self._wrap_result(linalg.permute(self, _as_dims(axes)))

    def transpose(self, *axes: ShapeArg) -> Self:
        """Transpose the array.

        Args:
            *axes: Optional explicit axis order, either as separate ``int``
                arguments or a single sequence. When omitted the last two axes
                are swapped (and all axes reversed for higher ranks).

        Returns:
            Self: A new array with the axes reordered; see :meth:`permute`.

        Raises:
            ShapeError: If an explicit axis order is invalid.
        """
        dims = _as_dims(axes)
        if not dims:
            dims = tuple(range(self.ndim - 1, -1, -1))
        return self.permute(*dims)

    def expand_dims(self, axis: Optional[AxisArg] = None) -> Self:
        """Insert singleton dimensions at the requested positions.

        Args:
            axis: Position(s) for the new axes. Negative values count from the
                final position; ``None`` (default) appends a single axis at the
                end.

        Returns:
            Self: A new array with the requested axes inserted.

        Raises:
            ShapeError: If an insertion position is out of range.
        """
        if axis is None:
            axis = self.ndim
        if isinstance(axis, int):
            axis = (axis,)
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        for ax in sorted(axis, reverse=True):
            if ax < 0:
                ax += len(new_shape) + 1
            if ax < 0 or ax > len(new_shape):
                raise ShapeError(
                    f"Axis {ax} is out of bounds for array of dimension {len(new_shape)}."
                )
            new_shape.insert(ax, 1)
            new_strides.insert(ax, 1)
        return self._array_view(tuple(new_shape), tuple(new_strides))

    def broadcast_to(self, shape: ShapeArg) -> Self:
        """Broadcast the array up to a larger shape.

        Leading dimensions are virtually prepended as needed, and each existing
        axis is either matched exactly or expanded from length 1.

        Args:
            shape: Target shape; must have at least ``ndim`` dimensions and be
                compatible with the current shape axis by axis.

        Returns:
            Self: A new array of the requested shape holding repeated data.

        Raises:
            ShapeError: If the target rank is smaller than ``ndim`` or an axis
                is neither equal nor ``1`` in the source.
        """
        target_shape = tuple(int(dim) for dim in shape)  # type: ignore[arg-type]
        if len(target_shape) < self.ndim:
            raise ShapeError(f"Cannot broadcast shape {self.shape} to {target_shape}")
        if any(dim < 0 for dim in target_shape):
            raise ShapeError("Broadcast dimensions must be non-negative")

        src_shape = (1,) * (len(target_shape) - self.ndim) + self.shape
        for src_dim, dst_dim in zip(src_shape, target_shape):
            if src_dim != dst_dim and src_dim != 1:
                raise ShapeError(
                    f"Cannot broadcast shape {self.shape} to {target_shape}"
                )

        src_shape = (1,) * (len(target_shape) - self.ndim) + self.shape
        strides = [0] * (len(target_shape) - self.ndim) + list(self.strides)
        for i, (src_dim, dst_dim) in enumerate(zip(src_shape, target_shape)):
            if src_dim == 1 and dst_dim != 1:
                strides[i] = 0
        return self._array_view(target_shape, strides)

    def repeat(self, repeats: Union[int, Sequence[int]], axis: Optional[int] = None) -> Self:
        """Repeat elements along an axis.

        Args:
            repeats: How many times each element is repeated. Only the first
                entry of a sequence is honoured; an ``int`` is treated as a
                1-tuple.
            axis: Axis along which to repeat. ``None`` (default) means axis 0.

        Returns:
            Self: A new array whose target axis has length
            ``shape[axis] * repeats``.

        Example:
            >>> Array([1, 2]).repeat(2)
            [1, 1, 2, 2]
        """
        if self.ndim == 0:
            return self._from_buffer(Buffer(list(self._flatten())), ())
        if axis is None:
            axis = 0
        axis = axis % self.ndim
        count = repeats if isinstance(repeats, int) else repeats[0]
        if count < 0:
            raise ShapeError("repeats must be non-negative")

        out_shape = list(self.shape)
        out_shape[axis] *= count
        out_shape_t = tuple(out_shape)
        source_values = list(self._flatten())
        values = []
        for index in range(_shape_size(out_shape_t)):
            out_coords = _coords_from_flat(index, out_shape_t)
            src_coords = list(out_coords)
            src_coords[axis] //= count
            values.append(source_values[_flat_from_coords(src_coords, self.shape)])
        return self._from_buffer(Buffer(values), out_shape_t)

    def tile(self, reps: ShapeArg) -> Self:
        """Repeat the whole array along each dimension.

        Args:
            reps: Number of copies per dimension, either an ``int`` or a
                sequence. The result has ``shape[i] * reps[i]`` along axis
                ``i``. When ``reps`` is longer than the array rank, leading
                dimensions of size one are added.

        Returns:
            Self: A new array made of repeated copies of ``self``.

        Example:
            >>> Array([[1, 2]]).tile(2)
            [[1, 2], [1, 2]]
        """
        if isinstance(reps, int):
            reps = (reps,)
        reps_t = tuple(int(rep) for rep in reps)
        if any(rep < 0 for rep in reps_t):
            raise ShapeError("tile repetitions must be non-negative")

        out_ndim = max(self.ndim, len(reps_t))
        src_shape = (1,) * (out_ndim - self.ndim) + self.shape
        rep_shape = reps_t + (1,) * (out_ndim - len(reps_t))
        out_shape = tuple(src_dim * rep for src_dim, rep in zip(src_shape, rep_shape))

        source_values = list(self._flatten())
        values = []
        for index in range(_shape_size(out_shape)):
            out_coords = _coords_from_flat(index, out_shape)
            src_coords = [
                coord % src_dim if src_dim else 0
                for coord, src_dim in zip(out_coords, src_shape)
            ]
            values.append(
                source_values[
                    _flat_from_coords(src_coords[-self.ndim :], self.shape)
                ]
                if self.ndim
                else source_values[0]
            )
        return self._from_buffer(Buffer(values), out_shape)

    def roll(self, shift: int, axis: Optional[int] = None) -> Self:
        """Cyclically shift elements along an axis.

        Args:
            shift: Number of positions to shift. Wraps around, and a shift
                larger than the axis length is reduced modulo that length.
            axis: Axis to shift along. ``None`` (default) flattens the array
                first and shifts along the resulting 1-D sequence.

        Returns:
            Self: A new array with the elements rotated in place along the axis.

        Raises:
            ShapeError: If ``axis`` is given and falls outside the array rank.
            NotImplementedError: If ``axis`` is given for an array with
                ``ndim > 1``.
        """
        if axis is None:
            flat = list(self.flatten())
            shift %= len(flat)
            return self.__class__(flat[-shift:] + flat[:-shift] if shift else flat)
        if axis < 0:
            axis += self.ndim
        if axis < 0 or axis >= self.ndim:
            raise ShapeError(
                f"Axis {axis} is out of bounds for array of dimension {self.ndim}."
            )
        if self.ndim == 1:
            values = list(self._flatten())
            shift %= len(values)
            return self._from_buffer(
                Buffer(values[-shift:] + values[:-shift] if shift else values),
                self.shape,
            )
        raise NotImplementedError("roll is only implemented for 1D arrays")

    def pad(self, padding: Any, mode: str = "constant", value: Any = 0) -> Self:
        """Pad a 2-D array with a constant.

        Args:
            padding: Either a single ``int`` applied to all four sides, or a
                ``(rows, columns)`` pair giving the total pad per vertical and
                horizontal side. A negative entry shrinks that side.
            mode: Padding strategy. Only ``"constant"`` (the default) is
                implemented; other values are currently ignored.
            value: The fill value written into the padded region.

        Returns:
            Self: A new array of shape ``(rows + 2*r, cols + 2*c)``.

        Raises:
            ShapeError: If the array is not 2-D, or ``padding`` is neither an
                ``int`` nor a 2-element pair.

        Example:
            >>> Array([[1, 2]]).pad(1, value=0)
            [[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]]
        """
        if self.ndim != 2:
            raise ShapeError("pad currently supports 2D arrays only")
        if isinstance(padding, int):
            padding = (padding, padding)
        if len(padding) != 2:
            raise ShapeError("padding must be an int or a pair")
        top_bottom, left_right = padding
        rows, cols = self.shape
        new_rows = rows + 2 * top_bottom
        new_cols = cols + 2 * left_right
        if new_rows < 0 or new_cols < 0:
            raise ShapeError("negative padding removes more than the array size")

        source_values = list(self._flatten())
        values = []
        for out_row in range(new_rows):
            src_row = out_row - top_bottom
            for out_col in range(new_cols):
                src_col = out_col - left_right
                if 0 <= src_row < rows and 0 <= src_col < cols:
                    values.append(source_values[src_row * cols + src_col])
                else:
                    values.append(value)
        return self._from_buffer(Buffer(values), (new_rows, new_cols))


__all__ = ["ArrayShapeOps"]
