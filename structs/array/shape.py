"""Shape-manipulation helpers for Array."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union, cast

from debug import get_logger
from debug.errors import ShapeError
from matlib import linalg, manipulation
from structs.dtypes import AxisArg, NestedList, Self, ShapeArg

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


class ArrayShapeOps(ArrayProtocol):
    """Mixin providing reshape/squeeze/transpose-like operations."""

    def reshape(self, *shape: ShapeArg) -> Self:
        """Return a reshaped copy of the array.

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

        flat = list(self.flatten())
        total = len(flat)

        if dims.count(-1) > 1:
            logger.error(
                "invalid reshape shape has multiple inferred dimensions", shape=dims
            )
            raise ShapeError("Only one dimension may be -1.")

        if -1 in dims:
            known = 1
            for s in dims:
                if s != -1:
                    known *= s

            if total % known != 0:
                logger.error("cannot infer reshape size", total=total, shape=dims)
                raise ShapeError(
                    f"Cannot reshape array of size {total} into shape {dims}"
                )

            inferred = total // known
            shape = tuple(inferred if s == -1 else s for s in dims)

        expected = 1
        for s in dims:
            if s < 0:
                logger.error("negative reshape dimension", shape=dims)
                raise ShapeError("Dimensions must be non-negative.")
            expected *= s

        if expected != total:
            logger.error(
                "reshape total mismatch", total=total, shape=dims, expected=expected
            )
            raise ShapeError(f"Cannot reshape array of size {total} into shape {dims}")

        flat_iter = iter(flat)

        def build(dim: int) -> Any:
            if dim == len(dims):
                return next(flat_iter)
            return [build(dim + 1) for _ in range(dims[dim])]

        return self.__class__(build(0))

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
        new_shape: list[int]
        if axis is None:
            new_shape = [dim for dim in self.shape if dim != 1]
        else:
            if isinstance(axis, int):
                axis = (axis,)
            new_shape = list(self.shape)
            for ax in sorted(axis, reverse=True):
                if new_shape[ax] != 1:
                    logger.error(
                        "cannot squeeze axis with size > 1", axis=ax, size=new_shape[ax]
                    )
                    raise ShapeError(
                        f"Cannot squeeze axis {ax} with size {new_shape[ax]}."
                    )
                del new_shape[ax]

        return self.reshape(tuple(new_shape))

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
        return self.reshape(tuple(new_shape))

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
        for ax in sorted(axis, reverse=True):
            if ax < 0:
                ax += len(new_shape) + 1
            if ax < 0 or ax > len(new_shape):
                raise ShapeError(
                    f"Axis {ax} is out of bounds for array of dimension {len(new_shape)}."
                )
            new_shape.insert(ax, 1)
        return self.reshape(tuple(new_shape))

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
        target_shape = tuple(shape)  # type: ignore[arg-type]
        if len(target_shape) < self.ndim:
            raise ShapeError(f"Cannot broadcast shape {self.shape} to {target_shape}")
        # Align source shape with target shape
        src_shape = (1,) * (len(target_shape) - self.ndim) + self.shape
        # Validate broadcasting rules
        for s, t in zip(src_shape, target_shape):
            if s != t and s != 1:
                raise ShapeError(
                    f"Cannot broadcast shape {self.shape} to {target_shape}"
                )

        def _broadcast(
            data: Any, src_shape: tuple[int, ...], dst_shape: tuple[int, ...]
        ) -> NestedList:
            if not dst_shape:
                return data
            src_dim = src_shape[0]
            dst_dim = dst_shape[0]
            # Virtual leading dimension
            if len(src_shape) > len(self.shape):
                return [
                    _broadcast(data, src_shape[1:], dst_shape[1:])
                    for _ in range(dst_dim)
                ]
            # Scalar
            if not isinstance(data, list):
                return [
                    _broadcast(data, src_shape[1:], dst_shape[1:])
                    for _ in range(dst_dim)
                ]
            # Broadcast singleton dimension
            if src_dim == 1:
                elem = data[0] if data else 0
                return [
                    _broadcast(elem, src_shape[1:], dst_shape[1:])
                    for _ in range(dst_dim)
                ]
            # Shapes are equal (validated already)
            return [
                _broadcast(data[i], src_shape[1:], dst_shape[1:])
                for i in range(src_dim)
            ]

        return self.__class__(_broadcast(self.data, src_shape, target_shape))

    def repeat(self, repeats: Union[int, Sequence[int]], axis: Optional[int] = None) -> Self:
        """Repeat elements along an axis.

        .. warning::
           ``ArrayConstructors.repeat(value, shape)`` shadows this method on
           the composed ``Array`` class, so on an ``Array`` instance this
           body is currently unreachable and ``arr.repeat(2)`` calls the
           constructor factory instead. See the note on ``tile`` below.

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
        if axis is None:
            axis = 0
        if isinstance(repeats, int):
            repeats = (repeats,)

        def _repeat(data: Any, current_axis: int) -> NestedList:
            if current_axis == axis:
                if isinstance(data, list):
                    return [item for item in data for _ in range(repeats[0])]
                return data
            if isinstance(data, list):
                return [_repeat(item, current_axis + 1) for item in data]
            return data

        return self.__class__(_repeat(self.data, 0))

    def tile(self, reps: ShapeArg) -> Self:
        """Repeat the whole array along each dimension.

        .. warning::
           ``ArrayConstructors.tile(value, reps)`` shadows this method on the
           composed ``Array`` class (it comes first in the MRO), so this body
           is currently unreachable on an ``Array`` instance. One of the two
           pairs needs renaming.

        Args:
            reps: Number of copies per dimension, either an ``int`` or a
                sequence. The result has ``shape[i] * reps[i]`` along axis ``i``;
                dimensions beyond ``len(reps)`` are left alone.

        Returns:
            Self: A new array made of repeated copies of ``self``.

        Example:
            >>> Array([[1, 2]]).tile(2)
            [[1, 2], [1, 2]]
        """
        if isinstance(reps, int):
            reps = (reps,)

        def _tile(data: Any, current_dim: int) -> NestedList:
            if current_dim >= len(reps):
                return data
            if isinstance(data, list):
                repeat_count = reps[current_dim]  # type: ignore[index]
                repeated = [item for item in data for _ in range(repeat_count)]
                return [_tile(item, current_dim + 1) for item in repeated]
            return data

        return self.__class__(_tile(self.data, 0))

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
            values = list(cast("List[Any]", self.data))
            shift %= len(values)
            return self.__class__(
                values[-shift:] + values[:-shift] if shift else values
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
        new_rows = len(self.data) + top_bottom + top_bottom  # type: ignore[arg-type]
        new_cols = len(self.data[0]) + left_right + left_right  # type: ignore[index,arg-type]
        result: List[List[Any]] = []
        for _ in range(top_bottom):
            result.append([value] * new_cols)
        for row in cast("List[Any]", self.data):
            result.append([value] * left_right + list(row) + [value] * left_right)
        for _ in range(top_bottom):
            result.append([value] * new_cols)
        return self.__class__(result)


__all__ = ["ArrayShapeOps"]