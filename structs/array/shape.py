"""Shape-manipulation helpers for Array."""

from __future__ import annotations

from debug import get_logger
from debug.errors import ShapeError
from matlib import linalg, manipulation

logger = get_logger(__name__)


class ArrayShapeOps:
    """Mixin providing reshape/squeeze/transpose-like operations."""

    def reshape(self, *shape):
        """
        Returns a reshaped copy of the array.

        Examples:
            a.reshape(2, 3)
            a.reshape((2, 3))
            a.reshape(-1, 3)
        """

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        flat = list(self._flatten())
        total = len(flat)

        if shape.count(-1) > 1:
            logger.error("invalid reshape shape has multiple inferred dimensions", shape=shape)
            raise ShapeError("Only one dimension may be -1.")

        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s

            if total % known != 0:
                logger.error("cannot infer reshape size", total=total, shape=shape)
                raise ShapeError(f"Cannot reshape array of size {total} into shape {shape}")

            inferred = total // known
            shape = tuple(inferred if s == -1 else s for s in shape)

        expected = 1
        for s in shape:
            if s < 0:
                logger.error("negative reshape dimension", shape=shape)
                raise ShapeError("Dimensions must be non-negative.")
            expected *= s

        if expected != total:
            logger.error("reshape total mismatch", total=total, shape=shape, expected=expected)
            raise ShapeError(f"Cannot reshape array of size {total} into shape {shape}")

        flat_iter = iter(flat)

        def build(dim):
            if dim == len(shape):
                return next(flat_iter)
            return [build(dim + 1) for _ in range(shape[dim])]

        return self.__class__(build(0))

    def flip(self, axis):
        """Flips the array along the specified axis."""
        if axis < 0:
            axis += self.ndim
        if axis < 0 or axis >= self.ndim:
            logger.error("invalid axis for flip", axis=axis, ndim=self.ndim)
            raise ShapeError(f"Axis {axis} is out of bounds for array of dimension {self.ndim}.")
        return self._wrap_result(manipulation.flip(self, axis))

    def squeeze(self, axis=None):
        """Removes axes of size 1 from the array."""
        if axis is None:
            new_shape = tuple(dim for dim in self.shape if dim != 1)
        else:
            if isinstance(axis, int):
                axis = (axis,)
            new_shape = list(self.shape)
            for ax in sorted(axis, reverse=True):
                if new_shape[ax] != 1:
                    logger.error("cannot squeeze axis with size > 1", axis=ax, size=new_shape[ax])
                    raise ShapeError(f"Cannot squeeze axis {ax} with size {new_shape[ax]}.")
                del new_shape[ax]
            new_shape = tuple(new_shape)

        return self.reshape(new_shape)

    def unsqueeze(self, axis):
        """Adds a new axis of size 1 at the specified position."""
        if isinstance(axis, int):
            axis = (axis,)
        new_shape = list(self.shape)
        for ax in sorted(axis):
            if ax < 0:
                ax += len(new_shape) + 1
            if ax < 0 or ax > len(new_shape):
                logger.error("invalid axis for unsqueeze", axis=ax, current_shape=self.shape)
                raise ShapeError(f"Axis {ax} is out of bounds for array of dimension {len(new_shape)}.")
            new_shape.insert(ax, 1)
        return self.reshape(tuple(new_shape))

    def swapaxes(self, axis1, axis2):
        """Swaps two axes of the array."""
        if axis1 < 0:
            axis1 += self.ndim
        if axis2 < 0:
            axis2 += self.ndim
        if axis1 < 0 or axis1 >= self.ndim or axis2 < 0 or axis2 >= self.ndim:
            logger.error("invalid axes for swapaxes", axis1=axis1, axis2=axis2, ndim=self.ndim)
            raise ShapeError(f"Axes {axis1} and {axis2} are out of bounds for array of dimension {self.ndim}.")
        new_axes = list(range(self.ndim))
        new_axes[axis1], new_axes[axis2] = new_axes[axis2], new_axes[axis1]
        return self.permute(*new_axes)

    def moveaxis(self, pos1, pos2):
        """Moves an axis from position pos1 to position pos2."""
        if pos1 < 0:
            pos1 += self.ndim
        if pos2 < 0:
            pos2 += self.ndim
        if pos1 < 0 or pos1 >= self.ndim or pos2 < 0 or pos2 >= self.ndim:
            logger.error("invalid axes for moveaxis", pos1=pos1, pos2=pos2, ndim=self.ndim)
            raise ShapeError(f"Axes {pos1} and {pos2} are out of bounds for array of dimension {self.ndim}.")
        axes = list(range(self.ndim))
        axes.pop(pos1)
        axes.insert(pos2, pos1)
        return self.permute(*axes)

    @property
    def T(self):
        """Transpose: swaps last two axes (generalises to N-D via permute)."""
        if self.ndim == 2:
            return self._wrap_result(linalg.transpose(self))
        if self.ndim < 2:
            return self
        axes = list(range(self.ndim - 1, -1, -1))
        return self._wrap_result(linalg.permute(self, axes))

    def permute(self, *axes):
        """Return a view with axes permuted."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return self._wrap_result(linalg.permute(self, axes))

    def transpose(self, *axes):
        """Transpose the array. With no axes, swap the last two dimensions."""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return self.permute(*axes)

    def expand_dims(self, axis=None):
        """Insert singleton dimensions at the requested axis or at the end if None."""
        if axis is None:
            axis = self.ndim
        if isinstance(axis, int):
            axis = (axis,)
        new_shape = list(self.shape)
        for ax in sorted(axis, reverse=True):
            if ax < 0:
                ax += len(new_shape) + 1
            if ax < 0 or ax > len(new_shape):
                raise ShapeError(f"Axis {ax} is out of bounds for array of dimension {len(new_shape)}.")
            new_shape.insert(ax, 1)
        return self.reshape(tuple(new_shape))

    def broadcast_to(self, shape):
        """Broadcast the array to the requested shape."""
        target_shape = tuple(shape)
        if len(target_shape) < self.ndim:
            raise ShapeError(
                f"Cannot broadcast shape {self.shape} to {target_shape}"
            )
        # Align source shape with target shape
        src_shape = (1,) * (len(target_shape) - self.ndim) + self.shape
        # Validate broadcasting rules
        for s, t in zip(src_shape, target_shape):
            if s != t and s != 1:
                raise ShapeError(
                    f"Cannot broadcast shape {self.shape} to {target_shape}"
                )

        def _broadcast(data, src_shape, dst_shape):
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

    def repeat(self, repeats, axis=None):
        """Repeat elements along the requested axis."""
        if axis is None:
            axis = 0
        if isinstance(repeats, int):
            repeats = (repeats,)

        def _repeat(data, current_axis):
            if current_axis == axis:
                if isinstance(data, list):
                    return [item for item in data for _ in range(repeats[0])]
                return data
            if isinstance(data, list):
                return [_repeat(item, current_axis + 1) for item in data]
            return data

        return self.__class__(_repeat(self.data, 0))

    def tile(self, reps):
        """Repeat the array along each dimension."""
        if isinstance(reps, int):
            reps = (reps,)

        def _tile(data, current_dim):
            if current_dim >= len(reps):
                return data
            if isinstance(data, list):
                repeat_count = reps[current_dim]
                repeated = [item for item in data for _ in range(repeat_count)]
                return [_tile(item, current_dim + 1) for item in repeated]
            return data

        return self.__class__(_tile(self.data, 0))

    def roll(self, shift, axis=None):
        """Roll array elements along an axis."""
        if axis is None:
            flat = list(self._flatten())
            shift %= len(flat)
            return self.__class__(flat[-shift:] + flat[:-shift] if shift else flat)
        if axis < 0:
            axis += self.ndim
        if axis < 0 or axis >= self.ndim:
            raise ShapeError(f"Axis {axis} is out of bounds for array of dimension {self.ndim}.")
        if self.ndim == 1:
            values = list(self.data)
            shift %= len(values)
            return self.__class__(values[-shift:] + values[:-shift] if shift else values)
        raise NotImplementedError("roll is only implemented for 1D arrays")

    def pad(self, padding, mode="constant", value=0):
        """Pad a 2D array with a constant value."""
        if self.ndim != 2:
            raise ShapeError("pad currently supports 2D arrays only")
        if isinstance(padding, int):
            padding = (padding, padding)
        if len(padding) != 2:
            raise ShapeError("padding must be an int or a pair")
        top_bottom, left_right = padding
        new_rows = len(self.data) + top_bottom + top_bottom
        new_cols = len(self.data[0]) + left_right + left_right
        result = []
        for _ in range(top_bottom):
            result.append([value] * new_cols)
        for row in self.data:
            result.append([value] * left_right + list(row) + [value] * left_right)
        for _ in range(top_bottom):
            result.append([value] * new_cols)
        return self.__class__(result)


__all__ = ["ArrayShapeOps"]
