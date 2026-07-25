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


__all__ = ["ArrayShapeOps"]
