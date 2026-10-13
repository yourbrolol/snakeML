"""Constructor-oriented helpers for Array.

Every method here is a ``classmethod`` factory returning a new ``Array``.
"""

from __future__ import annotations

from random import randint, random
from typing import Any, Optional, Union

from debug import get_logger
from debug.errors import ValidationError
from structs.dtypes import FillValue, Self, ShapeArg

from .protocol import ArrayProtocol


logger = get_logger(__name__)

__all__ = ["ArrayConstructors"]


class ArrayConstructors(ArrayProtocol):
    """Mixin providing constructor-style Array methods."""

    @classmethod
    def full(cls, shape: ShapeArg, value: FillValue = 0) -> Self:
        """Creates an Array of the given shape filled with a specified value.

        Args:
            shape: Target shape. A bare ``int`` is treated as 1-D of that
                length; otherwise a sequence of per-dimension sizes is required.
            value: A scalar written to every position, or a zero-argument
                callable invoked once per element for distinct values.

        Returns:
            ArrayT: A new array of class ``cls`` with the requested shape.

        Raises:
            ValidationError: If ``shape`` is neither an ``int`` nor a
                ``tuple``/``list``.

        Example:
            >>> Array.full((2, 2), 7)
            [[7, 7], [7, 7]]
        """
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for full", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")

        def build(dim: int) -> Any:
            if dim == len(shape):
                return value if not callable(value) else value()
            return [build(dim + 1) for _ in range(shape[dim])]

        return cls(build(0))

    @classmethod
    def eye(cls, size: int, k: int = 0) -> Self:
        """Creates an identity matrix of the given size.

        Args:
            size: Number of rows and columns; must be a positive integer.
            k: Diagonal offset. ``0`` (default) is the main diagonal, ``1`` the
                superdiagonal, ``-1`` the subdiagonal.

        Returns:
            ArrayT: A ``size`` x ``size`` array with ones on the chosen
            diagonal and zeros elsewhere.

        Raises:
            ValidationError: If ``size`` is not a positive integer.

        Example:
            >>> Array.eye(2)
            [[1, 0], [0, 1]]
        """
        if not isinstance(size, int) or size <= 0:
            logger.error("invalid size for eye", size=size)
            raise ValidationError("Size must be a positive integer.")

        result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

        return cls(result)

    @classmethod
    def zeros(cls, shape: ShapeArg) -> Self:
        """Creates an Array of the given shape filled with zeros.

        Args:
            shape: Target shape; an ``int`` means 1-D.

        Returns:
            ArrayT: A new zero-filled array of class ``cls``.

        Raises:
            ValidationError: If ``shape`` has an unsupported type.
        """
        return cls.full(shape, value=0)

    @classmethod
    def ones(cls, shape: ShapeArg) -> Self:
        """Creates an Array of the given shape filled with ones.

        Args:
            shape: Target shape; an ``int`` means 1-D.

        Returns:
            ArrayT: A new one-filled array of class ``cls``.

        Raises:
            ValidationError: If ``shape`` has an unsupported type.
        """
        return cls.full(shape, value=1)

    @classmethod
    def empty(cls, shape: ShapeArg) -> Self:
        """Creates an Array of the given shape without initializing values.

        Args:
            shape: Target shape; an ``int`` means 1-D.

        Returns:
            ArrayT: A new array of class ``cls`` whose every element is
            ``None`` (no allocation or zeroing is performed).

        Raises:
            ValidationError: If ``shape`` has an unsupported type.
        """
        return cls.full(shape, value=None)

    @classmethod
    def randn(cls, shape: ShapeArg) -> Self:
        """Creates an Array of the given shape filled with random floats in [0, 1).

        Args:
            shape: Target shape; an ``int`` means 1-D.

        Returns:
            ArrayT: A new array of class ``cls`` with independent uniform
            samples in ``[0, 1)`` at every position.

        Raises:
            ValidationError: If ``shape`` is neither an ``int`` nor a
                ``tuple``/``list``.
        """
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for randn", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")

        def build(dim: int) -> Any:
            if dim == len(shape):
                return random()
            return [build(dim + 1) for _ in range(shape[dim])]

        return cls(build(0))

    @classmethod
    def randint(
        cls, shape: ShapeArg, low: int = 0, high: int = 10
    ) -> Self:
        """Creates an Array of the given shape filled with random integers.

        Args:
            shape: Target shape; an ``int`` means 1-D.
            low: Inclusive lower bound of the sample range.
            high: Exclusive upper bound; samples land in ``[low, high - 1]``.

        Returns:
            ArrayT: A new array of class ``cls`` with independent integer
            samples at every position.

        Raises:
            ValidationError: If ``shape`` is neither an ``int`` nor a
                ``tuple``/``list``.
        """
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for randint", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")

        def build(dim: int) -> Any:
            if dim == len(shape):
                return randint(low, high - 1)
            return [build(dim + 1) for _ in range(shape[dim])]

        return cls(build(0))

    @classmethod
    def repeat(cls, value: FillValue, shape: ShapeArg) -> Self:
        """Creates an Array by repeating a value to fill the shape (alias to ``full``).

        Args:
            value: Scalar, or a callable producing one value per element.
            shape: Target shape; an ``int`` means 1-D.

        Returns:
            ArrayT: A new array of class ``cls``.

        Raises:
            ValidationError: If ``shape`` has an unsupported type.
        """
        return cls.full(shape, value=value)

    @classmethod
    def arange(
        cls,
        start: int,
        stop: Optional[int] = None,
        step: int = 1,
    ) -> Self:
        """Creates a 1D Array with evenly spaced values within a given interval.

        Args:
            start: First value, or the exclusive ``stop`` when ``stop`` is
                omitted (in which case the range starts at ``0``).
            stop: Exclusive upper bound. ``None`` (default) makes ``start`` the
                stop value instead.
            step: Spacing between consecutive values; must be non-zero. A
                negative step requires ``start > stop``.

        Returns:
            ArrayT: A new 1-D array of class ``cls``. Empty when the range
            cannot be satisfied, e.g. ``arange(5, 0)``.

        Raises:
            ValidationError: If ``step`` is ``0``.

        Example:
            >>> Array.arange(0, 4)
            [0, 1, 2, 3]
        """
        if stop is None:
            stop = start
            start = 0
        if step == 0:
            logger.error("step cannot be zero in arange", start=start, stop=stop)
            raise ValidationError("Step must be non-zero.")
        return cls(list(range(start, stop, step)))

    @classmethod
    def linspace(
        cls,
        start: float,
        stop: float,
        num: int = 50,
    ) -> Self:
        """Creates a 1D Array with evenly spaced values over a specified interval.

        Both endpoints are always included, so the result has exactly ``num``
        entries regardless of the span.

        Args:
            start: Value of the first element.
            stop: Value of the last element.
            num: Number of samples to generate; must be positive.

        Returns:
            ArrayT: A new 1-D array of class ``cls`` with ``num`` floats.

        Raises:
            ValidationError: If ``num`` is not positive.
        """
        if num <= 0:
            logger.error("num must be positive in linspace", num=num)
            raise ValidationError("Number of samples must be positive.")
        if num == 1:
            return cls([start])
        step = (stop - start) / (num - 1)
        return cls([start + i * step for i in range(num)])

    @classmethod
    def tile(cls, value: FillValue, reps: ShapeArg) -> Self:
        """Creates an Array by repeating a value along the specified dimensions.

        Args:
            value: Scalar, or a callable producing one value per element.
            reps: Target shape; an ``int`` means 1-D. Longer sequences define
                higher-rank output.

        Returns:
            ArrayT: A new array of class ``cls``.

        Raises:
            ValidationError: If ``reps`` is neither an ``int`` nor a
                ``tuple``/``list``.
        """
        if isinstance(reps, int):
            reps = (reps,)
        if not isinstance(reps, (tuple, list)):
            logger.error("invalid reps type for tile", reps=reps)
            raise ValidationError("Repetitions must be an int, tuple, or list.")

        return cls.full(reps, value)


__all__ = ["ArrayConstructors"]