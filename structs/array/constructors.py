"""Constructor-oriented helpers for Array."""

from __future__ import annotations

from debug import get_logger
from debug.errors import ValidationError
from random import random, randint

logger = get_logger(__name__)


class ArrayConstructors:
    """Mixin providing constructor-style Array methods."""

    @classmethod
    def full(cls, shape, value=0):
        """Creates an Array of the given shape filled with a specified value."""
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for full", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")

        def build(dim):
            if dim == len(shape):
                return value if not callable(value) else value()
            return [build(dim + 1) for _ in range(shape[dim])]

        return cls(build(0))
    
    @classmethod
    def eye(cls, size, k=0):
        """Creates an identity matrix of the given size."""
        if not isinstance(size, int) or size <= 0:
            logger.error("invalid size for eye", size=size)
            raise ValidationError("Size must be a positive integer.")
        
        result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

        return cls(result)

    @classmethod
    def zeros(cls, shape):
        """Creates an Array of the given shape filled with zeros."""
        return cls.full(shape, fill_value=0)

    @classmethod
    def ones(cls, shape):
        """Creates an Array of the given shape filled with ones."""
        return cls.full(shape, fill_value=1)

    @classmethod
    def empty(cls, shape):
        """Creates an Array of the given shape without initializing values."""
        return cls.full(shape, fill_value=None)

    @classmethod
    def randn(cls, shape):
        """Creates an Array of the given shape filled with random floats in [0, 1)."""

        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for randn", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")

        def build(dim):
            if dim == len(shape):
                return random()
            return [build(dim + 1) for _ in range(shape[dim])]

        return cls(build(0))

    @classmethod
    def randint(cls, shape, low=0, high=10):
        """Creates an Array of the given shape filled with random integers."""

        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            logger.error("invalid shape type for randint", shape=shape)
            raise ValidationError("Shape must be an int, tuple, or list.")

        def build(dim):
            if dim == len(shape):
                return randint(low, high - 1)
            return [build(dim + 1) for _ in range(shape[dim])]

        return cls(build(0))

    @classmethod
    def repeat(cls, value, shape):
        """Creates an Array by repeating a given value to fill the specified shape (alias to full())."""
        return cls.full(shape, fill_value=value)

    @classmethod
    def arange(cls, start, stop=None, step=1):
        """Creates a 1D Array with evenly spaced values within a given interval."""
        if stop is None:
            stop = start
            start = 0
        if step == 0:
            logger.error("step cannot be zero in arange", start=start, stop=stop)
            raise ValidationError("Step must be non-zero.")
        return cls(list(range(start, stop, step)))

    @classmethod
    def linspace(cls, start, stop, num=50):
        """Creates a 1D Array with evenly spaced values over a specified interval."""
        if num <= 0:
            logger.error("num must be positive in linspace", num=num)
            raise ValidationError("Number of samples must be positive.")
        if num == 1:
            return cls([start])
        step = (stop - start) / (num - 1)
        return cls([start + i * step for i in range(num)])

    @classmethod
    def tile(cls, value, reps):
        """Creates an Array by repeating a given value along specified dimensions (alias to full)."""
        if isinstance(reps, int):
            reps = (reps,)
        if not isinstance(reps, (tuple, list)):
            logger.error("invalid reps type for tile", reps=reps)
            raise ValidationError("Repetitions must be an int, tuple, or list.")

        return cls.full(reps, value)


__all__ = ["ArrayConstructors"]
