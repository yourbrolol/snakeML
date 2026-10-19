"""Parameter initializers producing :class:`~structs.Array` weights.

Every initializer is a callable object: ``__call__`` accepts a shape
(plus distribution parameters where relevant) and returns freshly
initialized weights. Collectively they cover constant fills and the
standard Xavier/Glorot and Kaiming schemes.
"""

from __future__ import annotations

import random
from math import sqrt
from typing import Any, Callable, Dict, Optional, Sequence

from matlib.rng import normal
from structs import Array

__all__ = [
    "Constant",
    "Initializer",
    "KaimingNormal",
    "KaimingUniform",
    "Normal",
    "Ones",
    "Uniform",
    "XavierNormal",
    "XavierUniform",
    "Zeros",
]


class Initializer:
    """Base parameter initializer class.

    Attributes:
        params: Hook for storing initializer-local state (unused here).
    """

    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}

    def __call__(self, shape: Sequence[int]) -> Array:
        raise NotImplementedError


class Constant(Initializer):
    """Constant parameter initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, value: float, shape: Sequence[int]) -> Array:
        """Return an array of the given shape filled with ``value``.

        Args:
            value: The constant fill value (callers pass ``0`` for zeros).
            shape: Target shape.

        Returns:
            An ``Array`` of ``shape`` filled with ``value``.
        """
        return Array.full(shape=shape, value=value)


class Zeros(Initializer):
    """Zero-valued parameter initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int]) -> Array:
        """Return an array of the given shape filled with zeros.

        Args:
            shape: Target shape.

        Returns:
            An ``Array`` of ``shape`` filled with ``0``.
        """
        return Array.zeros(shape=shape)


class Ones(Initializer):
    """One-valued parameter initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int]) -> Array:
        """Return an array of the given shape filled with ones.

        Args:
            shape: Target shape.

        Returns:
            An ``Array`` of ``shape`` filled with ``1``.
        """
        return Array.ones(shape=shape)


class Normal(Initializer):
    """Random normal initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int], mean: float = 0.0, std: float = 1.0) -> Array:
        """Return an array sampled from a normal distribution.

        Args:
            shape: Target shape.
            mean: Distribution mean.
            std: Distribution standard deviation.

        Returns:
            An ``Array`` of ``shape`` with ``N(mean, std^2)`` elements.
        """
        return Array.full(shape, lambda: normal(mean, std))


class Uniform(Initializer):
    """Random uniform initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int], low: float = 0.0, high: float = 1.0) -> Array:
        """Return an array sampled uniformly from ``[low, high)``.

        Args:
            shape: Target shape.
            low: Inclusive lower bound.
            high: Exclusive upper bound.

        Returns:
            An ``Array`` of ``shape`` with uniform samples.
        """
        return Array.full(shape, lambda: random.uniform(low, high))


class XavierNormal(Initializer):
    """Xavier/Glorot normal parameter initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int], fan_in: int, fan_out: int) -> Array:
        """Return an array sampled from a normal distribution with variance 2/(fan_in + fan_out).

        Args:
            shape: Target shape.
            fan_in: Number of inputs feeding each unit.
            fan_out: Number of outputs each unit feeds.

        Returns:
            An ``Array`` with ``N(0, 2/(fan_in + fan_out))`` elements.
        """
        sigma = sqrt(2 / (fan_in + fan_out))
        return Array.full(shape, lambda: normal(0, sigma))


class XavierUniform(Initializer):
    """Xavier/Glorot uniform parameter initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int], fan_in: int, fan_out: int) -> Array:
        """Return an array sampled uniformly from ``[-limit, limit]``.

        ``limit = sqrt(6 / (fan_in + fan_out))``.

        Args:
            shape: Target shape.
            fan_in: Number of inputs feeding each unit.
            fan_out: Number of outputs each unit feeds.

        Returns:
            An ``Array`` with Glorot-uniform elements.
        """
        limit = sqrt(6 / (fan_in + fan_out))
        return Array.full(shape, lambda: random.uniform(-limit, limit))


class KaimingNormal(Initializer):
    """Kaiming normal parameter initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int], fan_in: int) -> Array:
        """Return an array sampled from a normal distribution with std sqrt(2/fan_in).

        Args:
            shape: Target shape.
            fan_in: Number of inputs feeding each unit (He's variance scale).

        Returns:
            An ``Array`` with ``N(0, 2/fan_in)`` elements.
        """
        sigma = sqrt(2 / fan_in)
        return Array.full(shape, lambda: normal(0, sigma))


class KaimingUniform(Initializer):
    """Kaiming uniform parameter initializer."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, shape: Sequence[int], fan_in: int) -> Array:
        """Return an array sampled uniformly from ``[-limit, limit]``.

        ``limit = sqrt(6 / fan_in)``.

        Args:
            shape: Target shape.
            fan_in: Number of inputs feeding each unit (He's bound scale).

        Returns:
            An ``Array`` with Kaiming-uniform elements.
        """
        limit = sqrt(6 / fan_in)
        return Array.full(shape, lambda: random.uniform(-limit, limit))


if __name__ == "__main__":
    print(XavierNormal()([3, 3], 12, 12))
