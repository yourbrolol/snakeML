"""Random number generators for weight initialization.

Pure-:mod:`random` based sampling; currently provides Gaussian draws
via the Box-Muller transform.
"""

from __future__ import annotations

import math
import random

__all__ = ["normal"]


def normal(mean: float = 0.0, std: float = 1.0) -> float:
    """Generate a Gaussian random scalar using the Box-Muller transform.

    Args:
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.

    Returns:
        One sample from ``N(mean, std^2)``.

    Raises:
        ValueError: Propagated from :func:`math.log` should both uniform
            draws ever read ``0.0``.
    """
    u1 = random.random()
    u2 = random.random()

    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return mean + std * z
