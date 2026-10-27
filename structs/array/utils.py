"""Composite utility mixin for Array.

The former single-file ``ArrayUtilsOps`` implementation is split by concern
across :mod:`utils_helpers.buffer_core`, :mod:`utils_helpers.buffer_transform`,
:mod:`utils_helpers.buffer_select`, :mod:`utils_helpers.buffer_combine`, and
:mod:`utils_helpers.dunders`. This module preserves the public
``ArrayUtilsOps`` import path used by the rest of the package.
"""

from __future__ import annotations

from .utils_helpers import (
    ArrayBufferOps,
    ArrayCombineOps,
    ArrayDunderOps,
    ArraySelectOps,
    ArrayTransformOps,
)

__all__ = ["ArrayUtilsOps"]


class ArrayUtilsOps(
    ArrayBufferOps,
    ArrayTransformOps,
    ArraySelectOps,
    ArrayCombineOps,
    ArrayDunderOps,
):
    """Composite mixin preserving the historical ``ArrayUtilsOps`` API."""

    pass
