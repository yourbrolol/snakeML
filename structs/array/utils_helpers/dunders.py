"""Dunder helpers for Array utilities."""

from __future__ import annotations

from typing import Any

from ..protocol import ArrayProtocol

__all__ = ["ArrayDunderOps"]


class ArrayDunderOps(ArrayProtocol):
    """Containment, string, and truthiness helpers."""

    def __contains__(self, item: Any) -> bool:
        """Test whether a value occurs anywhere in the array.

        Args:
            item: The value to look for, compared with ``==`` against every
                flattened element.

        Returns:
            bool: ``True`` if any element matches, else ``False``.
        """
        return item in list(self.flatten())

    def __str__(self) -> str:
        """Readable representation of the array contents.

        Returns:
            str: Same as :meth:`__repr__`.
        """
        return repr(self)

    def __bool__(self) -> bool:
        """Truthiness of the array.

        Returns:
            bool: ``True`` only when the array is non-empty *and* its first
            flattened element is truthy. Note that this is an element test,
            not a general notion of array truthiness.
        """
        return bool(self.size and self.item(0))
