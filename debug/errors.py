"""Exception hierarchy for snakeML.

All library failures derive from :class:`LibraryError` so callers can
catch a single base class. Validation failures specialize into
:class:`ValidationError` (with :class:`ShapeError` and
:class:`TypeMismatchError` subclasses), while operational failures raise
:class:`OperationError`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "LibraryError",
    "OperationError",
    "ShapeError",
    "TypeMismatchError",
    "ValidationError",
]


class LibraryError(Exception):
    """Base exception for snakeML-specific failures.

    Attributes:
        message: Human-readable description of the failure.
        context: Optional key/value metadata attached to the failure.
        details: Optional opaque payload with extra debugging info.
    """

    def __init__(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Any] = None,
    ) -> None:
        self.message = message
        self.context = context or {}
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class ValidationError(LibraryError):
    """Raised when a value fails a library-level validation check."""


class ShapeError(ValidationError):
    """Raised when tensor-like shapes are invalid or incompatible."""


class TypeMismatchError(ValidationError):
    """Raised when a value is not of the expected type."""


class OperationError(LibraryError):
    """Raised when an operation cannot be completed successfully."""
