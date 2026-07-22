class LibraryError(Exception):
    """Base exception for snakeML-specific failures."""

    def __init__(self, message, *, context=None, details=None):
        self.message = message
        self.context = context or {}
        self.details = details
        super().__init__(message)

    def __str__(self):
        return self.message


class ValidationError(LibraryError):
    """Raised when a value fails a library-level validation check."""


class ShapeError(ValidationError):
    """Raised when tensor-like shapes are invalid or incompatible."""


class TypeMismatchError(ValidationError):
    """Raised when a value is not of the expected type."""


class OperationError(LibraryError):
    """Raised when an operation cannot be completed successfully."""
