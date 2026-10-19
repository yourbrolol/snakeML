"""Runtime validation helpers raising :mod:`debug.errors` on failure.

Each ``ensure_*`` helper returns the validated value (possibly coerced,
e.g. shapes converted to tuples) so calls can be chained inline, and
raises the appropriate :class:`debug.errors.ValidationError` subclass
with a descriptive message otherwise.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Type, Union

from .errors import ShapeError, TypeMismatchError, ValidationError

__all__ = ["ensure_shape", "ensure_type", "validate"]


def ensure_type(
    value: Any,
    expected_type: Union[Type[Any], Tuple[Type[Any], ...]],
    *,
    name: str = "value",
    allow_none: bool = False,
) -> Any:
    """Validate that a value matches the expected Python type.

    Args:
        value: The value under inspection.
        expected_type: A single type or tuple of types accepted for
            ``value`` (passed straight to :func:`isinstance`).
        name: Label used in the error message.
        allow_none: When ``True``, ``None`` short-circuits validation.

    Returns:
        The validated ``value`` unchanged.

    Raises:
        TypeMismatchError: If ``value`` fails the ``isinstance`` check.
    """
    if allow_none and value is None:
        return value
    if isinstance(expected_type, tuple):
        if isinstance(value, expected_type):
            return value
    elif isinstance(value, expected_type):
        return value
    raise TypeMismatchError(
        f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}."
    )


def ensure_shape(
    shape: Any,
    expected_shape: Sequence[int],
    *,
    name: str = "shape",
) -> Tuple[int, ...]:
    """Validate that a shape tuple matches the expected shape.

    Args:
        shape: The observed shape; any iterable of dimension sizes or
            ``None``.
        expected_shape: The shape ``shape`` must equal.
        name: Label used in the error message.

    Returns:
        ``shape`` tuple-converted on success.

    Raises:
        ShapeError: If ``shape`` is ``None`` or differs from
            ``expected_shape``.
    """
    if shape is None:
        raise ShapeError(f"{name} cannot be None.")
    actual = tuple(shape)
    expected = tuple(expected_shape)
    if actual != expected:
        raise ShapeError(f"{name} expected {expected}, got {actual}.")
    return actual


def validate(
    value: Any,
    *,
    expected_type: Union[Type[Any], Tuple[Type[Any], ...], None] = None,
    expected_shape: Sequence[int] = None,
    name: str = "value",
) -> Any:
    """Convenience helper running both type and shape validation.

    Args:
        value: The value under inspection.
        expected_type: Optional type (or tuple of types) forwarded to
            :func:`ensure_type`.
        expected_shape: Optional shape forwarded to :func:`ensure_shape`;
            requires ``value`` to expose a ``shape`` attribute.
        name: Label used in error messages.

    Returns:
        The validated ``value`` unchanged.

    Raises:
        ValidationError: If shape validation is requested but ``value``
            has no ``shape`` attribute.
        TypeMismatchError: If ``value`` fails its type check.
        ShapeError: If ``value``'s shape mismatches ``expected_shape``.
    """
    if expected_type is not None:
        ensure_type(value, expected_type, name=name)
    if expected_shape is not None:
        if hasattr(value, "shape"):
            ensure_shape(value.shape, expected_shape, name=name)
        else:
            raise ValidationError(f"{name} has no shape attribute for validation.")
    return value
