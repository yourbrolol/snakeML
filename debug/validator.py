from .errors import ShapeError, TypeMismatchError, ValidationError


def ensure_type(value, expected_type, *, name="value", allow_none=False):
    """Validate that a value matches the expected Python type."""
    if allow_none and value is None:
        return value
    if isinstance(expected_type, tuple):
        if isinstance(value, expected_type):
            return value
    elif isinstance(value, expected_type):
        return value
    raise TypeMismatchError(f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}.")


def ensure_shape(shape, expected_shape, *, name="shape"):
    """Validate that a shape tuple matches the expected shape."""
    if shape is None:
        raise ShapeError(f"{name} cannot be None.")
    actual = tuple(shape)
    expected = tuple(expected_shape)
    if actual != expected:
        raise ShapeError(f"{name} expected {expected}, got {actual}.")
    return actual


def validate(value, *, expected_type=None, expected_shape=None, name="value"):
    """Convenience helper that runs both type and shape validation when requested."""
    if expected_type is not None:
        ensure_type(value, expected_type, name=name)
    if expected_shape is not None:
        if hasattr(value, "shape"):
            ensure_shape(value.shape, expected_shape, name=name)
        else:
            raise ValidationError(f"{name} has no shape attribute for validation.")
    return value
