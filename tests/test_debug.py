import pytest

from debug import (
    clear_context,
    configure,
    current_context,
    describe,
    ensure_shape,
    ensure_type,
    get_logger,
    is_enabled,
    logger,
    operation_context,
    set_level,
    update_context,
)
from debug.errors import ShapeError, ValidationError


def test_logger_emits_formatted_messages(capsys):
    configure(enabled=True, level="info")
    logger.info("array initialized")

    captured = capsys.readouterr()
    assert "snakeML" in captured.out
    assert "array initialized" in captured.out


def test_operation_context_tracks_current_scope():
    with operation_context("array.reshape", shape=(2, 2), ndim=2):
        ctx = current_context()
        assert ctx["operation"] == "array.reshape"
        assert ctx["shape"] == (2, 2)

    assert current_context() == {}


def test_update_context_merges_metadata():
    clear_context()
    update_context(stage="init", layer="linear")

    assert current_context() == {"stage": "init", "layer": "linear"}
    clear_context()


def test_set_level_changes_verbosity():
    configure(enabled=True, level="warning")
    assert not is_enabled("info")
    assert is_enabled("error")


def test_logger_includes_extra_kwargs_as_context(capsys):
    configure(enabled=True, level="debug")
    module_logger = get_logger("debug.module")
    module_logger.info("hello", component="layer", step=2)

    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert "{component=layer, step=2}" in captured.out


def test_ensure_shape_raises_library_error():
    with pytest.raises(ShapeError):
        ensure_shape((2, 3), (2, 2), name="shape")


def test_ensure_type_raises_validation_error():
    with pytest.raises(ValidationError):
        ensure_type("bad", int, name="value")


def test_describe_handles_array_like_objects():
    class FakeArray:
        def __init__(self):
            self.shape = (2, 2)
            self.ndim = 2
            self.data = [[1, 2], [3, 4]]

    payload = describe(FakeArray())
    assert "FakeArray" in payload
    assert "shape=(2, 2)" in payload
