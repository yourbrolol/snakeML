"""Dispatch public math calls to the implementation for their operand type."""

from __future__ import annotations

import importlib
import pkgutil
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Optional, Tuple

_BASE_MODULES = ("_gen", "_list", "_array")
_GENERAL_MODULE = "_gen"
_DEFAULT_MODULE = "_list"

_modules: dict[str, ModuleType] = {}
_extra_modules_loaded = False


def _load_module(name: str) -> ModuleType:
    """Import and cache one matlib implementation package."""
    module = importlib.import_module(f".{name}", __package__)
    _modules[name] = module
    return module


for _base_name in _BASE_MODULES:
    _load_module(_base_name)


def _load_extra_modules() -> None:
    """Import additional implementation packages once they are needed."""
    global _extra_modules_loaded
    if _extra_modules_loaded:
        return

    _extra_modules_loaded = True
    package = importlib.import_module(__package__)
    for info in pkgutil.iter_modules(package.__path__):
        if not info.ispkg or info.name in _modules:
            continue
        module = importlib.import_module(f".{info.name}", __package__)
        if getattr(module, "supported", None):
            _modules[info.name] = module


def _matches(value: Any, supported: Tuple[str, ...]) -> bool:
    """Return whether a value's type or one of its bases is supported."""
    return any(base.__name__ in supported for base in type(value).__mro__)


def _known_match(value: Any) -> bool:
    """Return whether any loaded non-general module supports a value."""
    for name, module in _modules.items():
        if name == _GENERAL_MODULE:
            continue
        supported = tuple(getattr(module, "supported", ()))
        if supported and _matches(value, supported):
            return True
    return False


def _matching_modules(values: Tuple[Any, ...]) -> list[ModuleType]:
    """Return implementation modules matching any of the given values."""
    matched: list[ModuleType] = []
    for name, module in _modules.items():
        if name == _GENERAL_MODULE:
            continue
        supported = tuple(getattr(module, "supported", ()))
        if supported and any(_matches(value, supported) for value in values):
            matched.append(module)
    return matched


def _select_module(values: Tuple[Any, ...], default: str) -> ModuleType:
    """Select one implementation module for the supplied arguments."""
    if not _extra_modules_loaded and any(
        not _known_match(value) for value in values
    ):
        _load_extra_modules()

    matched = _matching_modules(values)
    if len(matched) > 1:
        names = ", ".join(module.__name__ for module in matched)
        raise TypeError(f"operands must share one implementation, got {names}")
    if matched:
        return matched[0]
    return _modules.get(default, _modules[_GENERAL_MODULE])


def route(
    func: Optional[Callable[..., Any]] = None,
    *,
    default: str = _DEFAULT_MODULE,
) -> Callable[..., Any]:
    """Wrap a protocol function so it receives the selected module first."""

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            values = tuple(args) + tuple(kwargs.values())
            module = _select_module(values, default)
            return function(module, *args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


__all__ = ["route"]
