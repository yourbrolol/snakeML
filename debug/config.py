from copy import deepcopy

LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}

_DEFAULTS = {
    "enabled": True,
    "level": "debug",
    "stream": None,
    "include_module": True,
    "include_context": True,
    "prefix": "snakeML",
}

CONFIG = deepcopy(_DEFAULTS)


def _normalize_level(level):
    if isinstance(level, str):
        normalized = LEVELS.get(level.lower())
        if normalized is None:
            raise ValueError(f"Unknown log level: {level}")
        return normalized
    if isinstance(level, int):
        return level
    raise ValueError(f"Unsupported log level type: {type(level).__name__}")


def configure(**overrides):
    """Update the global debug configuration."""
    for key, value in overrides.items():
        if key not in CONFIG:
            raise ValueError(f"Unsupported debug config option: {key}")
        if key == "level":
            _normalize_level(value)
        CONFIG[key] = value
    return get_config()


def get_config():
    """Return a copy of the active debug configuration."""
    return deepcopy(CONFIG)


def reset_config():
    """Restore the library debug defaults."""
    CONFIG.clear()
    CONFIG.update(deepcopy(_DEFAULTS))
    return get_config()


def set_level(level):
    """Set the minimum verbosity level for debug output."""
    _normalize_level(level)
    return configure(level=level)


def is_enabled(level_name="info"):
    """Return whether a message at the provided level should be emitted."""
    if not CONFIG["enabled"]:
        return False
    requested_level = _normalize_level(level_name)
    current_level = _normalize_level(CONFIG["level"])
    return requested_level >= current_level
