from copy import deepcopy
import os
from pathlib import Path

LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}

_DEFAULTS = {
    "enabled": True,
    "level": "info",
    "stream": None,
    "include_module": True,
    "include_context": True,
    "prefix": "snakeML",
    "save": False,
    "log_file": None,
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


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    return v in ("1", "true", "yes", "on")


def configure_from_env(path=".env", prefix="SNAKEML_"):
    """Load configuration from environment and an optional .env file.

    Environment variables with the given `prefix` (default `SNAKEML_`) will
    be mapped to debug config keys. Example: `SNAKEML_LEVEL=debug`.

    The .env file (if present) will be parsed for simple KEY=VALUE lines and
    will not override existing environment variables.
    """
    env = dict(os.environ)
    p = Path(path)
    if p.exists():
        try:
            for raw in p.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                # do not override existing os.environ
                env.setdefault(k, v)
        except Exception:
            # if .env parsing fails, silently ignore and continue with os.env
            pass

    updates = {}
    for k, v in env.items():
        if not k.startswith(prefix):
            continue
        key = k[len(prefix) :].lower()
        if key not in CONFIG:
            continue
        # interpret booleans for common flags
        if key in ("enabled", "save"):
            updates[key] = _parse_bool(v)
        else:
            updates[key] = v

    if updates:
        configure(**updates)

    return get_config()
