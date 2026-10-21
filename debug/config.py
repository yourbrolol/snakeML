"""Module-level configuration for :mod:`debug`.

A single process-wide ``CONFIG`` mapping drives every debug helper: which
messages are emitted, where they go and whether they are persisted to disk.
The public functions here are the only supported way of touching ``CONFIG``;
they are annotated so signatures reveal exactly what is accepted.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

#: Named log levels mapped to their numeric severities (mirrors :mod:`logging`).
LEVELS: Dict[str, int] = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}

#: The factory-reset configuration every process starts with.
_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "level": "info",
    "stream": None,
    "include_module": True,
    "include_context": True,
    "prefix": "snakeML",
    "save": False,
    "log_file": None,
}

#: The mutable, process-wide configuration snapshot.
CONFIG: Dict[str, Any] = deepcopy(_DEFAULTS)


def _normalize_level(level: Union[str, int]) -> int:
    """Coerce a user-supplied level to its numeric severity.

    Args:
        level: A level name (``"debug"``...) or an integer severity.

    Returns:
        The numeric severity understood by :func:`is_enabled`.

    Raises:
        ValueError: If ``level`` is not a name in :data:`LEVELS` nor an ``int``.
    """
    if isinstance(level, str):
        normalized: Optional[int] = LEVELS.get(level.lower())
        if normalized is None:
            raise ValueError(f"Unknown log level: {level}")
        return normalized
    if isinstance(level, int):
        return level
    raise ValueError(f"Unsupported log level type: {type(level).__name__}")


def configure(**overrides: Any) -> Dict[str, Any]:
    """Update the global debug configuration.

    Args:
        **overrides: Keyword arguments matching keys of ``_DEFAULTS``.
            ``level`` may be a name or integer, but each value must itself
            accept the new key.

    Returns:
        A copy of the post-update configuration.

    Raises:
        ValueError: If an unsupported option key is passed.
    """
    for key, value in overrides.items():
        if key not in CONFIG:
            raise ValueError(f"Unsupported debug config option: {key}")
        if key == "level":
            _normalize_level(value)
        CONFIG[key] = value
    return get_config()


def get_config() -> Dict[str, Any]:
    """Return a copy of the active debug configuration."""
    return deepcopy(CONFIG)


def reset_config() -> Dict[str, Any]:
    """Restore the library debug defaults.

    Returns:
        A copy of the restored configuration.
    """
    CONFIG.clear()
    CONFIG.update(deepcopy(_DEFAULTS))
    return get_config()


def set_level(level: Union[str, int]) -> Dict[str, Any]:
    """Set the minimum verbosity level for debug output.

    Args:
        level: A level name (``"warning"``) or numeric severity (``30``).

    Returns:
        A copy of the post-update configuration.
    """
    _normalize_level(level)
    return configure(level=level)


def is_enabled(level_name: Union[str, int] = "info") -> bool:
    """Return whether a message at the provided level should be emitted.

    Args:
        level_name: The level of the candidate message.

    Returns:
        ``True`` only when debugging is enabled and the candidate level is at
        least as severe as the configured minimum.
    """
    if not CONFIG["enabled"]:
        return False
    requested_level = _normalize_level(level_name)
    current_level = _normalize_level(CONFIG["level"])
    return requested_level >= current_level


def _parse_bool(value: Any) -> bool:
    """Interpret ``value`` as a truthy flag for config booleans.

    Args:
        value: A ``bool`` or a string such as ``"1"``/``"true"``/``"yes"``.

    Returns:
        ``True`` when the flag reads as enabled.
    """
    if isinstance(value, bool):
        return value
    v: str = str(value).strip().lower()
    return v in ("1", "true", "yes", "on")


def configure_from_env(
    path: Union[str, Path] = ".env", prefix: str = "SNAKEML_"
) -> Dict[str, Any]:
    """Load configuration from the environment and an optional ``.env`` file.

    Environment variables starting with ``prefix`` (default ``SNAKEML_``)
    are lowercased into config keys, e.g. ``SNAKEML_LEVEL=debug``. The
    ``.env`` file (if present) supplies fallback values and never overrides
    variables already present in :data:`os.environ`.

    Args:
        path: Path of an optional ``KEY=VALUE`` file to parse.
        prefix: Required environment-variable prefix.

    Returns:
        A copy of the configuration after any applied updates.
    """
    env: Dict[str, str] = dict(os.environ)
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
                env.setdefault(k, v)
        except Exception:
            # if .env parsing fails, silently ignore and continue with os.env
            pass

    updates: Dict[str, Any] = {}
    for k, v in env.items():
        if not k.startswith(prefix):
            continue
        key = k[len(prefix):].lower()
        if key not in CONFIG:
            continue
        if key in ("enabled", "save"):
            updates[key] = _parse_bool(v)
        else:
            updates[key] = v

    if updates:
        configure(**updates)

    return get_config()
