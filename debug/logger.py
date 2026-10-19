"""Level-gated logger for snakeML runtime debugging.

:class:`Logger` consults :mod:`debug.config` for enabled state, formats
messages via :func:`debug.formatter.format_message`, augments them with
bound and call-site context, prints them to the configured stream and
optionally appends each payload to ``config["log_file"]``.
"""

from __future__ import annotations

import inspect
import sys
from types import FrameType
from typing import Any, Dict, Optional, TextIO, Union

from .config import get_config, is_enabled
from .formatter import format_message

__all__ = ["Logger", "get_logger", "logger"]


class Logger:
    """Small logger tailored for snakeML's runtime debugging.

    Attributes:
        name: Module label rendered in every formatted payload.
        _default_context: Context merged ahead of per-call context by
            :meth:`bind`-style instances.
    """

    def __init__(self, name: str = "snakeML") -> None:
        self.name = name

    #: Context merged ahead of each call's context (populated by :meth:`bind`).
    _default_context: Optional[Dict[str, Any]]

    def _emit(
        self,
        level: str,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Render and dispatch one message through the debug pipeline.

        Args:
            level: Severity name, e.g. ``"debug"`` or ``"critical"``.
            message: ``printf``-style format string.
            *args: Substitutions for ``message`` (``%``-style).
            **kwargs: Either ``stream`` (override output handle) or
                ``context`` (extra metadata merged over bound context).
        """
        if not is_enabled(level):
            return
        if args:
            message = message % args
        config = get_config()
        stream = kwargs.pop("stream", None) or config.get("stream") or sys.stdout
        context = kwargs.pop("context", None)
        if context is None:
            context = {}
        else:
            try:
                context = dict(context)
            except Exception:
                context = {"context": context}
        defaults = getattr(self, "_default_context", None)
        if defaults is not None:
            context = {**defaults, **context}
        context.update(kwargs)
        payload = format_message(
            message, level=level, module=self.name, context=context or None
        )
        print(payload, file=stream)

        # optionally persist logs to a file when enabled in config
        try:
            if config.get("save") and config.get("log_file"):
                log_path: Any = config.get("log_file")
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(payload + "\n")
                except Exception:
                    # avoid raising from logging; fallback to stderr
                    try:
                        print(
                            f"[snakeML][ERROR] failed to write log to {log_path}",
                            file=sys.stderr,
                        )
                    except Exception:
                        pass
        except Exception:
            # defensive: never let logging crash the application
            pass

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._emit("debug", message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._emit("info", message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._emit("warning", message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._emit("error", message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._emit("critical", message, *args, **kwargs)

    def log(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        self._emit(level, message, *args, **kwargs)

    def bind(self, **default_context: Any) -> "Logger":
        """Create a logger that always carries the given context.

        Args:
            **default_context: Metadata merged ahead of each call's
                context for the returned logger.

        Returns:
            A new :class:`Logger` bound to these defaults.
        """
        bound = Logger(name=self.name)
        bound._default_context = default_context.copy()
        return bound


def get_logger(name: Optional[str] = None) -> Logger:
    """Return a :class:`Logger` for the calling module.

    Args:
        name: Explicit logger name; when omitted, inferred from the
            caller's ``__name__``.

    Returns:
        A named logger instance.
    """
    if name is None:
        caller = inspect.currentframe()
        name = (
            caller.f_back.f_globals.get("__name__", "snakeML")
            if caller is not None and caller.f_back is not None
            else "snakeML"
        )
    return Logger(name=name)


logger: Logger = get_logger()
