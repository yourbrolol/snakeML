import inspect
import sys

from .config import get_config, is_enabled
from .formatter import format_message


class Logger:
    """Small logger tailored for snakeML's runtime debugging."""

    def __init__(self, name="snakeML"):
        self.name = name

    def _emit(self, level, message, *args, **kwargs):
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
        if hasattr(self, "_default_context"):
            context = {**self._default_context, **context}
        context.update(kwargs)
        payload = format_message(message, level=level, module=self.name, context=context or None)
        print(payload, file=stream)

        # optionally persist logs to a file when enabled in config
        try:
            if config.get("save") and config.get("log_file"):
                log_path = config.get("log_file")
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(payload + "\n")
                except Exception:
                    # avoid raising from logging; fallback to stderr
                    try:
                        print(f"[snakeML][ERROR] failed to write log to {log_path}", file=sys.stderr)
                    except Exception:
                        pass
        except Exception:
            # defensive: never let logging crash the application
            pass

    def debug(self, message, *args, **kwargs):
        self._emit("debug", message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._emit("info", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._emit("warning", message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._emit("error", message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._emit("critical", message, *args, **kwargs)

    def log(self, level, message, *args, **kwargs):
        self._emit(level, message, *args, **kwargs)

    def bind(self, **default_context):
        bound = Logger(name=self.name)
        bound._default_context = default_context.copy()
        return bound


def get_logger(name=None):
    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "snakeML") if frame is not None else "snakeML"
    return Logger(name=name)


logger = get_logger()
