"""Logging utilities for Evret.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

EVRET_LOGGER_NAME = "evret"

class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for production logging."""

    _RESERVED = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=True)


def get_logger(name: str) -> logging.Logger:
    """Return a package-scoped logger.

    If `name` does not start with `evret`, it is namespaced under `evret`.
    """
    if name == EVRET_LOGGER_NAME or name.startswith(f"{EVRET_LOGGER_NAME}."):
        logger_name = name
    else:
        logger_name = f"{EVRET_LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)


def configure_logging(
    *,
    level: str = "INFO",
    structured: bool = False,
    force: bool = False,
) -> None:
    """Configure Evret logger handlers and formatter.

    Args:
        level: Log level string (e.g. DEBUG, INFO, WARNING, ERROR).
        structured: When True, emit JSON logs.
        force: When True, replace existing handlers on the evret root logger.
    """
    logger = logging.getLogger(EVRET_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.handlers and not force:
        return

    if force:
        logger.handlers.clear()

    handler = logging.StreamHandler()
    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )

    logger.addHandler(handler)