"""Logging utilities for Evret."""

from __future__ import annotations

import logging
import os

logging.getLogger("evret").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for Evret components.

    Respects EVRET_LOG_LEVEL environment variable (default: WARNING).

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    level = os.getenv("EVRET_LOG_LEVEL", "WARNING").upper()
    try:
        logger.setLevel(getattr(logging, level))
    except AttributeError:
        logger.setLevel(logging.WARNING)

    return logger
