"""
Logging Utility for AI Code Review Agent.

Provides a consistent, formatted logger for every module.
All loggers share the same handler format and respect LOG_LEVEL from config.
"""

import logging
import sys
from typing import Optional


# Registry to avoid duplicate handlers on the same logger
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a named logger.

    Args:
        name:  Module name — use __name__ when calling from a module.
        level: Override log level for this specific logger.
               Falls back to LOG_LEVEL env-var or INFO.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    # Lazy import to avoid circular dependency with config
    from backend.utils.config import get_config
    config = get_config()
    resolved_level = level or config.app.log_level

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, resolved_level.upper(), logging.INFO))

        formatter = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  →  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent log records bubbling up to the root logger (avoids duplicates)
    logger.propagate = False

    _loggers[name] = logger
    return logger
