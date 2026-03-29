"""
src/utils/logging.py

Structured logging using structlog.
All pipeline modules call get_logger(__name__).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


_configured = False


def configure_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    json_logs: bool = False,
    console: bool = True,
) -> None:
    """
    Call once at startup (e.g. in run_pipeline.py).
    Subsequent calls to get_logger() will inherit this configuration.
    """
    global _configured

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    if console:
        handlers.append(logging.StreamHandler(sys.stdout))

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "pipeline.log"))

    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=handlers,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=console)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    for handler in handlers:
        handler.setFormatter(formatter)

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Auto-configures with defaults if needed."""
    if not _configured:
        _auto_configure()
    return structlog.get_logger(name)


def _auto_configure() -> None:
    """Lazy default config — reads settings.yaml if available, else uses INFO/console."""
    try:
        from src.utils.config import settings
        log_cfg = settings.get("logging", {})
        configure_logging(
            level=log_cfg.get("level", "INFO"),
            log_dir=log_cfg.get("log_dir"),
            json_logs=log_cfg.get("json_logs", False),
            console=log_cfg.get("console", True),
        )
    except Exception:
        configure_logging()
