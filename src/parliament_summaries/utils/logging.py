from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_configured = False


def configure_logging(log_dir: Optional[str] = None) -> None:
    """Configure loguru with stdout + file sinks."""
    global _configured
    if _configured:
        return

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.add(path / "parliament.log", rotation="00:00", retention="7 days", level="DEBUG")
    _configured = True


def get_logger(name: str):
    """Return a child logger with consistent formatting."""
    return logger.bind(module=name)
