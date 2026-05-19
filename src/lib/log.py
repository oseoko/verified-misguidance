"""Logging helpers — file + stderr handlers, per-script log file.

Usage:
    from src.lib.log import setup_logger
    log = setup_logger("search")
    log.info("...")
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import paths

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logger(
    script_name: str,
    *,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configure root logger to file + stderr.

    Filename: ``{log_dir}/{script_name}_{YYYYMMDD_HHMMSS}.log``
    """
    log_dir = log_dir or paths.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / fname

    logger = logging.getLogger(script_name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(_DEFAULT_FMT)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info("log file: %s", log_path)
    return logger


__all__ = ["setup_logger"]
