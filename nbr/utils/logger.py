"""Logging utilities for training runs."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger


def setup_logging(output_dir: str | Path) -> None:
    """Configure loguru to log to stdout and a file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    logger.add(output_path / "train.log", rotation="10 MB")
    logger.info("Logging to {}", output_path)


def wandb_env() -> None:
    """Apply sane defaults for W&B from env vars."""
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    os.environ.setdefault("WANDB_START_METHOD", "thread")
