"""Logging configuration for the API."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Module-level reference to current log directory
_current_log_dir: Optional[Path] = None


def setup_logging(
    base_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True
) -> Path:
    """
    Configure logging with timestamped folder.

    Creates: logs/2024-01-30_10-30-00/api.log

    Args:
        base_dir: Base directory for logs (default: "logs")
        level: Logging level (default: INFO)
        console: Whether to also log to console (default: True)

    Returns:
        Path to the log directory for this run
    """
    global _current_log_dir

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(base_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    _current_log_dir = log_dir

    log_file = log_dir / "api.log"

    # Configure root logger
    handlers = [logging.FileHandler(log_file, encoding='utf-8')]
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # Override any existing config
    )

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return log_dir


def get_log_dir() -> Optional[Path]:
    """Get the current log directory for this run."""
    return _current_log_dir


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.

    Convenience wrapper for logging.getLogger().

    Usage:
        from sim_bench.api.logging import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
