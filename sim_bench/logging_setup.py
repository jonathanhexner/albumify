"""spec-041 follow-up — shared logging setup for every surface in the repo.

Originally lived at ``sim_bench/api/logging.py``. Moved here and
parameterized by ``surface`` so the FastAPI app, the v2 FC App, the
legacy FC App, and the headless CLI all configure logging the same
way: timestamped directory under ``logs/``, one file per surface.

Convention:

    logs/2026-05-24_10-30-00/
        api.log               (sim_bench/api/main.py)
        fc_app_v2.log         (app/face_clustering_v2/main.py)
        fc_app_legacy.log     (app/face_clustering/main.py)
        cli_run_v2.log        (scripts/run_v2.py)

Because ``setup_logging`` configures the ROOT logger, every module that
does ``logging.getLogger(__name__)`` automatically inherits both
handlers — file + console. No per-module wiring needed.

The arch test ``tests/architecture/test_logging_aligned.py`` enforces
that every entry point calls this function; new surfaces cannot be
added without going through it.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Module-level reference to the current run's log directory.
_current_log_dir: Optional[Path] = None


def setup_logging(
    surface: str,
    *,
    base_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
) -> Path:
    """Configure the root logger with a file handler scoped to one surface.

    Args:
        surface: short name of the running surface — used as the log file
                 name (``<surface>.log``). Examples: ``"api"``,
                 ``"fc_app_v2"``, ``"fc_app_legacy"``, ``"cli_run_v2"``.
        base_dir: parent directory for the timestamped subdir. Defaults
                 to ``"logs"`` (relative to the process CWD).
        level: log level. Defaults to ``INFO``.
        console: whether to also keep a ``StreamHandler``. Defaults to
                 ``True`` so the terminal still shows live logs.

    Returns:
        Path to the timestamped log directory for this process.
    """
    global _current_log_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(base_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    _current_log_dir = log_dir

    log_file = log_dir / f"{surface}.log"

    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # override any prior basicConfig
    )

    # Quiet noisy third-party loggers regardless of surface.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return log_dir


def get_log_dir() -> Optional[Path]:
    """Return the timestamped log directory of the current process,
    or None if ``setup_logging`` has not been called yet."""
    return _current_log_dir


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper for ``logging.getLogger``."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_log_dir", "get_logger"]
