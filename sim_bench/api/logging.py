"""Backward-compat shim. Moved to ``sim_bench.logging_setup``.

This module used to host ``setup_logging`` for the FastAPI app, with
the log file name hardcoded to ``api.log``. spec-041 follow-up moved
the logic up one level and parameterized it by ``surface`` so every
entry point in the repo uses the same convention. Old call sites that
do ``setup_logging()`` without a surface name continue to work and
default to ``surface="api"``.

Prefer ``from sim_bench.logging_setup import setup_logging`` in new code.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from sim_bench.logging_setup import (  # noqa: F401
    get_log_dir,
    get_logger,
    setup_logging as _setup_logging,
)


def setup_logging(*, base_dir: str = "logs", level=None, console: bool = True) -> Path:
    """Back-compat wrapper. New code should import from
    ``sim_bench.logging_setup`` and pass ``surface=`` explicitly."""
    warnings.warn(
        "sim_bench.api.logging.setup_logging is deprecated; use "
        "sim_bench.logging_setup.setup_logging(surface='api', ...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import logging as _logging
    return _setup_logging(
        "api",
        base_dir=base_dir,
        level=level if level is not None else _logging.INFO,
        console=console,
    )


__all__ = ["setup_logging", "get_log_dir", "get_logger"]
