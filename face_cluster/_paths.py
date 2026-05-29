"""Centralized path resolution for sim-bench.

Single source of truth for repo-root, data-dir, default DB, and the
Alembic config location. Every other module must import from here
rather than computing these paths inline.

spec-048 enforces this via tests/architecture/test_paths_module_sole_owner.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def sim_bench_data_dir() -> Path:
    d = Path.home() / ".sim_bench"
    d.mkdir(parents=True, exist_ok=True)
    return d


def profiles_dir() -> Path:
    return sim_bench_data_dir() / "profiles"


def default_db_path() -> Path:
    """Shared SQLite DB for action_log + training tables.

    Both ``face_cluster.run_history_db`` and ``face_cluster.training_db``
    historically returned this exact path; they now delegate here.
    """
    return sim_bench_data_dir() / "sim_bench.db"


def alembic_ini_path() -> Path:
    return repo_root() / "alembic.ini"


__all__ = [
    "repo_root",
    "sim_bench_data_dir",
    "profiles_dir",
    "default_db_path",
    "alembic_ini_path",
]
