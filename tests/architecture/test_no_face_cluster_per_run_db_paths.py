"""spec-056 — Arch guard: per-run-DB modules must live under sim_bench/.

After each PR in spec-056, the corresponding ``face_cluster.<module>``
import path is retired and must have zero callers in the source tree.

The ``ALLOW_LIST`` enumerates paths NOT yet migrated. PRs shrink this list:
- PR1 (this PR): removes ``face_cluster.db.schema``.
- PR2: removes ``face_cluster.repositories.cluster_analysis_repo``.
- PR3: removes ``face_cluster.run_store``.
- PR4: removes ``face_cluster.run_exporter``. Allow-list becomes empty.

The test is by-construction stable: once a path is dropped from the
allow-list, any reintroduced import anywhere under the working tree
fails the build.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths NOT YET migrated by spec-056 (shrinks with each PR).
ALLOW_LIST = {
    "face_cluster.run_store",
    "face_cluster.run_exporter",
    "face_cluster.repositories.cluster_analysis_repo",
}

# All four relocation targets — every PR removes one entry from ALLOW_LIST.
LEGACY_PATHS = {
    "face_cluster.db.schema",
    "face_cluster.run_store",
    "face_cluster.run_exporter",
    "face_cluster.repositories.cluster_analysis_repo",
}

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "specs",            # spec docs reference old paths intentionally
    "logs",
    ".claude",
}


def _iter_py_files() -> list[Path]:
    out: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        parts = set(path.relative_to(REPO_ROOT).parts)
        if parts & EXCLUDE_DIRS:
            continue
        out.append(path)
    return out


@pytest.mark.parametrize("legacy_path", sorted(LEGACY_PATHS - ALLOW_LIST))
def test_no_imports_of_relocated_path(legacy_path: str) -> None:
    """No .py file imports a relocated face_cluster.* path."""
    # Match `from <path>` and `import <path>` forms, with optional submodule.
    pattern = re.compile(
        rf"(?:from|import)\s+{re.escape(legacy_path)}(?:\s|\.|$)"
    )
    hits: list[str] = []
    for path in _iter_py_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not hits, (
        f"Found {len(hits)} import(s) of relocated path '{legacy_path}':\n"
        + "\n".join(hits)
    )


def test_allow_list_shrinks_to_empty_by_pr4() -> None:
    """Documentation guard: ALLOW_LIST must be empty after PR4 lands."""
    # Until PR4 completes, ALLOW_LIST is expected to be non-empty.
    # When spec-056 closes out, this assertion should be flipped to
    # `assert ALLOW_LIST == set()` and the prior body removed.
    assert ALLOW_LIST.issubset(LEGACY_PATHS), (
        "ALLOW_LIST contains entries not in LEGACY_PATHS"
    )
