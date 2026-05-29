"""spec-048 — `face_cluster/_paths.py` must be the sole owner of:
- `Path.home() / ".sim_bench"` literals
- `Path(__file__).resolve().parents[...]` repo-root walks
- hardcoded `"sim_bench.db"` / `"face_history.db"` strings

Any other module that needs these must import from `_paths`. This prevents
regression of spec-046 smell S6 (inlined path resolution scattered across
data-layer modules).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_CLUSTER = PROJECT_ROOT / "face_cluster"
PATHS_MODULE = FACE_CLUSTER / "_paths.py"

# Patterns that should appear ONLY in _paths.py
_HOME_PATTERN = re.compile(r"""Path\.home\(\)\s*/\s*['"]\.sim_bench['"]""")
_PARENTS_PATTERN = re.compile(r"""Path\(__file__\)\.resolve\(\)\.parents\[""")
_DB_LITERAL_PATTERN = re.compile(r"""['"](sim_bench|face_history)\.db['"]""")


def _python_files_in_face_cluster() -> list[Path]:
    return [p for p in FACE_CLUSTER.rglob("*.py") if "docs" not in p.parts]


@pytest.mark.parametrize(
    "pattern,description",
    [
        (_HOME_PATTERN, "Path.home() / '.sim_bench'"),
        (_PARENTS_PATTERN, "Path(__file__).resolve().parents[...]"),
        (_DB_LITERAL_PATTERN, "hardcoded 'sim_bench.db' / 'face_history.db'"),
    ],
)
def test_path_pattern_only_in_paths_module(pattern: re.Pattern[str], description: str) -> None:
    offenders: list[str] = []
    for py_file in _python_files_in_face_cluster():
        if py_file == PATHS_MODULE:
            continue
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rel = py_file.relative_to(PROJECT_ROOT)
                offenders.append(f"{rel}:{lineno}  {line.strip()}")
    assert not offenders, (
        f"Found {description} outside face_cluster/_paths.py — "
        f"these must use _paths helpers:\n  " + "\n  ".join(offenders)
    )
