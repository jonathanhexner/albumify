"""spec-048 — rebuild_golden.py must not depend on legacy free functions.

The legacy modules ``face_cluster.run_history_db`` and
``face_cluster.run_history`` are scheduled for deletion 2026-06-08.
If rebuild_golden imports anything from them (other than ``RunRow``
as a type), the fixture becomes unregenerable after that date.

Smell S4 in the spec-046 audit.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "tests" / "face_clustering" / "fixtures" / "rebuild_golden.py"

# Type-only imports from these modules are allowed (no runtime call dep).
_ALLOWED_TYPE_IMPORTS = {"RunRow", "HistoryFilters"}
_BANNED_MODULES = {"face_cluster.run_history_db", "face_cluster.run_history"}


def _imported_names_from_banned(tree: ast.AST) -> list[str]:
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _BANNED_MODULES:
            for alias in node.names:
                if alias.name not in _ALLOWED_TYPE_IMPORTS:
                    offenders.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _BANNED_MODULES:
                    offenders.append(f"import {alias.name}")
    return offenders


def test_rebuild_golden_imports_no_legacy_free_functions() -> None:
    assert SCRIPT.exists(), f"rebuild_golden.py missing at {SCRIPT}"
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    offenders = _imported_names_from_banned(tree)
    assert not offenders, (
        "rebuild_golden.py must not import from legacy modules slated for "
        "deletion 2026-06-08. Use RunHistoryRepository instead:\n  "
        + "\n  ".join(offenders)
    )
