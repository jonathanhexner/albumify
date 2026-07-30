"""spec-042 arch test — every public class/function in new v2 code has a docstring.

Catches "forgot to document" at PR time without policing prose quality
(that's reviewer comments). Presence only.

Public = name does not start with underscore. Test files, ``__init__.py``,
and modules under ``__pycache__`` are excluded.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCAN_DIRS = [
    REPO_ROOT / "face_cluster" / "views",
    REPO_ROOT / "app" / "face_clustering_v2",
]

_EXCLUDE_BASENAMES = {"__init__.py"}

# Pre-existing files that predate spec-042's documentation contract.
# The contract is binding for NEW code in the spec-042 rebuild. Existing
# files get cleaned up as they're touched — when a legacy file's
# contents are rewritten (or a new tab moves logic out of it), remove
# its entry from this set so the docstring guard catches future drift.
_LEGACY_ALLOWLIST = frozenset({
    # face_cluster/views (pre-spec-042; view layer existed earlier).
    "_base.py",
    "cluster_debug_view.py",
    "cluster_view.py",
    "face_view.py",
    "merge_view.py",
    "run_overview.py",
    # app/face_clustering_v2 (pre-spec-042; shipped by spec-040 T4 and
    # spec-041 follow-up). Will be cleaned up incrementally.
    "_profile_bar.py",
    "main.py",
    "pipeline.py",
    "ui_spec.py",
    "widget_factory.py",
    "clusters_tab.py",  # to be replaced by cluster_analysis_tab.py
    "run_tab.py",
    "clusters_tab.py",
})


def _py_files() -> list[Path]:
    out: list[Path] = []
    for d in _SCAN_DIRS:
        if not d.exists():
            continue
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if p.name in _EXCLUDE_BASENAMES:
                continue
            if p.name in _LEGACY_ALLOWLIST:
                continue
            out.append(p)
    return sorted(out)


def _public_defs_without_docstring(path: Path) -> list[str]:
    """Return ``[qualname, ...]`` of public defs/classes lacking a docstring."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:  # pragma: no cover — would surface elsewhere
        return [f"<syntax error in {path}>"]

    offenders: list[str] = []
    # Module-level walk only — we check top-level classes and functions, plus
    # methods of top-level classes. Nested helpers and closures are skipped.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node) is None:
                offenders.append(f"{path.name}::{node.name}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node) is None:
                offenders.append(f"{path.name}::{node.name}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if sub.name.startswith("_") or sub.name in {"__init__"}:
                        # __init__ is conventionally documented in the class
                        # docstring; private methods don't need a docstring.
                        continue
                    if ast.get_docstring(sub) is None:
                        offenders.append(f"{path.name}::{node.name}.{sub.name}")
    return offenders


def test_every_public_def_has_a_docstring():
    files = _py_files()
    if not files:
        pytest.skip("No v2 view / tab files to scan yet")
    all_offenders: list[str] = []
    for path in files:
        all_offenders.extend(_public_defs_without_docstring(path))
    assert not all_offenders, (
        "spec-042 documentation contract — public classes / functions in v2 "
        "code MUST have a docstring. Missing:\n"
        + "\n".join(f"  - {o}" for o in all_offenders)
    )
