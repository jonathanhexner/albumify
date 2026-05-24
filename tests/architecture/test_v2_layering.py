"""spec-042 arch test — enforce the two-layer split.

* ``face_cluster/views/*.py`` is the backend layer. Must NOT import
  ``streamlit``. Tested via source scan because importing the module
  to inspect ``sys.modules`` would itself trigger the prohibited
  import in any module that has it.

* ``app/face_clustering_v2/tabs/*.py`` is the UI layer. Must NOT
  bypass services by importing the DB layer or parsing run JSON
  directly. Permitted: imports from ``face_cluster.views.*`` (the
  service classes), ``app.face_clustering_v2.components.*``,
  ``app.face_clustering_v2.widget_factory``, ``streamlit``.

Both rules catch the drift class that spec-042 set out to prevent —
service logic leaking into UI files, or Streamlit assumptions leaking
into the backend.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


_VIEWS_DIR = REPO_ROOT / "face_cluster" / "views"
_V2_TABS_DIR = REPO_ROOT / "app" / "face_clustering_v2" / "tabs"


def test_views_layer_does_not_import_streamlit():
    """No file under face_cluster/views/ may import streamlit."""
    if not _VIEWS_DIR.exists():
        pytest.skip(f"{_VIEWS_DIR} not present yet")
    offenders: list[str] = []
    for path in _py_files(_VIEWS_DIR):
        src = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import\s+streamlit|from\s+streamlit\b)", src, re.MULTILINE):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "face_cluster/views/ MUST NOT import streamlit. Offenders:\n"
        + "\n".join(f"  - {p}" for p in offenders)
    )


_TABS_FORBIDDEN_PATTERNS = [
    # Direct DB access — should go through a service.
    (r"^\s*import\s+sqlite3\b", "imports sqlite3 directly — use a service"),
    (r"^\s*from\s+sqlite3\b",   "imports sqlite3 directly — use a service"),
    # Direct DB-helper access — services own this.
    (r"^\s*from\s+face_cluster\.run_history_db\b",
     "imports face_cluster.run_history_db directly — use a HistoryService method"),
    (r"^\s*from\s+face_cluster\.run_history\b(?!_)",
     "imports face_cluster.run_history directly — use a HistoryService method"),
    # JSON file parsing — services own this.
    (r"json\.loads\(.*\.read_text", "parses JSON inline — push into the service"),
]


def test_tabs_layer_does_not_bypass_services():
    """v2 tab files may only call into services / components / widget_factory."""
    if not _V2_TABS_DIR.exists():
        pytest.skip(f"{_V2_TABS_DIR} not present yet")
    offenders: list[str] = []
    for path in _py_files(_V2_TABS_DIR):
        src = path.read_text(encoding="utf-8")
        for pattern, reason in _TABS_FORBIDDEN_PATTERNS:
            if re.search(pattern, src, re.MULTILINE):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}: {reason} (pattern: {pattern!r})"
                )
    assert not offenders, (
        "app/face_clustering_v2/tabs/ MUST NOT bypass services. Offenders:\n"
        + "\n".join(f"  - {o}" for o in offenders)
    )
