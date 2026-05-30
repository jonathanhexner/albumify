"""spec-064 — architecture guards for the v2 Face Analysis tab + Service.

Mirrors ``test_recluster_tab.py``. The tab must be pure orchestration: no
SQL, no FS, no ``cfg.get`` literals, no ``AsyncHandle`` (SIGHTING-079).
LOC budget <= 80.

The :class:`FaceAnalysisService` must be Streamlit-free.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TAB_FILE = REPO_ROOT / "app" / "face_clustering_v2" / "tabs" / "face_analysis_tab.py"
SERVICE_FILE = REPO_ROOT / "face_cluster" / "views" / "face_view.py"


_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bsqlite3\b"), "raw sqlite3 access"),
    (re.compile(r"\bopen\s*\("), "raw file open"),
    (re.compile(r"\.read_text\s*\("), "raw filesystem read"),
    (re.compile(r"\.read_bytes\s*\("), "raw filesystem read"),
    (re.compile(r"\bRunStore\s*\("), "direct RunStore construction (use Service / Repository)"),
    (re.compile(r"json\.loads?\s*\("), "JSON parsing (belongs in Service / Repository)"),
    (re.compile(r"\bAsyncHandle\b"), "AsyncHandle (SIGHTING-079: sync + st.spinner only)"),
]


def _scan(path: Path):
    src = path.read_text(encoding="utf-8")
    return [(label, m.group(0)) for pat, label in _FORBIDDEN_PATTERNS for m in pat.finditer(src)]


def test_tab_has_no_direct_db_or_filesystem_access() -> None:
    """Tab is rendering only — no DB, no FS, no JSON parsing, no AsyncHandle."""
    hits = [
        f"{TAB_FILE.relative_to(REPO_ROOT)}: {label} ({token!r})"
        for label, token in _scan(TAB_FILE)
    ]
    assert not hits, "Forbidden direct-access patterns found:\n  " + "\n  ".join(hits)


def test_tab_has_no_cfg_get_literals() -> None:
    """No ``cfg.get('field', '?')`` field-name duplication."""
    pat = re.compile(r"\bcfg\s*\.\s*get\s*\(")
    assert not pat.search(TAB_FILE.read_text(encoding="utf-8")), (
        f"`cfg.get(` literal found in {TAB_FILE.relative_to(REPO_ROOT)}"
    )


def test_tab_loc_budget_under_80() -> None:
    """Tab files are orchestration only — ~80 LOC ceiling."""
    n_lines = sum(1 for _ in TAB_FILE.open(encoding="utf-8"))
    assert n_lines <= 80, (
        f"{TAB_FILE.relative_to(REPO_ROOT)} has {n_lines} lines; budget is 80."
    )


def test_service_is_streamlit_free() -> None:
    """FaceAnalysisService must not import streamlit — Service runs headless."""
    src = SERVICE_FILE.read_text(encoding="utf-8")
    bad = re.compile(r"^\s*(import\s+streamlit|from\s+streamlit)", re.MULTILINE)
    assert not bad.search(src), (
        f"{SERVICE_FILE.relative_to(REPO_ROOT)} imports streamlit — "
        f"Services must be Streamlit-free."
    )


def test_service_methods_have_return_annotations() -> None:
    """Every public FaceAnalysisService method has a typed return annotation."""
    from face_cluster.views.face_view import FaceAnalysisService

    public = [
        name for name, _ in inspect.getmembers(FaceAnalysisService, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    assert public, "FaceAnalysisService exposes no public methods?"
    bad = []
    for name in public:
        sig = inspect.signature(getattr(FaceAnalysisService, name))
        ret = sig.return_annotation
        if ret is inspect.Signature.empty:
            bad.append(f"{name}: missing return annotation")
        elif ret is dict or (isinstance(ret, str) and ret == "dict"):
            bad.append(f"{name}: returns dict")
    assert not bad, "Service return-annotation violations:\n  " + "\n  ".join(bad)
