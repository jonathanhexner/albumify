"""spec-066 — architecture guards for the v2 Gallery tab.

Mirrors test_face_analysis_tab.py. The tab is pure orchestration: no SQL,
no FS reads, no JSON parsing, no AsyncHandle, no cfg.get literals. LOC <= 80.
The Gallery has no new Service — it composes ClusterAnalysisService.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TAB_FILE = REPO_ROOT / "app" / "face_clustering_v2" / "tabs" / "gallery_tab.py"

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
    hits = [f"{TAB_FILE.relative_to(REPO_ROOT)}: {label} ({tok!r})" for label, tok in _scan(TAB_FILE)]
    assert not hits, "Forbidden direct-access patterns found:\n  " + "\n  ".join(hits)


def test_tab_has_no_cfg_get_literals() -> None:
    pat = re.compile(r"\bcfg\s*\.\s*get\s*\(")
    assert not pat.search(TAB_FILE.read_text(encoding="utf-8")), (
        f"`cfg.get(` literal found in {TAB_FILE.relative_to(REPO_ROOT)}"
    )


def test_tab_loc_budget_under_80() -> None:
    n_lines = sum(1 for _ in TAB_FILE.open(encoding="utf-8"))
    assert n_lines <= 80, f"{TAB_FILE.relative_to(REPO_ROOT)} has {n_lines} lines; budget is 80."
