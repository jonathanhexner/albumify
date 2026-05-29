"""spec-045 Phase 7 — architecture guards for the Cluster Analysis tab + Service.

Locks the Tab → Service → Repository → DB layering. Failures here mean
someone broke the contract that makes the v2 stack independently testable.

Cases mirror spec §8.5 #1–#5.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

TAB_FILE = REPO_ROOT / "app" / "face_clustering_v2" / "tabs" / "cluster_analysis_tab.py"
COMPONENT_DIR = REPO_ROOT / "app" / "face_clustering_v2" / "components"
CLUSTER_COMPONENT_FILES = [
    COMPONENT_DIR / "cluster_picker.py",
    COMPONENT_DIR / "cluster_metrics.py",
    COMPONENT_DIR / "face_grid.py",
    COMPONENT_DIR / "nearest_clusters.py",
    COMPONENT_DIR / "force_merge.py",
    COMPONENT_DIR / "cluster_debug.py",
]


# Token-level patterns the tab + components MUST NOT contain.
# Tab/components are pure orchestration / rendering — DB and FS access
# belongs in the Repository, JSON parsing belongs in RunStore, etc.
_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bsqlite3\b"), "raw sqlite3 access"),
    (re.compile(r"\bopen\s*\("), "raw file open"),
    (re.compile(r"\.read_text\s*\("), "raw filesystem read"),
    (re.compile(r"\.read_bytes\s*\("), "raw filesystem read"),
    (re.compile(r"\bRunStore\s*\("), "direct RunStore construction (use Repository)"),
    (re.compile(r"json\.loads?\s*\("), "JSON parsing (belongs in Repository / RunStore)"),
]


def _scan(path: Path):
    src = path.read_text(encoding="utf-8")
    return [(label, m.group(0)) for pat, label in _FORBIDDEN_PATTERNS for m in pat.finditer(src)]


def test_tab_has_no_direct_db_or_filesystem_access():  # #1
    """The tab and every component is rendering only — no DB, no FS, no JSON parsing."""
    hits = []
    for f in [TAB_FILE] + CLUSTER_COMPONENT_FILES:
        for label, token in _scan(f):
            hits.append(f"{f.relative_to(REPO_ROOT)}: {label} ({token!r})")
    assert not hits, "Forbidden direct-access patterns found:\n  " + "\n  ".join(hits)


def test_tab_has_no_cfg_get_literals():  # #2
    """No ``cfg.get('field', '?')`` field-name duplication — config flow goes
    through typed Service / Repository methods."""
    pat = re.compile(r"\bcfg\s*\.\s*get\s*\(")
    hits = []
    for f in [TAB_FILE] + CLUSTER_COMPONENT_FILES:
        if pat.search(f.read_text(encoding="utf-8")):
            hits.append(str(f.relative_to(REPO_ROOT)))
    assert not hits, f"`cfg.get(` literal found in v2 tab/components: {hits}"


def test_service_returns_typed_objects():  # #3
    """Every public Service method returns a typed object (dataclass / AsyncHandle), never a bare dict."""
    from face_cluster.views.cluster_analysis import ClusterAnalysisService

    public = [
        name for name, _ in inspect.getmembers(ClusterAnalysisService, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    assert public, "ClusterAnalysisService exposes no public methods?"
    bad = []
    for name in public:
        sig = inspect.signature(getattr(ClusterAnalysisService, name))
        ret = sig.return_annotation
        if ret is inspect.Signature.empty:
            bad.append(f"{name}: missing return annotation")
        elif ret is dict or (isinstance(ret, str) and ret == "dict"):
            bad.append(f"{name}: returns dict")
    assert not bad, "Service return-annotation violations:\n  " + "\n  ".join(bad)


def test_repository_takes_typed_config():  # #4
    """ClusterAnalysisRepository.__init__ must take the typed Config, not a Session or kwargs soup."""
    import typing

    from face_cluster.repositories.cluster_analysis_repo import (
        ClusterAnalysisRepoConfig,
        ClusterAnalysisRepository,
    )

    sig = inspect.signature(ClusterAnalysisRepository.__init__)
    params = list(sig.parameters.values())
    assert len(params) == 2, f"expected (self, config), got {[p.name for p in params]}"
    # `from __future__ import annotations` turns annotations into strings —
    # resolve via get_type_hints so the check is meaningful.
    hints = typing.get_type_hints(ClusterAnalysisRepository.__init__)
    assert hints.get("config") is ClusterAnalysisRepoConfig, (
        f"config arg type-hint = {hints.get('config')!r}; expected ClusterAnalysisRepoConfig"
    )


def test_force_merge_preview_fields_match_writer():  # #5
    """Drift guard: every field the legacy snapshot writer expects to derive
    from a preview must exist on ForceMergePreview. Catches the case where
    someone adds a new gate to the preview but forgets to pipe it through."""
    from dataclasses import fields
    from face_cluster.views.cluster_analysis import ForceMergePreview, ForceMergeResult

    required_on_preview = {
        "cluster_a", "cluster_b", "exemplar_dist", "threshold",
        "passes_exemplar", "passes_support", "passes_diameter",
        "is_candidate", "n_gates_passed",
    }
    preview_fields = {f.name for f in fields(ForceMergePreview)}
    missing = required_on_preview - preview_fields
    assert not missing, f"ForceMergePreview missing required fields: {sorted(missing)}"

    # ForceMergeResult is the writer's output; preview must carry enough
    # context to interpret the result. parent_run_dir / snapshot_dir are
    # writer-only; cluster_a/b must be present in both.
    result_fields = {f.name for f in fields(ForceMergeResult)}
    shared = {"new_cluster_id"}  # writer's "what we made" must match preview's "what we asked for"
    # (preview has cluster_a/b; writer has new_cluster_id = min(a, b)).
    assert "new_cluster_id" in result_fields, "ForceMergeResult.new_cluster_id missing"
