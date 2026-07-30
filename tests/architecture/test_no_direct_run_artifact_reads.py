"""Architecture invariant: UI code does not read run-directory artifacts directly.

spec-030 / FR-008 / FR-012 — every consumer of run data must go through
`sim_bench.run_db.store.RunStore` (or `face_cluster.loader.load_pipeline_result`
which delegates to it).  Direct `pd.read_csv("faces.csv")`, `np.load("...npy")`,
`json.load(...)` of a run-internal artifact, or `sqlite3.connect` of a run DB
are forbidden in `app/face_clustering/`.

This test is a ratchet: if a new direct-read appears anywhere in app/ that
targets a run artifact, the test fails.  Known-deliberate exceptions are
allow-listed below — each carries a reason.

Allow-list semantics:
  * file path is allow-listed in `_DELIBERATE_DIRECT_READS`
  * the surrounding code is a filesystem inspector (run_panels, history_tab) —
    these UIs literally exist to show what's on disk
  * the file is a sidecar that isn't part of the run-internal layout (e.g.,
    `merge_decisions.json` carrying user approval state)
  * the file is `pipeline_run.json` itself, used for live-pipeline progress
    polling before the DB is fully written
"""
from __future__ import annotations

import re
from pathlib import Path

# (file_relative_to_app, justification, regexes_to_allow)
_DELIBERATE_DIRECT_READS = {
    # Filesystem-inspector UI panels — they exist to surface raw on-disk artifacts.
    "run_panels.py": "filesystem inspector showing raw run-directory contents",
    # The History tab reads JSON sidecars to render a lightweight per-run summary
    # without paying the cost of a full PipelineResult load.
    "tabs/history_tab.py": "lightweight per-run summary; no PipelineResult cost",
    # User-approval sidecar lives alongside the run, not inside it.
    "tabs/labeling_review_tab.py": "user-approval sidecar (merge_decisions.json), not run-internal",
    # User comments per face — UI annotation sidecar, not run-internal data.
    "face_popup.py": "user-annotation sidecar (face_comments.json), not run-internal",
    # Crops + faces.csv + crop_manifest.json: planned for cutover in Phase 4 of
    # spec-030, alongside the removal of the legacy writer that produces them.
    # Until then this is the only consumer of those legacy artifacts.
    "cache_helpers.py": "Phase 4: removed when legacy CSV/JSON writes go away",
}

_FORBIDDEN_PATTERNS = (
    r"pd\.read_csv\b",
    r"np\.load\b",
    r"json\.load\(",
    r"json\.loads\(\s*\(?\s*[^)]*\.read_text",   # path.read_text() then json.loads(...)
    r"sqlite3\.connect\b",
)


def _scan(file_path: Path) -> list[tuple[int, str]]:
    """Return [(line_no, matched_text)] for every forbidden pattern hit in the file."""
    text = file_path.read_text(encoding="utf-8")
    hits = []
    for pat in _FORBIDDEN_PATTERNS:
        for m in re.finditer(pat, text):
            line_no = text[: m.start()].count("\n") + 1
            line = text.splitlines()[line_no - 1]
            hits.append((line_no, line.strip()))
    return hits


def test_no_new_direct_run_artifact_reads_in_app():
    """Every direct-read site in app/face_clustering/ must be allow-listed.

    To resolve a failure: route the read through `RunStore` (preferred) or
    `load_pipeline_result()` if you need a `PipelineResult`.  If the read is
    genuinely deliberate (filesystem inspector, sidecar file, transitional
    legacy support), add the file to `_DELIBERATE_DIRECT_READS` with a
    one-line justification.
    """
    app = Path(__file__).resolve().parents[2] / "app" / "face_clustering"
    assert app.is_dir(), f"app/face_clustering/ not found at {app}"

    violations: list[str] = []
    for py in sorted(app.rglob("*.py")):
        rel = py.relative_to(app).as_posix()
        if rel in _DELIBERATE_DIRECT_READS:
            continue
        if rel.startswith("__pycache__"):
            continue
        for line_no, snippet in _scan(py):
            violations.append(f"{rel}:{line_no}  {snippet[:120]}")

    assert not violations, (
        "Forbidden direct file reads found in app/face_clustering/. Route through "
        "sim_bench.run_db.store.RunStore or face_cluster.loader.load_pipeline_result, "
        "or add to _DELIBERATE_DIRECT_READS with a justification.\n\n"
        + "\n".join(violations)
    )


def test_allow_list_entries_still_exist():
    """Catch stale entries: every allow-listed file must actually exist.
    Prevents the allow-list from accumulating zombie entries after refactors."""
    app = Path(__file__).resolve().parents[2] / "app" / "face_clustering"
    missing = [
        rel for rel in _DELIBERATE_DIRECT_READS
        if not (app / rel).is_file()
    ]
    assert not missing, (
        f"Allow-list entries point at files that no longer exist: {missing}. "
        f"Remove them from _DELIBERATE_DIRECT_READS."
    )
