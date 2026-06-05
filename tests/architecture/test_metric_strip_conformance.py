"""spec-078 — lock metric-strip conformance in the v2 app.

Every metric strip must go through ``render_metric_strip`` (driven by a
``ColumnSpec`` list) — no hand-written ``st.metric`` / ``cN.metric`` calls in
v2 components or tabs. The single allowed site is ``metric_strip.py`` itself,
which owns the one real ``st.metric`` call.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = [
    REPO_ROOT / "app" / "face_clustering_v2" / "components",
    REPO_ROOT / "app" / "face_clustering_v2" / "tabs",
]
ALLOWED = {"metric_strip.py"}  # the renderer itself
_METRIC = re.compile(r"\.metric\s*\(")


def test_no_bare_st_metric_in_v2():
    offenders: list[str] = []
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for path in sorted(d.glob("*.py")):
            if path.name in ALLOWED:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _METRIC.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "Hand-written st.metric found — migrate onto render_metric_strip + a "
        "ColumnSpec list (face_cluster/views/metric_specs.py):\n  "
        + "\n  ".join(offenders)
    )
