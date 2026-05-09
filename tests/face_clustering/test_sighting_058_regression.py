"""Regression test for SIGHTING-058 — the exact UI symptoms the user reported
on `face_clustering_20260508_000446`.

The original failure modes were:
  - C0 vs C1 iter 1 row (action="passed", all gates green) rendered as REJECTED
  - Margin badge for runs with merge_margin=0 rendered as the literal "inf"
  - Support gate showed "191/0" because required_support was dropped on the
    way through the lossy DB schema
  - Diameter showed "0.972/n/a" because max_allowed_diameter was dropped
  - "Actual merges: 0" despite 4 merges actually executing

This test reproduces the same data shape (iter-1 "passed" + iter-2 "merged"
for the same pair, with `merge_margin=0` in config), routes it through the
spec-030 Phase 3 path (RunExporter → disk → loader → MergeAnalysisView →
panel labelling), and asserts the user-visible values are correct.

It does NOT depend on the machine-specific run directory — the row shapes
are reproduced synthetically.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from face_cluster.loader import load_pipeline_result
from face_cluster.run_exporter import RunExporter
from face_cluster.types import ClusterResult, FaceRecord
from face_cluster.views.merge_view import MergeAnalysisView

# Panel helpers live under app/face_clustering, not on the default import path.
APP_DIR = Path(__file__).resolve().parents[2] / "app" / "face_clustering"
sys.path.insert(0, str(APP_DIR))

from _merge_decisions_panel import _outcome_label_color, _format_margin_value  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: synthesize the C0/C1 + C0/C11 scenario from the broken run.
# Two clusters merge across iterations; merge_margin=0 disables the margin gate.
# ---------------------------------------------------------------------------

def _face(face_id: int) -> FaceRecord:
    rng = np.random.default_rng(face_id)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    return FaceRecord(
        face_id=face_id,
        image_id=f"img_{face_id // 10}.jpg",
        image_path=f"/tmp/img_{face_id // 10}.jpg",
        bbox=(0.0, 0.0, 100.0, 100.0),
        area=10000.0,
        blur_score=80.0,
        is_core=True,
        face_index=face_id % 10,
        embedding=emb,
        embedding_normalized=emb,
    )


def _cluster_result(clusters: dict, n: int) -> ClusterResult:
    labels = np.full(n, -1, dtype=np.int32)
    for cid, members in clusters.items():
        for idx in members:
            labels[idx] = cid
    return ClusterResult(
        labels=labels,
        clusters=clusters,
        cluster_stats={cid: {"diameter": 0.4, "mean_dist": 0.2} for cid in clusters},
        exemplars={cid: members[:1] for cid, members in clusters.items()},
        n_clusters=len(clusters),
        n_noise=0,
    )


def _row(iteration: int, ca: int, cb: int, *, action: str, actually_merged: bool) -> dict:
    """Synthesize a merge_log row with margin disabled (margin_gap=inf)."""
    return {
        "iteration": iteration,
        "cluster_a": ca, "cluster_b": cb,
        "cluster_a_size": 22, "cluster_b_size": 33,
        "exemplar_dist": 0.580,
        "threshold_used": 0.6,
        "T_a": None, "T_b": None, "T_global": None,
        "p25_cross_dist": 0.597,
        "passes_cross": True,
        "support": 191,
        "unique_support": 16,
        "required_support": 2,
        "post_diameter": 0.972,
        "max_allowed_diameter": 2.294,
        "margin_gap": float("inf"),       # margin gate disabled
        "margin_dist_to_b": 0.0,
        "margin_competitor_dist": 0.0,
        "margin_competitor_id": -1,
        "passes_exemplar": True,
        "passes_support": True,
        "passes_margin": True,             # passes when disabled
        "passes_diameter": True,
        "action": action,
        "actually_merged": actually_merged,
        "rejection_reason": None,
    }


@pytest.fixture
def broken_run_layout(tmp_path: Path):
    """A v4 layout that mirrors the data shape of face_clustering_20260508_000446.

    Key reproductions:
      - C0+C1 has TWO rows: iter 1 action=passed, iter 2 action=merged
      - merge_margin=0 in config so margin_gap=inf
      - All four gates pass on the iter-1 row (the source of "4/4 REJECTED")
    """
    out = tmp_path / "broken"
    faces = [_face(i) for i in range(6)]
    base = _cluster_result({0: [0, 1, 2], 1: [3, 4, 5]}, n=6)
    merged = _cluster_result({0: [0, 1, 2, 3, 4, 5]}, n=6)

    log = [
        _row(1, 0, 1, action="passed", actually_merged=False),  # the user's confused row
        _row(2, 0, 1, action="merged", actually_merged=True),
    ]

    # Config carries merge_enabled=True and merge_margin=0 — the gate-disabled scenario.
    config = SimpleNamespace(
        merge_enabled=True,
        merge_margin=0.0,
        merge_candidate_threshold=0.65,
    )
    RunExporter(out / "_v4").export(
        faces=faces,
        base_cluster_result=base,
        merged_cluster_result=merged,
        core_indices=list(range(6)),
        merge_log=log,
        merge_metadata={"n_iterations": 2},
        config=config,
        source_album="album",
        producer="albumify",
        run_id="20260508_000446_synth",
        started_at="2026-05-08T00:04:46",
        finished_at="2026-05-08T00:05:31",
    )
    return out


# ---------------------------------------------------------------------------
# Tests — assert the post-fix user-visible state
# ---------------------------------------------------------------------------

def test_loader_finds_v4_layout(broken_run_layout):
    result = load_pipeline_result(broken_run_layout)
    assert result.merge_log is not None
    assert len(result.merge_log) == 2
    assert result.merged_cluster_result is not None  # merge stage ran


def test_view_reports_correct_actual_merges(broken_run_layout):
    """User saw 'Actual merges: 0' on a run that merged 4 pairs.  After the fix
    the count must reflect actually_merged=True rows."""
    result = load_pipeline_result(broken_run_layout)
    view = MergeAnalysisView.compute(result)
    assert len(view.merges) == 1, (
        f"expected 1 actual merge (iter 2 C0+C1), got {len(view.merges)}: {view.merges}"
    )
    assert view.merges[0].actually_merged is True
    assert view.merges[0].action == "merged"


def test_iter1_passed_row_labelled_passed_not_rejected(broken_run_layout):
    """The original confused symptom: iter 1 C0+C1 row (action='passed',
    4/4 gates) was labelled REJECTED.  Must now label PASSED."""
    result = load_pipeline_result(broken_run_layout)
    view = MergeAnalysisView.compute(result)

    # The "passed" row is in all_rejection_rows because actually_merged=False.
    iter1 = [r for r in view.all_rejection_rows
             if r.iteration == 1 and (r.cluster_a, r.cluster_b) == (0, 1)]
    assert iter1, "iter 1 C0+C1 row not present in view"
    label, color = _outcome_label_color(iter1[0])
    assert (label, color) == ("PASSED", "orange"), (
        f"iter-1 'passed' row mislabelled: got {label!r}/{color!r}"
    )
    assert iter1[0].n_gates_passed == 4, "all 4 gates should have passed"


def test_iter2_merged_row_labelled_merged(broken_run_layout):
    result = load_pipeline_result(broken_run_layout)
    view = MergeAnalysisView.compute(result)
    iter2 = [r for r in view.merges
             if r.iteration == 2 and (r.cluster_a, r.cluster_b) == (0, 1)]
    assert iter2, "iter 2 C0+C1 merge not present in view.merges"
    label, color = _outcome_label_color(iter2[0])
    assert (label, color) == ("MERGED", "green")


def test_margin_badge_shows_disabled_not_inf(broken_run_layout):
    """merge_margin=0 in config; merger returns margin_gap=inf.  Old UI
    rendered the literal string 'inf'; must now render 'disabled'."""
    result = load_pipeline_result(broken_run_layout)
    view = MergeAnalysisView.compute(result)
    rows = view.merges + view.all_rejection_rows
    assert rows
    for r in rows:
        text = _format_margin_value(r)
        assert text == "disabled", (
            f"row iter={r.iteration} pair=({r.cluster_a},{r.cluster_b}) "
            f"margin_gap={r.margin_gap!r} rendered as {text!r}, expected 'disabled'"
        )


def test_support_and_diameter_no_longer_show_zeros(broken_run_layout):
    """Original UI showed 'Support 191/0' and 'Diameter 0.972/n/a' because
    required_support and max_allowed_diameter were dropped on disk.  The v4
    schema preserves them; the rendered values must be the input values."""
    result = load_pipeline_result(broken_run_layout)
    view = MergeAnalysisView.compute(result)

    # Pick the iter-1 passed row — it carries the same numbers as the broken run.
    passed = [r for r in view.all_rejection_rows
              if r.iteration == 1 and (r.cluster_a, r.cluster_b) == (0, 1)][0]
    assert passed.support == 191
    assert passed.required_support == 2          # was 0 in the broken UI
    assert passed.unique_support == 16
    assert passed.post_diameter == pytest.approx(0.972, rel=1e-6)
    assert passed.max_allowed_diameter == pytest.approx(2.294, rel=1e-6)  # was 0/n/a in the broken UI


def test_cross_table_consistency_n_merges(broken_run_layout):
    """Anti-regression for the original 'Actual merges: 0 vs n_merges=4' split:
    view.merges count == metadata.n_merges == count of actually_merged=True rows."""
    result = load_pipeline_result(broken_run_layout)
    view = MergeAnalysisView.compute(result)
    n_actually = sum(1 for r in (view.merges + view.all_rejection_rows) if r.actually_merged)
    assert len(view.merges) == n_actually
