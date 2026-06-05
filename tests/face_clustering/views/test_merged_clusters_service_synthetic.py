"""spec-065 — synthetic-data tests for MergedClustersService.

Builds the spec-045 synthetic run dir, seeds a few merge_decisions rows
covering both ``actually_merged=True`` and ``False``, and exercises the
Service's read surface. Covers spec §"Tests" cases (4 synthetic + 1
opt-in real-fixture smoke).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from face_cluster.types import MergeDecisionRow
from face_cluster.views.merged_clusters import (
    MergedClustersService,
    MergeDecisionCriteria,
)
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


def _seed_merge_decisions(run_dir: Path) -> None:
    """Seed 3 rows: (0,1) merged at iter=1, (1,2) rejected at iter=1, (0,2) merged at iter=2."""
    conn = sqlite3.connect(str(run_dir / "face_clustering.db"))
    try:
        triples = [
            # iter, ca, cb, ca_size, cb_size, exemplar_dist, threshold, support, required_support,
            # post_diameter, max_allowed, margin_gap, margin_dist_to_b, margin_competitor_dist,
            # margin_competitor_id, passes_exemplar, passes_support, passes_margin, passes_diameter,
            # action, actually_merged
            (1, 0, 1, 10, 10, 0.30, 0.45, 5, 1, 0.20, 0.40, 0.10, 0.30, 0.50, -1,  1, 1, 1, 1, "merged",   1),
            (1, 1, 2, 10, 10, 0.55, 0.45, 0, 1, 0.40, 0.40, 0.05, 0.55, 0.60,  0,  0, 0, 1, 1, "rejected", 0),
            (2, 0, 2, 20, 10, 0.35, 0.45, 4, 1, 0.25, 0.40, 0.15, 0.35, 0.55, -1,  1, 1, 1, 1, "merged",   1),
        ]
        conn.executemany(
            "INSERT INTO merge_decisions ("
            "iteration, cluster_a, cluster_b, cluster_a_size, cluster_b_size, "
            "exemplar_dist, threshold_used, support, required_support, "
            "post_diameter, max_allowed_diameter, margin_gap, margin_dist_to_b, "
            "margin_competitor_dist, margin_competitor_id, "
            "passes_exemplar, passes_support, passes_margin, passes_diameter, "
            "action, actually_merged"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            triples,
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = _build_synthetic_run_dir(tmp_path)
    _seed_merge_decisions(d)
    return d


@pytest.fixture
def service(run_dir: Path) -> MergedClustersService:
    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    return MergedClustersService(repo)


def test_list_merge_decisions_no_criteria_returns_all_typed(service):
    rows = service.list_merge_decisions()
    assert isinstance(rows, list)
    assert all(isinstance(r, MergeDecisionRow) for r in rows)
    assert len(rows) == 3


def test_list_merge_decisions_filters_actually_merged(service):
    merged = service.list_merge_decisions(MergeDecisionCriteria(actually_merged=True))
    assert len(merged) == 2
    assert all(r.actually_merged for r in merged)

    rejected = service.list_merge_decisions(MergeDecisionCriteria(actually_merged=False))
    assert len(rejected) == 1
    assert not rejected[0].actually_merged


def test_list_merge_decisions_filters_by_cluster_and_iteration(service):
    rows = service.list_merge_decisions(MergeDecisionCriteria(cluster_a=0, iteration=2))
    assert len(rows) == 1
    assert rows[0].cluster_a == 0
    assert rows[0].cluster_b == 2
    assert rows[0].iteration == 2


def test_list_iterations_and_clusters_in_log(service):
    assert service.list_iterations() == [1, 2]
    assert service.list_clusters_in_log() == [0, 1, 2]


# --- spec-071: gate badges, summary, pair faces -------------------------------

def test_gate_badges_maps_every_gate(service):
    row = next(r for r in service.list_merge_decisions() if r.action == "rejected")
    badges = service.gate_badges(row)
    assert [b.name for b in badges] == ["cross", "exemplar", "support", "margin", "diameter"]
    by = {b.name: b for b in badges}
    assert by["exemplar"].passed is False   # seeded passes_exemplar=0
    assert by["support"].passed is False    # seeded passes_support=0
    assert by["margin"].passed is True
    assert by["diameter"].passed is True
    assert all(isinstance(b.detail, str) and b.detail for b in badges)


def test_summary_counts_and_top_gate(service):
    s = service.summary()
    assert s.n_merged == 2          # the two action="merged" rows
    assert s.n_rejected == 1        # the one action="rejected" row
    # that row failed exemplar + support; either is a valid top.
    assert s.top_rejection_gate in ("exemplar", "support")


def test_pair_faces_resolves_cluster_members(service):
    from sim_bench.db.face_clustering.cluster_analysis_repo import ClusterAnalysisCriteria
    row = next(r for r in service.list_merge_decisions() if r.actually_merged)
    pf = service.pair_faces(row)
    assert pf.cluster_a == row.cluster_a and pf.cluster_b == row.cluster_b
    # Faces must match what the repo assigns to each cluster (resolution logic).
    assigns = service._repo.find_assignments(ClusterAnalysisCriteria())
    expected_a = sorted(a.face_id for a in assigns if a.cluster_id == row.cluster_a)
    assert pf.a_face_ids == expected_a
    assert isinstance(pf.a_crops, list) and isinstance(pf.b_crops, list)


# ---------------------------------------------------------------------------
# Opt-in real-fixture smoke
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_fixture_list_merge_decisions(v2_budapest_run_dir: Path) -> None:
    """Smoke: real Budapest run yields a typed list (may be empty)."""
    repo = ClusterAnalysisRepository(
        ClusterAnalysisRepoConfig(run_dir=v2_budapest_run_dir, read_only=True)
    )
    svc = MergedClustersService(repo)
    rows = svc.list_merge_decisions()
    assert isinstance(rows, list)
    assert all(isinstance(r, MergeDecisionRow) for r in rows)
