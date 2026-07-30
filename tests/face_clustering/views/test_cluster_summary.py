"""spec-074 — ClusterAnalysisService.cluster_summary() (all-clusters overview).

Reuses the synthetic per-run DB builder. Verifies the summary returns one
enriched row per cluster with a real nearest-cluster id / distance / size.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from face_cluster.views.cluster_analysis import (
    CLUSTER_SUMMARY_COLUMNS,
    ClusterAnalysisService,
    ClusterSummaryRow,
)
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


@pytest.fixture
def service(tmp_path: Path) -> ClusterAnalysisService:
    run_dir = _build_synthetic_run_dir(tmp_path)
    return ClusterAnalysisService(ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir)))


def test_summary_has_one_row_per_cluster(service):
    summary = service.cluster_summary()
    assert {r.cluster_id for r in summary} == set(service.get_cluster_ids())
    assert all(isinstance(r, ClusterSummaryRow) for r in summary)


def test_summary_rows_carry_real_size_and_diameter(service):
    for r in service.cluster_summary():
        assert r.size >= 1
        assert r.diameter >= 0.0
        assert r.avg_intra_dist >= 0.0


def test_nearest_is_another_cluster_with_matching_size(service):
    summary = service.cluster_summary()
    if len(summary) < 2:
        pytest.skip("synthetic run has <2 clusters")
    by_id = {r.cluster_id: r for r in summary}
    for r in summary:
        assert r.nearest_cluster_id != r.cluster_id
        assert r.nearest_cluster_id in by_id
        assert 0.0 <= r.nearest_cluster_dist <= 2.0
        assert r.nearest_cluster_size == by_id[r.nearest_cluster_id].size


def test_registry_columns_read_a_summary_row(service):
    summary = service.cluster_summary()
    if not summary:
        pytest.skip("no clusters")
    r = summary[0]
    # every column reads without error; cluster_id column is the raw int (for sort + click)
    vals = {c.label: c.read(r) for c in CLUSTER_SUMMARY_COLUMNS}
    assert vals["Cluster"] == r.cluster_id
    assert vals["Faces"] == r.size


def test_nearest_pairs_sorted_and_structured(service):
    pairs = service.nearest_cluster_pairs(top_n=10)
    ids = set(service.get_cluster_ids())
    if len(ids) < 2:
        pytest.skip("need >=2 clusters")
    assert [p.exemplar_dist for p in pairs] == sorted(p.exemplar_dist for p in pairs)
    for p in pairs:
        assert p.cluster_a in ids and p.cluster_b in ids and p.cluster_a != p.cluster_b
        assert 0.0 <= p.exemplar_dist <= 2.0
        assert isinstance(p.evaluated, bool) and isinstance(p.merged, bool)
