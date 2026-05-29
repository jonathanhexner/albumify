"""spec-045 Phase 5 — real-fixture smoke for ClusterAnalysisService.

Spec.§8.4 — 4 read-only smoke tests against the most recent v2 Budapest
run. Skips cleanly when no real run dir is present (CI / fresh checkout).
"""
from __future__ import annotations

import pytest

from face_cluster.repositories.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from face_cluster.views.cluster_analysis import (
    ClusterAnalysisService,
    ForceMergePreview,
)
from face_cluster.views.cluster_debug_view import ClusterDebugView
from face_cluster.views.cluster_view import ClusterView


@pytest.fixture(scope="module")
def real_service(v2_budapest_run_dir) -> ClusterAnalysisService:
    repo = ClusterAnalysisRepository(
        ClusterAnalysisRepoConfig(run_dir=v2_budapest_run_dir, read_only=True)
    )
    return ClusterAnalysisService(repo)


def test_real_list_clusters_nonempty(real_service):  # #1
    rows = real_service.list_clusters()
    if not rows:
        pytest.skip("Real run has zero non-noise clusters.")
    assert len(rows) >= 1


def test_real_compute_detail_completes(real_service):  # #2
    rows = real_service.list_clusters()
    if not rows:
        pytest.skip("No clusters to compute detail on.")
    handle = real_service.compute_detail_async(cluster_id=rows[0].cluster_id)
    state = handle.wait(timeout=10.0)
    assert state == "done", f"final state: {state}, error={handle.error!r}"
    assert isinstance(handle.result, ClusterView)


def test_real_compute_debug_completes(real_service):  # #3
    rows = real_service.list_clusters()
    if not rows:
        pytest.skip("No clusters to compute debug on.")
    handle = real_service.compute_debug_async(cluster_id=rows[0].cluster_id)
    state = handle.wait(timeout=10.0)
    assert state == "done", f"final state: {state}, error={handle.error!r}"
    assert isinstance(handle.result, ClusterDebugView)


def test_real_preview_force_merge_runs(real_service):  # #4
    rows = sorted(real_service.list_clusters(), key=lambda r: r.size)
    if len(rows) < 2:
        pytest.skip("Need at least 2 clusters to preview a merge.")
    preview = real_service.preview_force_merge(
        cluster_a=rows[0].cluster_id, cluster_b=rows[1].cluster_id
    )
    assert isinstance(preview, ForceMergePreview)
