"""spec-077 — ImageMetricsService smoke + registry checks.

Real-fixture smoke (skips when no run present) + a registry sanity test that
needs no data.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from face_cluster.views.image_metrics import (
    DEFAULT_IMAGE_COLUMNS,
    IMAGE_METRIC_COLUMNS,
    ImageMetricsService,
)
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)

_REF = Path.home() / ".sim_bench" / "runs" / "6437d335de914755bc3edb825c9591c0"


def test_registry_labels_and_defaults():
    labels = [c.label for c in IMAGE_METRIC_COLUMNS]
    assert "Faces" in labels and "Gate passed" in labels and "Composite" in labels
    assert set(DEFAULT_IMAGE_COLUMNS).issubset(set(labels))


@pytest.mark.slow
def test_list_images_real_run():
    if not (_REF / "face_clustering.db").is_file():
        pytest.skip("reference run not present")
    svc = ImageMetricsService(ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=_REF, read_only=True)))
    rows = svc.list_images()
    assert len(rows) >= 1
    r = rows[0]
    assert r.n_faces >= 0
    assert isinstance(r.filter_passed, bool)
    # the registry reads every column off a real row without error
    for c in IMAGE_METRIC_COLUMNS:
        _ = c.read(r)
