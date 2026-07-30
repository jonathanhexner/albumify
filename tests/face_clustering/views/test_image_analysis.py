"""spec-081 — Image Analysis: disposition classification + image_detail passthrough."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.face_clustering_v2.components.image_analysis import _COLOUR, _disposition
from face_cluster.views.image_metrics import ImageMetricsService
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)

_REF = Path.home() / ".sim_bench" / "runs" / "6437d335de914755bc3edb825c9591c0"


def _face(cluster_id=None, rejection_reason=None):
    return SimpleNamespace(cluster_id=cluster_id, rejection_reason=rejection_reason)


def test_disposition_three_way():
    assert _disposition(_face(cluster_id=3)) == "clustered"
    assert _disposition(_face(rejection_reason="top_k_per_image")) == "filtered"
    assert _disposition(_face(cluster_id=None, rejection_reason=None)) == "noise"
    # noise label (negative cluster id) is not 'clustered'
    assert _disposition(_face(cluster_id=-1)) == "noise"


def test_every_disposition_has_a_colour():
    for d in ("clustered", "noise", "filtered"):
        assert d in _COLOUR


@pytest.mark.slow
def test_image_detail_real_run():
    if not (_REF / "face_clustering.db").is_file():
        pytest.skip("reference run not present")
    svc = ImageMetricsService(ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=_REF, read_only=True)))
    imgs = svc.list_images()
    det = svc.image_detail(imgs[0].image_path)
    assert det.image_path == imgs[0].image_path
    assert len(det.faces) == imgs[0].n_faces
    for f in det.faces:
        assert _disposition(f) in ("clustered", "noise", "filtered")
        assert len(f.bbox) == 4
