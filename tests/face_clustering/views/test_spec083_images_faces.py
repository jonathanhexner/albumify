"""spec-083 — per-image passed-face count + Image Analysis box selection."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.face_clustering_v2.components.image_analysis import faces_to_box

_REF = Path.home() / ".sim_bench" / "runs" / "6437d335de914755bc3edb825c9591c0"


def _face(cluster_id=None, rejection_reason=None, bbox=(0, 0, 10, 10)):
    return SimpleNamespace(cluster_id=cluster_id, rejection_reason=rejection_reason, bbox=bbox)


# ---- faces_to_box (pure) ----------------------------------------------------

def test_filtered_hidden_by_default():
    faces = [_face(cluster_id=1), _face(rejection_reason="blur"), _face(cluster_id=None)]
    boxed = faces_to_box(faces, show_filtered=False)
    assert len(boxed) == 2  # clustered + noise, not the filtered one


def test_filtered_shown_when_requested():
    faces = [_face(cluster_id=1), _face(rejection_reason="blur")]
    assert len(faces_to_box(faces, show_filtered=True)) == 2


def test_invalid_bbox_never_boxed():
    faces = [_face(cluster_id=1, bbox=None), _face(cluster_id=2, bbox=(1, 2, 3))]
    assert faces_to_box(faces, show_filtered=True) == []


# ---- ImageRow.n_passed (real run) -------------------------------------------

@pytest.mark.slow
def test_n_passed_matches_face_table():
    if not (_REF / "face_clustering.db").is_file():
        pytest.skip("reference run not present")
    from sqlalchemy import func, select
    from sim_bench.db.face_clustering.cluster_analysis_repo import (
        ClusterAnalysisRepoConfig, ClusterAnalysisRepository,
    )
    from sim_bench.run_db.models import Face

    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=_REF, read_only=True))
    rows = repo._run_store.list_images()
    # every image: 0 <= n_passed <= n_faces
    for r in rows:
        assert 0 <= r.n_passed <= r.n_faces, r.image_path
    # grand total passed == count of faces with rejection_reason IS NULL
    with repo._run_store._sessionmaker() as s:
        expected = s.execute(
            select(func.count()).select_from(Face).where(Face.rejection_reason.is_(None))
        ).scalar_one()
    assert sum(r.n_passed for r in rows) == expected
