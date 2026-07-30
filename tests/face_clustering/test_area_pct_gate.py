"""spec-073 — area-% quality gate.

Resolution-independent face-size filter: a face passes iff
``area_ratio*100 >= min_face_area_pct``. Disabled when the threshold is
None; permissive when the face has no area_ratio (legacy runs).
"""
from __future__ import annotations

import numpy as np

from face_cluster.config import PipelineConfig
from face_cluster.quality import QualityGater
from face_cluster.types import FaceRecord


def _face(face_id: int, area_ratio, image_id="img"):
    return FaceRecord(
        face_id=face_id,
        image_id=image_id,
        bbox=(0.0, 0.0, 10.0, 10.0),
        landmarks=None,
        aligned_face=None,
        embedding=np.ones(4, dtype=np.float32),
        embedding_normalized=np.ones(4, dtype=np.float32) / 2.0,
        blur_score=999.0,           # sharp — won't fail blur
        area=100.0,
        is_core=False,
        image_path="p.jpg",
        face_index=face_id,
        area_ratio=area_ratio,
    )


def _gater(pct):
    # large per-image cap so top-k doesn't pre-drop our single faces
    return QualityGater(PipelineConfig(min_face_area_pct=pct, max_faces_per_image_core=50))


def test_gate_disabled_when_threshold_none():
    g = _gater(None)
    _, _, verdicts = g.select_core_set([_face(1, 0.001)])
    assert "area_pct" not in verdicts[0].gates  # gate not evaluated


def test_small_face_rejected_below_threshold():
    g = _gater(2.0)                         # need >= 2% of the image
    _, _, verdicts = g.select_core_set([_face(1, 0.005)])  # 0.5%
    gate = verdicts[0].gates["area_pct"]
    assert gate.passed is False and verdicts[0].rejection_reason == "area_pct"


def test_large_face_passes():
    g = _gater(2.0)
    _, _, verdicts = g.select_core_set([_face(1, 0.05)])   # 5% >= 2%
    assert verdicts[0].gates["area_pct"].passed is True


def test_permissive_when_area_ratio_missing():
    g = _gater(2.0)
    _, _, verdicts = g.select_core_set([_face(1, None)])
    assert verdicts[0].gates["area_pct"].passed is True
