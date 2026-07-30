"""spec-079 Stage 3 — the bridge must plumb pose into FaceRecord.

Regression guard for the divergence that made Albumify (20 identities) differ
from FC v2 (8): insightface_detect_faces serializes pose as a [yaw,pitch,roll]
LIST under the 'pose' key, but the bridge read only a dict under 'pose_scores'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.cluster_people import FaceForClustering
from sim_bench.pipeline.steps.face_cluster_bridge import faces_to_face_records


def _face_and_ctx(if_face: dict):
    # Reproduce the real Windows mismatch: insightface_faces is keyed by a
    # CANONICAL forward-slash path, while FaceForClustering.original_path
    # str()s to backslashes on Windows. The lookup must normalize.
    canonical = "D:/album/img1.jpg"
    face = FaceForClustering(
        original_path=Path(canonical), face_index=0,
        embedding=np.ones(512, dtype=np.float32), bbox={"x": 0, "y": 0, "w": 10, "h": 10},
    )
    ctx = PipelineContext()
    ctx.insightface_faces = {canonical: {"faces": [{"face_index": 0, **if_face}]}}
    return [face], np.ones((1, 512), dtype=np.float32), ctx


def ut_PoseListKey_IsPlumbed():
    faces, emb, ctx = _face_and_ctx({"pose": [12.0, 7.0, 3.0]})
    recs = faces_to_face_records(faces, emb, context=ctx)
    assert recs[0].pose == (12.0, 7.0, 3.0)


def ut_LegacyPoseScoresDict_StillPlumbed():
    faces, emb, ctx = _face_and_ctx({"pose_scores": {"yaw": 1.0, "pitch": 2.0, "roll": 3.0}})
    recs = faces_to_face_records(faces, emb, context=ctx)
    assert recs[0].pose == (1.0, 2.0, 3.0)


def ut_NoPose_StaysNone():
    faces, emb, ctx = _face_and_ctx({})
    recs = faces_to_face_records(faces, emb, context=ctx)
    assert recs[0].pose is None


def test_bridge_pose_plumbing_suite():
    ut_PoseListKey_IsPlumbed()
    ut_LegacyPoseScoresDict_StillPlumbed()
    ut_NoPose_StaysNone()
