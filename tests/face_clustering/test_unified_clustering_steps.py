"""spec-040 Phase 3: unified face-clustering step chain — end-to-end synthetic.

Walks the 8 new clustering steps in order against a hand-built
context.face_records list. Asserts that the chain produces
context.people_clusters without going through the bridge.
"""
from __future__ import annotations

import numpy as np
import pytest

from face_cluster.types import FaceRecord
from sim_bench.pipeline.context import PipelineContext
import sim_bench.pipeline.steps.all_steps  # noqa: F401  (registers steps)
from sim_bench.pipeline.steps.face_clustering_steps import (
    AssignPeopleClustersStep,
    AttachHoldoutFacesStep,
    BuildFaceKNNGraphStep,
    ClusterFaceComponentsStep,
    MergeFaceClustersStep,
    QualityGateFacesStep,
    SelectFaceExemplarsStep,
    ApplyDiameterCapStep,
)


def _synthetic_face(face_id: int, identity: int, image_idx: int) -> FaceRecord:
    """FaceRecord with a cluster-friendly embedding: same identity = same direction."""
    rng = np.random.default_rng(seed=identity * 1000 + face_id)
    base = np.zeros(64, dtype=np.float32)
    base[identity % 64] = 1.0
    # small noise so embeddings aren't bit-identical
    emb = base + 0.05 * rng.standard_normal(64).astype(np.float32)
    norm = float(np.linalg.norm(emb))
    return FaceRecord(
        face_id=face_id,
        image_id=f"img_{image_idx:03d}",
        bbox=(0.0, 0.0, 100.0, 100.0),
        embedding=emb,
        embedding_normalized=emb / norm if norm > 0 else emb,
        area=10000.0,
        blur_score=100.0,  # well above default blur_min=50
        is_core=False,  # will be set by quality_gate_faces
        image_path=f"/fake/img_{image_idx:03d}.jpg",
        face_index=0,
    )


@pytest.fixture
def two_identity_context() -> PipelineContext:
    """5 faces of identity A across 3 images + 4 faces of identity B across 3 images."""
    ctx = PipelineContext()
    face_id = 0
    img_idx = 0
    for _ in range(5):
        ctx.face_records.append(_synthetic_face(face_id, identity=0, image_idx=img_idx))
        face_id += 1
        img_idx += 1
    for _ in range(4):
        ctx.face_records.append(_synthetic_face(face_id, identity=1, image_idx=img_idx))
        face_id += 1
        img_idx += 1
    return ctx


def _run_chain(ctx: PipelineContext, config: dict) -> None:
    """Run the 8 unified clustering steps in order."""
    for step_cls in (
        QualityGateFacesStep,
        BuildFaceKNNGraphStep,
        ClusterFaceComponentsStep,
        SelectFaceExemplarsStep,
        MergeFaceClustersStep,
        AttachHoldoutFacesStep,
        ApplyDiameterCapStep,
        AssignPeopleClustersStep,
    ):
        step_cls().process(ctx, config)


def test_unified_chain_produces_two_clusters(two_identity_context):
    config = {
        "K": 3,
        "distance_threshold": 0.6,
        "min_cluster_size": 2,
        "blur_min": 0.0,        # disable gates we don't simulate
        "yaw_max": 999.0,
        "pitch_max": 999.0,
        "roll_max": 999.0,
        "max_faces_per_image_core": 50,
    }
    _run_chain(two_identity_context, config)

    # 2 identities → 2 clusters expected.
    assert two_identity_context.people_clusters, "people_clusters is empty"
    n_clusters = len(two_identity_context.people_clusters)
    assert n_clusters >= 2, f"Expected ≥2 clusters, got {n_clusters}"

    # Every assigned face is a FaceRecord (no bridges / proxies).
    for cid, faces in two_identity_context.people_clusters.items():
        for f in faces:
            assert isinstance(f, FaceRecord), (
                f"cluster {cid}: expected FaceRecord, got {type(f).__name__} — "
                "no bridges allowed (spec-040)"
            )


def test_unified_chain_writes_no_intermediate_dicts(two_identity_context):
    """The new chain must not populate the legacy dict-of-dicts state."""
    config = {"K": 3, "distance_threshold": 0.6, "min_cluster_size": 2,
              "blur_min": 0.0, "yaw_max": 999.0, "pitch_max": 999.0,
              "roll_max": 999.0, "max_faces_per_image_core": 50}
    _run_chain(two_identity_context, config)

    # The new chain should not touch context.insightface_faces or
    # context.face_embeddings (those are written by producer steps, not
    # by the clustering chain).
    assert two_identity_context.insightface_faces == {}, (
        "Clustering chain wrote to context.insightface_faces — bridge regression"
    )
    assert two_identity_context.face_embeddings == {}, (
        "Clustering chain wrote to context.face_embeddings — bridge regression"
    )


def test_empty_face_records_produces_empty_clusters():
    ctx = PipelineContext()
    _run_chain(ctx, {"K": 3, "distance_threshold": 0.6, "min_cluster_size": 2})
    assert ctx.people_clusters == {}


# ---------------------------------------------------------------------------
# spec-040 T3 (REVIEW.md C3): apply_diameter_cap is no longer a no-op.
# ---------------------------------------------------------------------------

_BASE_CFG = {
    "K": 3, "distance_threshold": 0.6, "min_cluster_size": 2,
    "blur_min": 0.0, "yaw_max": 999.0, "pitch_max": 999.0, "roll_max": 999.0,
    "max_faces_per_image_core": 50,
}


def test_apply_diameter_cap_no_op_when_disabled(two_identity_context):
    """Default config has cap disabled → cap_summary signals disabled."""
    _run_chain(two_identity_context, _BASE_CFG)
    summary = two_identity_context.cap_summary
    assert summary["enabled"] is False
    assert summary["applied"] is False
    assert summary["reason"] == "disabled by config"


def test_apply_diameter_cap_runs_and_keeps_tight_clusters(two_identity_context):
    """Cap on with permissive thresholds: runs, keeps clusters, reports kept count.

    The two-identity fixture has 5 faces of identity A + 4 of identity B, each
    cluster's intra-distance is ~0.1 (small noise around an orthogonal basis).
    A permissive threshold (1.0) keeps everything.
    """
    cfg = {
        **_BASE_CFG,
        "merge_enabled": True,  # need a merged_cluster_result for the cap to run
        "cluster_diameter_cap_enabled": True,
        "max_full_diameter": 1.0,
        "max_exemplar_diameter": 1.0,
    }
    _run_chain(two_identity_context, cfg)
    summary = two_identity_context.cap_summary
    assert summary["enabled"] is True
    assert summary["applied"] is True
    assert summary["n_clusters_inspected"] >= 1
    assert summary["n_split"] == 0, "permissive threshold should not split any cluster"
    # cap_decisions is the dict-list form ready for export.
    assert isinstance(two_identity_context.cap_decisions, list)


def test_apply_diameter_cap_signals_no_merge_output():
    """Cap on but cluster chain hasn't produced a merge result → reason is logged."""
    from sim_bench.pipeline.steps.face_clustering_steps import ApplyDiameterCapStep
    ctx = PipelineContext()
    # Intentionally do NOT run the chain — merged_cluster_result stays None.
    ApplyDiameterCapStep().process(ctx, {**_BASE_CFG, "cluster_diameter_cap_enabled": True})
    summary = ctx.cap_summary
    assert summary["enabled"] is True
    assert summary["applied"] is False
    assert summary["reason"] == "no merge output to inspect"
