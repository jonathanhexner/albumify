"""build_scene_distance step + cluster_scenes precomputed branch (spec-103).

Covers: (1) the step writes a precomputed scene distance to context; (2) cluster_scenes clusters that
distance when present; (3) A5 GATE -- with boost=0 the precomputed path is EQUIVALENT to the embedding
path (same partition), i.e. the new branch is byte-identical to before when it adds no time signal, and
(4) when scene_distance is None, cluster_scenes clusters the embeddings exactly as before.
"""

from __future__ import annotations

import numpy as np

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.build_scene_distance import BuildSceneDistanceStep
from sim_bench.pipeline.steps.cluster_scenes import ClusterScenesStep
from sim_bench.scene_cluster.geo_time_fusion import SceneDistanceResult

CLUSTER_CFG = {"method": "hdbscan", "min_cluster_size": 2, "min_samples": 2}


def _ctx_with_two_visual_groups():
    """6 images: 3 in visual group A (~[1,0,..]), 3 in group B (~[0,1,..]); ids carry timestamps."""
    rng = np.random.RandomState(0)
    ids = [
        "20250101_120000", "20250101_120005", "20250101_120010",   # group A, seconds apart
        "20250101_130000", "20250101_130005", "20250101_130010",   # group B, an hour later
    ]
    base_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    base_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    emb = {}
    for i, s in enumerate(ids):
        base = base_a if i < 3 else base_b
        emb[s] = base + 0.02 * rng.randn(8)
    ctx = PipelineContext()
    ctx.scene_embeddings = emb
    return ctx, ids


def _partition(clusters: dict) -> set:
    return frozenset(frozenset(v) for k, v in clusters.items() if int(k) >= 0)


def ut_step_writes_precomputed_distance():
    ctx, ids = _ctx_with_two_visual_groups()
    BuildSceneDistanceStep().process(ctx, {"boost": 0.6, "tau_sec": 60.0})
    assert isinstance(ctx.scene_distance, SceneDistanceResult)
    assert ctx.scene_distance.image_ids == ids
    assert ctx.scene_distance.distance_matrix.shape == (6, 6)
    # every image has a capture time from its filename
    assert all("time" in sig for sig in ctx.scene_distance_signal.values())


def ut_cluster_scenes_uses_precomputed_when_present():
    ctx, ids = _ctx_with_two_visual_groups()
    BuildSceneDistanceStep().process(ctx, {"boost": 0.6, "tau_sec": 60.0})
    ClusterScenesStep().process(ctx, CLUSTER_CFG)
    # two visual groups -> two scenes, no image lost
    assert len(_partition(ctx.scene_clusters)) == 2
    assert set().union(*_partition(ctx.scene_clusters)) == set(ids)


def ut_A5_boost_zero_equals_embedding_path():
    """Path A with boost=0 == today's clustering: same partition whether we cluster the precomputed
    pure-visual distance or the raw embeddings. Guards the 'byte-identical when absent' contract."""
    ctx_emb, _ = _ctx_with_two_visual_groups()
    ClusterScenesStep().process(ctx_emb, CLUSTER_CFG)          # embedding path (scene_distance is None)
    part_emb = _partition(ctx_emb.scene_clusters)

    ctx_pre, _ = _ctx_with_two_visual_groups()
    BuildSceneDistanceStep().process(ctx_pre, {"boost": 0.0, "tau_sec": 60.0})  # pure visual
    ClusterScenesStep().process(ctx_pre, CLUSTER_CFG)          # precomputed path
    part_pre = _partition(ctx_pre.scene_clusters)

    assert part_emb == part_pre


def ut_cluster_scenes_unchanged_when_distance_absent():
    ctx, ids = _ctx_with_two_visual_groups()
    assert ctx.scene_distance is None
    ClusterScenesStep().process(ctx, CLUSTER_CFG)
    assert len(_partition(ctx.scene_clusters)) == 2


def test_build_scene_distance_suite():
    ut_step_writes_precomputed_distance()
    ut_cluster_scenes_uses_precomputed_when_present()
    ut_A5_boost_zero_equals_embedding_path()
    ut_cluster_scenes_unchanged_when_distance_absent()
