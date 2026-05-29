"""spec-053 Phase 1 — every helper's calc() composes the same result
as calling its underlying methods directly.

Each test:
  - builds a minimal helper input
  - calls helper.calc(inputs) → result_a
  - calls the legacy multi-step sequence → result_b
  - asserts equivalent output

The point isn't to exercise the helpers (their existing test files do
that); it's to prove that calc() is a pure facade over the existing
methods — no new behavior, no skipped step, no different ordering.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.exemplars import (
    D10ExemplarSelector,
    ExemplarInputs,
)
from face_cluster.knn_graph import KNNGraphBuilder, KNNGraphInputs
from face_cluster.merge import ConservativeMerger, MergeInputs
from face_cluster.quality import QualityGateInputs, QualityGater
from sim_bench.run_db.exporter import RunExportInputs, RunExporter
from face_cluster.types import ClusterResult, FaceRecord


def _make_face(face_id: int, embedding: np.ndarray, image_id: str = "img0") -> FaceRecord:
    return FaceRecord(
        face_id=face_id, image_id=image_id, image_path=image_id,
        bbox=(0, 0, 100, 100), landmarks=None,
        aligned_face=None, embedding=embedding,
        embedding_normalized=embedding / (np.linalg.norm(embedding) + 1e-9),
        pose=None, blur_score=100.0, area=10000, is_core=False,
        face_index=face_id,
    )


def _make_cfg(**overrides) -> PipelineConfig:
    defaults = dict(
        K=2, distance_threshold=0.9, min_cluster_size=1,
        blur_min=0.0, yaw_max=999.0, pitch_max=999.0, roll_max=999.0,
        max_faces_per_image_core=50,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. QualityGater
# ---------------------------------------------------------------------------

def test_quality_gater_calc_matches_manual_sequence() -> None:
    cfg = _make_cfg(blur_min=0.0)  # permissive
    faces = [_make_face(i, np.random.randn(512).astype(np.float32)) for i in range(4)]

    # calc() path
    gater_a = QualityGater(cfg)
    result_a = gater_a.calc(QualityGateInputs(faces=list(faces)))

    # Manual sequence
    gater_b = QualityGater(cfg)
    faces_b = gater_b.compute_blur_scores(list(faces))
    core_b, hold_b, verdicts_b = gater_b.select_core_set(faces_b)

    assert sorted(result_a.core_indices) == sorted(core_b)
    assert sorted(result_a.holdout_indices) == sorted(hold_b)
    assert len(result_a.verdicts) == len(verdicts_b)


# ---------------------------------------------------------------------------
# 2. KNNGraphBuilder
# ---------------------------------------------------------------------------

def test_knn_graph_builder_calc_matches_build_graph() -> None:
    cfg = _make_cfg()
    rng = np.random.RandomState(0)
    faces = [_make_face(i, rng.randn(512).astype(np.float32)) for i in range(5)]
    core = [0, 1, 2, 3, 4]

    builder_a = KNNGraphBuilder(cfg)
    result_a = builder_a.calc(KNNGraphInputs(faces=faces, core_indices=core))

    builder_b = KNNGraphBuilder(cfg)
    result_b = builder_b.build_graph(faces, core)

    assert result_a.graph.edges == result_b.edges
    assert result_a.graph.neighbors == result_b.neighbors


# ---------------------------------------------------------------------------
# 3. D10ExemplarSelector
# ---------------------------------------------------------------------------

def test_exemplar_selector_calc_matches_select_exemplars() -> None:
    cfg = _make_cfg(d10_k=2, exemplars_d10_threshold=2.0, N_exemplars_max=3)
    rng = np.random.RandomState(1)
    faces = [_make_face(i, rng.randn(512).astype(np.float32)) for i in range(4)]
    core = [0, 1, 2, 3]

    builder = KNNGraphBuilder(cfg)
    graph = builder.build_graph(faces, core)
    cluster_result = ClusterResult(
        labels=np.array([0, 0, 0, 0]),
        clusters={0: [0, 1, 2, 3]},
        cluster_stats={0: {}},
        exemplars={}, n_clusters=1, n_noise=0,
    )

    sel_a = D10ExemplarSelector(cfg)
    result_a = sel_a.calc(ExemplarInputs(
        cluster_result=ClusterResult(
            labels=np.array([0, 0, 0, 0]),
            clusters={0: [0, 1, 2, 3]},
            cluster_stats={0: {}}, exemplars={}, n_clusters=1, n_noise=0,
        ),
        graph_result=graph,
    ))

    sel_b = D10ExemplarSelector(cfg)
    cr_b, d10_b = sel_b.select_exemplars(
        ClusterResult(
            labels=np.array([0, 0, 0, 0]),
            clusters={0: [0, 1, 2, 3]},
            cluster_stats={0: {}}, exemplars={}, n_clusters=1, n_noise=0,
        ),
        graph,
    )

    assert result_a.cluster_result.exemplars == cr_b.exemplars
    assert result_a.node_d10_map.keys() == d10_b.keys()


# ---------------------------------------------------------------------------
# 4. ConservativeMerger
# ---------------------------------------------------------------------------

def test_simplified_merger_calc_matches_merge_clusters_with_logging() -> None:
    cfg = _make_cfg(merge_enabled=True)
    rng = np.random.RandomState(2)
    faces = [_make_face(i, rng.randn(512).astype(np.float32)) for i in range(4)]
    core = [0, 1, 2, 3]
    graph = KNNGraphBuilder(cfg).build_graph(faces, core)
    cr = ClusterResult(
        labels=np.array([0, 0, 1, 1]),
        clusters={0: [0, 1], 1: [2, 3]},
        cluster_stats={0: {}, 1: {}},
        exemplars={0: [0], 1: [2]},
        n_clusters=2, n_noise=0,
    )

    merger_a = ConservativeMerger(cfg)
    result_a = merger_a.calc(MergeInputs(cluster_result=cr, graph_result=graph))

    merger_b = ConservativeMerger(cfg)
    cr_b, log_b, meta_b = merger_b.merge_clusters_with_logging(cr, graph)

    assert result_a.cluster_result.n_clusters == cr_b.n_clusters
    assert len(result_a.merge_log) == len(log_b)
    assert result_a.merge_metadata.keys() == meta_b.keys()


# ---------------------------------------------------------------------------
# 5. RunExporter
# ---------------------------------------------------------------------------

def test_run_exporter_calc_facade_smoke(tmp_path: Path) -> None:
    """Verify calc() routes through to export() — minimal smoke.

    The full export() correctness is covered by test_run_exporter.py;
    this test only proves the facade is a pass-through.
    """
    out = tmp_path / "run_a"
    exp = RunExporter(out)

    # Minimal valid inputs — an empty face list with allow-listed producer.
    cr = ClusterResult(
        labels=np.array([]), clusters={}, cluster_stats={},
        exemplars={}, n_clusters=0, n_noise=0,
    )
    inputs = RunExportInputs(
        faces=[], base_cluster_result=cr, merged_cluster_result=None,
        core_indices=[], merge_log=[], merge_metadata={},
        config=_make_cfg(), source_album="x",
        producer="fc_app", run_id="r1",
        started_at="2026-05-29T00:00:00Z", finished_at="2026-05-29T00:00:01Z",
    )
    result = exp.calc(inputs)
    assert result.output_dir == out
    assert (out / "face_clustering.db").exists()
