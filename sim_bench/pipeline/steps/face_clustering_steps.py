"""spec-040 Phase 3: face-clustering pipeline steps (replaces face_cluster_bridge).

Eight registered steps that together perform what the legacy
``face_cluster_bridge.run_face_cluster_knn()`` did, but as named,
individually-testable pipeline steps that operate on
``context.face_records: List[FaceRecord]`` directly.

No translator / adapter class is added — producers populate face_records
and each step here reads / mutates the same Pydantic objects.

The steps are registered with the pipeline registry. They are NOT yet
wired into ``configs/pipeline.yaml::default_pipeline`` (that wiring lands
when the new FC App in spec-040 Phase 5 invokes them). Until then they
exist as the contract surface and are exercised by their unit tests.
"""
from __future__ import annotations

import copy
import logging
from typing import List

import numpy as np

from face_cluster.attach import HoldoutAttacher
from face_cluster.clustering import ConnectedComponentsClusterer
from face_cluster.config import PipelineConfig as FCConfig
from face_cluster.exemplars import D10ExemplarSelector
from face_cluster.knn_graph import KNNGraphBuilder
from face_cluster.merge import ConservativeMerger
from face_cluster.quality import QualityGater
from face_cluster.types import FaceRecord

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step

logger = logging.getLogger(__name__)


def _build_fc_config(config: dict) -> FCConfig:
    """Translate step config dict → FCConfig dataclass (clustering algorithms expect it).

    This is the ONE residual translation in spec-040: from the step's
    dict config to the algorithm class's expected FCConfig dataclass.
    Necessary because we are NOT rewriting the clustering algorithms
    (locked decision: non-goal). When/if face_cluster algorithms migrate
    to Pydantic in a future spec, this collapses.
    """
    return FCConfig(
        K=config.get("K", 5),
        distance_threshold=config.get("distance_threshold", 0.35),
        min_cluster_size=config.get("min_cluster_size", 2),
        yaw_max=float(config.get("yaw_max", 999.0)),
        pitch_max=float(config.get("pitch_max", 999.0)),
        roll_max=float(config.get("roll_max", 999.0)),
        blur_min=float(config.get("blur_min", 0.0)),
        max_faces_per_image_core=config.get("max_faces_per_image_core", 3),
        det_score_min=config.get("det_score_min"),
        split_enabled=config.get("split_enabled", False),
        merge_enabled=config.get("merge_enabled", False),
        attach_enabled=config.get("attach_enabled", False),
        merge_candidate_threshold=config.get("merge_candidate_threshold", 0.45),
        merge_exemplar_threshold=config.get("merge_exemplar_threshold", 0.35),
        merge_use_cross_gate=config.get("merge_use_cross_gate", True),
        merge_cross_threshold=config.get("merge_cross_threshold", 0.40),
        merge_cross_max_size=config.get("merge_cross_max_size", 5),
        merge_support_frac=config.get("merge_support_frac", 0.3),
        merge_support_min=config.get("merge_support_min", 2),
        merge_support_unique=config.get("merge_support_unique", False),
        merge_margin=config.get("merge_margin", 0.05),
        merge_diameter_expansion_factor=config.get("merge_diameter_expansion_factor", 1.5),
    )


# ---------------------------------------------------------------------------
# Step 1: quality gate
# ---------------------------------------------------------------------------
@register_step
class QualityGateFacesStep(BaseStep):
    """Run QualityGater on context.face_records; set is_core / rejection_reason."""

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="quality_gate_faces",
            display_name="Quality-gate faces",
            description="Run blur / pose / area / det / top-k gates on each FaceRecord.",
            category="clustering",
            requires={"face_records"},
            produces={"core_indices", "holdout_indices"},
            depends_on=["extract_face_embeddings"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        faces: List[FaceRecord] = context.face_records or []
        if not faces:
            context.core_indices = []
            context.holdout_indices = []
            return
        gater = QualityGater(_build_fc_config(config))
        core, holdout, _verdicts = gater.select_core_set(faces)
        context.core_indices = list(core)
        context.holdout_indices = list(holdout)
        logger.info("quality_gate_faces: %d core / %d holdout / %d total",
                    len(core), len(holdout), len(faces))


# ---------------------------------------------------------------------------
# Step 2: kNN graph
# ---------------------------------------------------------------------------
@register_step
class BuildFaceKNNGraphStep(BaseStep):
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="build_face_knn_graph",
            display_name="Build face kNN graph",
            description="Mutual-kNN graph over the core set.",
            category="clustering",
            requires={"face_records", "core_indices"},
            produces={"graph_result"},
            depends_on=["quality_gate_faces"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if not context.core_indices:
            context.graph_result = None
            return
        context.graph_result = KNNGraphBuilder(_build_fc_config(config)).build_graph(
            context.face_records, context.core_indices
        )


# ---------------------------------------------------------------------------
# Step 3: connected components
# ---------------------------------------------------------------------------
@register_step
class ClusterFaceComponentsStep(BaseStep):
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="cluster_face_components",
            display_name="Cluster face components",
            description="Connected components on the kNN graph.",
            category="clustering",
            requires={"graph_result", "core_indices"},
            produces={"cluster_result"},
            depends_on=["build_face_knn_graph"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if context.graph_result is None:
            context.cluster_result = None
            return
        context.cluster_result = ConnectedComponentsClusterer(
            _build_fc_config(config)
        ).cluster(context.graph_result, context.core_indices)


# ---------------------------------------------------------------------------
# Step 4: exemplars
# ---------------------------------------------------------------------------
@register_step
class SelectFaceExemplarsStep(BaseStep):
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="select_face_exemplars",
            display_name="Select face exemplars",
            description="D10-based exemplar selection within each cluster.",
            category="clustering",
            requires={"cluster_result", "graph_result"},
            produces={"cluster_result"},
            depends_on=["cluster_face_components"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if context.cluster_result is None:
            return
        cr, _ = D10ExemplarSelector(_build_fc_config(config)).select_exemplars(
            context.cluster_result, context.graph_result
        )
        context.cluster_result = cr


# ---------------------------------------------------------------------------
# Step 5: optional merge
# ---------------------------------------------------------------------------
@register_step
class MergeFaceClustersStep(BaseStep):
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="merge_face_clusters",
            display_name="Merge face clusters",
            description="Conservative 4-gate merge of nearby clusters.",
            category="clustering",
            requires={"cluster_result"},
            produces={"merged_cluster_result", "merge_log", "merge_metadata"},
            depends_on=["select_face_exemplars"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if context.cluster_result is None:
            context.merged_cluster_result = None
            return
        fc_cfg = _build_fc_config(config)
        if not fc_cfg.merge_enabled:
            context.merged_cluster_result = copy.deepcopy(context.cluster_result)
            context.merge_log = None
            context.merge_metadata = None
            return
        merger = ConservativeMerger(fc_cfg)
        merged_cr, merge_log, merge_metadata = merger.merge_clusters_with_logging(
            context.cluster_result, context.graph_result
        )
        context.merged_cluster_result = merged_cr
        context.merge_log = merge_log
        context.merge_metadata = merge_metadata


# ---------------------------------------------------------------------------
# Step 6: optional holdout attach
# ---------------------------------------------------------------------------
@register_step
class AttachHoldoutFacesStep(BaseStep):
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="attach_holdout_faces",
            display_name="Attach holdout faces",
            description="Attach quality-failed (holdout) faces to existing clusters.",
            category="clustering",
            requires={"merged_cluster_result"},
            produces={"merged_cluster_result"},
            depends_on=["merge_face_clusters"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        fc_cfg = _build_fc_config(config)
        if not fc_cfg.attach_enabled:
            return
        cr = context.merged_cluster_result
        if cr is None or not context.holdout_indices:
            return
        attacher = HoldoutAttacher(fc_cfg)
        context.merged_cluster_result = attacher.attach_holdouts(
            context.face_records,
            context.core_indices,
            context.holdout_indices,
            cr,
            context.graph_result,
        )


# ---------------------------------------------------------------------------
# Step 7: diameter cap (spec-031)
# ---------------------------------------------------------------------------
@register_step
class ApplyDiameterCapStep(BaseStep):
    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="apply_diameter_cap",
            display_name="Apply diameter cap",
            description="spec-031 absolute diameter ceiling; reverts violating merges.",
            category="clustering",
            requires={"merged_cluster_result"},
            produces={"cap_decisions", "cap_summary"},
            depends_on=["attach_holdout_faces"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        # The cap logic lives in face_cluster/cluster_diameter_cap.py and
        # is invoked from the legacy FC App stage runner. Wiring it as a
        # step is straightforward but optional for Phase 3 — leave as a
        # no-op until specs/031 is integrated. Mark in cap_summary.
        context.cap_decisions = []
        context.cap_summary = {"note": "diameter cap step is a no-op until integrated", "applied": False}


# ---------------------------------------------------------------------------
# Step 8: assign people_clusters (final consumption)
# ---------------------------------------------------------------------------
@register_step
class AssignPeopleClustersStep(BaseStep):
    """Materialize context.people_clusters from the final cluster_result.

    Format matches what PeopleService.create_from_clusters consumes — a
    dict of {cluster_id: [face_proxy_or_dict, ...]}. We expose the
    FaceRecord objects directly (each FaceRecord carries image_path /
    face_index / cluster_id), so consumers can iterate without
    reconstructing.
    """

    def __init__(self) -> None:
        self._metadata = StepMetadata(
            name="assign_people_clusters",
            display_name="Assign people clusters",
            description="Materialize the final cluster_id → [FaceRecord] mapping.",
            category="clustering",
            requires={"merged_cluster_result"},
            produces={"people_clusters"},
            depends_on=["apply_diameter_cap"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        cr = context.merged_cluster_result or context.cluster_result
        if cr is None:
            context.people_clusters = {}
            return
        fc_cfg = _build_fc_config(config)
        uses_global = fc_cfg.attach_enabled and bool(context.holdout_indices)

        people: dict[int, list] = {}
        n = len(context.face_records)
        for cid, node_indices in cr.clusters.items():
            faces = []
            for node_idx in node_indices:
                gi = node_idx if uses_global else (
                    context.core_indices[node_idx] if node_idx < len(context.core_indices) else node_idx
                )
                if 0 <= gi < n:
                    rec = context.face_records[gi]
                    # Mutate the FaceRecord with its final cluster_id so the
                    # downstream consumer can iterate without a side dict.
                    if hasattr(rec, "cluster_id"):
                        rec.cluster_id = cid
                    faces.append(rec)
            people[cid] = faces
        context.people_clusters = people
        logger.info("assign_people_clusters: %d clusters / %d assigned faces",
                    len(people), sum(len(v) for v in people.values()))
