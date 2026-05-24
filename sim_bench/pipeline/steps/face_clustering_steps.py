"""spec-040 Phase 3: face-clustering pipeline steps (replaces face_cluster_bridge).

Eight registered steps that together perform what the legacy
``face_cluster_bridge.run_face_cluster_knn()`` did, but as named,
individually-testable pipeline steps that operate on
``context.face_records: List[FaceRecord]`` directly.

No translator / adapter class is added — producers populate face_records
and each step here reads / mutates the same Pydantic objects.

spec-041 update: each step receives a full FCConfig-shaped dict via
``FCParams.to_step_configs()`` broadcast. No per-step translator —
each ``process()`` builds its own ``FCConfig(**config)`` directly,
and defaults come from ``FCConfig.__init__`` itself.

The steps are registered with the pipeline registry.
"""
from __future__ import annotations

import copy
import logging
from typing import List

import numpy as np

from face_cluster.attach import HoldoutAttacher
from face_cluster.cluster_diameter_cap import apply_diameter_cap, decisions_to_dict_list
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
            depends_on=[],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        faces: List[FaceRecord] = context.face_records or []
        if not faces:
            context.core_indices = []
            context.holdout_indices = []
            return
        gater = QualityGater(FCConfig(**config))
        core, holdout, _verdicts = gater.select_core_set(faces)
        # Defensive: faces with no embedding must never enter the core set.
        # If they do, downstream kNN builds np.array([None, ndarray, ...]) and
        # fails with "inhomogeneous shape". Filter explicitly and log loud so
        # a producer regression (cf. spec-040 A1 dual-write) is diagnosable.
        n_dropped = 0
        filtered_core: list[int] = []
        for i in core:
            if faces[i].embedding_normalized is None:
                n_dropped += 1
            else:
                filtered_core.append(i)
        if n_dropped:
            logger.warning(
                "quality_gate_faces: dropped %d core faces with None embedding_normalized "
                "(producer dual-write regression?)", n_dropped,
            )
        context.core_indices = filtered_core
        context.holdout_indices = list(holdout)
        logger.info("quality_gate_faces: %d core / %d holdout / %d total",
                    len(filtered_core), len(holdout), len(faces))


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
        context.graph_result = KNNGraphBuilder(FCConfig(**config)).build_graph(
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
            FCConfig(**config)
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
        cr, _ = D10ExemplarSelector(FCConfig(**config)).select_exemplars(
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
            # spec-041 audit fix: merger.merge_clusters_with_logging takes
            # (cluster_result, graph_result) — graph_result was undeclared.
            requires={"cluster_result", "graph_result"},
            produces={"merged_cluster_result", "merge_log", "merge_metadata"},
            depends_on=["select_face_exemplars"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if context.cluster_result is None:
            context.merged_cluster_result = None
            return
        fc_cfg = FCConfig(**config)
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
            # spec-041 audit fix: attacher.attach_holdouts reads
            # face_records, core_indices, holdout_indices, graph_result —
            # all four were undeclared.
            requires={
                "merged_cluster_result",
                "face_records",
                "core_indices",
                "holdout_indices",
                "graph_result",
            },
            produces={"merged_cluster_result"},
            depends_on=["merge_face_clusters"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        fc_cfg = FCConfig(**config)
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
            # spec-041 audit fix: cap algorithm reads merged_cluster_result
            # AND cluster_result (pre-merge baseline), plus face_records and
            # core_indices to build the per-cluster face subset. Also
            # OVERWRITES merged_cluster_result when caps fire.
            requires={
                "merged_cluster_result",
                "cluster_result",
                "face_records",
                "core_indices",
            },
            produces={"cap_decisions", "cap_summary", "merged_cluster_result"},
            depends_on=["attach_holdout_faces"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """spec-040 T3 — wire the spec-031 diameter cap into the v2 chain.

        Mirrors the legacy invocation in ``face_cluster.pipeline._cap_diameters``:
        operate in graph-local indices (the same space merge runs in), use
        ``core_indices`` to map face_records → core_faces, then call
        ``apply_diameter_cap``. The result is a NEW ClusterResult (cap never
        edits in place); we replace ``context.merged_cluster_result`` with it
        so downstream steps see the post-cap clustering.

        Skipped when ``cluster_diameter_cap_enabled=False`` (default) or
        when there is no merge result to inspect — both paths leave the
        clustering unchanged but populate ``cap_summary`` so consumers can
        tell the difference between "off" and "ran but kept everything".
        """
        fc_cfg = FCConfig(**config)
        context.cap_decisions = []

        if not fc_cfg.cluster_diameter_cap_enabled:
            context.cap_summary = {"enabled": False, "applied": False, "reason": "disabled by config"}
            return
        merged_cr = getattr(context, "merged_cluster_result", None)
        base_cr = getattr(context, "cluster_result", None)
        if merged_cr is None or base_cr is None:
            context.cap_summary = {"enabled": True, "applied": False, "reason": "no merge output to inspect"}
            return

        # Cap operates on the core-face subset in graph-local indices.
        core_faces = [context.face_records[i] for i in context.core_indices]

        result = apply_diameter_cap(
            merged_result=merged_cr,
            base_result=base_cr,
            faces=core_faces,
            max_full_diameter=fc_cfg.max_full_diameter,
            max_exemplar_diameter=fc_cfg.max_exemplar_diameter,
        )

        context.merged_cluster_result = result.cluster_result
        context.cap_decisions = decisions_to_dict_list(result.decisions)
        context.cap_summary = {
            "enabled": True,
            "applied": True,
            "max_full_diameter": fc_cfg.max_full_diameter,
            "max_exemplar_diameter": fc_cfg.max_exemplar_diameter,
            "n_clusters_inspected": len(result.decisions),
            "n_kept": result.n_kept,
            "n_split": result.n_split,
        }
        logger.info(
            "apply_diameter_cap: kept=%d split=%d -> %d clusters",
            result.n_kept, result.n_split, result.cluster_result.n_clusters,
        )


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
            # spec-041 audit fix: reads merged_cluster_result (preferred)
            # or cluster_result (fallback), plus face_records / core_indices
            # / holdout_indices to map cluster-local indices back to faces.
            requires={
                "merged_cluster_result",
                "cluster_result",
                "face_records",
                "core_indices",
                "holdout_indices",
            },
            produces={"people_clusters"},
            depends_on=["apply_diameter_cap"],
            config_schema={"type": "object"},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        cr = context.merged_cluster_result or context.cluster_result
        if cr is None:
            context.people_clusters = {}
            return
        fc_cfg = FCConfig(**config)
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
