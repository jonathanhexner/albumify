"""spec-040 Phase 5: thin FC App runner over the unified pipeline framework.

Replaces ``face_cluster_legacy.pipeline.FaceClusteringPipeline`` (the
hand-written stage runner) with a thin wrapper that invokes
``sim_bench.pipeline.PipelineExecutor`` with the spec-040 Phase 3 unified
clustering step chain.

The new FC App (when built) calls this module. Equivalence with the
legacy runner is the Phase 6 acceptance gate.

Usage:
    result = FCAppRunner().run(
        source_dir=Path("test_data/album"),
        output_dir=Path("runs/fc_v2_001"),
        step_configs={
            "cluster_people": {"K": 5, "distance_threshold": 0.35, ...},
            # ... etc
        },
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor, PipelineResult
from sim_bench.pipeline.registry import get_registry
from sim_bench.run_db.store import RunStore

logger = logging.getLogger(__name__)


# The unified face-clustering step chain. Order matters (deps).
# Reads context.face_records (populated by upstream detect/embed steps
# in the wider default_pipeline; for FC App standalone invocation the
# caller is responsible for populating context.face_records before
# running this chain).
UNIFIED_CLUSTERING_STEPS = [
    "quality_gate",  # spec-053: consolidated step (was "quality_gate_faces")
    "build_face_knn_graph",
    "cluster_face_components",
    "select_face_exemplars",
    "merge_face_clusters",
    "attach_holdout_faces",
    "apply_diameter_cap",
    "assign_people_clusters",
]


@dataclass
class FCAppRunResult:
    """Return value of FCAppRunner.run().

    Mirrors the load-bearing fields of the legacy ``PipelineResult`` so
    callers (and equivalence tests) can compare side-by-side without
    knowing which runner produced the output.
    """
    success: bool
    n_clusters: int
    n_faces_assigned: int
    n_noise: int
    people_clusters: Dict[int, List[Any]] = field(default_factory=dict)
    step_results: List[Any] = field(default_factory=list)
    error_message: Optional[str] = None


class FCAppRunner:
    """The new FC App's entry point. Uses the unified pipeline framework."""

    def __init__(self, step_list: Optional[List[str]] = None):
        # Importing all_steps triggers @register_step decorators.
        import sim_bench.pipeline.steps.all_steps  # noqa: F401
        self._steps = step_list or UNIFIED_CLUSTERING_STEPS
        self._executor = PipelineExecutor(get_registry())

    def run(
        self,
        context: PipelineContext,
        step_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> FCAppRunResult:
        """Run the unified clustering chain on a context with face_records already populated.

        The caller is responsible for:
        - populating ``context.face_records: List[FaceRecord]`` (face detection,
          embedding extraction, etc. — typically done by upstream pipeline steps).

        This method runs the 8-step unified clustering chain and returns
        an FCAppRunResult.
        """
        if not context.face_records:
            logger.warning("FCAppRunner.run: context.face_records is empty — nothing to cluster")
            return FCAppRunResult(
                success=True,
                n_clusters=0,
                n_faces_assigned=0,
                n_noise=0,
                people_clusters={},
            )

        config = PipelineConfig(step_configs=step_configs or {}, fail_fast=True)
        pipeline_result: PipelineResult = self._executor.execute(
            context, self._steps, config=config
        )

        n_clusters = len(context.people_clusters)
        n_assigned = sum(len(v) for v in context.people_clusters.values())
        n_noise = len(context.face_records) - n_assigned

        return FCAppRunResult(
            success=pipeline_result.success,
            n_clusters=n_clusters,
            n_faces_assigned=n_assigned,
            n_noise=n_noise,
            people_clusters=context.people_clusters,
            step_results=pipeline_result.step_results,
            error_message=pipeline_result.error_message,
        )

    def recluster(
        self,
        prior_run_dir: Path,
        step_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> FCAppRunResult:
        """Re-run the 8-step clustering chain on a prior run's face_records.

        Loads ``face_records`` from ``RunStore(prior_run_dir).faces()`` and
        invokes :meth:`run` with the new ``step_configs``. The expensive
        producer chain (detect / align / embed) is skipped — clustering is
        the only thing that re-runs. ``context.parent_run_id`` is set to
        ``prior_run_dir.name`` so downstream writers (RunExporter) can
        record lineage in ``run_metadata.parent_run_id``.

        Args:
            prior_run_dir: A complete v5 run dir (RunStore.faces() must work).
            step_configs: Per-step config dict; typically ``FCParams.to_step_configs()``.

        Returns:
            FCAppRunResult — same shape as :meth:`run`.

        Raises:
            RunStoreError: If ``prior_run_dir`` is missing or malformed.
        """
        store = RunStore(prior_run_dir)
        face_records = store.faces()
        context = PipelineContext()
        context.face_records = face_records
        context.parent_run_id = prior_run_dir.name
        return self.run(context, step_configs=step_configs)


__all__ = ["FCAppRunner", "FCAppRunResult", "UNIFIED_CLUSTERING_STEPS"]
