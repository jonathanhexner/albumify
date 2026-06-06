"""The one runner: execute a ``PipelineSpec`` into a v5 run dir.

Any frontend (FC v2 UI, Albumify API) or the standalone CLI builds a
``PipelineSpec`` and calls ``run_pipeline()``. There is ONE executor pass over
the declared steps — no hardcoded per-app step list, no second runner call.
Identical spec -> identical run.

This replaces the hand-rolled orchestration in
``app/face_clustering_v2/pipeline.py::run_v2_pipeline`` (manual image discovery,
a producer pass + a separate ``FCAppRunner`` pass): both producer and clustering
steps are now just entries in ``spec.steps``, run together by the shared executor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import sim_bench.pipeline.steps.all_steps  # noqa: F401 — registers all steps
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_registry
from sim_bench.pipeline.spec import PipelineSpec, validate_spec_or_raise
from sim_bench.run_db.exporter import RunExporter

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Outcome of one ``run_pipeline`` call."""

    success: bool
    run_id: str
    run_dir: Path
    db_path: Path
    n_images: int
    n_faces: int
    n_clusters: int
    n_noise: int
    started_at: str
    finished_at: str
    error_message: Optional[str] = None


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _empty_cluster_result(n_faces: int):
    from face_cluster.types import ClusterResult
    import numpy as np

    return ClusterResult(
        labels=np.full(n_faces, -1, dtype=int),
        clusters={}, cluster_stats={}, exemplars={},
        n_clusters=0, n_noise=n_faces,
    )


def run_pipeline(
    *,
    source_dir: Path,
    run_dir: Path,
    run_id: str,
    album: str,
    spec: PipelineSpec,
    producer: str = "fc_app",
    progress_cb: Optional[Callable[[str, float, str], None]] = None,
) -> RunResult:
    """Validate ``spec`` then execute it end-to-end into ``run_dir``."""
    validate_spec_or_raise(spec)

    source_dir = Path(source_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()

    context = PipelineContext(source_directory=source_dir)
    config = PipelineConfig(
        step_configs=spec.step_configs, fail_fast=True, progress_callback=progress_cb,
    )
    executor = PipelineExecutor(get_registry())
    result = executor.execute(context, spec.steps, config)

    n_faces = len(getattr(context, "face_records", []) or [])
    images = [str(p) for p in (getattr(context, "image_paths", []) or [])]
    db_path = run_dir / "face_clustering.db"

    if not result.success:
        logger.error("run_pipeline failed: %s", result.error_message)
        return RunResult(
            success=False, run_id=run_id, run_dir=run_dir, db_path=db_path,
            n_images=len(images), n_faces=n_faces, n_clusters=0, n_noise=n_faces,
            started_at=started_at, finished_at=_now(),
            error_message=result.error_message,
        )

    people = getattr(context, "people_clusters", {}) or {}
    n_clusters = len(people)
    n_assigned = sum(len(v) for v in people.values())
    finished_at = _now()

    base_cr = getattr(context, "cluster_result", None) or _empty_cluster_result(n_faces)
    merged_cr = getattr(context, "merged_cluster_result", None)
    RunExporter(run_dir).export(
        faces=context.face_records,
        base_cluster_result=base_cr,
        merged_cluster_result=merged_cr,
        core_indices=getattr(context, "core_indices", []) or [],
        merge_log=getattr(context, "merge_log", None),
        merge_metadata=getattr(context, "merge_metadata", None),
        config=None,
        source_album=album,
        producer=producer,
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        image_paths=images,
        filters=getattr(context, "filters", None),
        filter_verdicts=getattr(context, "filter_verdicts", None),
    )

    return RunResult(
        success=True, run_id=run_id, run_dir=run_dir, db_path=db_path,
        n_images=len(images), n_faces=n_faces, n_clusters=n_clusters,
        n_noise=n_faces - n_assigned, started_at=started_at, finished_at=finished_at,
    )


__all__ = ["PipelineSpec", "RunResult", "run_pipeline"]
