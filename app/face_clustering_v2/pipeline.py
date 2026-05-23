"""spec-040 Phase 5b: extractable v2 pipeline runner used by the FC App v2 UI.

The Streamlit Run tab is a thin wrapper around ``run_v2_pipeline`` below.
Splitting the logic out lets the equivalence + e2e tests call it directly
without spawning a browser, while the UI layer just collects config and
renders progress / status.

Pipeline:

    discover -> detect_persons -> insightface_detect_faces ->
    detect_face_orientation -> align_faces -> extract_face_embeddings ->
    FCAppRunner (8-step unified clustering chain) -> RunExporter (v5)

Producer tag: ``fc_app_v2`` (vs legacy ``fc_app`` and Albumify ``albumify``).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import sim_bench.pipeline.steps.all_steps  # noqa: F401  -- registers steps
from face_cluster.fc_app_runner import FCAppRunner, UNIFIED_CLUSTERING_STEPS
from face_cluster.fc_params import FCParams
from face_cluster.run_exporter import RunExporter
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_registry

logger = logging.getLogger(__name__)


PRODUCER_STEPS: List[str] = [
    "detect_persons",
    "insightface_detect_faces",
    "detect_face_orientation",
    "align_faces",
    "extract_face_embeddings",
]


@dataclass
class V2RunResult:
    """Return value of ``run_v2_pipeline``. Mirrors fields the UI needs."""
    success: bool
    output_dir: Path
    db_path: Path
    n_images: int
    n_faces: int
    n_clusters: int
    n_noise: int
    run_id: str
    started_at: str
    finished_at: str
    error_message: Optional[str] = None
    action_id: Optional[int] = None


def _discover_jpgs(src_dir: Path) -> List[Path]:
    """Sorted list of supported images under src_dir (one level deep)."""
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )


_DEPRECATION_WARNED = False


def run_v2_pipeline(
    src_dir: Path,
    output_dir: Path,
    *,
    params: Optional[FCParams] = None,
    step_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    producer: str = "fc_app_v2",
    progress_cb: Optional[Callable[[str, float, str], None]] = None,
) -> V2RunResult:
    """Run the v2 pipeline end-to-end and write a v5 face_clustering.db.

    Preferred call shape (spec-041):

        result = run_v2_pipeline(src, out, params=FCParams(K=5, ...))

    Parameters
    ----------
    src_dir:
        Directory containing source images. One-level deep scan; subdirs
        ignored. JPG / PNG only — HEIC support is environment-dependent.
    output_dir:
        Destination for the run artifacts (face_clustering.db, crops/, etc.).
        Created if missing.
    params:
        ``FCParams`` container holding the full clustering knob set. When
        provided, ``step_configs`` is derived from ``params.to_step_configs()``.
        Mutually exclusive with ``step_configs``.
    step_configs:
        Legacy per-step config dict. Deprecated in favor of ``params``;
        emits ``DeprecationWarning`` once per process. Removed in a
        future spec.
    producer:
        Producer tag written into the v5 ``run_metadata.producer`` column
        and the global ``action_log`` row. Defaults to ``fc_app_v2``.
    progress_cb:
        Optional ``(step_name, fraction, message) -> None`` callback for UI
        progress bars. Ignored if None.
    """
    global _DEPRECATION_WARNED
    if params is not None and step_configs is not None:
        raise ValueError(
            "run_v2_pipeline: pass either 'params' or 'step_configs', not both."
        )
    if params is not None:
        step_configs = params.to_step_configs()
    elif step_configs is not None:
        if not _DEPRECATION_WARNED:
            warnings.warn(
                "run_v2_pipeline(step_configs=...) is deprecated; pass an "
                "FCParams instance via params= instead (spec-041).",
                DeprecationWarning,
                stacklevel=2,
            )
            _DEPRECATION_WARNED = True
    # else: both None — defaults take effect inside FCAppRunner.

    src_dir = Path(src_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _discover_jpgs(src_dir)
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Action log row — survives the run even on error so the UI can render
    # history. Producer column makes the new FC App's runs distinguishable
    # from legacy and Albumify rows.
    try:
        from face_cluster import run_history_db
        action_id = run_history_db.start_action("fc_app_v2_run", payload={
            "run_id": run_id,
            "source_dir": str(src_dir),
            "output_dir": str(output_dir),
            "source_album": src_dir.name,
            "producer": producer,
            "n_images": len(images),
        })
    except Exception as e:  # pragma: no cover — action_log is best-effort
        logger.warning("action_log start failed (non-fatal): %s", e)
        action_id = None

    if not images:
        finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        if action_id is not None:
            _safe_complete_action(action_id, ok=False, message="no images discovered")
        return V2RunResult(
            success=False, output_dir=output_dir,
            db_path=output_dir / "face_clustering.db",
            n_images=0, n_faces=0, n_clusters=0, n_noise=0,
            run_id=run_id, started_at=started_at, finished_at=finished_at,
            error_message=f"No images (.jpg/.jpeg/.png) found under {src_dir}",
            action_id=action_id,
        )

    context = PipelineContext(source_directory=src_dir)
    context.image_paths = images
    if progress_cb:
        context.progress_callback = progress_cb  # type: ignore[attr-defined]

    step_configs = step_configs or {}
    producer_configs = {name: step_configs.get(name, {}) for name in PRODUCER_STEPS}
    clustering_configs = {name: step_configs.get(name, {}) for name in UNIFIED_CLUSTERING_STEPS}
    context.step_configs = {**producer_configs, **clustering_configs}

    # 1. Producer chain — populates context.face_records via A1 dual-write.
    executor = PipelineExecutor(get_registry())
    producer_result = executor.execute(
        context, PRODUCER_STEPS,
        config=PipelineConfig(step_configs=producer_configs, fail_fast=True),
    )
    if not producer_result.success:
        finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        if action_id is not None:
            _safe_complete_action(action_id, ok=False, message=producer_result.error_message)
        return V2RunResult(
            success=False, output_dir=output_dir,
            db_path=output_dir / "face_clustering.db",
            n_images=len(images), n_faces=0, n_clusters=0, n_noise=0,
            run_id=run_id, started_at=started_at, finished_at=finished_at,
            error_message=(
                f"Producer chain failed at {producer_result.failed_step}: "
                f"{producer_result.error_message}"
            ),
            action_id=action_id,
        )

    # 2. Clustering chain — FCAppRunner is the canonical entry.
    fc_result = FCAppRunner().run(context, step_configs=clustering_configs)
    if not fc_result.success:
        finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        if action_id is not None:
            _safe_complete_action(action_id, ok=False, message=fc_result.error_message)
        return V2RunResult(
            success=False, output_dir=output_dir,
            db_path=output_dir / "face_clustering.db",
            n_images=len(images), n_faces=len(context.face_records),
            n_clusters=0, n_noise=0,
            run_id=run_id, started_at=started_at, finished_at=finished_at,
            error_message=f"Clustering chain failed: {fc_result.error_message}",
            action_id=action_id,
        )

    # 3. Export the v5 run directory.
    base_cr = getattr(context, "cluster_result", None) or _empty_cluster_result(len(context.face_records))
    merged_cr = getattr(context, "merged_cluster_result", None)
    finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    RunExporter(output_dir).export(
        faces=context.face_records,
        base_cluster_result=base_cr,
        merged_cluster_result=merged_cr,
        core_indices=getattr(context, "core_indices", []) or [],
        merge_log=getattr(context, "merge_log", None),
        merge_metadata=getattr(context, "merge_metadata", None),
        config=None,  # per-step Pydantic configs replace the dataclass FCConfig at the boundary
        source_album=src_dir.name,
        producer="fc_app",  # RunExporter producer allow-list — see _VALID_PRODUCERS
        run_id=run_id, started_at=started_at, finished_at=finished_at,
        image_paths=[str(p) for p in images],
        filters=getattr(context, "filters", None),
    )

    if action_id is not None:
        _safe_complete_action(action_id, ok=True, result_fields={
            "n_faces": len(context.face_records),
            "n_clusters": fc_result.n_clusters,
            "n_noise": fc_result.n_noise,
            "run_id": run_id,
        })

    return V2RunResult(
        success=True, output_dir=output_dir,
        db_path=output_dir / "face_clustering.db",
        n_images=len(images),
        n_faces=len(context.face_records),
        n_clusters=fc_result.n_clusters,
        n_noise=fc_result.n_noise,
        run_id=run_id, started_at=started_at, finished_at=finished_at,
        action_id=action_id,
    )


def _empty_cluster_result(n_faces: int):
    """Fallback when the chain produced no clustering — every face is noise."""
    from face_cluster.types import ClusterResult
    import numpy as np
    return ClusterResult(
        labels=np.full(n_faces, -1, dtype=int),
        clusters={},
        cluster_stats={},
        exemplars={},
        n_clusters=0, n_noise=n_faces,
    )


def _safe_complete_action(
    action_id: int, *, ok: bool,
    result_fields: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> None:
    try:
        from face_cluster import run_history_db
        if ok:
            run_history_db.complete_action(action_id, result_fields=result_fields)
        else:
            run_history_db.fail_action(action_id, message or "unknown error")
    except Exception as e:  # pragma: no cover
        logger.warning("action_log complete failed (non-fatal): %s", e)
