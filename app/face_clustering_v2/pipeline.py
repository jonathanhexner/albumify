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
from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
from face_cluster.fc_params import FCParams
from sim_bench.pipeline.run import run_pipeline
from sim_bench.pipeline.spec import PipelineSpec

logger = logging.getLogger(__name__)


PRODUCER_STEPS: List[str] = [
    "detect_persons",
    "insightface_detect_faces",
    "detect_face_orientation",
    "align_faces",
    "extract_face_embeddings",
]

# spec-079: FC v2's full producer chain = discovery + the producer steps. Same
# list run_profile.py feeds the shared runner, so the UI and the headless/test
# path build an identical spec.
FC_V2_PRODUCER: List[str] = ["discover_images"] + PRODUCER_STEPS


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


_DEPRECATION_WARNED = False


def run_v2_pipeline(
    src_dir: Path,
    run_dir: Path,
    *,
    run_id: str,
    album: str,
    params: Optional[FCParams] = None,
    step_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    producer: str = "fc_app_v2",
    profile: Optional[str] = None,
    progress_cb: Optional[Callable[[str, float, str], None]] = None,
) -> V2RunResult:
    """Run the v2 pipeline end-to-end into a pre-allocated run directory.

    spec-050: callers allocate ``run_dir`` and ``run_id`` (typically via
    ``face_cluster.run_layout.allocate_run_dir``) and pass them in. The
    pipeline no longer invents a timestamp-based ``run_id``; the UUID
    used for the directory name IS the ``run_id`` written to action_log.

    Preferred call shape:

        from face_cluster.run_layout import allocate_run_dir
        run_dir, run_id = allocate_run_dir(Path("~/.sim_bench/runs"), "Budapest")
        result = run_v2_pipeline(src, run_dir, run_id=run_id, album="Budapest",
                                 params=FCParams(...))

    Parameters
    ----------
    src_dir:
        Directory containing source images. One-level deep scan; subdirs
        ignored. JPG / PNG only — HEIC support is environment-dependent.
    run_dir:
        Pre-allocated output directory (``<base>/<uuid>/``). Created if
        missing — but the caller's allocator already created it, so this
        is just a guard.
    run_id:
        Stable identifier for the run, written to
        ``action_log.run_id``. Caller's contract: ``run_id == run_dir.name``.
    album:
        Free-form album label. Written to ``action_log.source_album``
        and to the v5 export's metadata.
    params:
        ``FCParams`` container holding the full clustering knob set.
        Mutually exclusive with ``step_configs``.
    step_configs:
        Legacy per-step config dict. Deprecated in favor of ``params``;
        emits ``DeprecationWarning`` once per process.
    producer:
        Producer tag written into the v5 ``run_metadata.producer`` column
        and the global ``action_log`` row. Defaults to ``fc_app_v2``.
    profile:
        Name of the FCParams profile this run used (e.g. ``profile_4.json``),
        recorded in the ``action_log`` payload so the Overview tab can build
        a per-profile breakdown (spec-066). ``None`` when no profile was
        loaded — the dashboard shows those as ``(none)``.
    progress_cb:
        Optional ``(step_name, fraction, message) -> None`` callback.
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
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # spec-079: build ONE spec (discovery + producer + the unified clustering
    # chain) and hand it to the single shared runner. This is the SAME spec
    # scripts/run_profile.py feeds run_pipeline — the UI no longer runs a
    # bespoke two-pass orchestration (producer executor + FCAppRunner). Config
    # flows through the typed FCParams contract.
    if params is not None:
        spec = PipelineSpec.from_fcparams(
            params, producer_steps=FC_V2_PRODUCER, clustering_steps=UNIFIED_CLUSTERING_STEPS,
        )
    else:
        steps = FC_V2_PRODUCER + UNIFIED_CLUSTERING_STEPS
        spec = PipelineSpec(steps=steps, step_configs=step_configs or {})

    # Action log row — survives the run even on error so the UI can render
    # history. Producer column makes the new FC App's runs distinguishable
    # from legacy and Albumify rows.
    # spec-043: routed through RunHistoryRepository instead of the deprecated
    # run_history_db free functions.
    try:
        from face_cluster.repositories import RunHistoryRepository
        action_id = RunHistoryRepository().start_action("fc_app_v2_run", payload={
            "run_id": run_id,
            "source_dir": str(src_dir),
            "output_dir": str(output_dir),
            "source_album": album,
            "producer": producer,
            "profile": profile,  # spec-066: feeds the Overview per-profile chart
        })
    except Exception as e:  # pragma: no cover — action_log is best-effort
        logger.warning("action_log start failed (non-fatal): %s", e)
        action_id = None

    # The one runner: validate spec, run all steps in a single executor pass,
    # export the v5 run dir. Discovery is the discover_images step inside the
    # spec — no hand-rolled _discover_jpgs.
    result = run_pipeline(
        source_dir=src_dir, run_dir=output_dir, run_id=run_id, album=album,
        spec=spec, producer="fc_app", progress_cb=progress_cb,
    )
    finished_at = result.finished_at

    if result.n_images == 0 or not result.success:
        # No-images takes priority: an empty source dir makes the producer chain
        # fail deep (extract_face_embeddings has nothing to do), but the UX
        # contract is a clean "No images" message, not the internal step error.
        if result.n_images == 0:
            message = f"No images (.jpg/.jpeg/.png) found under {src_dir}"
        else:
            message = result.error_message or "unknown error"
        if action_id is not None:
            _safe_complete_action(action_id, ok=False, message=message)
        return V2RunResult(
            success=False, output_dir=output_dir, db_path=result.db_path,
            n_images=result.n_images, n_faces=result.n_faces,
            n_clusters=0, n_noise=result.n_faces,
            run_id=run_id, started_at=started_at, finished_at=finished_at,
            error_message=message, action_id=action_id,
        )

    if action_id is not None:
        _safe_complete_action(action_id, ok=True, result_fields={
            "n_faces": result.n_faces,
            "n_clusters": result.n_clusters,
            "n_noise": result.n_noise,
            "run_id": run_id,
        })

    return V2RunResult(
        success=True, output_dir=output_dir, db_path=result.db_path,
        n_images=result.n_images, n_faces=result.n_faces,
        n_clusters=result.n_clusters, n_noise=result.n_noise,
        run_id=run_id, started_at=started_at, finished_at=finished_at,
        action_id=action_id,
    )


def _safe_complete_action(
    action_id: int, *, ok: bool,
    result_fields: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> None:
    # spec-043: routed through RunHistoryRepository instead of the deprecated
    # run_history_db free functions.
    try:
        from face_cluster.repositories import RunHistoryRepository
        repo = RunHistoryRepository()
        if ok:
            repo.complete_action(action_id, result_fields=result_fields)
        else:
            repo.fail_action(action_id, message or "unknown error")
    except Exception as e:  # pragma: no cover
        logger.warning("action_log complete failed (non-fatal): %s", e)
