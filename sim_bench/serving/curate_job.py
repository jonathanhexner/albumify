"""Run one curation job from storage and write the normalized album back (spec-110 T2.1, AC3/AC4).

Storage-reference, not inline bytes: the worker is handed a ``job_id``, reads that job's staged
images from a :class:`~sim_bench.serving.job_store.JobStore`, runs the real pipeline over a scratch
copy, normalizes the finished context into the Album contract (spec-106) and writes ``album.json``
next to the inputs.

Two deliberate choices make a re-run reproduce its result (AC4):
  * the pipeline is handed the job's input paths **explicitly** (no folder scan), so it sees exactly
    the staged N images whatever else sits beside them. Those paths are the shared input store's
    (spec-112 A2), the same in every job, so the path-keyed feature cache carries across jobs;
  * ``album.json`` / ``display/`` are **overwritten**, so a re-run cannot accumulate stale picks.

``created_at`` is wall-clock and therefore differs between runs by design; compare two runs with
:func:`stable_album`, which drops it.

spec-117: given the job database, the curator writes the run's live progress and heartbeat to the job row
(:class:`~sim_bench.serving.db_progress.DbProgressWriter`, replacing ``progress.json``) and, given a
:class:`~sim_bench.serving.worker_persister.WorkerPersister`, the results themselves: the job ends
``succeeded`` or ``failed`` here, in the worker, whether or not the app is still watching.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sim_bench.pipeline.run_spec import RunSpec
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.run_output import RunOutput
from sim_bench.serving.run_settings import RunSettings
from sim_bench.serving.pipeline_runner import SpecBuilder, SpecInputs, SpecResult

logger = logging.getLogger(__name__)

__all__ = ["CurateJobInputs", "CurateJobResult", "JobCurator", "run_pipeline", "stable_album",
           "feature_cache_path", "FEATURE_CACHE_ENV"]

# (image paths, plan, on_step=, fail_fast=, album_name=) -> (finished PipelineContext, PipelineResult). Injectable
# so the orchestration above can be tested without the real models.
PipelineRunner = Callable[..., Tuple[object, object]]
OutputBuilder = Callable[[object, object], RunOutput]


# The worker's feature cache (faces, embeddings, quality, scene features). Default: the PC's
# ~/.sim_bench/sim_bench.db, shared with the app. The L4 worker points this at a Modal Volume.
FEATURE_CACHE_ENV = "ALBUMIFY_FEATURE_CACHE"


def feature_cache_path() -> Path:
    env = os.environ.get(FEATURE_CACHE_ENV)
    if env:
        return Path(env)
    from sim_bench.api.database.session import get_database_path
    return get_database_path()


def run_pipeline(images: List[Path], plan: SpecResult, *, on_step: Optional[Callable] = None,
                 on_progress: Optional[Callable[[str, float, str], None]] = None,
                 fail_fast: bool = True, album_name: Optional[str] = None) -> Tuple[object, object]:
    """The production runner: execute the resolved plan over exactly ``images``, with the feature
    cache attached (spec-112 TW1.3: job runs had none, so every job recomputed everything)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from sim_bench.api.database.models import UniversalCache
    from sim_bench.pipeline.cache_handler import UniversalCacheHandler
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.run import execute_spec

    cache_db = feature_cache_path()
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{cache_db.as_posix()}",
                           connect_args={"check_same_thread": False, "timeout": 60})
    UniversalCache.__table__.create(engine, checkfirst=True)
    session = sessionmaker(bind=engine)()
    logger.info("feature cache: %s", cache_db)
    try:
        context = PipelineContext(image_paths=list(images),
                                  cache_handler=UniversalCacheHandler(session))
        context.album_name = album_name   # spec-088: names the export folder
        return context, execute_spec(plan.as_pipeline_spec(), context, fail_fast=fail_fast,
                                     progress_cb=on_progress, on_step_complete=on_step)
    finally:
        session.close()
        engine.dispose()


def stable_album(album: Optional[Dict]) -> Optional[Dict]:
    """``album`` minus the fields that legitimately change between identical runs (``created_at``)."""
    if album is None:
        return None
    return {k: v for k, v in album.items() if k != "created_at"}


@dataclass(frozen=True, slots=True)
class CurateJobInputs:
    """Per-call data for :meth:`JobCurator.calc`."""

    job_id: str
    pipeline: str = "default"
    device: str = "cpu"
    runner: str = "local"            # recorded in run.json settings (spec-115 R12)
    # spec-115 W2: the server makes display images and face crops from its own copies (LocalRenders), so the
    # worker returns results only. True keeps the old worker-side renders (debug / comparison only).
    write_renders: bool = False
    run_spec: Optional[Dict] = None  # spec-116 R2: what to run (RunSpec.to_dict); None = the named pipeline


@dataclass(frozen=True, slots=True)
class CurateJobResult:
    """What the worker returns to its caller; the album itself also lands in the store."""

    ok: bool
    job_id: str
    n_inputs: int = 0
    device: str = "cpu"
    album: Optional[Dict] = None
    album_path: Optional[str] = None
    bundle_path: Optional[str] = None
    display_paths: List[str] = field(default_factory=list)
    flipped: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)
    timing: Dict = field(default_factory=dict)
    failed_step: Optional[str] = None
    error: Optional[str] = None
    settings: Dict = field(default_factory=dict)   # RunSettings (spec-115 R12)


class JobCurator:
    """Curate a job whose images live in a store. Config in ``__init__``, per-call data in Inputs."""

    def __init__(self, store: JobStore, *,
                 spec_builder: Optional[SpecBuilder] = None,
                 runner: PipelineRunner = run_pipeline,
                 clock: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(),
                 output_builder: OutputBuilder = RunOutput.from_context,
                 db=None, persister=None):
        self._store = store
        self._db = db                       # ControlDB: live progress + heartbeat in the job row (spec-117 R5)
        self._persister = persister         # WorkerPersister: the worker writes the results (spec-117 R6)
        self._spec_builder = spec_builder or SpecBuilder()
        self._runner = runner
        self._clock = clock
        self._output_builder = output_builder

    def calc(self, inputs: CurateJobInputs) -> CurateJobResult:
        store, job_id = self._store, inputs.job_id
        sources = store.input_images(job_id)   # validates job_id
        if not sources:
            return CurateJobResult(ok=False, job_id=job_id, device=inputs.device,
                                   error=f"job {job_id!r} has no staged inputs under "
                                         f"{store.paths(job_id).input_dir}")

        spec = RunSpec.from_dict(inputs.run_spec) if inputs.run_spec else None
        plan = self._spec_builder.place(spec, inputs.device) if spec else \
            self._spec_builder.calc(SpecInputs(pipeline=inputs.pipeline, device=inputs.device))
        logger.info("job %s: %d inputs, pipeline=%s device=%s (%d steps)",
                    job_id, len(sources), inputs.pipeline, inputs.device, len(plan.steps))

        from sim_bench.serving.db_progress import DbProgressWriter, NullProgress

        progress = (DbProgressWriter(self._db, job_id, plan.steps, total=len(sources))
                    if self._db is not None else NullProgress())
        try:
            with progress:
                context, result = self._runner(sources, plan, on_step=progress.on_step,
                                               on_progress=progress.on_fraction,
                                               fail_fast=spec.fail_fast if spec else True,
                                               album_name=spec.album_name if spec else None)
                out = self._finish(inputs, plan, sources, context, result)
                if out.ok and self._persister is not None:
                    progress.set_stage("saving results")
                    out = self._persist(inputs, out)
        except Exception as e:
            if self._persister is not None:
                self._persister.fail(job_id, f"{type(e).__name__}: {e}")
            raise
        if not out.ok and self._persister is not None:
            self._persister.fail(job_id, out.error, out.failed_step)
        return out

    def _persist(self, inputs: CurateJobInputs, out: CurateJobResult) -> CurateJobResult:
        """The worker writes its results (spec-117 R6); a failure here fails the run with the reason."""
        from sim_bench.serving.worker_persister import PersistInputs

        try:
            self._persister.calc(PersistInputs(job_id=inputs.job_id, runner=inputs.runner))
            return out
        except Exception as e:
            logger.exception("job %s: results not saved", inputs.job_id)
            failed = dataclasses.replace(out, ok=False, error=f"saving results failed: {type(e).__name__}: {e}")
            self._store.write_run_record(inputs.job_id, _run_record(failed, inputs))
            return failed

    def _finish(self, inputs: CurateJobInputs, plan, sources: List[Path], context, result) -> CurateJobResult:
        store, job_id = self._store, inputs.job_id

        timing = _timing(result)
        settings = RunSettings.collect(plan, pipeline=inputs.pipeline, runner=inputs.runner,
                                       device=inputs.device).to_dict()
        base = dict(job_id=job_id, n_inputs=len(sources), device=inputs.device,
                    flipped=list(plan.flipped), refused=list(plan.refused), timing=timing, settings=settings)
        if not getattr(result, "success", False):
            failed = CurateJobResult(ok=False, failed_step=getattr(result, "failed_step", None),
                                     error=getattr(result, "error_message", None), **base)
            store.write_run_record(job_id, _run_record(failed, inputs))
            return failed

        from sim_bench.album.normalizer import normalize

        album = normalize(context, job_id=job_id, created_at=self._clock()).to_dict()
        album_path = store.write_album(job_id, album)

        bundle_path = self._write_bundle(job_id, context, sources, album, inputs.write_renders)
        store.write_run_output(job_id, self._output_builder(context, result).to_json())

        displays: List[Path] = []
        if inputs.write_renders:
            by_stem = {p.stem: p for p in sources}
            picked = [by_stem[p["id"]] for p in album["picks"] if p["id"] in by_stem]
            displays = store.write_displays(job_id, picked)

        out = CurateJobResult(ok=True, album=album, album_path=str(album_path),
                              bundle_path=str(bundle_path),
                              display_paths=[str(p) for p in displays], **base)
        store.write_run_record(job_id, _run_record(out, inputs))
        logger.info("job %s: %d picks, %d dropped -> %s",
                    job_id, len(album["picks"]), len(album["dropped"]), album_path)
        return out


    def _write_bundle(self, job_id: str, context, sources: List[Path], album: Dict,
                      write_crops: bool = False) -> Path:
        """Per-photo / per-face rows + embeddings (+ worker face crops only if asked) beside the album."""
        from sim_bench.album.results_bundle import BundleInputs, ResultsBundler

        album_ids = [p["id"] for p in album["picks"]] + [d["id"] for d in album["dropped"]]
        bundle = ResultsBundler().calc(BundleInputs(
            context=context, input_ids=[p.stem for p in sources], album_ids=album_ids))
        if not write_crops:
            return self._store.write_results(job_id, bundle.manifest(), bundle.embeddings,
                                             bundle.embedding_face_ids, {})
        crops = {fid: data for fid, data in
                 ((fid, _encode_crop(arr)) for fid, arr in bundle.crops.items()) if data}
        uncropped = [f for f in bundle.faces if f["face_id"] not in crops]
        if uncropped:
            # align_faces skips gate-rejected faces, so those have no crop by design; only a
            # face that reached a person without a crop is worth a warning
            in_person = sum(f["person_id"] is not None for f in uncropped)
            log = logger.warning if in_person else logger.info
            log("job %s: %d of %d faces have no aligned crop (%d of them assigned to a person)",
                job_id, len(uncropped), len(bundle.faces), in_person)
        return self._store.write_results(job_id, bundle.manifest(), bundle.embeddings,
                                         bundle.embedding_face_ids, crops)


def _encode_crop(aligned) -> Optional[bytes]:
    """RGB aligned face (align_faces) -> 112x112 JPEG bytes, or None if unusable."""
    import io

    from sim_bench.pipeline.steps.face_cluster_export import _crop_from_aligned

    img = _crop_from_aligned(aligned)
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _timing(result) -> Dict:
    steps = getattr(result, "step_results", None) or []
    return {
        "total_ms": getattr(result, "total_duration_ms", None),
        "steps": [{"step": getattr(s, "step_name", None), "ms": getattr(s, "duration_ms", None)}
                  for s in steps],
    }


def _run_record(result: CurateJobResult, inputs: CurateJobInputs) -> Dict:
    """Debug trail beside the album: what ran, where, how long, what failed."""
    return {
        "job_id": result.job_id,
        "ok": result.ok,
        "pipeline": inputs.pipeline,
        "device": result.device,
        "n_inputs": result.n_inputs,
        "device_flipped_steps": result.flipped,
        "device_refused_steps": result.refused,
        "timing": result.timing,
        "failed_step": result.failed_step,
        "error": result.error,
        "settings": result.settings,
    }
