"""Run one curation job from storage and write the normalized album back (spec-110 T2.1, AC3/AC4).

Storage-reference, not inline bytes: the worker is handed a ``job_id``, reads that job's staged
images from a :class:`~sim_bench.serving.job_store.JobStore`, runs the real pipeline over a scratch
copy, normalizes the finished context into the Album contract (spec-106) and writes ``album.json``
next to the inputs.

Two deliberate choices make a re-run reproduce its result (AC4):
  * the run happens in a **scratch copy** of ``input/``, because the pipeline writes straighten
    derivatives beside its inputs -- a second run must see the same N images, not N + derivatives;
  * ``album.json`` / ``display/`` are **overwritten**, so a re-run cannot accumulate stale picks.

``created_at`` is wall-clock and therefore differs between runs by design; compare two runs with
:func:`stable_album`, which drops it.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder, SpecInputs, SpecResult

logger = logging.getLogger(__name__)

__all__ = ["CurateJobInputs", "CurateJobResult", "JobCurator", "run_pipeline", "stable_album"]

# (work_dir, plan) -> (finished PipelineContext, PipelineResult). Injectable so the orchestration
# above can be tested without the real models.
PipelineRunner = Callable[[Path, SpecResult], Tuple[object, object]]


def run_pipeline(work_dir: Path, plan: SpecResult) -> Tuple[object, object]:
    """The production runner: execute the resolved plan over ``work_dir``."""
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.run import execute_spec

    context = PipelineContext(source_directory=work_dir)
    return context, execute_spec(plan.as_pipeline_spec(), context, fail_fast=True)


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
    write_displays: bool = True


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


class JobCurator:
    """Curate a job whose images live in a store. Config in ``__init__``, per-call data in Inputs."""

    def __init__(self, store: JobStore, *,
                 spec_builder: Optional[SpecBuilder] = None,
                 runner: PipelineRunner = run_pipeline,
                 clock: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat()):
        self._store = store
        self._spec_builder = spec_builder or SpecBuilder()
        self._runner = runner
        self._clock = clock

    def calc(self, inputs: CurateJobInputs) -> CurateJobResult:
        store, job_id = self._store, inputs.job_id
        sources = store.input_images(job_id)   # validates job_id
        if not sources:
            return CurateJobResult(ok=False, job_id=job_id, device=inputs.device,
                                   error=f"job {job_id!r} has no staged inputs under "
                                         f"{store.paths(job_id).input_dir}")

        plan = self._spec_builder.calc(SpecInputs(pipeline=inputs.pipeline, device=inputs.device))
        logger.info("job %s: %d inputs, pipeline=%s device=%s (%d steps)",
                    job_id, len(sources), inputs.pipeline, inputs.device, len(plan.steps))

        scratch = Path(tempfile.mkdtemp(prefix=f"curate_{job_id}_"))
        try:
            for src in sources:
                shutil.copyfile(src, scratch / src.name)
            context, result = self._runner(scratch, plan)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

        timing = _timing(result)
        base = dict(job_id=job_id, n_inputs=len(sources), device=inputs.device,
                    flipped=list(plan.flipped), refused=list(plan.refused), timing=timing)
        if not getattr(result, "success", False):
            failed = CurateJobResult(ok=False, failed_step=getattr(result, "failed_step", None),
                                     error=getattr(result, "error_message", None), **base)
            store.write_run_record(job_id, _run_record(failed, inputs))
            return failed

        from sim_bench.album.normalizer import normalize

        album = normalize(context, job_id=job_id, created_at=self._clock()).to_dict()
        album_path = store.write_album(job_id, album)

        bundle_path = self._write_bundle(job_id, context, sources, album)

        displays: List[Path] = []
        if inputs.write_displays:
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


    def _write_bundle(self, job_id: str, context, sources: List[Path], album: Dict) -> Path:
        """Per-photo / per-face rows + embeddings + face crops beside the album (spec-112 TA1.2)."""
        from sim_bench.album.results_bundle import BundleInputs, ResultsBundler

        album_ids = [p["id"] for p in album["picks"]] + [d["id"] for d in album["dropped"]]
        bundle = ResultsBundler().calc(BundleInputs(
            context=context, input_ids=[p.stem for p in sources], album_ids=album_ids))
        crops = {fid: data for fid, data in
                 ((fid, _encode_crop(arr)) for fid, arr in bundle.crops.items()) if data}
        if len(crops) < len(bundle.faces):
            logger.warning("job %s: %d of %d faces have no aligned crop",
                           job_id, len(bundle.faces) - len(crops), len(bundle.faces))
        return self._store.write_results(job_id, bundle.manifest(), bundle.embeddings,
                                         bundle.embedding_face_ids, crops)


def _encode_crop(aligned) -> Optional[bytes]:
    """BGR aligned face (align_faces) -> 112x112 JPEG bytes, or None if unusable."""
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
    }
