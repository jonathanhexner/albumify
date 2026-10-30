"""The control plane's one job flow (spec-112 TA4.2):

    create (queued) -> stage (staging) -> runner.run (running) -> load (loading) -> succeeded
                                     any failure or exception ---------------------> failed

Same code whatever the runner; only ``runner`` differs. The runner writes files, this service is
the only thing that writes the DB. Synchronous by design: the API calls it from a background task,
the CLI calls it directly.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import (FAILED, LOADING, QUEUED, RUNNING, STAGING, SUCCEEDED, TERMINAL,
                                    JobRepo, this_worker)
from sim_bench.control.loader import LoadInputs, ResultsLoader
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.runners import CANCELLED, Runner, make_runner
from sim_bench.serving.serving_config import ServingConfig
from sim_bench.serving.staging import downsample, select_images

logger = logging.getLogger(__name__)

__all__ = ["SubmitInputs", "SubmitResult", "Estimate", "JobService", "RemoteError", "new_job_id"]


def new_job_id(source: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(source).name).strip("-")[:40] or "job"
    # random suffix: two submits of one folder in the same second must not collide (review L4)
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class RemoteError(RuntimeError):
    """The worker side (Modal) failed an owner request; the API answers 502."""


class _Cancelled(Exception):
    """Raised inside ``run`` once the owner cancelled the job."""


@dataclass(frozen=True, slots=True)
class SubmitInputs:
    source: Optional[Path] = None      # folder of photos; None = inputs already staged
    job_id: Optional[str] = None
    runner: Optional[str] = None       # overrides ServingConfig.runner
    pipeline: Optional[str] = None
    start: int = 0
    n: Optional[int] = None
    tag: Optional[str] = None          # e.g. regression:slovakia (spec-114); None = an owner's album
    rerun_of: Optional[str] = None     # spec-115 R5: the same photos as this earlier job (its manifest)


@dataclass(frozen=True, slots=True)
class SubmitResult:
    job_id: str
    status: str
    error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Estimate:
    """What a job would cost before it starts (spec-115 R2). GPU figures only for the L4 runner."""

    runner: str
    n_photos: int
    gpu_minutes: Optional[float] = None
    dollars: Optional[float] = None
    allowed: bool = True
    reason: Optional[str] = None


RunnerFactory = Callable[[str, JobStore, str, str], Runner]


def _default_factory(name: str, store: JobStore, pipeline: str, device: str) -> Runner:
    return make_runner(name, store, pipeline=pipeline, device=device)


class JobService:
    def __init__(self, db: ControlDB, config: ServingConfig, *,
                 store: Optional[JobStore] = None,
                 runner_factory: RunnerFactory = _default_factory):
        self.db, self.config = db, config
        self.store = store or JobStore(config.jobs_root)
        self.repo = JobRepo(db)
        self._factory = runner_factory
        self._active: Dict[str, Runner] = {}          # job id -> the runner working on it (for cancel)
        self._cancel_requested: Set[str] = set()
        self._modal = threading.Lock()                 # one Modal job at a time: one cache writer (R4, review M4)

    def estimate(self, source: Optional[Path], start: int = 0, n: Optional[int] = None,
                 runner: Optional[str] = None, rerun_of: Optional[str] = None) -> Estimate:
        """Photo count and, for the L4, GPU minutes and dollars against the spend cap (spec-115 R2)."""
        cfg = self.config.with_runner(runner)
        if rerun_of is not None:
            count = len(self.store.input_keys(rerun_of) or [])
        elif source is None or not Path(source).is_dir():
            raise ValueError(f"source folder not found: {source}")
        else:
            count = len(select_images(Path(source), start, n))
        if cfg.runner != "modal_gpu":
            return Estimate(runner=cfg.runner, n_photos=count)
        minutes = round((cfg.l4_fixed_seconds + count * cfg.l4_seconds_per_photo) / 60, 1)
        dollars = round(minutes / 60 * cfg.l4_dollars_per_hour, 2)
        reason = None
        if count > cfg.max_photos_l4:
            reason = f"{count} photos is over the L4 limit of {cfg.max_photos_l4}; choose a smaller range"
        elif minutes > cfg.max_gpu_minutes:
            reason = f"about {minutes} GPU minutes is over the limit of {cfg.max_gpu_minutes:g}"
        return Estimate(runner=cfg.runner, n_photos=count, gpu_minutes=minutes, dollars=dollars,
                        allowed=reason is None, reason=reason)

    def cancel(self, job_id: str) -> None:
        """Stop a queued or running job (spec-115 R4). It ends ``failed`` / "cancelled by owner"."""
        job = self.repo.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in TERMINAL:
            raise ValueError(f"job {job_id!r} is already {job.status}")
        if job.status == LOADING:
            raise ValueError(f"job {job_id!r} has finished running and is loading its results")
        self._cancel_requested.add(job_id)
        runner = self._active.get(job_id)
        if runner is not None:
            runner.cancel(job_id)
        elif job.status == QUEUED:
            self.repo.fail(job_id, CANCELLED)

    def run_settings(self, job_id: str) -> Optional[Dict]:
        """The settings the run recorded in run.json (spec-115 R12), if any."""
        path = self.store.paths(job_id).run_path
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("settings") or None
        except (OSError, ValueError):     # not there yet, or still being copied back
            return None

    def delete_remote_inputs(self, job_id: str, remote=None) -> Dict:
        """Owner's "delete my photos from Modal" (spec-115 R13); photos another job used within 7 days stay."""
        job = self.repo.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in TERMINAL:     # its photos are in use (review M7)
            raise ValueError(f"job {job_id!r} is {job.status}; delete its photos once it has finished")
        keys = self.store.input_keys(job_id)
        if not keys:
            raise ValueError(f"job {job_id!r} has no photos in the shared store")
        if remote is None:
            from sim_bench.serving.runners import ModalVolumeRemote
            remote = ModalVolumeRemote()
        try:
            res = remote.delete_inputs(job_id, keys)
        except Exception as e:
            raise RemoteError(f"Modal did not delete the photos: {type(e).__name__}: {e}") from e
        self.repo.log(job_id, f"deleted {len(res.get('deleted', []))} photos from Modal "
                              f"({res.get('kept', 0)} kept: used by another job in the last 7 days)")
        return res

    def _check_cancel(self, job_id: str) -> None:
        if job_id in self._cancel_requested:
            raise _Cancelled()

    def _discard_partial_results(self, job_id: str) -> None:
        """A cancelled job keeps its staged photos (shared) and nothing else it produced."""
        paths = self.store.paths(job_id)
        for f in (paths.album_path, paths.run_path):
            f.unlink(missing_ok=True)
        for d in (paths.results_dir, paths.display_dir, paths.faces_dir):
            shutil.rmtree(d, ignore_errors=True)

    def create(self, inputs: SubmitInputs) -> str:
        """Record the job (queued) and return its id; ``run`` does the work."""
        if inputs.rerun_of is not None:
            if self.repo.get(inputs.rerun_of) is None:
                raise ValueError(f"no job {inputs.rerun_of!r} to re-run")
            if not self.store.input_keys(inputs.rerun_of):
                raise ValueError(f"job {inputs.rerun_of!r} predates shared staging; re-run it from its folder")
        elif inputs.source is None and not inputs.job_id:
            raise ValueError("either a source folder, a job to re-run, or an already-staged job_id is required")
        if inputs.source is not None and not Path(inputs.source).is_dir():
            raise ValueError(f"source folder not found: {inputs.source}")
        cfg = self.config.with_runner(inputs.runner)
        if cfg.runner == "modal_gpu":      # every L4 path is checked against the spend cap (review L2)
            est = self.estimate(inputs.source, inputs.start, inputs.n, cfg.runner,
                                rerun_of=inputs.rerun_of or (inputs.job_id if inputs.source is None else None))
            if not est.allowed:
                raise ValueError(est.reason)
        source = inputs.source if inputs.source is not None else (
            self.repo.get(inputs.rerun_of).source if inputs.rerun_of else None)
        job_id = inputs.job_id or new_job_id(Path(source) if source else Path(inputs.rerun_of or "job"))
        self.store.paths(job_id)   # validates the id before anything is written
        self.repo.create(job_id, runner=cfg.runner, pipeline=inputs.pipeline or cfg.pipeline,
                         source=str(source) if source else None, tag=inputs.tag)
        if inputs.rerun_of:
            self.repo.log(job_id, f"re-run of {inputs.rerun_of}")
        return job_id

    def _stage(self, job_id: str, source: Path, start: int, n: Optional[int]) -> None:
        """Put the folder's photos in the shared input store (only new ones are resized) and point
        the job at them (spec-112 A2)."""
        keys, fresh = [], 0
        for photo in select_images(source, start, n):
            try:
                key = self.store.inputs.key_for(photo)
                if not self.store.inputs.has(key):
                    self.store.inputs.put_bytes(key, downsample(photo))
                    fresh += 1
                keys.append(key)
            except Exception as e:   # one unreadable file must not sink the job; it is logged
                logger.warning("skipping unreadable image %s: %s", photo, e)
        if not keys:
            raise ValueError(f"no readable images in {source}")
        self.store.write_input_manifest(job_id, keys)
        self.repo.log(job_id, f"staged {len(keys)} photos ({fresh} new, {len(keys) - fresh} already staged)")

    def run(self, job_id: str, inputs: SubmitInputs) -> SubmitResult:
        """Stage -> run -> load. Never leaves the job in a live status (AC4)."""
        cfg = self.config.with_runner(inputs.runner)
        pipeline = inputs.pipeline or cfg.pipeline
        try:
            self._check_cancel(job_id)
            self.repo.transition(job_id, STAGING, worker=this_worker())
            if inputs.rerun_of is not None:
                keys = self.store.input_keys(inputs.rerun_of)
                self.store.write_input_manifest(job_id, keys)
                self.repo.log(job_id, f"staged {len(keys)} photos (the same photos as {inputs.rerun_of})")
            elif inputs.source is not None:
                self._stage(job_id, Path(inputs.source), inputs.start, inputs.n)
            n = len(self.store.input_images(job_id))
            if n == 0:
                raise ValueError(f"job {job_id!r} has no staged inputs")
            self._check_cancel(job_id)
            self.repo.transition(job_id, RUNNING, f"{n} photos on {cfg.runner} ({pipeline})",
                                 n_inputs=n)

            runner = self._factory(cfg.runner, self.store, pipeline, cfg.device)
            gated = cfg.runner.startswith("modal")
            if gated:
                self._wait_for_modal(job_id)
            try:
                self._check_cancel(job_id)
                self._active[job_id] = runner
                try:
                    outcome = runner.run(job_id)
                finally:
                    self._active.pop(job_id, None)
            finally:
                if gated:
                    self._modal.release()
            for note in outcome.notes:
                self.repo.log(job_id, note)
            if outcome.error == CANCELLED or job_id in self._cancel_requested:
                raise _Cancelled()
            if not outcome.ok:
                self.repo.fail(job_id, outcome.error or "run failed", outcome.failed_step)
                return SubmitResult(job_id, FAILED, outcome.error)
            if outcome.wall_s is not None:
                self.repo.log(job_id, f"pipeline {outcome.wall_s}s")

            self.repo.transition(job_id, LOADING)
            res = ResultsLoader(self.db, self.store).calc(LoadInputs(job_id))
            for w in res.warnings:
                self.repo.log(job_id, w, kind="warning")
            self.repo.transition(job_id, SUCCEEDED,
                                 f"{res.n_picks} picks of {res.n_photos} photos, "
                                 f"{res.n_faces} faces, {res.n_people} people")
            return SubmitResult(job_id, SUCCEEDED)
        except _Cancelled:
            self._discard_partial_results(job_id)
            if self.repo.get(job_id).status not in TERMINAL:
                self.repo.fail(job_id, CANCELLED)
            return SubmitResult(job_id, FAILED, CANCELLED)
        except Exception as e:  # any crash ends in `failed` with the error recorded
            if job_id in self._cancel_requested and self.repo.get(job_id).status in TERMINAL:
                return SubmitResult(job_id, FAILED, CANCELLED)   # cancelled while queued (review L3)
            logger.exception("job %s failed", job_id)
            msg = f"{type(e).__name__}: {e}"
            if self.repo.get(job_id).status not in TERMINAL:
                self.repo.fail(job_id, msg)
            return SubmitResult(job_id, FAILED, msg)
        finally:
            self._cancel_requested.discard(job_id)

    def _wait_for_modal(self, job_id: str) -> None:
        """One Modal job at a time; a job waiting its turn can still be cancelled (review M1)."""
        if not self._modal.acquire(blocking=False):
            self.repo.log(job_id, "waiting for the other Modal job to finish")
            while not self._modal.acquire(timeout=1):
                self._check_cancel(job_id)

    def submit(self, inputs: SubmitInputs) -> SubmitResult:
        return self.run(self.create(inputs), inputs)

    def load_existing(self, job_id: str, *, runner: str, source: Optional[str] = None,
                      tag: Optional[str] = None) -> SubmitResult:
        """Record + load a job whose files are already in the store (e.g. an earlier Modal run)."""
        self.repo.create(job_id, runner=runner, source=source, tag=tag)
        try:
            self.repo.transition(job_id, RUNNING, "files already present; not re-run",
                                 worker=this_worker())
            self.repo.transition(job_id, LOADING)
            res = ResultsLoader(self.db, self.store).calc(LoadInputs(job_id))
            self.repo.transition(job_id, SUCCEEDED, f"{res.n_picks} picks of {res.n_photos} photos",
                                 n_inputs=res.n_photos)
            return SubmitResult(job_id, SUCCEEDED)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.repo.fail(job_id, msg)
            return SubmitResult(job_id, FAILED, msg)
