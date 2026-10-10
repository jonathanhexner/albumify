"""The control plane's one job flow (spec-112 TA4.2):

    create (queued) -> stage (staging) -> runner.run (running) -> load (loading) -> succeeded
                                     any failure or exception ---------------------> failed

Same code whatever the runner; only ``runner`` differs. The runner writes files, this service is
the only thing that writes the DB. Synchronous by design: the API calls it from a background task,
the CLI calls it directly.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import FAILED, LOADING, RUNNING, STAGING, SUCCEEDED, JobRepo
from sim_bench.control.loader import LoadInputs, ResultsLoader
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.runners import Runner, make_runner
from sim_bench.serving.serving_config import ServingConfig
from sim_bench.serving.staging import collect_images

logger = logging.getLogger(__name__)

__all__ = ["SubmitInputs", "SubmitResult", "JobService", "new_job_id"]


def new_job_id(source: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(source).name).strip("-")[:40] or "job"
    return f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}"


@dataclass(frozen=True, slots=True)
class SubmitInputs:
    source: Optional[Path] = None      # folder of photos; None = inputs already staged
    job_id: Optional[str] = None
    runner: Optional[str] = None       # overrides ServingConfig.runner
    pipeline: Optional[str] = None
    start: int = 0
    n: Optional[int] = None


@dataclass(frozen=True, slots=True)
class SubmitResult:
    job_id: str
    status: str
    error: Optional[str] = None


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

    def create(self, inputs: SubmitInputs) -> str:
        """Record the job (queued) and return its id; ``run`` does the work."""
        if inputs.source is None and not inputs.job_id:
            raise ValueError("either a source folder or an already-staged job_id is required")
        if inputs.source is not None and not Path(inputs.source).is_dir():
            raise ValueError(f"source folder not found: {inputs.source}")
        cfg = self.config.with_runner(inputs.runner)
        job_id = inputs.job_id or new_job_id(inputs.source)
        self.store.paths(job_id)   # validates the id before anything is written
        self.repo.create(job_id, runner=cfg.runner, pipeline=inputs.pipeline or cfg.pipeline,
                         source=str(inputs.source) if inputs.source else None)
        return job_id

    def run(self, job_id: str, inputs: SubmitInputs) -> SubmitResult:
        """Stage -> run -> load. Never leaves the job in a live status (AC4)."""
        cfg = self.config.with_runner(inputs.runner)
        pipeline = inputs.pipeline or cfg.pipeline
        try:
            self.repo.transition(job_id, STAGING)
            if inputs.source is not None:
                images = collect_images(Path(inputs.source), inputs.start, inputs.n)
                if not images:
                    raise ValueError(f"no readable images in {inputs.source}")
                self.store.stage_inputs(job_id, images)
            n = len(self.store.input_images(job_id))
            if n == 0:
                raise ValueError(f"job {job_id!r} has no staged inputs")
            self.repo.transition(job_id, RUNNING, f"{n} photos on {cfg.runner} ({pipeline})",
                                 n_inputs=n)

            outcome = self._factory(cfg.runner, self.store, pipeline, cfg.device).run(job_id)
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
        except Exception as e:  # any crash ends in `failed` with the error recorded
            logger.exception("job %s failed", job_id)
            msg = f"{type(e).__name__}: {e}"
            self.repo.fail(job_id, msg)
            return SubmitResult(job_id, FAILED, msg)

    def submit(self, inputs: SubmitInputs) -> SubmitResult:
        return self.run(self.create(inputs), inputs)

    def load_existing(self, job_id: str, *, runner: str, source: Optional[str] = None) -> SubmitResult:
        """Record + load a job whose files are already in the store (e.g. an earlier Modal run)."""
        self.repo.create(job_id, runner=runner, source=source)
        try:
            self.repo.transition(job_id, RUNNING, "files already present; not re-run")
            self.repo.transition(job_id, LOADING)
            res = ResultsLoader(self.db, self.store).calc(LoadInputs(job_id))
            self.repo.transition(job_id, SUCCEEDED, f"{res.n_picks} picks of {res.n_photos} photos",
                                 n_inputs=res.n_photos)
            return SubmitResult(job_id, SUCCEEDED)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self.repo.fail(job_id, msg)
            return SubmitResult(job_id, FAILED, msg)
