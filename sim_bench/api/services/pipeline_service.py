"""Pipeline service - every album run goes through here, on this PC or on the L4 (spec-116).

    start_pipeline   build the RunSpec, check it, create the run row and a job with the same id
    execute_pipeline JobService stages the photos and runs them on the chosen machine; the worker writes the run's
                     rows itself (spec-117 R6, WorkerPersister -> RunPersister, original paths). A worker that did
                     not (an in-process fake) gets them written here, the same way
    live / get_status  read the job row only: stage, step, in-step fraction, heartbeat (spec-117 R5)

The models never load in this process: runs on this PC use a child process, like every job (spec-104).
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from sim_bench.api.database.models import Album, Person, PipelineRun, PipelineResult
from sim_bench.api.schemas.pipeline import RunRow
from sim_bench.api.services.config_service import ConfigService
from sim_bench.control.jobs import FAILED, SUCCEEDED, TERMINAL
from sim_bench.pipeline.run_spec import RunSpec
from sim_bench.pipeline.spec import validate_spec_or_raise
from sim_bench.serving.job_service import Estimate, JobService, SubmitInputs
from sim_bench.serving.worker_persister import write_run_rows

# No more hardcoded pipeline - loaded from config service

_RAN_ON = {"local": "This PC", "modal_gpu": "L4 (GPU)", "modal_cpu": "Modal CPU"}


@dataclass
class JobProgress:
    """Progress update for a running job."""
    step: str
    progress: float
    message: str


@dataclass
class JobState:
    """State for a running job."""
    run_id: str
    album_id: str
    inputs: SubmitInputs
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    completed: bool = False


# Shared job storage across all PipelineService instances
_jobs: dict[str, JobState] = {}


class PipelineService:
    """Service for running pipelines."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None,
        job_service: Optional[JobService] = None,
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)
        self._job_service = job_service

    @property
    def _service(self) -> JobService:
        if self._job_service is None:
            from sim_bench.api.routers.jobs import get_job_service   # lazy: the router imports this layer's peers
            self._job_service = get_job_service()
        return self._job_service

    def start_pipeline(
        self,
        album_id: str,
        steps: list[str] = None,
        pipeline_name: str = "default_pipeline",
        step_configs: dict[str, dict] = None,
        fail_fast: bool = True,
        runner: str = "local",
    ) -> str:
        """
        Start a pipeline run.

        Args:
            album_id: Album to process
            steps: Explicit step list (overrides pipeline_name if provided)
            pipeline_name: Name of pipeline from config (e.g., "default_pipeline", "minimal_pipeline")
            step_configs: Step configuration overrides
            fail_fast: Stop on first error
            runner: Where it runs: local | modal_gpu | modal_cpu

        Returns the run id, which is also the id of its job (R11). Raises ``ValueError`` when the album or the
        spec is bad, or the L4 spend cap refuses it (R1); then nothing is left behind.
        """
        self._logger.info(f"Starting pipeline for album {album_id} on {runner}")

        album = self._session.query(Album).filter(Album.id == album_id).first()
        if album is None:
            self._logger.error(f"Album not found: {album_id}")
            raise ValueError(f"Album not found: {album_id}")

        # Load steps from config if not explicitly provided
        if steps is None:
            config_service = ConfigService(self._session)
            config = config_service.get_default_profile().config
            steps = config.get(pipeline_name, config.get("default_pipeline", []))
            self._logger.info(f"Using pipeline '{pipeline_name}' with {len(steps)} steps")

        # Before anything is created, so a bad spec costs neither a run row nor L4 time.
        spec = RunSpec.build(steps, step_configs, fail_fast=fail_fast, album_name=album.name)
        validate_spec_or_raise(spec.to_pipeline_spec())

        run_id = str(uuid.uuid4())
        self._logger.info(f"Created pipeline run {run_id} with steps: {steps}")
        inputs = SubmitInputs(source=Path(album.source_path), job_id=run_id, runner=runner,
                              pipeline=pipeline_name.removesuffix("_pipeline"), run_spec=spec.to_dict())
        self._service.create(inputs)     # the spend cap lives here

        run = PipelineRun(
            id=run_id,
            album_id=album_id,
            pipeline_name=pipeline_name,
            steps=steps,
            step_configs=step_configs or {},
            fail_fast=fail_fast,
            status="pending"
        )
        self._session.add(run)
        self._session.commit()

        _jobs[run_id] = JobState(run_id=run_id, album_id=album_id, inputs=inputs)
        return run_id

    def execute_pipeline(self, job_id: str) -> None:
        """Run it to an end state, synchronously: completed with every row, or failed with none (R8)."""
        self._logger.info(f"Executing pipeline {job_id}")

        job = _jobs.get(job_id)
        if job is None:
            self._logger.error(f"Job not found: {job_id}")
            raise ValueError(f"Job not found: {job_id}")

        run = self._session.query(PipelineRun).filter(PipelineRun.id == job_id).first()
        if run is None:
            raise ValueError(f"Pipeline run not found: {job_id}")

        run.status = "running"
        run.started_at = datetime.utcnow()
        self._session.commit()     # holds no transaction open during the run

        def write_rows(_job_id: str) -> None:       # only when the worker did not write them itself
            write_run_rows(self._session, self._service.store, job_id, job.inputs.runner)

        res = self._service.run(job_id, replace(job.inputs, on_loaded=write_rows))
        self._session.expire_all()                  # the worker wrote the run row in its own session
        if res.status != SUCCEEDED:
            self._logger.error(f"Pipeline {job_id} failed: {res.error}")
            self._mark_failed(job_id, res.error)
        else:
            self._logger.info(f"Pipeline {job_id} completed")
        job.completed = True

    def _mark_failed(self, run_id: str, error: Optional[str]) -> None:
        self._session.rollback()
        run = self._session.query(PipelineRun).filter(PipelineRun.id == run_id).first()
        failed_job = self._service.repo.get(run_id)
        step = failed_job.failed_step if failed_job else None
        run.status = "failed"
        run.error_message = f"{step}: {error}" if step else error
        run.completed_at = datetime.utcnow()
        run.progress = 1.0
        self._session.commit()

    def cancel(self, run_id: str) -> None:
        """Stop a run on either machine (R7). Raises ``KeyError`` if it has no job, ``ValueError`` if too late."""
        self._service.cancel(run_id)

    def estimate(self, album_id: str, runner: str) -> Estimate:
        album = self._session.query(Album).filter(Album.id == album_id).first()
        if album is None:
            raise ValueError(f"Album not found: {album_id}")
        return self._service.estimate(Path(album.source_path), runner=runner)

    def get_status(self, job_id: str) -> Optional[PipelineRun]:
        """Get the status of a pipeline run; a run whose process died ends failed instead of running forever."""
        run = self._session.query(PipelineRun).filter(PipelineRun.id == job_id).first()
        if run is not None and run.status in ("pending", "running"):
            self._service.repo.reap_orphans()
            job = self._service.repo.get(job_id)
            if job is not None and job.status == FAILED:
                self._mark_failed(job_id, job.error)
                run = self._session.query(PipelineRun).filter(PipelineRun.id == job_id).first()
        return run

    def live(self, run: PipelineRun) -> Dict:
        """What the job row says about a run in flight (spec-117 R5): machine, stage, step, in-step fraction and
        photo n of N, heartbeat. The same for a run on this PC and on the L4."""
        job = self._service.repo.get(run.id)
        if job is None:                  # made before spec-116
            return {}
        out: Dict = {"runner": job.runner}
        if job.gpu:
            out["gpu"] = job.gpu
        if run.status in ("pending", "running") and job.status not in TERMINAL:
            progress = job.progress or {}
            out["stage"] = progress.get("stage") or job.status if job.status == "running" else job.status
            out["current_step"] = progress.get("current_step")
            out["completed_steps"] = progress.get("completed_steps") or []
            out["started_at"] = job.started_at or run.started_at
            for key in ("step_fraction", "photos_done", "photos_total"):
                if progress.get(key) is not None:
                    out[key] = progress[key]
            if job.heartbeat_at is not None:
                out["heartbeat_at"] = job.heartbeat_at
        return out

    def list_runs(self) -> List[RunRow]:
        """Every run, newest first, for the Albums page (R10). Old runs have no job and ran on this PC."""
        runs = (self._session.query(PipelineRun).join(Album)
                .order_by(PipelineRun.created_at.desc()).all())
        jobs = {j.id: j for j in self._service.repo.list(100_000)}
        # columns only: the page redraws on every click and a result row carries every photo's metrics
        counts = {run_id: (photos, picks) for run_id, photos, picks in self._session.query(
            PipelineResult.run_id, PipelineResult.total_images, PipelineResult.num_selected).all()}
        people = dict(self._session.query(Person.run_id, func.count()).group_by(Person.run_id).all())
        rows = []
        for run in runs:
            job = jobs.get(run.id)
            photos, picks = counts.get(run.id, (None, None))
            rows.append(RunRow(
                run_id=run.id, album_id=run.album_id, album_name=run.album.name, folder=run.album.source_path,
                ran_on=_RAN_ON.get(job.runner, job.runner) if job else _RAN_ON["local"],
                created_at=run.created_at, photos=photos, picks=picks, people=people.get(run.id),
                status=run.status, comparable=bool(job and job.status == SUCCEEDED)))
        return rows

    def get_result(self, job_id: str) -> Optional[PipelineResult]:
        """Get the result of a completed pipeline run."""
        return self._session.query(PipelineResult).filter(PipelineResult.run_id == job_id).first()

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to a run's end state (the websocket polls ``get_status``; steps are not pushed any more)."""
        job = _jobs.get(job_id)
        if job is None:
            self._logger.warning(f"Subscribe failed - job not found: {job_id}")
            raise ValueError(f"Job not found: {job_id}")

        queue = asyncio.Queue()
        job.subscribers.append(queue)
        self._logger.debug(f"Client subscribed to job {job_id}")
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from progress updates."""
        job = _jobs.get(job_id)
        if job and queue in job.subscribers:
            job.subscribers.remove(queue)
            self._logger.debug(f"Client unsubscribed from job {job_id}")
