"""The worker writes its own results (spec-117 R6): rows in the database, files in the object store.

Runs in the worker, after the pipeline, on this PC (the child process) and on the L4 alike:

    job  running -> loading
    files   album.json, run.json, results/*    -> object store <job_id>/...   (R2; a no-op for the local folder)
    rows    ResultsLoader: photos, faces, job_people, scenes  (+ display images and face crops, LocalRenders)
            the album run, if the job is one: pipeline_results, people, metric rows (RunPersister), with the
            ORIGINAL photo paths from the job's manifest (PathMap), and the run marked completed
    job  loading -> succeeded

Nothing is copied back to the PC. The job rows and the run rows are two transactions (two databases when
``DATABASE_URL`` is unset); a failure in either fails the job, and RunPersister leaves no partial run rows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Tuple

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import LOADING, SUCCEEDED, TERMINAL, JobRepo
from sim_bench.control.loader import LoadInputs, ResultsLoader
from sim_bench.serving.job_store import JobStore
from sim_bench.storage.object_store import job_key

logger = logging.getLogger(__name__)

__all__ = ["WorkerPersister", "PersistInputs", "PersistResult", "write_run_rows", "fail_run_row", "RESULT_FILES"]

# What the app may read of a finished job besides its rows (run settings, a run's own face embeddings).
RESULT_FILES = ("album.json", "run.json", "results/bundle.json", "results/run_output.json",
                "results/embeddings.npy", "results/embedding_face_ids.npy")

ApiSessions = Callable[[], object]          # () -> sqlalchemy Session on the API tables


@dataclass(frozen=True, slots=True)
class PersistInputs:
    job_id: str
    runner: str = "local"            # an L4 run keeps no worker export folder (spec-116)


@dataclass(frozen=True, slots=True)
class PersistResult:
    job_id: str
    n_photos: int = 0
    n_picks: int = 0
    n_faces: int = 0
    n_people: int = 0
    run_rows: bool = False           # the album run's rows were written
    uploaded: int = 0                # result files put in the object store
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def write_run_rows(session, store: JobStore, job_id: str, runner: str) -> bool:
    """The album run behind job ``job_id`` (same id, spec-116 R11): its rows and its completed status, one commit.
    False when the job is not an album run (Jobs page, regression). Raises when the run's output is missing."""
    from sim_bench.api.database.models import PipelineRun
    from sim_bench.api.services.run_persister import RunPersister
    from sim_bench.serving.path_map import PathMap
    from sim_bench.serving.run_output import RunOutput

    run = session.get(PipelineRun, job_id)
    if run is None:
        return False
    text = store.read_run_output(job_id)
    if text is None:
        raise FileNotFoundError(f"run {job_id} returned no results/run_output.json")
    output = RunOutput.from_json(text).remap(PathMap.from_job(store, job_id))
    if runner != "local":
        output.fc_export_dir = None       # that folder stayed on the worker
    RunPersister(session).persist(run.album_id, job_id, output)
    run.completed_steps = [{"step": s, "duration_ms": ms, "status": "completed", "error": None}
                           for s, ms in output.step_timings.items()]
    run.status = "completed"
    run.current_step = None
    run.completed_at = datetime.utcnow()
    run.progress = 1.0
    session.commit()
    return True


def fail_run_row(session, job_id: str, error: Optional[str], failed_step: Optional[str] = None) -> None:
    """Mark the album run behind ``job_id`` failed (no-op for a job that is not one)."""
    from sim_bench.api.database.models import PipelineRun

    session.rollback()
    run = session.get(PipelineRun, job_id)
    if run is None or run.status in ("completed", "failed"):
        return
    run.status = "failed"
    run.error_message = f"{failed_step}: {error}" if failed_step else error
    run.completed_at = datetime.utcnow()
    run.progress = 1.0
    session.commit()


class WorkerPersister:
    """Config in ``__init__``: the job database, the job store (with its object store), the API tables."""

    def __init__(self, db: ControlDB, store: JobStore, *, api_sessions: Optional[ApiSessions] = None):
        self.db, self.store = db, store
        self._repo = JobRepo(db)
        self._api_sessions = api_sessions

    def calc(self, inputs: PersistInputs) -> PersistResult:
        job_id = inputs.job_id
        self._repo.transition(job_id, LOADING)
        uploaded = self._upload(job_id)
        loaded = ResultsLoader(self.db, self.store).calc(LoadInputs(job_id))
        for w in loaded.warnings:
            self._repo.log(job_id, w, kind="warning")
        run_rows = False
        if self._api_sessions is not None:
            session = self._api_sessions()
            try:
                run_rows = write_run_rows(session, self.store, job_id, inputs.runner)
            finally:
                session.close()
        self._repo.transition(job_id, SUCCEEDED, f"{loaded.n_picks} picks of {loaded.n_photos} photos, "
                                                 f"{loaded.n_faces} faces, {loaded.n_people} people")
        logger.info("job %s: results written by the worker (%d files stored, run rows: %s)", job_id, uploaded,
                    run_rows)
        return PersistResult(job_id=job_id, n_photos=loaded.n_photos, n_picks=loaded.n_picks,
                             n_faces=loaded.n_faces, n_people=loaded.n_people, run_rows=run_rows,
                             uploaded=uploaded, warnings=tuple(loaded.warnings))

    def fail(self, job_id: str, error: Optional[str], failed_step: Optional[str] = None) -> None:
        """The run failed in the worker: the job (and its album run) end failed with the reason, written here so
        nobody has to be watching (R9). Never raises: the caller is already handling a failure."""
        try:
            job = self._repo.get(job_id)
            if job is not None and job.status not in TERMINAL:
                self._repo.fail(job_id, error or "run failed", failed_step)
        except Exception as e:
            logger.warning("job %s: could not record the failure: %s", job_id, type(e).__name__)
        if self._api_sessions is None:
            return
        try:
            session = self._api_sessions()
            try:
                fail_run_row(session, job_id, error, failed_step)
            finally:
                session.close()
        except Exception as e:
            logger.warning("job %s: could not mark its run failed: %s", job_id, type(e).__name__)

    def _upload(self, job_id: str) -> int:
        """The result files the app reads, into the object store (nothing to do when it is the job folder)."""
        objects = self.store.objects
        if objects.is_local:
            return 0
        root = self.store.paths(job_id).root
        n = 0
        for rel in RESULT_FILES:
            path = root / rel
            if path.is_file():
                objects.put_file(job_key(job_id, rel), path)
                n += 1
        return n

