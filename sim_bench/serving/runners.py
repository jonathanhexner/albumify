"""Where a job runs (spec-112 TA4.3, spec-115 W3/W4). Every runner has the same contract:

    run(job_id) -> RunOutcome   once the run has ended
    cancel(job_id)              stop it; run() then returns ok=False, error "cancelled by owner"

spec-117: the worker writes its own progress and results to the database (WorkerPersister), on this PC and on
the L4, so the job row is the outcome; RunOutcome only carries what the control plane logs.

    LocalRunner   JobCurator in a child process: torch / ONNX memory is only returned to the OS
                  when a process exits (spec-104, SIGHTING-117), so a long-lived API must not run
                  the models in-process.
    ModalRunner   upload the photos the Modal input store lacks (by content key), spawn curate_job on the
                  deployed app and follow the job row by call id (spec-117 R8); nothing comes back.
"""
from __future__ import annotations

import json
import os
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

from sim_bench.serving.job_store import JobStore
from sim_bench.serving.run_settings import COMMIT_ENV, current_commit

logger = logging.getLogger(__name__)

__all__ = ["RunOutcome", "Runner", "LocalRunner", "ModalRunner", "ModalVolumeRemote",
           "make_runner", "CANCELLED"]

CANCELLED = "cancelled by owner"
# The job database the local worker writes its progress and results to (spec-117 R6). An env var, not an argument:
# a Postgres URL carries a password, and a command line is visible to every process on the machine.
CONTROL_DB_ENV = "ALBUMIFY_CONTROL_DB_URL"


@dataclass(frozen=True, slots=True)
class RunOutcome:
    ok: bool
    error: Optional[str] = None
    failed_step: Optional[str] = None
    wall_s: Optional[float] = None
    notes: Tuple[str, ...] = ()        # facts for the job's event log (e.g. "uploaded 3 new photos")


class Runner(Protocol):
    name: str

    def run(self, job_id: str) -> RunOutcome: ...

    def cancel(self, job_id: str) -> None: ...


def _outcome_from_record(store: JobStore, job_id: str, fallback_error: str,
                         notes: Sequence[str] = ()) -> RunOutcome:
    path = store.paths(job_id).run_path
    if not path.is_file():
        return RunOutcome(ok=False, error=fallback_error, notes=tuple(notes))
    rec = json.loads(path.read_text(encoding="utf-8"))
    total = (rec.get("timing") or {}).get("total_ms")
    return RunOutcome(ok=bool(rec.get("ok")), error=rec.get("error"),
                      failed_step=rec.get("failed_step"),
                      wall_s=round(total / 1000, 1) if total else None, notes=tuple(notes))


class LocalRunner:
    name = "local"

    def __init__(self, store: JobStore, *, pipeline: str = "default", device: str = "cpu",
                 timeout_s: int = 6 * 3600, db=None):
        self._store, self._pipeline, self._device, self._timeout = store, pipeline, device, timeout_s
        self._db_url = db.url if db is not None else None
        self._procs: dict = {}
        self._cancelled: set = set()

    def command(self, job_id: str) -> list:
        return [sys.executable, "-m", "sim_bench.serving.runners", "--root", str(self._store.root),
                "--job-id", job_id, "--pipeline", self._pipeline, "--device", self._device]

    def run(self, job_id: str) -> RunOutcome:
        run_path = self._store.paths(job_id).run_path
        if run_path.exists():
            run_path.unlink()   # a stale record from an earlier run must not read as this one's
        logger.info("job %s: local child process (%s, %s)", job_id, self._pipeline, self._device)
        # UTF-8 both ways: with the Windows locale code page (e.g. cp1255) one undecodable
        # byte killed the reader thread and lost the child's output, error tail included.
        proc = subprocess.Popen(self.command(job_id), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                env=self.child_env())
        self._procs[job_id] = proc
        if job_id in self._cancelled:          # cancelled while the child was starting (review M3)
            proc.kill()
        try:
            _, stderr = proc.communicate(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return RunOutcome(ok=False, error=f"local run exceeded {self._timeout}s")
        finally:
            self._procs.pop(job_id, None)
        if job_id in self._cancelled:
            self._cancelled.discard(job_id)
            return RunOutcome(ok=False, error=CANCELLED)
        tail = (stderr or "").strip().splitlines()[-5:]
        return _outcome_from_record(
            self._store, job_id,
            f"worker process exited {proc.returncode} without a run record: {' | '.join(tail)}")

    def child_env(self) -> dict:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", COMMIT_ENV: current_commit() or ""}
        if self._db_url:
            env[CONTROL_DB_ENV] = self._db_url
        return env

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.kill()


INPUTS_VOLUME, JOBS_VOLUME = "albumify-inputs", "albumify-jobs"


class ModalVolumeRemote:
    """The Modal Volume side: which photos the inputs Volume lacks, upload them, delete them (spec-115 R13).
    Runs are started by ``DetachedModalRemote`` (spec-117). ``modal`` is imported lazily."""

    def __init__(self, gpu: bool = True):
        self.gpu = gpu

    def missing(self, keys: Sequence[str]) -> List[str]:
        """Input-store keys not yet on the ``albumify-inputs`` Volume (one listing)."""
        import modal
        vol = modal.Volume.from_name(INPUTS_VOLUME, create_if_missing=True)
        present = {e.path.lstrip("/") for e in vol.listdir("/", recursive=True)}
        return [k for k in keys if k not in present]

    def upload(self, keys: Sequence[str], store: JobStore) -> None:
        import modal
        if not keys:
            return
        import io
        import json
        import time
        vol = modal.Volume.from_name(INPUTS_VOLUME, create_if_missing=True)
        claim = json.dumps({"uploaded": time.time()}).encode()
        with vol.batch_upload() as batch:
            for k in keys:
                batch.put_file(str(store.inputs.path(k)), "/" + k)
                # a just-uploaded photo counts as used now (review M2); the worker adds the job id later
                batch.put_file(io.BytesIO(claim), "/" + k.split("/", 1)[0] + "/.used.json")

    def delete_inputs(self, job_id: str, keys: Sequence[str]) -> dict:
        import modal
        from sim_bench.serving.modal_app import cpu_app, delete_inputs
        with modal.enable_output(), cpu_app.run():
            return delete_inputs.remote(job_id, list(keys))


class ModalRunner:
    """A detached run on the deployed Modal app (spec-117 R8): upload the photos it lacks, spawn, store the call id
    on the job, then follow the JOB ROW until the worker ends it. Nothing is copied back: the worker writes the
    rows and the files. If this process goes away, the run goes on and still lands in the database.

    ``volumes`` uploads photos (``ModalVolumeRemote``); ``remote`` starts / follows / cancels by call id
    (``DetachedModalRemote``). A test fake may be both."""

    def __init__(self, store: JobStore, *, gpu: bool = True, pipeline: str = "default",
                 remote=None, volumes=None, db=None, poll_s: float = 2.0):
        from sim_bench.serving.detached_modal import DetachedModalRemote

        self.name = "modal_gpu" if gpu else "modal_cpu"
        self._store, self._pipeline, self._db, self._poll_s = store, pipeline, db, poll_s
        self._remote = remote if remote is not None else DetachedModalRemote(gpu=gpu)
        self._volumes = volumes if volumes is not None else (remote if remote is not None
                                                             else ModalVolumeRemote(gpu=gpu))
        self._cancelled = threading.Event()
        self._call_id: Optional[str] = None

    def run(self, job_id: str) -> RunOutcome:
        from sim_bench.control.jobs import JobRepo

        paths = self._store.paths(job_id)
        keys = self._store.input_keys(job_id)
        if not keys:
            return RunOutcome(ok=False, error=f"job {job_id!r} has no input manifest under {paths.input_dir}; "
                                              "re-run it from its source folder")
        if self._db is None:
            return RunOutcome(ok=False, error="an L4 run needs the job database")
        if getattr(self._remote, "needs_shared_db", False) and self._db.is_sqlite:
            return RunOutcome(ok=False, error="L4 runs need DATABASE_URL (Neon) in .env: the L4 worker writes its "
                                              "status and results to the database, and cannot reach a file here")
        repo = JobRepo(self._db)
        repo.set_progress(job_id, {"stage": "uploading"})
        missing = self._volumes.missing(keys)
        self._volumes.upload(missing, self._store)
        notes = [f"uploaded {len(missing)} new photos to Modal ({len(keys) - len(missing)} already there)"]
        if self._cancelled.is_set():
            return RunOutcome(ok=False, error=CANCELLED, notes=tuple(notes))
        # "waiting for gpu" until the worker's first write: a queued run never reads "Running" (R9)
        repo.set_progress(job_id, {"stage": "waiting for gpu"}, heartbeat=False)
        self._call_id = self._remote.start(job_id, keys, self._store.read_run_spec(job_id),
                                           self._store.manifest(job_id), self._pipeline)
        repo.set_progress(job_id, None, heartbeat=False, call_id=self._call_id)
        notes.append(f"started on Modal, call {self._call_id}")
        if self._cancelled.is_set():               # cancel arrived while the call was being spawned
            self._remote.cancel(self._call_id)
            return RunOutcome(ok=False, error=CANCELLED, notes=tuple(notes))
        return self._follow(repo, job_id, notes)

    def _follow(self, repo, job_id: str, notes: List[str]) -> RunOutcome:
        from sim_bench.control.jobs import SUCCEEDED, TERMINAL
        from sim_bench.serving.detached_modal import call_ended

        while True:
            job = repo.get(job_id)
            if job.status in TERMINAL:
                logger.info("job %s: %s ended %s; %s", job_id, self.name, job.status, "; ".join(notes))
                return RunOutcome(ok=job.status == SUCCEEDED, error=job.error, failed_step=job.failed_step,
                                  notes=tuple(notes))
            if self._cancelled.is_set():
                self._remote.cancel(self._call_id)
                return RunOutcome(ok=False, error=CANCELLED, notes=tuple(notes))
            status = self._remote.status(self._call_id)
            if call_ended(status):
                job = repo.get(job_id)              # the worker may have finished just now
                if job.status in TERMINAL:
                    continue
                return RunOutcome(ok=False, error=f"the L4 worker ended without saving its results ({status})",
                                  notes=tuple(notes))
            self._cancelled.wait(self._poll_s)

    def cancel(self, job_id: str) -> None:
        self._cancelled.set()


def make_runner(name: str, store: JobStore, *, pipeline: str, device: str = "cpu", db=None) -> Runner:
    if name == "local":
        return LocalRunner(store, pipeline=pipeline, device=device, db=db)
    if name in ("modal_gpu", "modal_cpu"):
        return ModalRunner(store, gpu=name == "modal_gpu", pipeline=pipeline, db=db)
    raise ValueError(f"unknown runner {name!r}")


def _child_main() -> int:
    """Entry for LocalRunner's child process: curate one job, record it, exit."""
    import argparse

    from sim_bench.serving.curate_job import CurateJobInputs, JobCurator

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--pipeline", default="default")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from sim_bench.storage.database import load_env
    load_env()        # spec-117: the worker reads the same DATABASE_URL / R2_* as the API (env already set wins)
    curator, store, _ = worker_curator(a.root, runner="local")
    res = curator.calc(CurateJobInputs(job_id=a.job_id, pipeline=a.pipeline, device=a.device, runner="local",
                                       run_spec=store.read_run_spec(a.job_id)))
    return 0 if res.ok else 1


def require_shared_db(runner: str) -> None:
    """A Modal worker must write to the shared database (Neon). Without ``DATABASE_URL`` it would quietly write to a
    SQLite file inside the container and its results would vanish with it, after the GPU time is paid
    (spec-117 review M1). Stop before any work instead; the watchdog then shows the reason on the job."""
    if not runner.startswith("modal"):
        return
    from sim_bench.storage.database import ENV_VAR, DatabaseConfig

    url = os.environ.get(ENV_VAR, "")
    if not url or DatabaseConfig(url).is_sqlite:
        raise RuntimeError(f"{runner} worker has no shared {ENV_VAR} (add it to the Modal secret); stopped before "
                           "any work so results are not written to a file inside the container")


def worker_curator(root, *, inputs_root=None, runner: str = "local", control_db_url: Optional[str] = None):
    """The curator a real worker runs (spec-117): progress and results go to the job database, files to the object
    store, the album run's rows to the API tables. Same on this PC and on the L4."""
    require_shared_db(runner)
    from sim_bench.api.database.session import get_session_direct
    from sim_bench.control.db import ControlDB
    from sim_bench.serving.curate_job import JobCurator
    from sim_bench.serving.worker_persister import WorkerPersister
    from sim_bench.storage.object_store import make_object_store

    store = JobStore(root, inputs_root=inputs_root, objects=make_object_store(root))
    db = ControlDB(control_db_url or os.environ.get(CONTROL_DB_ENV) or None)
    persister = WorkerPersister(db, store, api_sessions=get_session_direct)
    return JobCurator(store, db=db, persister=persister), store, db


if __name__ == "__main__":
    sys.exit(_child_main())
