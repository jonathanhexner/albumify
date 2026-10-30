"""Where a job runs (spec-112 TA4.3, spec-115 W3/W4). Every runner has the same contract:

    run(job_id) -> RunOutcome,  with the job's results (album.json, run.json, results/) in the LOCAL job
                                store afterwards; the server renders display/ and faces/ itself
    cancel(job_id)              stop it; run() then returns ok=False, error "cancelled by owner"

so the control plane loads results the same way whichever runner ran. Runners never touch the DB.

    LocalRunner   JobCurator in a child process: torch / ONNX memory is only returned to the OS
                  when a process exits (spec-104, SIGHTING-117), so a long-lived API must not run
                  the models in-process.
    ModalRunner   upload the photos the Modal input store lacks (by content key), run curate_job as a
                  cancellable call, copy back the results only (spec-115 R8/R9).
"""
from __future__ import annotations

import json
import os
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

from sim_bench.serving.job_store import JobStore
from sim_bench.serving.run_settings import COMMIT_ENV, current_commit

logger = logging.getLogger(__name__)

__all__ = ["RunOutcome", "Runner", "LocalRunner", "ModalRunner", "ModalVolumeRemote",
           "make_runner", "CANCELLED"]

CANCELLED = "cancelled by owner"


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
                 timeout_s: int = 6 * 3600):
        self._store, self._pipeline, self._device, self._timeout = store, pipeline, device, timeout_s
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
                                env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                     COMMIT_ENV: current_commit() or ""})
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

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.kill()


INPUTS_VOLUME, JOBS_VOLUME = "albumify-inputs", "albumify-jobs"


class ModalVolumeRemote:
    """The real Modal calls. Imported lazily: the API must start without Modal installed/authed."""

    def __init__(self, gpu: bool = True):
        self.gpu = gpu
        self._call = None
        self._cancelled = False

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

    def run_job(self, job_id: str, keys: Sequence[str], pipeline: str) -> dict:
        import modal
        from sim_bench.serving.modal_app import app, cpu_app, curate_job, curate_job_cpu
        fn, run_app = (curate_job, app) if self.gpu else (curate_job_cpu, cpu_app)
        if self._cancelled:                    # cancelled while photos were uploading: no GPU at all
            return {"ok": False, "error": CANCELLED}
        with modal.enable_output(), run_app.run():
            self._call = fn.spawn(job_id, pipeline, list(keys), current_commit())
            if self._cancelled:                # cancel arrived while the app was starting (review M3)
                self._call.cancel()
            try:
                return self._call.get()
            except Exception as e:
                if self._cancelled:
                    return {"ok": False, "error": CANCELLED}
                return {"ok": False, "error": f"Modal call failed: {e}"}
            finally:
                self._call = None

    def cancel(self) -> None:
        self._cancelled = True
        if self._call is not None:
            self._call.cancel()

    def download_results(self, job_id: str, store: JobStore) -> int:
        """Copy back the job's results only (spec-115 R8): album.json, run.json, results/*."""
        import modal
        from modal.volume import FileEntryType
        vol = modal.Volume.from_name(JOBS_VOLUME)
        remote = [f"/{job_id}/album.json", f"/{job_id}/run.json"]
        try:
            remote += [e.path if e.path.startswith("/") else "/" + e.path
                       for e in vol.listdir(f"/{job_id}/results", recursive=True) if e.type == FileEntryType.FILE]
        except Exception as e:                   # the worker failed before writing results (review M5)
            logger.info("job %s: no results folder on the Volume (%s)", job_id, type(e).__name__)
        root, n = store.paths(job_id).root, 0
        for path in remote:
            dest = root / Path(path.lstrip("/")).relative_to(job_id)
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")    # readers never see half a file
            try:
                with open(part, "wb") as fh:
                    for chunk in vol.read_file(path):
                        fh.write(chunk)
                os.replace(part, dest)
                n += 1
            except FileNotFoundError:
                part.unlink(missing_ok=True)      # album.json is absent when the run failed
        return n

    def delete_inputs(self, job_id: str, keys: Sequence[str]) -> dict:
        import modal
        from sim_bench.serving.modal_app import cpu_app, delete_inputs
        with modal.enable_output(), cpu_app.run():
            return delete_inputs.remote(job_id, list(keys))


class ModalRunner:
    def __init__(self, store: JobStore, *, gpu: bool = True, pipeline: str = "default",
                 remote=None):
        self.name = "modal_gpu" if gpu else "modal_cpu"
        self._store, self._pipeline = store, pipeline
        self._remote = remote or ModalVolumeRemote(gpu=gpu)
        self._cancelled = False

    def run(self, job_id: str) -> RunOutcome:
        paths = self._store.paths(job_id)
        keys = self._store.input_keys(job_id)
        if not keys:
            return RunOutcome(ok=False, error=f"job {job_id!r} has no input manifest under {paths.input_dir}; "
                                              "re-run it from its source folder")
        for stale in (paths.run_path, paths.album_path):
            if stale.exists():
                stale.unlink()
        missing = self._remote.missing(keys)
        self._remote.upload(missing, self._store)
        notes = [f"uploaded {len(missing)} new photos to Modal ({len(keys) - len(missing)} already there)"]
        if self._cancelled:
            return RunOutcome(ok=False, error=CANCELLED, notes=tuple(notes))
        res = self._remote.run_job(job_id, keys, self._pipeline)
        if res.get("error") == CANCELLED:
            return RunOutcome(ok=False, error=CANCELLED, notes=tuple(notes))
        n = self._remote.download_results(job_id, self._store)
        notes.append(f"copied back {n} result files")
        logger.info("job %s: %s ok=%s; %s", job_id, self.name, res.get("ok"), "; ".join(notes))
        return _outcome_from_record(self._store, job_id,
                                    res.get("error") or "Modal run returned no run record", notes)

    def cancel(self, job_id: str) -> None:
        self._cancelled = True
        self._remote.cancel()


def make_runner(name: str, store: JobStore, *, pipeline: str, device: str = "cpu") -> Runner:
    if name == "local":
        return LocalRunner(store, pipeline=pipeline, device=device)
    if name in ("modal_gpu", "modal_cpu"):
        return ModalRunner(store, gpu=name == "modal_gpu", pipeline=pipeline)
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
    res = JobCurator(JobStore(a.root)).calc(
        CurateJobInputs(job_id=a.job_id, pipeline=a.pipeline, device=a.device))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(_child_main())
