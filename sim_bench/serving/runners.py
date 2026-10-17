"""Where a job runs (spec-112 TA4.3). Every runner has the same contract:

    run(job_id) -> RunOutcome,  with the job's files (album.json, results/, display/, faces/,
                                run.json) in the LOCAL job store afterwards

so the control plane loads results the same way whichever runner ran. Runners never touch the DB.

    LocalRunner   JobCurator in a child process: torch / ONNX memory is only returned to the OS
                  when a process exits (spec-104, SIGHTING-117), so a long-lived API must not run
                  the models in-process.
    ModalRunner   upload the staged inputs to the Modal jobs Volume, call curate_job, copy the job
                  directory back. Phase A shape; Phase B swaps the Volume for R2 (spec-112 B1).
"""
from __future__ import annotations

import json
import os
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from sim_bench.serving.job_store import JobStore

logger = logging.getLogger(__name__)

__all__ = ["RunOutcome", "Runner", "LocalRunner", "ModalRunner", "ModalVolumeRemote",
           "make_runner"]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    ok: bool
    error: Optional[str] = None
    failed_step: Optional[str] = None
    wall_s: Optional[float] = None


class Runner(Protocol):
    name: str

    def run(self, job_id: str) -> RunOutcome: ...


def _outcome_from_record(store: JobStore, job_id: str, fallback_error: str) -> RunOutcome:
    path = store.paths(job_id).run_path
    if not path.is_file():
        return RunOutcome(ok=False, error=fallback_error)
    rec = json.loads(path.read_text(encoding="utf-8"))
    total = (rec.get("timing") or {}).get("total_ms")
    return RunOutcome(ok=bool(rec.get("ok")), error=rec.get("error"),
                      failed_step=rec.get("failed_step"),
                      wall_s=round(total / 1000, 1) if total else None)


class LocalRunner:
    name = "local"

    def __init__(self, store: JobStore, *, pipeline: str = "default", device: str = "cpu",
                 timeout_s: int = 6 * 3600):
        self._store, self._pipeline, self._device, self._timeout = store, pipeline, device, timeout_s

    def command(self, job_id: str) -> list:
        return [sys.executable, "-m", "sim_bench.serving.runners", "--root", str(self._store.root),
                "--job-id", job_id, "--pipeline", self._pipeline, "--device", self._device]

    def run(self, job_id: str) -> RunOutcome:
        run_path = self._store.paths(job_id).run_path
        if run_path.exists():
            run_path.unlink()   # a stale record from an earlier run must not read as this one's
        logger.info("job %s: local child process (%s, %s)", job_id, self._pipeline, self._device)
        try:
            # UTF-8 both ways: with the Windows locale code page (e.g. cp1255) one undecodable
            # byte killed the reader thread and lost the child's output, error tail included.
            proc = subprocess.run(self.command(job_id), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self._timeout,
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return RunOutcome(ok=False, error=f"local run exceeded {self._timeout}s")
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        return _outcome_from_record(
            self._store, job_id,
            f"worker process exited {proc.returncode} without a run record: {' | '.join(tail)}")


class ModalVolumeRemote:
    """The real Modal calls. Imported lazily: the API must start without Modal installed/authed."""

    def __init__(self, gpu: bool = True):
        self.gpu = gpu

    def stage(self, job_id: str, images) -> None:
        import modal
        from sim_bench.serving.modal_app import cpu_app, stage_job
        with modal.enable_output(), cpu_app.run():
            stage_job.remote(job_id, images)

    def curate(self, job_id: str, pipeline: str) -> dict:
        import modal
        from sim_bench.serving.modal_app import app, cpu_app, curate_job, curate_job_cpu
        fn, run_app = (curate_job, app) if self.gpu else (curate_job_cpu, cpu_app)
        with modal.enable_output(), run_app.run():
            return fn.remote(job_id, pipeline)

    def download(self, job_id: str, store: JobStore) -> int:
        """Copy every file of the job except input/ from the Volume into ``store``."""
        import modal
        from modal.volume import FileEntryType
        vol = modal.Volume.from_name("albumify-jobs")
        root = store.paths(job_id).root
        n = 0
        for entry in vol.listdir(f"/{job_id}", recursive=True):
            rel = Path(entry.path).relative_to(job_id) if not entry.path.startswith("/") \
                else Path(entry.path.lstrip("/")).relative_to(job_id)
            if entry.type != FileEntryType.FILE or rel.parts[0] == "input":
                continue
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in vol.read_file(entry.path):
                    fh.write(chunk)
            n += 1
        return n


class ModalRunner:
    def __init__(self, store: JobStore, *, gpu: bool = True, pipeline: str = "default",
                 remote=None):
        self.name = "modal_gpu" if gpu else "modal_cpu"
        self._store, self._pipeline = store, pipeline
        self._remote = remote or ModalVolumeRemote(gpu=gpu)

    def run(self, job_id: str) -> RunOutcome:
        paths = self._store.paths(job_id)
        images = [(p.name, p.read_bytes()) for p in self._store.input_images(job_id)]
        if not images:
            return RunOutcome(ok=False, error=f"no staged inputs under {paths.input_dir}")
        for stale in (paths.run_path, paths.album_path):
            if stale.exists():
                stale.unlink()
        self._remote.stage(job_id, images)
        res = self._remote.curate(job_id, self._pipeline)
        n = self._remote.download(job_id, self._store)
        logger.info("job %s: %s ok=%s, %d files copied back", job_id, self.name, res.get("ok"), n)
        return _outcome_from_record(self._store, job_id,
                                    res.get("error") or "Modal run returned no run record")


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
