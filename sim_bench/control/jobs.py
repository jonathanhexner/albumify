"""Job rows + the status set they move through (spec-112 TA2.2).

    queued -> staging -> running -> loading -> succeeded
       \\________\\__________\\__________\\----> failed
    succeeded | failed -> queued        (re-run the same job id)

Every transition appends a ``status`` event, so ``job_events`` is the job's full history. An
invalid transition raises: a job must never claim ``succeeded`` without having loaded its results,
and a crash must end in ``failed`` with the error recorded, never stuck in ``running`` (AC4).

A process that is killed (out of memory, closed terminal) cannot record its own failure. The job
keeps the ``worker`` that runs it (``host:pid:start-time``); ``reap_orphans`` fails a live job whose
worker is gone (spec-114 AC6).
"""
from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from typing import Callable, Dict, FrozenSet, List, Optional

import psutil

from sqlalchemy import select

from sim_bench.control.db import ControlDB
from sim_bench.control.models import Job, JobEvent

__all__ = ["JobRepo", "InvalidTransition", "TRANSITIONS", "TERMINAL", "LIVE", "this_worker", "worker_alive",
           "QUEUED", "STAGING", "RUNNING", "LOADING", "SUCCEEDED", "FAILED"]

QUEUED, STAGING, RUNNING, LOADING, SUCCEEDED, FAILED = (
    "queued", "staging", "running", "loading", "succeeded", "failed")

TRANSITIONS: Dict[str, FrozenSet[str]] = {
    QUEUED: frozenset({STAGING, RUNNING, FAILED}),
    STAGING: frozenset({RUNNING, FAILED}),
    RUNNING: frozenset({LOADING, FAILED}),
    LOADING: frozenset({SUCCEEDED, FAILED}),
    SUCCEEDED: frozenset({QUEUED}),
    FAILED: frozenset({QUEUED}),
}
TERMINAL = frozenset({SUCCEEDED, FAILED})
LIVE = frozenset({STAGING, RUNNING, LOADING})   # a process is working on it (queued: none yet)


def this_worker() -> str:
    """``host:pid:start`` of this process; the start time tells a reused pid apart."""
    pid = os.getpid()
    return f"{socket.gethostname()}:{pid}:{int(psutil.Process(pid).create_time())}"


def worker_alive(worker: Optional[str]) -> bool:
    """False when the worker process is gone. A worker on another host is assumed alive."""
    if not worker:
        return False                      # written before workers were recorded: nothing runs it
    host, pid, start = worker.rsplit(":", 2)
    if host != socket.gethostname():
        return True
    try:
        return int(psutil.Process(int(pid)).create_time()) == int(start)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


class InvalidTransition(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobRepo:
    """All writes to ``jobs`` / ``job_events`` go through here."""

    def __init__(self, db: ControlDB):
        self._db = db

    def create(self, job_id: str, *, runner: str, pipeline: str = "default",
               source: Optional[str] = None, tag: Optional[str] = None) -> Job:
        """New job in ``queued``; an existing finished job is re-queued (same id = re-run)."""
        with self._db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                job = Job(id=job_id, status=QUEUED, runner=runner, pipeline=pipeline,
                          source=source, tag=tag, created_at=_now())
                s.add(job)
                s.flush()
                _event(s, job_id, "status", QUEUED)
                return job
            _check(job.status, QUEUED)
            job.status, job.runner, job.pipeline = QUEUED, runner, pipeline
            job.source = source or job.source
            job.tag = tag or job.tag
            job.error = job.failed_step = None
            job.started_at = job.finished_at = None
            _event(s, job_id, "status", QUEUED)
            return job

    def transition(self, job_id: str, to: str, message: str = "", **fields) -> Job:
        with self._db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            _check(job.status, to)
            job.status = to
            if to == RUNNING:
                job.started_at = _now()
            if to in TERMINAL:
                job.finished_at = _now()
            for k, v in fields.items():
                setattr(job, k, v)
            _event(s, job_id, "error" if to == FAILED else "status",
                   f"{to}: {message}" if message else to)
            return job

    def fail(self, job_id: str, error: str, failed_step: Optional[str] = None) -> Job:
        return self.transition(job_id, FAILED, error, error=error, failed_step=failed_step)

    def log(self, job_id: str, message: str, kind: str = "info") -> None:
        with self._db.session() as s:
            _event(s, job_id, kind, message)

    def get(self, job_id: str) -> Optional[Job]:
        with self._db.session() as s:
            return s.get(Job, job_id)

    def set_tag(self, job_id: str, tag: Optional[str]) -> None:
        with self._db.session() as s:
            s.get(Job, job_id).tag = tag

    def reap_orphans(self, alive: Callable[[Optional[str]], bool] = worker_alive) -> List[str]:
        """Fail every live job whose worker process is gone; return their ids."""
        with self._db.session() as s:
            ids = [j.id for j in s.scalars(select(Job).where(Job.status.in_(LIVE)))
                   if not alive(j.worker)]
        for job_id in ids:
            self.fail(job_id, "interrupted: the process running this job ended before it finished "
                              "(killed, out of memory, or closed)")
        return ids

    def list(self, limit: int = 100) -> List[Job]:
        with self._db.session() as s:
            return list(s.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)))

    def events(self, job_id: str) -> List[JobEvent]:
        with self._db.session() as s:
            return list(s.scalars(select(JobEvent).where(JobEvent.job_id == job_id)
                                  .order_by(JobEvent.id)))


def _check(current: str, to: str) -> None:
    if to not in TRANSITIONS.get(current, frozenset()):
        raise InvalidTransition(f"job cannot go {current} -> {to}")


def _event(s, job_id: str, kind: str, message: str) -> None:
    s.add(JobEvent(job_id=job_id, ts=_now(), kind=kind, message=message))
