"""End detached runs whose worker is gone (spec-117 R9).

A detached L4 run has no process on this PC to fail it, so its job row is judged by what the worker writes:

    no heartbeat yet, call still running    queued on Modal: left alone (the app shows "Waiting for a GPU")
    no heartbeat yet, call ended            failed: "the L4 worker ended before it started (...)"
    heartbeat older than grace_s            failed: "the L4 worker stopped (no heartbeat for N s)"; the call is
                                            cancelled so a hung container stops billing
    heartbeat fresh, call ended, job live   failed: "the L4 worker ended without saving its results (...)"

The job row is read again right before it is failed, so a worker that finished in between is never overruled.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

from sqlalchemy import select

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import LIVE, JobRepo
from sim_bench.control.models import Job
from sim_bench.serving.detached_modal import call_ended

logger = logging.getLogger(__name__)

__all__ = ["RunWatchdog"]


def _utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)   # SQLite drops the zone


class RunWatchdog:
    def __init__(self, db: ControlDB, remote, *, grace_s: float = 90.0,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.db, self.remote, self.grace_s, self._clock = db, remote, grace_s, clock
        self._repo = JobRepo(db)

    def reconcile(self, job_ids: Optional[List[str]] = None) -> int:
        """Fail every live detached job (one with a call id) whose worker is gone; returns how many."""
        with self.db.session() as s:
            q = select(Job).where(Job.status.in_(LIVE), Job.call_id.is_not(None))
            if job_ids is not None:
                q = q.where(Job.id.in_(job_ids))
            jobs = [(j.id, j.call_id, j.heartbeat_at) for j in s.scalars(q)]
        failed = 0
        for job_id, call_id, beat in jobs:
            reason = self._reason(call_id, beat)
            if reason is None:
                continue
            job = self._repo.get(job_id)
            if job is None or job.status not in LIVE:
                continue                       # the worker finished meanwhile
            self._repo.fail(job_id, reason)
            failed += 1
            logger.warning("job %s: %s", job_id, reason)
        return failed

    def _reason(self, call_id: str, beat: Optional[datetime]) -> Optional[str]:
        if beat is not None:
            age = (self._clock() - _utc(beat)).total_seconds()
            if age > self.grace_s:
                try:
                    self.remote.cancel(call_id)            # a hung container must not keep billing
                except Exception as e:
                    logger.info("call %s: cancel after lost heartbeat failed: %s", call_id, type(e).__name__)
                return f"the L4 worker stopped (no heartbeat for {round(age)} s)"
        status = self._status(call_id)
        if status is None or not call_ended(status):
            return None
        if beat is None:
            return f"the L4 worker ended before it started ({status})"
        return f"the L4 worker ended without saving its results ({status})"

    def _status(self, call_id: str) -> Optional[str]:
        try:
            return self.remote.status(call_id)
        except Exception as e:      # Modal unreachable from here: judge by the heartbeat only
            logger.info("call %s: status unknown: %s", call_id, type(e).__name__)
            return None
