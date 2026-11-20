"""spec-117 R8 / R9: detached L4 runs survive the PC; a vanished worker ends failed with the reason within the
grace period; cancel works by call id; a queued run is not "running"."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import FAILED, RUNNING, STAGING, SUCCEEDED, JobRepo
from sim_bench.serving.run_watchdog import RunWatchdog

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


class FakeCalls:
    def __init__(self, status="running"):
        self.state, self.cancelled = status, []

    def status(self, call_id):
        if isinstance(self.state, Exception):
            raise self.state
        return self.state(call_id) if callable(self.state) else self.state

    def cancel(self, call_id):
        self.cancelled.append(call_id)


@pytest.fixture
def db(control_db_url):
    return ControlDB(control_db_url)


def _detached(db, job_id="j1", beat_age_s=None, call_id="call-1"):
    """A live L4 job started by a process that is long gone (another host), with a call id."""
    repo = JobRepo(db)
    repo.create(job_id, runner="modal_gpu")
    repo.transition(job_id, STAGING, worker="pc-that-restarted:1:1")
    repo.transition(job_id, RUNNING)
    repo.set_progress(job_id, {"stage": "waiting for gpu"}, heartbeat=False, call_id=call_id)
    if beat_age_s is not None:
        repo.set_progress(job_id, {"stage": "running"}, heartbeat=False,
                          heartbeat_at=NOW - timedelta(seconds=beat_age_s))
    return repo


def _dog(db, calls):
    return RunWatchdog(db, calls, grace_s=90, clock=lambda: NOW)


class ut_RunWatchdog:
    def test_a_stopped_heartbeat_fails_the_job_and_cancels_the_call(self, db):
        repo = _detached(db, beat_age_s=100)
        calls = FakeCalls("running")
        assert _dog(db, calls).reconcile() == 1
        job = repo.get("j1")
        assert job.status == FAILED and "no heartbeat for 100 s" in job.error
        assert calls.cancelled == ["call-1"]

    def test_a_fresh_heartbeat_is_left_alone(self, db):
        repo = _detached(db, beat_age_s=10)
        assert _dog(db, FakeCalls("running")).reconcile() == 0
        assert repo.get("j1").status == RUNNING

    def test_a_queued_call_without_heartbeat_is_left_alone(self, db):
        repo = _detached(db)
        assert _dog(db, FakeCalls("running")).reconcile() == 0
        assert repo.get("j1").progress["stage"] == "waiting for gpu"

    def test_a_call_that_died_before_starting_fails(self, db):
        repo = _detached(db)
        assert _dog(db, FakeCalls("failed: FunctionTimeoutError: boot")).reconcile() == 1
        assert "ended before it started (failed: FunctionTimeoutError: boot)" in repo.get("j1").error

    def test_a_call_that_ended_without_results_fails(self, db):
        repo = _detached(db, beat_age_s=5)
        assert _dog(db, FakeCalls("done")).reconcile() == 1
        assert "ended without saving its results" in repo.get("j1").error

    def test_a_worker_that_just_finished_is_never_overruled(self, db):
        repo = _detached(db, beat_age_s=5)

        def finishes(call_id):              # the worker ends the job while the watchdog asks Modal
            repo.transition("j1", "loading")
            repo.transition("j1", SUCCEEDED)
            return "done"

        assert _dog(db, FakeCalls(finishes)).reconcile() == 0
        assert repo.get("j1").status == SUCCEEDED

    def test_modal_unreachable_judges_by_heartbeat_only(self, db):
        repo = _detached(db, beat_age_s=5)
        assert _dog(db, FakeCalls(ConnectionError("offline"))).reconcile() == 0
        assert repo.get("j1").status == RUNNING

    def test_jobs_without_a_call_id_are_not_its_business(self, db):
        repo = JobRepo(db)
        repo.create("local", runner="local")
        repo.transition("local", STAGING, worker="x:1:1")
        assert _dog(db, FakeCalls("done")).reconcile() == 0
        assert repo.get("local").status == STAGING


class ut_DetachedSurvival:
    def test_a_restarted_api_does_not_reap_a_detached_run(self, db):
        """R8: the process that started the run is gone (API restart); the run is judged by its heartbeat."""
        repo = _detached(db, beat_age_s=5)
        assert repo.reap_orphans(alive=lambda w: False) == []
        assert repo.get("j1").status == RUNNING

    def test_cancel_by_call_id_after_a_restart(self, db, tmp_path):
        """R9: no runner in this process knows the run; cancel still reaches Modal by the stored call id."""
        from sim_bench.serving.job_service import JobService
        from sim_bench.serving.runners import CANCELLED
        from sim_bench.serving.serving_config import ServingConfig

        _detached(db, beat_age_s=5)
        svc = JobService(db, ServingConfig(jobs_root=tmp_path / "jobs"))
        calls = FakeCalls("running")
        svc._detached = calls
        svc.cancel("j1")
        job = svc.repo.get("j1")
        assert calls.cancelled == ["call-1"] and job.status == FAILED and job.error == CANCELLED
