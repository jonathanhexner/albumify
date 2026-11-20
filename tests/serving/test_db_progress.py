"""spec-117 R5: status and progress come from the job row only, written by the worker; progress.json is retired."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import JobRepo
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
from sim_bench.serving.db_progress import DbProgressWriter
from sim_bench.serving.job_store import JobStore

from tests.serving.test_run_output import RESULT, finished_context


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _step(name, ok=True):
    return SimpleNamespace(step_name=name, duration_ms=5, success=ok, error_message=None if ok else "boom")


@pytest.fixture
def db(control_db_url):
    d = ControlDB(control_db_url)
    JobRepo(d).create("j1", runner="modal_gpu")
    return d


class ut_DbProgressWriter:
    def test_steps_and_in_step_fraction_reach_the_job_row(self, db):
        clock = Clock()
        w = DbProgressWriter(db, "j1", ["discover_images", "triage_content", "select_best"], total=715,
                             clock=clock)
        w.on_step(_step("discover_images"))
        clock.t = 10
        w.on_fraction("triage_content", 0.43, "307/715 photos")
        job = JobRepo(db).get("j1")
        assert job.progress["current_step"] == "triage_content"
        assert (job.progress["step_fraction"], job.progress["photos_done"], job.progress["photos_total"]) == \
            (0.43, 307, 715)
        assert [d["step"] for d in job.progress["completed_steps"]] == ["discover_images"]
        assert job.heartbeat_at is not None

    def test_fractions_are_throttled_to_one_write_per_interval(self, db):
        clock = Clock()
        w = DbProgressWriter(db, "j1", ["triage_content"], total=100, clock=clock)
        clock.t = 10
        w.on_fraction("triage_content", 0.10)
        clock.t = 11                                     # 1 s later: not written
        w.on_fraction("triage_content", 0.20)
        assert JobRepo(db).get("j1").progress["step_fraction"] == 0.10
        clock.t = 12.5                                   # 2.5 s after the last write: written
        w.on_fraction("triage_content", 0.30)
        assert JobRepo(db).get("j1").progress["photos_done"] == 30     # n of N from the fraction

    def test_the_end_of_a_step_is_always_written(self, db):
        clock = Clock()
        w = DbProgressWriter(db, "j1", ["a", "b"], clock=clock)
        clock.t = 10
        w.on_fraction("a", 0.5)
        w.on_fraction("a", 1.0)                           # same instant, but a step end
        assert JobRepo(db).get("j1").progress["step_fraction"] == 1.0
        w.on_step(_step("a"))
        progress = JobRepo(db).get("j1").progress
        assert progress["current_step"] == "b" and "step_fraction" not in progress

    def test_heartbeat_thread_beats_at_least_every_interval(self, db):
        # Polls to a deadline instead of one fixed sleep: one heartbeat write to Neon takes ~0.65 s (SIGHTING-155),
        # so a 0.7 s window failed there although the thread beat fine.
        w = DbProgressWriter(db, "j1", ["a"], heartbeat_s=0.2)
        with w:
            deadline = time.monotonic() + 10.0
            first = JobRepo(db).get("j1").heartbeat_at
            while first is None and time.monotonic() < deadline:
                time.sleep(0.05)
                first = JobRepo(db).get("j1").heartbeat_at
            later = first
            while later == first and time.monotonic() < deadline:
                time.sleep(0.05)
                later = JobRepo(db).get("j1").heartbeat_at
        assert first is not None and later > first

    def test_a_failed_write_never_stops_the_run(self, db, monkeypatch):
        w = DbProgressWriter(db, "j1", ["a"])
        monkeypatch.setattr(JobRepo, "set_progress", lambda *a, **k: (_ for _ in ()).throw(OSError("Neon down")))
        w.on_step(_step("a"))
        w.on_fraction("a", 1.0)
        w.heartbeat()                                     # nothing raised


def test_curator_writes_progress_to_the_job_row_not_a_file(tmp_path, control_db_url):
    """The worker's curator, given the job database: the row follows the steps; no progress.json anywhere."""
    db = ControlDB(control_db_url)
    JobRepo(db).create("j1", runner="local")
    store = JobStore(tmp_path / "jobs")
    store.inputs.put_bytes("00000000000000aa/a.jpg", b"x")
    store.write_input_manifest("j1", ["00000000000000aa/a.jpg"])
    snapshots = []

    def runner(images, plan, on_step, on_progress, **_):
        snapshots.append(JobRepo(db).get("j1").progress)
        on_progress("discover_images", 1.0, "")
        on_step(_step("discover_images"))
        snapshots.append(JobRepo(db).get("j1").progress)
        return finished_context(), RESULT

    JobCurator(store, runner=runner, db=db).calc(CurateJobInputs(job_id="j1", run_spec={
        "steps": ["discover_images", "select_best"], "step_configs": {}}))
    assert snapshots[0] == {"stage": "running", "current_step": "discover_images", "completed_steps": [],
                            "photos_total": 1}
    assert snapshots[1]["current_step"] == "select_best"
    assert snapshots[1]["completed_steps"][0]["step"] == "discover_images"
    assert not list((tmp_path / "jobs").rglob("progress.json"))
