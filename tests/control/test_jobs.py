"""spec-112 TA2 - control DB + job status set (AC4: a crash ends `failed`, never stuck)."""
from __future__ import annotations

import pytest

from sim_bench.control.db import ControlDB, _redact, default_url
from sim_bench.control.jobs import (
    FAILED, LOADING, QUEUED, RUNNING, STAGING, SUCCEEDED, InvalidTransition, JobRepo, this_worker,
    worker_alive,
)


@pytest.fixture
def repo(tmp_path):
    return JobRepo(ControlDB(f"sqlite:///{(tmp_path / 'c.db').as_posix()}"))


class ut_JobRepo:
    def test_create_queues_and_logs(self, repo):
        job = repo.create("j1", runner="local", source="D:/trip")
        assert job.status == QUEUED and job.runner == "local"
        assert [e.message for e in repo.events("j1")] == [QUEUED]

    def test_happy_path_records_every_status(self, repo):
        repo.create("j1", runner="local")
        for to in (STAGING, RUNNING, LOADING, SUCCEEDED):
            repo.transition("j1", to)
        job = repo.get("j1")
        assert job.status == SUCCEEDED and job.started_at and job.finished_at
        assert [e.message for e in repo.events("j1")] == [QUEUED, STAGING, RUNNING, LOADING,
                                                           SUCCEEDED]

    @pytest.mark.parametrize("status", [QUEUED, STAGING, RUNNING, LOADING])
    def test_any_live_status_can_fail_with_its_error(self, repo, status):
        repo.create("j1", runner="local")
        path = [STAGING, RUNNING, LOADING]
        for to in path[:path.index(status) + 1] if status != QUEUED else []:
            repo.transition("j1", to)
        repo.fail("j1", "boom", failed_step="score_ava")
        job = repo.get("j1")
        assert job.status == FAILED and job.error == "boom" and job.failed_step == "score_ava"
        last = repo.events("j1")[-1]
        assert last.kind == "error" and "boom" in last.message

    def test_cannot_succeed_without_loading(self, repo):
        repo.create("j1", runner="local")
        repo.transition("j1", RUNNING)
        with pytest.raises(InvalidTransition):
            repo.transition("j1", SUCCEEDED)

    def test_rerun_requeues_and_clears_the_error(self, repo):
        repo.create("j1", runner="local")
        repo.fail("j1", "boom")
        job = repo.create("j1", runner="modal_gpu")
        assert job.status == QUEUED and job.error is None and job.runner == "modal_gpu"

    def test_cannot_requeue_a_running_job(self, repo):
        repo.create("j1", runner="local")
        repo.transition("j1", RUNNING)
        with pytest.raises(InvalidTransition):
            repo.create("j1", runner="local")

    def test_unknown_job_raises(self, repo):
        with pytest.raises(KeyError):
            repo.transition("nope", RUNNING)


class ut_OrphanJobs:
    """spec-114 AC6: a job whose process was killed must not stay live forever."""

    def test_job_of_a_dead_worker_is_failed_as_interrupted(self, repo):
        repo.create("dead", runner="local")
        repo.transition("dead", STAGING, worker="myhost:4242:1")
        repo.transition("dead", RUNNING)
        assert repo.reap_orphans(alive=lambda w: False) == ["dead"]
        job = repo.get("dead")
        assert job.status == FAILED and "interrupted" in job.error

    def test_live_worker_queued_and_finished_jobs_are_left_alone(self, repo):
        for jid in ("live", "queued", "done"):
            repo.create(jid, runner="local")
        repo.transition("live", STAGING, worker=this_worker())
        repo.transition("done", RUNNING, worker="gone:1:1")
        repo.transition("done", LOADING)
        repo.transition("done", SUCCEEDED)
        assert repo.reap_orphans() == []
        assert [repo.get(j).status for j in ("live", "queued", "done")] == [STAGING, QUEUED, SUCCEEDED]

    def test_worker_alive_rules(self):
        import socket
        assert worker_alive(this_worker())
        assert not worker_alive(None)                              # recorded before workers existed
        assert not worker_alive(f"{socket.gethostname()}:{2**22 + 7}:1")   # no such process here
        pid = this_worker().split(":")[1]
        assert not worker_alive(f"{socket.gethostname()}:{pid}:1")  # same pid, other start time: reused
        assert worker_alive("another-host:1:1")                   # cannot check: assume alive

    def test_tag_is_kept_on_rerun_unless_replaced(self, repo):
        repo.create("j", runner="local", tag="regression:quick")
        repo.transition("j", RUNNING)
        repo.fail("j", "x")
        assert repo.create("j", runner="local").tag == "regression:quick"


class ut_ControlDB:
    def test_older_db_gets_the_new_nullable_columns(self, tmp_path):
        import sqlite3
        path = tmp_path / "old.db"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE jobs (id VARCHAR(64) PRIMARY KEY, status VARCHAR(16) NOT NULL, "
                    "runner VARCHAR(16) NOT NULL, pipeline VARCHAR(32) NOT NULL, source TEXT, "
                    "n_inputs INTEGER NOT NULL, error TEXT, failed_step VARCHAR(64), theme JSON, "
                    "album_created_at VARCHAR(40), created_at DATETIME NOT NULL, started_at DATETIME, "
                    "finished_at DATETIME)")
        con.execute("INSERT INTO jobs VALUES ('old', 'succeeded', 'local', 'default', NULL, 3, NULL, "
                    "NULL, NULL, NULL, '2026-10-01 10:00:00', NULL, NULL)")
        con.commit()
        con.close()
        repo = JobRepo(ControlDB(f"sqlite:///{path.as_posix()}"))
        job = repo.get("old")
        assert job.status == SUCCEEDED and job.tag is None and job.worker is None


    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///x.db")
        assert default_url() == "sqlite:///x.db"

    def test_default_is_local_sqlite(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        assert default_url().startswith("sqlite:///") and default_url().endswith("control.db")

    def test_password_is_never_logged(self):
        assert _redact("postgresql://u:secret@ep-x.neon.tech/db") == \
            "postgresql://***@ep-x.neon.tech/db"
