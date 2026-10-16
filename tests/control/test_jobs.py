"""spec-112 TA2 - control DB + job status set (AC4: a crash ends `failed`, never stuck)."""
from __future__ import annotations

import pytest

from sim_bench.control.db import ControlDB, _redact, default_url
from sim_bench.control.jobs import (
    FAILED, LOADING, QUEUED, RUNNING, STAGING, SUCCEEDED, InvalidTransition, JobRepo,
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


class ut_ControlDB:
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
