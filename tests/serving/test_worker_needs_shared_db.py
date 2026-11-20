"""spec-117 review M1: a Modal worker without a shared DATABASE_URL stops at once instead of writing its results
to a SQLite file inside the container (lost when the container ends, after the GPU time is paid)."""
from __future__ import annotations

import pytest

from sim_bench.serving.runners import require_shared_db, worker_curator


class ut_RequireSharedDb:
    @pytest.mark.parametrize("runner", ["modal_gpu", "modal_cpu"])
    def test_a_modal_worker_without_database_url_stops(self, monkeypatch, runner):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError, match="no shared DATABASE_URL"):
            require_shared_db(runner)

    def test_a_modal_worker_on_a_sqlite_url_stops(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:////tmp/sim_bench.db")
        with pytest.raises(RuntimeError, match="no shared DATABASE_URL"):
            require_shared_db("modal_gpu")

    def test_a_modal_worker_on_postgres_goes_on(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@example.neon.tech/db")
        require_shared_db("modal_gpu")                    # nothing raised

    def test_the_local_worker_may_use_sqlite(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        require_shared_db("local")                        # nothing raised

    def test_worker_curator_checks_before_touching_any_store(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(RuntimeError):
            worker_curator(tmp_path / "jobs", runner="modal_gpu")
        assert not (tmp_path / "jobs").exists()
