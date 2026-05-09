"""Tests that _list_available_runs reads from action_log and returns expected shape/sort."""
import pytest
from pathlib import Path
from unittest.mock import patch

from face_cluster.run_history_db import start_action, complete_action, init_table


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "hist.db"
    init_table(db_path=db_path)
    return db_path


def _seed_runs(db, n: int = 3):
    """Insert n completed pipeline_run rows."""
    for i in range(n):
        aid = start_action("pipeline_run", payload={
            "run_id":     f"run_{i:03d}",
            "output_dir": f"/results/run_{i:03d}",
            "album":      f"Album{i}",
        }, db_path=db)
        complete_action(aid, result_fields={
            "run_id":    f"run_{i:03d}",
            "output_dir": f"/results/run_{i:03d}",
            "n_faces":   10 + i,
            "n_clusters": 2 + i,
            "n_noise":   1,
        }, db_path=db)


class ut_HistoryTabData:

    def test_list_returns_all_runs(self, tmp_path, db, monkeypatch):
        import face_cluster.run_history_db as rh
        monkeypatch.setattr(rh, "get_db_path", lambda: db)

        _seed_runs(db, n=4)

        # Import after monkeypatching
        from app.face_clustering.run_panels import _list_available_runs

        runs = _list_available_runs(complete_only=False)
        assert len(runs) == 4

    def test_list_complete_only_filters(self, tmp_path, db, monkeypatch):
        import face_cluster.run_history_db as rh
        monkeypatch.setattr(rh, "get_db_path", lambda: db)

        _seed_runs(db, n=2)
        # Add one running row
        start_action("pipeline_run", payload={"run_id": "running_r"}, db_path=db)

        from app.face_clustering.run_panels import _list_available_runs
        complete = _list_available_runs(complete_only=True)
        all_runs = _list_available_runs(complete_only=False)

        assert len(complete) == 2
        assert len(all_runs) == 3

    def test_sorted_newest_first(self, tmp_path, db, monkeypatch):
        import face_cluster.run_history_db as rh
        monkeypatch.setattr(rh, "get_db_path", lambda: db)

        _seed_runs(db, n=3)

        from app.face_clustering.run_panels import _list_available_runs
        runs = _list_available_runs(complete_only=False)

        dates = [r["started"] for r in runs]
        assert dates == sorted(dates, reverse=True)

    def test_required_keys_present(self, tmp_path, db, monkeypatch):
        import face_cluster.run_history_db as rh
        monkeypatch.setattr(rh, "get_db_path", lambda: db)

        _seed_runs(db, n=1)

        from app.face_clustering.run_panels import _list_available_runs
        runs = _list_available_runs()
        assert runs, "Expected at least one run"
        row = runs[0]
        for key in ("run_id", "output_folder", "album", "started", "status",
                    "faces", "clusters", "noise", "_dir"):
            assert key in row, f"Missing key: {key}"
