"""Tests for face_cluster.run_history_db — action_log CRUD."""
import json
import pytest
from pathlib import Path

from face_cluster.run_history_db import (
    start_action, complete_action, fail_action,
    list_actions, get_action, init_table,
)


class ut_RunHistoryDB:

    def test_start_action_returns_id(self, tmp_path):
        db = tmp_path / "test.db"
        aid = start_action("pipeline_run", payload={"run_id": "r1"}, db_path=db)
        assert isinstance(aid, int)
        assert aid > 0

    def test_complete_roundtrip(self, tmp_path):
        db = tmp_path / "test.db"
        aid = start_action("recluster", payload={"run_id": "r2", "output_dir": "/out"}, db_path=db)
        complete_action(aid, result_fields={
            "n_faces": 50, "n_clusters": 5, "n_noise": 2,
            "run_id": "r2", "output_dir": "/out",
        }, db_path=db)

        row = get_action(aid, db_path=db)
        assert row["status"] == "complete"
        assert row["n_faces"] == 50
        assert row["n_clusters"] == 5
        assert row["duration_s"] is not None
        assert row["duration_s"] >= 0

    def test_fail_action(self, tmp_path):
        db = tmp_path / "test.db"
        aid = start_action("ml_train", payload={}, db_path=db)
        fail_action(aid, "Something went wrong", db_path=db)

        row = get_action(aid, db_path=db)
        assert row["status"] == "failed"
        assert "Something went wrong" in row["error"]
        assert row["duration_s"] is not None

    def test_list_actions_sorted_newest_first(self, tmp_path):
        db = tmp_path / "test.db"
        for i in range(3):
            aid = start_action("merge_apply", payload={"round": i}, db_path=db)
            complete_action(aid, db_path=db)

        rows = list_actions(types=["merge_apply"], db_path=db)
        assert len(rows) == 3
        # started_at must be non-decreasing when reversed
        dates = [r["started_at"] for r in rows]
        assert dates == sorted(dates, reverse=True)

    def test_list_actions_type_filter(self, tmp_path):
        db = tmp_path / "test.db"
        a1 = start_action("pipeline_run", db_path=db)
        a2 = start_action("profile_save", db_path=db)
        complete_action(a1, db_path=db)
        complete_action(a2, db_path=db)

        runs_only = list_actions(types=["pipeline_run"], db_path=db)
        assert all(r["action_type"] == "pipeline_run" for r in runs_only)
        assert len(runs_only) == 1

    def test_hot_fields_promoted(self, tmp_path):
        db = tmp_path / "test.db"
        aid = start_action("pipeline_run", payload={
            "run_id": "abc", "output_dir": "/out/abc",
            "album": "MyAlbum",
        }, db_path=db)

        row = get_action(aid, db_path=db)
        assert row["run_id"] == "abc"
        assert row["output_dir"] == "/out/abc"
        assert row["album"] == "MyAlbum"

    def test_payload_merge_on_complete(self, tmp_path):
        db = tmp_path / "test.db"
        aid = start_action("recluster", payload={"config": {"K": 5}}, db_path=db)
        complete_action(aid, payload_update={"summary": {"n_faces": 10}}, db_path=db)

        row = get_action(aid, db_path=db)
        payload = json.loads(row["payload_json"])
        assert payload["config"]["K"] == 5
        assert payload["summary"]["n_faces"] == 10

    def test_init_table_idempotent(self, tmp_path):
        db = tmp_path / "test.db"
        init_table(db_path=db)
        init_table(db_path=db)  # second call must not raise
        rows = list_actions(db_path=db)
        assert rows == []
