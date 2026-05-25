"""spec-042 H1 — HistoryService unit tests against synthetic data.

26+ cases covering every public method's contract. Uses an in-memory
SQLite (via the ``synthetic_action_log_db`` fixture) seeded by
``_seed.py`` row factories. No real-data dependency — these tests run
in milliseconds on CI without any fixtures present.

Per the spec-042 testing strategy, this is the **load-bearing** layer.
The real-fixture integration tests (test_history_service_real.py) are
smoke confirmations only.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from face_cluster.repositories import RunHistoryRepoConfig, RunHistoryRepository
from face_cluster.views.history import (
    ActionTypeFormat,
    HistoryQuery,
    HistoryService,
    LoadedRun,
    RunDetail,
)
from tests.face_clustering.views._seed import insert_action, seed_runs


def _service(db_path):
    """spec-043 migration helper: construct a HistoryService with an
    in-memory-ish Repository wired to ``db_path``. Replaces the
    pre-migration ``HistoryService(db_path=...)`` shape."""
    return HistoryService(
        repo=RunHistoryRepository(RunHistoryRepoConfig(db_path=db_path)),
    )


# ===========================================================================
# list_runs — filter + ordering
# ===========================================================================

def test_list_runs_empty_db_returns_empty_list(synthetic_action_log_db):
    service = _service(synthetic_action_log_db)
    assert service.list_runs(HistoryQuery()) == []


def test_list_runs_no_filters_returns_all(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db, albums=("Budapest", "Paris"), runs_per_album=3)
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery())
    assert len(rows) == 6


def test_list_runs_filter_by_album(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db, albums=("Budapest", "Paris"), runs_per_album=3)
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery(album="Budapest"))
    assert len(rows) == 3
    assert all(r.display_album == "Budapest" for r in rows)


def test_list_runs_filter_by_date_from(synthetic_action_log_db):
    base = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
    insert_action(synthetic_action_log_db, started_at=base, source_album="A")
    insert_action(synthetic_action_log_db, started_at=base + timedelta(days=20), source_album="B")
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery(date_from=date(2026, 5, 20)))
    assert len(rows) == 1
    assert rows[0].display_album == "B"


def test_list_runs_filter_by_date_to_inclusive(synthetic_action_log_db):
    """date_to includes the entire end day."""
    base = datetime(2026, 5, 20, 23, 59, tzinfo=timezone.utc)
    insert_action(synthetic_action_log_db, started_at=base, source_album="A")
    insert_action(synthetic_action_log_db,
                  started_at=base + timedelta(days=1, minutes=1),
                  source_album="B")
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery(date_to=date(2026, 5, 20)))
    assert [r.display_album for r in rows] == ["A"]


def test_list_runs_text_filter_matches_album(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, source_album="BudapestTrip")
    insert_action(synthetic_action_log_db, source_album="Paris")
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery(text="Buda"))
    assert len(rows) == 1
    assert rows[0].display_album == "BudapestTrip"


def test_list_runs_text_filter_matches_run_name(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, source_album="A", run_name="exp-merge-1")
    insert_action(synthetic_action_log_db, source_album="A", run_name="baseline")
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery(text="merge"))
    assert len(rows) == 1
    assert rows[0].run_name == "exp-merge-1"


def test_list_runs_text_filter_matches_comment(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, source_album="A", comment="best merge so far")
    insert_action(synthetic_action_log_db, source_album="A", comment="baseline")
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery(text="best"))
    assert len(rows) == 1
    assert rows[0].comment == "best merge so far"


def test_list_runs_multi_filter_is_AND_not_OR(synthetic_action_log_db):
    base = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
    # Budapest today
    insert_action(synthetic_action_log_db, source_album="Budapest", started_at=base)
    # Paris today
    insert_action(synthetic_action_log_db, source_album="Paris", started_at=base)
    # Budapest yesterday
    insert_action(synthetic_action_log_db, source_album="Budapest",
                  started_at=base - timedelta(days=1))
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery(
        album="Budapest", date_from=date(2026, 5, 25),
    ))
    assert len(rows) == 1
    assert rows[0].display_album == "Budapest"


def test_list_runs_newest_first(synthetic_action_log_db):
    """Returned rows are sorted by started_at DESC."""
    base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    insert_action(synthetic_action_log_db, source_album="A",
                  started_at=base, run_name="oldest")
    insert_action(synthetic_action_log_db, source_album="A",
                  started_at=base + timedelta(days=5), run_name="middle")
    insert_action(synthetic_action_log_db, source_album="A",
                  started_at=base + timedelta(days=10), run_name="newest")
    service = _service(synthetic_action_log_db)
    rows = service.list_runs(HistoryQuery())
    assert [r.run_name for r in rows] == ["newest", "middle", "oldest"]


# ===========================================================================
# list_albums
# ===========================================================================

def test_list_albums_distinct_and_sorted(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db,
              albums=("Budapest", "Paris", "Budapest", "Athens"),
              runs_per_album=1)
    service = _service(synthetic_action_log_db)
    assert service.list_albums() == ["Athens", "Budapest", "Paris"]


def test_list_albums_empty_db_returns_empty_list(synthetic_action_log_db):
    service = _service(synthetic_action_log_db)
    assert service.list_albums() == []


# ===========================================================================
# get_run_detail
# ===========================================================================

def test_get_run_detail_returns_full_shape(synthetic_action_log_db):
    rid = insert_action(
        synthetic_action_log_db,
        source_album="Budapest",
        run_name="exp-1",
        config_json=json.dumps({"K": 5, "distance_threshold": 0.35}),
    )
    service = _service(synthetic_action_log_db)
    detail = service.get_run_detail(rid)
    assert isinstance(detail, RunDetail)
    assert detail.row.id == rid
    assert detail.config == {"K": 5, "distance_threshold": 0.35}
    assert detail.parent_row is None
    assert detail.config_delta == []
    assert detail.summary is None  # no pipeline_run.json on disk
    assert detail.has_required_artifacts is False


def test_get_run_detail_with_parent_computes_config_delta(synthetic_action_log_db):
    parent_id = insert_action(
        synthetic_action_log_db,
        source_album="A",
        run_name="parent",
        config_json=json.dumps({"K": 5, "distance_threshold": 0.35}),
    )
    child_id = insert_action(
        synthetic_action_log_db,
        source_album="A",
        run_name="child",
        config_json=json.dumps({"K": 7, "distance_threshold": 0.35}),
        parent_run_id=parent_id,
    )
    service = _service(synthetic_action_log_db)
    detail = service.get_run_detail(child_id)
    assert detail.parent_row is not None
    assert detail.parent_row.id == parent_id
    diffs = {d.field: (d.parent_value, d.child_value) for d in detail.config_delta}
    assert diffs.get("K") == (5, 7)
    assert "distance_threshold" not in diffs  # unchanged → not in delta


def test_get_run_detail_raises_on_missing_id(synthetic_action_log_db):
    service = _service(synthetic_action_log_db)
    with pytest.raises(ValueError, match="No run with id=999999"):
        service.get_run_detail(999999)


def test_get_run_detail_parses_pipeline_run_json(synthetic_action_log_db, tmp_path):
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    prun = {
        "summary": {"n_faces": 340, "n_core": 282, "n_clusters": 33, "n_noise": 58},
        "stages": {
            "detect_persons": {"elapsed_s": 54.2},
            "extract_face_embeddings": {"elapsed_s": 62.5},
        },
        "merge_metadata": {"merge_candidate_threshold": 0.45},
        "merge_log": [
            {"actually_merged": True, "pair": [1, 2]},
            {"actually_merged": False, "pair": [3, 4]},
            {"actually_merged": True, "pair": [5, 6]},
        ],
    }
    (out_dir / "pipeline_run.json").write_text(json.dumps(prun), encoding="utf-8")
    rid = insert_action(synthetic_action_log_db, output_dir=str(out_dir))
    service = _service(synthetic_action_log_db)
    detail = service.get_run_detail(rid)
    assert detail.summary is not None
    assert detail.summary.n_faces == 340
    assert detail.summary.n_clusters_base == 33
    assert detail.summary.merge_count == 2  # only the 'actually_merged' entries
    assert detail.summary.merge_candidate_threshold == 0.45
    assert detail.summary.stage_durations["detect_persons"] == 54.2


# ===========================================================================
# update_comment
# ===========================================================================

def test_update_comment_persists(synthetic_action_log_db):
    rid = insert_action(synthetic_action_log_db)
    service = _service(synthetic_action_log_db)
    service.update_comment(rid, "important run")
    assert service.get_run_detail(rid).row.comment == "important run"


def test_update_comment_idempotent(synthetic_action_log_db):
    rid = insert_action(synthetic_action_log_db)
    service = _service(synthetic_action_log_db)
    service.update_comment(rid, "foo")
    service.update_comment(rid, "foo")
    assert service.get_run_detail(rid).row.comment == "foo"


def test_update_comment_rejects_overlength(synthetic_action_log_db):
    rid = insert_action(synthetic_action_log_db)
    service = _service(synthetic_action_log_db)
    with pytest.raises(ValueError, match="exceeds"):
        service.update_comment(rid, "x" * 2049)


# ===========================================================================
# load_run
# ===========================================================================

def test_load_run_raises_on_incomplete_run(synthetic_action_log_db):
    rid = insert_action(synthetic_action_log_db, status="failed")
    service = _service(synthetic_action_log_db)
    with pytest.raises(ValueError, match="not complete"):
        service.load_run(rid)


def test_load_run_raises_when_output_dir_missing(synthetic_action_log_db):
    rid = insert_action(synthetic_action_log_db, status="complete", output_dir=None)
    service = _service(synthetic_action_log_db)
    with pytest.raises(ValueError, match="no output_dir"):
        service.load_run(rid)


def test_load_run_raises_when_artifacts_missing(synthetic_action_log_db, tmp_path):
    out_dir = tmp_path / "empty_run"
    out_dir.mkdir()  # no faces.csv / clusters.csv / embeddings.npy
    rid = insert_action(
        synthetic_action_log_db,
        status="complete",
        output_dir=str(out_dir),
    )
    service = _service(synthetic_action_log_db)
    with pytest.raises(ValueError, match="missing required artifacts"):
        service.load_run(rid)


# ===========================================================================
# list_other_actions
# ===========================================================================

def test_list_other_actions_filters_by_type(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, action_type="merge_apply",
                  payload_json=json.dumps({"round": 1, "n_approved": 3,
                                            "clusters_before": 10}))
    insert_action(synthetic_action_log_db, action_type="fc_app_v2_run")
    insert_action(synthetic_action_log_db, action_type="ml_train",
                  payload_json=json.dumps({"model_type": "lr", "accuracy": 0.85}))
    service = _service(synthetic_action_log_db)
    rows = service.list_other_actions()
    types = {r.action_type for r in rows}
    assert types == {"merge_apply", "ml_train"}  # fc_app_v2_run excluded


def test_list_other_actions_honors_limit(synthetic_action_log_db):
    for _ in range(150):
        insert_action(synthetic_action_log_db, action_type="profile_save",
                      payload_json=json.dumps({"profile_name": "p"}))
    service = _service(synthetic_action_log_db)
    rows = service.list_other_actions(limit=50)
    assert len(rows) == 50


def test_list_other_actions_details_formatted_by_type(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, action_type="merge_apply",
                  n_clusters=25,
                  payload_json=json.dumps({"round": 2, "n_approved": 5,
                                            "clusters_before": 30}))
    service = _service(synthetic_action_log_db)
    rows = service.list_other_actions()
    assert "round=2" in rows[0].details
    assert "approved=5" in rows[0].details
    assert "30 -> 25" in rows[0].details


# ===========================================================================
# get_action_payload
# ===========================================================================

def test_get_action_payload_returns_parsed_dict(synthetic_action_log_db):
    rid = insert_action(synthetic_action_log_db,
                        payload_json=json.dumps({"foo": "bar", "n": 42}))
    service = _service(synthetic_action_log_db)
    assert service.get_action_payload(rid) == {"foo": "bar", "n": 42}


def test_get_action_payload_missing_id_returns_empty_dict(synthetic_action_log_db):
    service = _service(synthetic_action_log_db)
    assert service.get_action_payload(999999) == {}


# ===========================================================================
# ActionTypeFormat — typed dispatch
# ===========================================================================

def test_action_type_format_merge_apply():
    out = ActionTypeFormat.format(
        "merge_apply",
        {"round": 3, "n_approved": 7, "clusters_before": 20},
        {"n_clusters": 13},
    )
    assert "round=3" in out
    assert "approved=7" in out
    assert "20 -> 13" in out


def test_action_type_format_profile_save():
    out = ActionTypeFormat.format("profile_save", {"profile_name": "strict"}, {})
    assert "profile=strict" in out


def test_action_type_format_ml_train():
    out = ActionTypeFormat.format(
        "ml_train", {"model_type": "lr", "accuracy": 0.92}, {},
    )
    assert "model=lr" in out
    assert "0.92" in out


def test_action_type_format_unknown_type_returns_empty():
    assert ActionTypeFormat.format("nonexistent_type", {}, {}) == ""
