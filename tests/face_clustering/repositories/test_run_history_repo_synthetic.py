"""spec-043 Phase 2 — RunHistoryRepository unit tests against synthetic data.

Every public method covered with at least one positive case + one
edge case. Tests construct a fresh repository via the
``synthetic_action_log_db`` fixture (in-memory-ish; per-test tmp_path).
No real DB dependency — runs in milliseconds on CI without
~/.sim_bench/sim_bench.db present.

Per spec-043 §5.1: ≥24 cases. Below the actual count.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from face_cluster.repositories import (
    NotFoundError,
    RunHistoryCriteria,
    RunHistoryRepoConfig,
    RunHistoryRepository,
    ValidationError,
)
from tests.face_clustering.views._seed import insert_action, seed_runs


def _repo(db_path) -> RunHistoryRepository:
    return RunHistoryRepository(RunHistoryRepoConfig(db_path=db_path))


# ===========================================================================
# find — composable criteria
# ===========================================================================

def test_find_empty_db_returns_empty_list(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    assert repo.find(RunHistoryCriteria()) == []


def test_find_no_filters_returns_all(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db, albums=("A", "B"), runs_per_album=3)
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria())
    assert len(rows) == 6


def test_find_filter_by_album(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db, albums=("Budapest", "Paris"), runs_per_album=3)
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(album="Budapest"))
    assert len(rows) == 3
    assert all(r.display_album == "Budapest" for r in rows)


def test_find_filter_by_date_from(synthetic_action_log_db):
    base = datetime(2026, 5, 10, 12, tzinfo=timezone.utc)
    insert_action(synthetic_action_log_db, started_at=base, source_album="A")
    insert_action(synthetic_action_log_db, started_at=base + timedelta(days=20), source_album="B")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(date_from=date(2026, 5, 20)))
    assert [r.display_album for r in rows] == ["B"]


def test_find_filter_by_date_to_inclusive(synthetic_action_log_db):
    base = datetime(2026, 5, 20, 23, 59, tzinfo=timezone.utc)
    insert_action(synthetic_action_log_db, started_at=base, source_album="A")
    insert_action(synthetic_action_log_db, started_at=base + timedelta(days=1, minutes=1), source_album="B")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(date_to=date(2026, 5, 20)))
    assert [r.display_album for r in rows] == ["A"]


def test_find_inverted_date_range_raises(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    with pytest.raises(ValidationError, match="date_from"):
        repo.find(RunHistoryCriteria(
            date_from=date(2026, 6, 1), date_to=date(2026, 5, 1),
        ))


def test_find_text_filter_matches_album(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, source_album="BudapestTrip")
    insert_action(synthetic_action_log_db, source_album="Paris")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(text="Buda"))
    assert [r.display_album for r in rows] == ["BudapestTrip"]


def test_find_text_filter_matches_run_name(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, source_album="A", run_name="exp-merge-1")
    insert_action(synthetic_action_log_db, source_album="A", run_name="baseline")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(text="merge"))
    assert len(rows) == 1
    assert rows[0].run_name == "exp-merge-1"


def test_find_text_filter_matches_comment(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, source_album="A", comment="best merge so far")
    insert_action(synthetic_action_log_db, source_album="A", comment="baseline")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(text="best"))
    assert len(rows) == 1
    assert rows[0].comment == "best merge so far"


def test_find_filter_by_ids(synthetic_action_log_db):
    a = insert_action(synthetic_action_log_db, source_album="A")
    b = insert_action(synthetic_action_log_db, source_album="B")
    insert_action(synthetic_action_log_db, source_album="C")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(ids=[a, b]))
    assert {r.id for r in rows} == {a, b}


def test_find_multi_filter_is_AND(synthetic_action_log_db):
    base = datetime(2026, 5, 25, 12, tzinfo=timezone.utc)
    insert_action(synthetic_action_log_db, source_album="Budapest", started_at=base)
    insert_action(synthetic_action_log_db, source_album="Paris", started_at=base)
    insert_action(synthetic_action_log_db, source_album="Budapest",
                  started_at=base - timedelta(days=1))
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(
        album="Budapest", date_from=date(2026, 5, 25),
    ))
    assert len(rows) == 1
    assert rows[0].display_album == "Budapest"


def test_find_pagination_limit_and_offset(synthetic_action_log_db):
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i in range(10):
        insert_action(synthetic_action_log_db, source_album="A",
                      started_at=base + timedelta(hours=i),
                      run_name=f"run-{i:02d}")
    page1 = _repo(synthetic_action_log_db).find(RunHistoryCriteria(limit=3, offset=0))
    page2 = _repo(synthetic_action_log_db).find(RunHistoryCriteria(limit=3, offset=3))
    assert len(page1) == 3
    assert len(page2) == 3
    # Newest-first by default: page1 has run-09, run-08, run-07; page2 has run-06...
    assert page1[0].run_name == "run-09"
    assert page2[0].run_name == "run-06"


def test_find_order_by_asc(synthetic_action_log_db):
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    insert_action(synthetic_action_log_db, source_album="A",
                  started_at=base, run_name="oldest")
    insert_action(synthetic_action_log_db, source_album="A",
                  started_at=base + timedelta(days=10), run_name="newest")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(order_by="started_at_asc"))
    assert [r.run_name for r in rows] == ["oldest", "newest"]


def test_find_filter_by_producer(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, producer="fc_app_v2")
    insert_action(synthetic_action_log_db, producer="fc_app")
    rows = _repo(synthetic_action_log_db).find(RunHistoryCriteria(producer="fc_app_v2"))
    assert len(rows) == 1


def test_find_filter_by_action_types(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, action_type="fc_app_v2_run")
    insert_action(synthetic_action_log_db, action_type="merge_apply")
    insert_action(synthetic_action_log_db, action_type="profile_save")
    rows = _repo(synthetic_action_log_db).find(
        RunHistoryCriteria(action_types=["merge_apply", "profile_save"]),
    )
    assert {r.action_type for r in rows} == {"merge_apply", "profile_save"}


# ===========================================================================
# find_one, count, distinct_albums, get_by_id
# ===========================================================================

def test_find_one_returns_first_match(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db, albums=("Budapest",), runs_per_album=3)
    row = _repo(synthetic_action_log_db).find_one(RunHistoryCriteria(album="Budapest"))
    assert row is not None
    assert row.display_album == "Budapest"


def test_find_one_no_match_returns_none(synthetic_action_log_db):
    insert_action(synthetic_action_log_db, source_album="A")
    row = _repo(synthetic_action_log_db).find_one(RunHistoryCriteria(album="Nonexistent"))
    assert row is None


def test_count_matches_find_length(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db, albums=("A",), runs_per_album=7)
    repo = _repo(synthetic_action_log_db)
    assert repo.count(RunHistoryCriteria(album="A")) == 7


def test_distinct_albums_sorted(synthetic_action_log_db):
    seed_runs(synthetic_action_log_db, albums=("Paris", "Athens", "Budapest"), runs_per_album=1)
    assert _repo(synthetic_action_log_db).distinct_albums() == ["Athens", "Budapest", "Paris"]


def test_distinct_albums_empty_db(synthetic_action_log_db):
    assert _repo(synthetic_action_log_db).distinct_albums() == []


def test_get_by_id_returns_existing_row(synthetic_action_log_db):
    rid = insert_action(synthetic_action_log_db, source_album="X")
    row = _repo(synthetic_action_log_db).get_by_id(rid)
    assert row is not None and row.id == rid


def test_get_by_id_returns_none_for_missing(synthetic_action_log_db):
    assert _repo(synthetic_action_log_db).get_by_id(999_999) is None


# ===========================================================================
# Mutations: start_action / complete_action / fail_action
# ===========================================================================

def test_start_action_inserts_running_row(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    rid = repo.start_action("fc_app_v2_run", payload={
        "run_id": "20260525_120000",
        "source_album": "Budapest",
        "producer": "fc_app_v2",
    })
    row = repo.get_by_id(rid)
    assert row is not None
    assert row.action_type == "fc_app_v2_run"
    assert row.status == "running"
    assert row.run_id == "20260525_120000"
    assert row.display_album == "Budapest"


def test_complete_action_sets_complete_and_merges_result(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    rid = repo.start_action("fc_app_v2_run", payload={"source_album": "X"})
    repo.complete_action(rid, result_fields={"n_clusters": 33, "n_faces": 340})
    row = repo.get_by_id(rid)
    assert row.status == "complete"
    assert row.n_clusters == 33
    assert row.n_faces == 340
    assert row.duration_s is not None and row.duration_s >= 0


def test_complete_action_raises_on_missing_id(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    with pytest.raises(NotFoundError):
        repo.complete_action(999_999, result_fields={"n_clusters": 1})


def test_fail_action_sets_failed_and_records_error(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    rid = repo.start_action("fc_app_v2_run", payload={"source_album": "X"})
    repo.fail_action(rid, "boom")
    row = repo.get_by_id(rid)
    assert row.status == "failed"


def test_fail_action_raises_on_missing_id(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    with pytest.raises(NotFoundError):
        repo.fail_action(999_999, "boom")


# ===========================================================================
# Mutations: update_comment
# ===========================================================================

def test_update_comment_persists(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    rid = insert_action(synthetic_action_log_db)
    repo.update_comment(rid, "important")
    assert repo.get_by_id(rid).comment == "important"


def test_update_comment_idempotent(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    rid = insert_action(synthetic_action_log_db)
    repo.update_comment(rid, "foo")
    repo.update_comment(rid, "foo")
    assert repo.get_by_id(rid).comment == "foo"


def test_update_comment_raises_on_overlength(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    rid = insert_action(synthetic_action_log_db)
    with pytest.raises(ValidationError, match="2048"):
        repo.update_comment(rid, "x" * 2049)


def test_update_comment_raises_on_missing_id(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    with pytest.raises(NotFoundError):
        repo.update_comment(999_999, "anything")


# ===========================================================================
# Config: read_only flag
# ===========================================================================

def test_read_only_repository_rejects_mutations(synthetic_action_log_db):
    # Insert via writable repo first (so there's a row to read).
    rid = insert_action(synthetic_action_log_db)
    # Now construct a read-only one and confirm mutations raise.
    repo = RunHistoryRepository(
        RunHistoryRepoConfig(db_path=synthetic_action_log_db, read_only=True),
    )
    with pytest.raises(ValidationError, match="read_only"):
        repo.update_comment(rid, "x")
    with pytest.raises(ValidationError, match="read_only"):
        repo.start_action("anything")
    # Reads still work
    assert repo.get_by_id(rid) is not None


# ===========================================================================
# Validation guards
# ===========================================================================

def test_find_rejects_unknown_text_fields(synthetic_action_log_db):
    repo = _repo(synthetic_action_log_db)
    with pytest.raises(ValidationError, match="text_fields"):
        repo.find(RunHistoryCriteria(text="x", text_fields=("evil; DROP TABLE",)))
