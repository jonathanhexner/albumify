"""T013 — Tests for face_cluster.run_history search helper."""
from datetime import date, timezone, datetime

import pytest

from face_cluster import run_history_db
from face_cluster.run_history import HistoryFilters, RunRow, search, distinct_albums, get_run_by_id

_UNKNOWN = "(unknown)"


def _insert(db, source_album, run_name="run", started_at=None, comment=None, run_kind="base"):
    started_at = started_at or datetime.now(timezone.utc).isoformat()
    action_id = run_history_db.start_action(
        "pipeline_run",
        payload={
            "source_album": source_album,
            "run_name": run_name,
            "run_kind": run_kind,
        },
        db_path=db,
    )
    # Backfill started_at if provided (for date-range tests)
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE action_log SET started_at=?, comment=? WHERE id=?",
                 (started_at, comment, action_id))
    conn.commit()
    conn.close()
    run_history_db.complete_action(action_id, db_path=db)
    return action_id


class ut_RunHistorySearch:
    def test_no_filters_returns_all(self, tmp_path):
        db = tmp_path / "test.db"
        _insert(db, "AlbumA")
        _insert(db, "AlbumB")
        _insert(db, "AlbumC")
        rows = search(HistoryFilters(), db_path=db)
        albums = {r.source_album for r in rows}
        assert {"AlbumA", "AlbumB", "AlbumC"}.issubset(albums)

    def test_album_filter(self, tmp_path):
        db = tmp_path / "test.db"
        _insert(db, "AlbumA")
        _insert(db, "AlbumA")
        _insert(db, "AlbumB")
        rows = search(HistoryFilters(album="AlbumA"), db_path=db)
        assert all(r.source_album == "AlbumA" for r in rows)
        assert len(rows) == 2

    def test_date_from_filter(self, tmp_path):
        db = tmp_path / "test.db"
        _insert(db, "AlbumA", started_at="2026-01-10T00:00:00+00:00")
        _insert(db, "AlbumA", started_at="2026-03-01T00:00:00+00:00")
        rows = search(HistoryFilters(date_from=date(2026, 2, 1)), db_path=db)
        assert all(r.started_at >= "2026-02-01" for r in rows)
        assert len(rows) == 1

    def test_date_to_filter(self, tmp_path):
        db = tmp_path / "test.db"
        _insert(db, "AlbumA", started_at="2026-01-10T00:00:00+00:00")
        _insert(db, "AlbumA", started_at="2026-03-01T00:00:00+00:00")
        rows = search(HistoryFilters(date_to=date(2026, 2, 1)), db_path=db)
        assert len(rows) == 1
        assert rows[0].started_at.startswith("2026-01-10")

    def test_text_search_on_comment(self, tmp_path):
        db = tmp_path / "test.db"
        _insert(db, "AlbumA", comment="tight threshold test")
        _insert(db, "AlbumA", comment="loose params")
        rows = search(HistoryFilters(text="tight"), db_path=db)
        assert len(rows) == 1
        assert rows[0].comment == "tight threshold test"

    def test_text_search_on_album(self, tmp_path):
        db = tmp_path / "test.db"
        _insert(db, "SpecialAlbum")
        _insert(db, "OtherAlbum")
        rows = search(HistoryFilters(text="Special"), db_path=db)
        assert len(rows) == 1
        assert rows[0].source_album == "SpecialAlbum"

    def test_null_source_album_returns_unknown(self, tmp_path):
        """Pre-013 rows with NULL source_album must not crash and show (unknown)."""
        db = tmp_path / "test.db"
        # Insert without source_album (simulates pre-013 row)
        aid = run_history_db.start_action("pipeline_run", payload={}, db_path=db)
        run_history_db.complete_action(aid, db_path=db)
        rows = search(HistoryFilters(), db_path=db)
        null_rows = [r for r in rows if r.source_album == _UNKNOWN]
        assert len(null_rows) >= 1

    def test_distinct_albums(self, tmp_path):
        db = tmp_path / "test.db"
        _insert(db, "AlbumA")
        _insert(db, "AlbumA")
        _insert(db, "AlbumB")
        albums = distinct_albums(db_path=db)
        assert albums == ["AlbumA", "AlbumB"]

    def test_get_run_by_id(self, tmp_path):
        db = tmp_path / "test.db"
        action_id = _insert(db, "AlbumA", run_name="run_007")
        row = get_run_by_id(action_id, db_path=db)
        assert row is not None
        assert row.id == action_id
        assert row.source_album == "AlbumA"

    def test_get_run_by_id_missing_returns_none(self, tmp_path):
        db = tmp_path / "test.db"
        run_history_db.init_table(db_path=db)
        assert get_run_by_id(99999, db_path=db) is None

    def test_reserved_rows_excluded(self, tmp_path):
        """Reservation rows (status='reserved') must not appear in search results."""
        from face_cluster.run_naming import RunDirSpec, allocate_run_dir
        db = tmp_path / "test.db"
        spec = RunDirSpec(source_album="AlbumX", kind="remerge")
        allocate_run_dir(spec, tmp_path / "results", db_path=db)
        # The reservation is in run_dir_reservations, not action_log, so this
        # test simply confirms search() does not crash and returns only action_log rows
        rows = search(HistoryFilters(), db_path=db)
        assert all(r.status != "reserved" for r in rows)
