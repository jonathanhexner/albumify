"""spec-117 R4: copy every Albumify table to the one database; counts match per table; a non-empty target is
refused; the SQLite sources are only read. SQLite to SQLite here (the parity schema in the needs_neon run)."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from scripts.migrate_to_neon import CountMismatch, NeonMigrator, NotEmpty
from sim_bench.api.database.models import (Album, Base, ConfigProfile, PipelineResult, PipelineRun, Person,
                                           UniversalCache)
from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import JobRepo
from sim_bench.control.models import Person as JobPerson, Photo
from sim_bench.storage.database import DatabaseConfig

from tests._db import fresh_database_url


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def sources(tmp_path):
    """A sim_bench.db with one album run, and a spec-116 control.db (its job table still named ``people``)."""
    api = tmp_path / "sim_bench.db"
    engine = DatabaseConfig.sqlite(api).make_engine()
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        s.add(Album(id="alb", name="Trip", source_path="D:/Trip"))
        s.add(PipelineRun(id="run1", album_id="alb", steps=["discover_images"], status="completed",
                          completed_steps=[{"step": "discover_images", "duration_ms": 5}]))
        s.add(PipelineResult(id="res1", run_id="run1", total_images=2, image_metrics={"D:/Trip/a.jpg": {"iqa": 0.5}},
                             selected_images=["D:/Trip/a.jpg"]))
        s.add(Person(id="p1", album_id="alb", run_id="run1", person_index=0, face_instances=[{"face_index": 0}]))
        s.add(ConfigProfile(id="cfg1", name="default", config={"default_pipeline": ["discover_images"]}, is_default=True))
        for i in range(1203):                        # more than two batches
            s.add(UniversalCache(image_path=f"staged:{i:016x}/a.jpg", feature_type="f", model_name="m",
                                 data_blob=bytes([i % 256]) * 8, image_mtime=0.0))
        s.commit()
    engine.dispose()

    control = tmp_path / "control.db"
    db = ControlDB(f"sqlite:///{control.as_posix()}")
    repo = JobRepo(db)
    repo.create("run1", runner="local", source="D:/Trip")
    repo.log("run1", "staged 2 photos")
    with db.session() as s:
        s.add(Photo(job_id="run1", photo_id="a", in_album=True, status="pick", decisions=[{"step": "x"}]))
        s.add(JobPerson(job_id="run1", person_id="person_0", face_count=3, sample_ids=[1, 2], label="Ann"))
    db.engine.dispose()
    con = sqlite3.connect(control)
    con.execute("ALTER TABLE job_people RENAME TO people")          # what the live control.db looks like
    con.commit()
    con.close()
    return {"api": f"sqlite:///{api.as_posix()}", "control": f"sqlite:///{control.as_posix()}"}, api, control


class ut_NeonMigrator:
    def test_copy_moves_every_row_and_counts_match(self, sources, tmp_path):
        urls, api, control = sources
        before = (_sha(api), _sha(control))
        target = fresh_database_url(tmp_path, "target.db")
        m = NeonMigrator(urls, target, batch=500)
        plan = {p.target_table: p.rows for p in m.plan()}
        assert plan["universal_cache"] == 1203 and plan["job_people"] == 1 and plan["jobs"] == 1
        m.copy()
        counts = m.verify()
        assert counts["universal_cache"] == (1203, 1203) and counts["job_people"] == (1, 1)
        assert counts["job_events"][0] >= 2 and counts["job_events"][0] == counts["job_events"][1]
        assert (_sha(api), _sha(control)) == before            # the sources are the backup: untouched

    def test_values_round_trip_with_their_types(self, sources, tmp_path):
        urls, _, _ = sources
        target = fresh_database_url(tmp_path, "target.db")
        NeonMigrator(urls, target).copy()
        engine = DatabaseConfig(target).make_engine()
        with sessionmaker(bind=engine)() as s:
            res = s.get(PipelineResult, "res1")
            assert res.image_metrics == {"D:/Trip/a.jpg": {"iqa": 0.5}} and res.selected_images == ["D:/Trip/a.jpg"]
            assert s.get(PipelineRun, "run1").completed_steps[0]["step"] == "discover_images"
            assert s.scalars(select(UniversalCache.data_blob).where(
                UniversalCache.image_path == f"staged:{7:016x}/a.jpg")).one() == bytes([7]) * 8
        engine.dispose()
        db = ControlDB(target)
        with db.session() as s:
            p = s.get(JobPerson, ("run1", "person_0"))
            assert (p.label, p.sample_ids) == ("Ann", [1, 2])
            assert s.get(Photo, ("run1", "a")).decisions == [{"step": "x"}]
        assert isinstance(JobRepo(db).get("run1").created_at, datetime)

    def test_new_rows_after_the_copy_get_fresh_ids(self, sources, tmp_path):
        urls, _, _ = sources
        target = fresh_database_url(tmp_path, "target.db")
        NeonMigrator(urls, target).copy()
        repo = JobRepo(ControlDB(target))
        repo.log("run1", "after the move")                       # autoincrement id past the copied ones
        assert [e.message for e in repo.events("run1")][-1] == "after the move"

    def test_a_non_empty_target_is_refused(self, sources, tmp_path):
        urls, _, _ = sources
        target = fresh_database_url(tmp_path, "target.db")
        NeonMigrator(urls, target).copy()
        with pytest.raises(NotEmpty, match="refusing"):
            NeonMigrator(urls, target).copy()

    def test_verify_reports_a_difference(self, sources, tmp_path):
        urls, _, _ = sources
        target = fresh_database_url(tmp_path, "target.db")
        m = NeonMigrator(urls, target)
        m.copy()
        con = sqlite3.connect(Path(urls["api"][len("sqlite:///"):]))
        con.execute("INSERT INTO albums (id, name, source_path) VALUES ('late', 'Late', 'D:/x')")
        con.commit()
        con.close()
        with pytest.raises(CountMismatch, match="albums"):
            m.verify()

    def test_skip_leaves_a_table_out(self, sources, tmp_path):
        urls, _, _ = sources
        target = fresh_database_url(tmp_path, "target.db")
        m = NeonMigrator(urls, target, skip=["universal_cache"])
        assert "universal_cache" not in {p.target_table for p in m.plan()}
        m.copy()
        assert m.target_rows().get("universal_cache", 0) == 0
