"""spec-117 R3: RunPersister writes people before the face rows that point at them, so Postgres (which enforces
``face_metric_rows.person_id -> people.id``) accepts a run. SQLite with foreign keys ON stands in for Postgres here;
the needs_neon parity run checks the real thing."""
from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import Album, Base, FaceMetricRow, Person, PipelineRun
from sim_bench.api.services.run_persister import RunPersister
from sim_bench.serving.run_output import RunOutput
from sim_bench.storage.database import DatabaseConfig

from tests.serving.test_run_output import RESULT, finished_context


def test_face_rows_land_after_their_people_with_foreign_keys_enforced(tmp_path):
    engine = DatabaseConfig.sqlite(tmp_path / "fk.db").make_engine(sqlite_foreign_keys=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Album(id="alb", name="Trip", source_path="D:/Trip"))
    s.add(PipelineRun(id="run", album_id="alb", steps=[], status="running"))
    s.commit()
    output = RunOutput.from_context(finished_context(), RESULT)
    people = RunPersister(s).persist("alb", "run", output)
    s.commit()
    face = s.query(FaceMetricRow).one()
    assert face.person_id == people[0].id and s.get(Person, face.person_id) is not None
