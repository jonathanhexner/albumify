"""spec-112 TA3 - files -> rows -> album. AC2 (nothing lost, idempotent), AC3 (built album ==
worker album), AC5 (edits round-trip).

The job's files are produced by the real JobCurator (album normalizer + results bundler) over a fake
finished context, so the loader is tested against exactly what the worker writes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sim_bench.control.album_builder import (
    DEMOTED_DETAIL, AlbumBuilder, BuildInputs, EditError, record_edit,
)
from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import JobRepo
from sim_bench.control.loader import LoadInputs, MissingResults, ResultsLoader
from sim_bench.control.models import Face, Photo
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator, stable_album
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder

from tests.control.fakes import NAMES, fake_pipeline as _runner


@pytest.fixture
def env(tmp_path):
    store = JobStore(tmp_path / "jobs")
    store.stage_inputs("trip", [(n, b"jpeg") for n in NAMES])
    JobCurator(store, spec_builder=SpecBuilder(), runner=_runner,
               clock=lambda: "2026-10-10T00:00:00+00:00").calc(
        CurateJobInputs(job_id="trip", pipeline="minimal"))
    db = ControlDB(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    JobRepo(db).create("trip", runner="local")
    return SimpleNamespace(store=store, db=db, loader=ResultsLoader(db, store),
                           builder=AlbumBuilder(db))


def _rows(db, model):
    with db.session() as s:
        return list(s.query(model).filter_by(job_id="trip"))


class ut_ResultsLoader:
    def test_every_staged_photo_and_face_becomes_a_row(self, env):
        res = env.loader.calc(LoadInputs("trip"))
        assert (res.n_photos, res.n_in_album, res.n_picks, res.n_faces) == (6, 5, 2, 3)
        assert (res.n_people, res.n_scenes) == (2, 3) and res.warnings == []

    def test_photo_filtered_before_scoring_is_a_row_with_its_reason(self, env):
        env.loader.calc(LoadInputs("trip"))
        q = next(p for p in _rows(env.db, Photo) if p.photo_id == "q")
        assert q.in_album is False and q.status is None
        assert q.decisions[0]["reason"] == "IQA 0.05 < 0.20"

    def test_keys_point_at_the_job_files(self, env):
        env.loader.calc(LoadInputs("trip"))
        a = next(p for p in _rows(env.db, Photo) if p.photo_id == "a")
        assert a.input_key == "input/a.jpg" and a.display_key == "display/a.jpg"
        f0 = next(f for f in _rows(env.db, Face) if f.face_id == 0)
        assert f0.crop_key == "faces/0.jpg" and f0.person_id == 3 and f0.photo_id == "a"
        assert (env.store.paths("trip").root / f0.crop_key).is_file()

    def test_reload_is_idempotent(self, env):
        first = env.loader.calc(LoadInputs("trip"))
        second = env.loader.calc(LoadInputs("trip"))
        assert first == second
        assert len(_rows(env.db, Photo)) == 6 and len(_rows(env.db, Face)) == 3

    def test_missing_bundle_is_reported(self, env, tmp_path):
        env.store.paths("trip").bundle_path.unlink()
        with pytest.raises(MissingResults, match="bundle.json"):
            env.loader.calc(LoadInputs("trip"))

    def test_unknown_job_is_rejected(self, env):
        env.store.stage_inputs("other", [("x.jpg", b"x")])
        env.store.write_album("other", {"picks": [], "dropped": []})
        env.store.write_results("other", {"photos": [], "faces": []})
        with pytest.raises(KeyError):
            env.loader.calc(LoadInputs("other"))


class ut_AlbumBuilder:
    def test_built_album_equals_the_worker_album(self, env):
        """AC3: tables -> album reproduces album.json exactly."""
        env.loader.calc(LoadInputs("trip"))
        built = env.builder.calc(BuildInputs("trip")).to_dict()
        assert stable_album(built) == stable_album(env.store.read_album("trip"))
        assert built["created_at"] == env.store.read_album("trip")["created_at"]

    def test_promote_moves_a_dropped_photo_into_the_picks_in_time_order(self, env):
        env.loader.calc(LoadInputs("trip"))
        record_edit(env.db, "trip", "promote", {"photo_id": "b"})
        album = env.builder.calc(BuildInputs("trip")).to_dict()
        assert [p["id"] for p in album["picks"]] == ["c", "a", "b"]
        assert "b" not in {d["id"] for d in album["dropped"]}
        assert album["picks"][1]["beat"] == []            # a no longer "beat" b
        assert album["counts"]["kept"] == 3

    def test_demote_then_promote_round_trips(self, env):
        env.loader.calc(LoadInputs("trip"))
        record_edit(env.db, "trip", "demote", {"photo_id": "a"})
        album = env.builder.calc(BuildInputs("trip")).to_dict()
        a = next(d for d in album["dropped"] if d["id"] == "a")
        assert a["reason"] == "not_selected" and a["reason_detail"] == DEMOTED_DETAIL
        record_edit(env.db, "trip", "promote", {"photo_id": "a"})
        assert [p.id for p in env.builder.calc(BuildInputs("trip")).picks] == ["c", "a"]

    def test_rename_person(self, env):
        env.loader.calc(LoadInputs("trip"))
        record_edit(env.db, "trip", "rename", {"person_id": "person_3", "label": " Mom "})
        people = {p.id: p.label for p in env.builder.calc(BuildInputs("trip")).people}
        assert people == {"person_3": "Mom", "person_5": ""}

    def test_edits_survive_a_reload(self, env):
        env.loader.calc(LoadInputs("trip"))
        record_edit(env.db, "trip", "promote", {"photo_id": "d"})
        env.loader.calc(LoadInputs("trip"))
        assert "d" in [p.id for p in env.builder.calc(BuildInputs("trip")).picks]

    @pytest.mark.parametrize("kind,payload", [
        ("promote", {"photo_id": "q"}),          # never in the album
        ("promote", {"photo_id": "zzz"}),
        ("rename", {"person_id": "person_99", "label": "x"}),
        ("explode", {}),
    ])
    def test_invalid_edits_are_refused(self, env, kind, payload):
        env.loader.calc(LoadInputs("trip"))
        with pytest.raises(EditError):
            record_edit(env.db, "trip", kind, payload)
