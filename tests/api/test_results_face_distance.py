"""spec-116 R14 / R6: face tabs read the run's own embeddings and rows, whichever machine ran it."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sim_bench.api.database.models import (Album, Base, FaceMetricRow, Person, PipelineResult, PipelineRun,
                                           UniversalCache)
from sim_bench.api.database.session import get_session
from sim_bench.api.routers import results as results_router
from sim_bench.api.services import run_embeddings
from sim_bench.api.services.face_service import FaceService
from sim_bench.api.services.run_embeddings import RunEmbeddings
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.serving_config import ServingConfig

RUN = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs")
    faces = [{"face_id": 0, "face_index": 0, "photo_id": "a"}, {"face_id": 1, "face_index": 1, "photo_id": "a"},
             {"face_id": 2, "face_index": 0, "photo_id": "b"}]
    vectors = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    store.write_results(RUN, {"faces": faces}, vectors, [0, 1, 2], {})
    monkeypatch.setattr(run_embeddings, "load_serving_config", lambda: ServingConfig(jobs_root=tmp_path / "jobs"))
    run_embeddings._loaded.clear()

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([Album(id="alb", name="Trip", source_path="D:/Trip"),
                     PipelineRun(id=RUN, album_id="alb", steps=[], status="completed"),
                     PipelineRun(id="old", album_id="alb", steps=[], status="completed")])
    session.commit()
    app = FastAPI()
    app.include_router(results_router.router)
    app.dependency_overrides[get_session] = lambda: session
    c = TestClient(app)
    c.session = session
    return c


def _distance(c, run, a="D:/Trip/a.jpg", b="D:/Trip/b.jpg", face_b=0):
    return c.get(f"/api/v1/results/{run}/face-distance",
                 params={"image_path_a": a, "image_path_b": b, "face_index_b": face_b})


class ut_FaceDistance:
    def test_reads_run_embeddings(self, world):
        r = _distance(world, RUN)
        assert r.status_code == 200, r.text
        assert r.json()["cosine_distance"] == pytest.approx(1.0) and r.json()["verdict"] == "different"
        assert _distance(world, RUN, b="D:/Trip/a.jpg", face_b=1).json()["cosine_distance"] == pytest.approx(0.0)

    def test_a_windows_path_finds_the_same_photo(self, world):
        assert _distance(world, RUN, a="D:\\Trip\\a.jpg").status_code == 200

    def test_an_old_run_falls_back_to_the_cache(self, world):
        emb = np.array([1, 0, 0, 0], dtype=np.float32)
        from sim_bench.pipeline.serializers import Serializers
        for name in ("a", "b"):
            world.session.add(UniversalCache(
                image_path=f"D:/Old/{name}.jpg:face_0", feature_type="face_embedding", model_name="m",
                data_blob=Serializers.numpy_serialize(emb), image_mtime=0.0))
        world.session.commit()
        r = _distance(world, "old", "D:/Old/a.jpg", "D:/Old/b.jpg")
        assert r.status_code == 200 and r.json()["cosine_distance"] == pytest.approx(0.0)

    def test_a_face_nobody_has_is_a_404(self, world):
        assert _distance(world, "old", "D:/Old/none.jpg").status_code == 404

    def test_a_run_in_flight_is_looked_up_again(self, tmp_path):
        reader = RunEmbeddings(tmp_path)
        assert not reader.has("late")
        JobStore(tmp_path).write_results("late", {"faces": [{"face_id": 0, "face_index": 0, "photo_id": "a"}]},
                                         np.ones((1, 4)), [0], {})
        assert reader.has("late")


class ut_PeopleAndFaces:
    def test_a_staged_run_lists_its_faces_without_the_cache(self, world):
        s = world.session
        s.add(PipelineResult(id="r", run_id=RUN, image_metrics={"D:\\Trip\\a.jpg": {"frontal_scores": [
            {"face_index": 0, "frontal_score": 0.7}]}}))
        s.add(Person(id="p", album_id="alb", run_id=RUN, person_index=0, face_count=1, image_count=1,
                     face_instances=[{"image_path": "D:\\Trip\\a.jpg", "face_index": 0}]))
        s.add(FaceMetricRow(run_id=RUN, image_path="D:\\Trip\\a.jpg", face_index=0, person_id="p", bbox_x=0.1,
                            bbox_y=0.2, bbox_w=0.3, bbox_h=0.4, filter_passed=True))
        s.commit()
        faces = FaceService(s).get_all_faces("alb", RUN)
        assert [(f.face_key, f.status, f.frontal_score) for f in faces] == [("D:/Trip/a.jpg:face_0", "assigned", 0.7)]
        assert faces[0].bbox["x"] == 0.1
