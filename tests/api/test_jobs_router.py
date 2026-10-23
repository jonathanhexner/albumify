"""spec-112 TA5.2 - jobs API on local SQLite + local job store + an in-process fake runner.

The router is mounted on a bare FastAPI app so the test does not start the legacy API (sim_bench.db,
step registry). AC3 over HTTP: the album endpoint returns exactly the worker's album.json.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from sim_bench.api.routers import jobs as jobs_router
from sim_bench.control.db import ControlDB
from sim_bench.serving.curate_job import stable_album
from sim_bench.serving.job_service import JobService
from sim_bench.serving.serving_config import ServingConfig

from tests.control.fakes import NAMES
from tests.control.test_job_service import FailingRunner, _in_process


@pytest.fixture
def client(tmp_path):
    svc = JobService(ControlDB(f"sqlite:///{(tmp_path / 'c.db').as_posix()}"),
                     ServingConfig(runner="local", pipeline="minimal", jobs_root=tmp_path / "jobs"),
                     runner_factory=_in_process)
    app = FastAPI()
    app.include_router(jobs_router.router)
    app.dependency_overrides[jobs_router.get_job_service] = lambda: svc
    src = tmp_path / "trip"
    src.mkdir()
    for n in NAMES:
        Image.new("RGB", (900, 600), "gray").save(src / n)
    c = TestClient(app)
    c.svc, c.src = svc, src
    return c


def _submit(client, **extra):
    r = client.post("/jobs", json={"source": str(client.src), "job_id": "trip", **extra})
    assert r.status_code == 202, r.text
    return r.json()


class ut_JobsRouter:
    def test_submit_runs_to_succeeded_with_history(self, client):
        assert _submit(client) == {"job_id": "trip", "status": "queued"}
        job = client.get("/jobs/trip").json()      # TestClient runs background tasks inline
        assert job["status"] == "succeeded" and job["n_inputs"] == 6
        assert [e["kind"] for e in job["events"]][0] == "status"
        assert [j["id"] for j in client.get("/jobs").json()] == ["trip"]

    def test_album_endpoint_equals_worker_album(self, client):
        """AC3 over the API."""
        _submit(client)
        album = client.get("/jobs/trip/album").json()
        assert stable_album(album) == stable_album(client.svc.store.read_album("trip"))

    def test_photos_include_the_ones_the_album_never_saw(self, client):
        _submit(client)
        photos = {p["photo_id"]: p for p in client.get("/jobs/trip/photos").json()}
        assert len(photos) == 6 and photos["q"]["in_album"] is False
        assert photos["q"]["decisions"][0]["step"] == "filter_quality"
        assert photos["a"]["has_display"] and photos["a"]["n_faces"] == 1

    def test_images_and_crops_are_served(self, client):
        _submit(client)
        assert client.get("/jobs/trip/photos/a/image").headers["content-type"] == "image/jpeg"
        assert client.get("/jobs/trip/photos/b/image").status_code == 200   # dropped: input copy
        assert client.get("/jobs/trip/faces/0/crop").status_code == 200
        assert client.get("/jobs/trip/photos/zzz/image").status_code == 404
        faces = client.get("/jobs/trip/faces", params={"person_id": 3}).json()
        assert [f["face_id"] for f in faces] == [0, 1]

    def test_other_jobs_photo_is_404(self, client, tmp_path):
        """spec-115 R7: photos live in a shared store, but a job only serves the ones its manifest lists."""
        _submit(client)
        other = tmp_path / "other"
        other.mkdir()
        Image.new("RGB", (900, 600), "red").save(other / "x.jpg")
        assert client.post("/jobs", json={"source": str(other), "job_id": "other"}).status_code == 202
        foreign = "inputs/" + client.svc.store.input_keys("other")[0]
        from sim_bench.control.models import Photo
        with client.svc.db.session() as s:
            row = s.get(Photo, ("trip", "b"))
            row.input_key, row.display_key = foreign, None
        assert client.get("/jobs/trip/photos/b/image").status_code == 404
        assert client.get("/jobs/other/photos/x/image").status_code == 200

    def test_edit_round_trip(self, client):
        """AC5 over the API."""
        _submit(client)
        r = client.post("/jobs/trip/edits", json={"kind": "promote", "payload": {"photo_id": "b"}})
        assert r.status_code == 201
        assert "b" in [p["id"] for p in client.get("/jobs/trip/album").json()["picks"]]
        r = client.post("/jobs/trip/edits", json={"kind": "promote", "payload": {"photo_id": "q"}})
        assert r.status_code == 422

    def test_bad_requests(self, client, tmp_path):
        assert client.post("/jobs", json={"source": str(tmp_path / "nope")}).status_code == 422
        assert client.post("/jobs", json={"source": str(client.src),
                                          "runner": "lambda"}).status_code == 422
        assert client.post("/jobs", json={"source": str(client.src),
                                          "job_id": "../x"}).status_code == 422
        assert client.get("/jobs/nope").status_code == 404
        assert client.get("/jobs/nope/album").status_code == 404

    def test_album_of_a_failed_job_is_409(self, client):
        client.svc._factory = lambda *a: FailingRunner()
        _submit(client)
        assert client.get("/jobs/trip").json()["status"] == "failed"
        assert client.get("/jobs/trip/album").status_code == 409

    def test_load_existing_files(self, client):
        _submit(client)
        client.svc.store.paths("trip")  # files are there from the run
        r = client.post("/jobs/trip/load", json={"runner": "modal_gpu"})
        assert r.status_code == 200 and r.json()["status"] == "succeeded"
        assert client.post("/jobs/none/load", json={}).status_code == 404


class ut_CompareRouter:
    """spec-114: GET /jobs/{b}/compare?with={a}."""

    def test_compare_two_runs_with_each_photos_decision(self, client):
        _submit(client)
        _submit(client, job_id="trip2")
        client.post("/jobs/trip2/edits", json={"kind": "demote", "payload": {"photo_id": "a"}})
        r = client.get("/jobs/trip2/compare", params={"with": "trip"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert (out["a"], out["b"]) == ("trip", "trip2") and out["picks"]["missing"] == ["a"]
        assert out["decisions"]["trip"]["a"]["state"] == "pick"
        assert out["decisions"]["trip2"]["a"]["state"] != "pick"
        assert {c["name"] for c in out["checks"]} >= {"kept", "people", "person #1 faces"}

    def test_compare_refuses_unknown_unfinished_and_missing_reference(self, client):
        _submit(client)
        assert client.get("/jobs/trip/compare", params={"with": "nope"}).status_code == 404
        assert client.get("/jobs/trip/compare").status_code == 422          # `with` is required
        client.svc._factory = lambda *a: FailingRunner()
        _submit(client, job_id="bad")
        assert client.get("/jobs/bad/compare", params={"with": "trip"}).status_code == 409

    def test_tag_is_listed_and_a_dead_jobs_status_is_corrected(self, client):
        client.svc.repo.create("tagged", runner="local", tag="regression:quick")
        client.svc.repo.transition("tagged", "staging", worker="this-host-is-gone:1:1")
        client.svc.repo.create("orphan", runner="local")
        client.svc.repo.transition("orphan", "running")              # no worker recorded
        jobs = {j["id"]: j for j in client.get("/jobs").json()}
        assert jobs["tagged"]["tag"] == "regression:quick"
        assert jobs["orphan"]["status"] == "failed" and "interrupted" in jobs["orphan"]["error"]


class ut_WorkbenchRoutes:
    """spec-115 W4: estimate, cancel, re-run, settings, delete photos from Modal."""

    def test_estimate_for_l4_and_local(self, client):
        l4 = client.get("/jobs/estimate", params={"source": str(client.src), "runner": "modal_gpu"}).json()
        assert l4["n_photos"] == 6 and l4["allowed"] and l4["gpu_minutes"] > 0 and l4["dollars"] is not None
        local = client.get("/jobs/estimate", params={"source": str(client.src), "runner": "local"}).json()
        assert local["gpu_minutes"] is None
        assert client.get("/jobs/estimate", params={"source": "Z:/nope"}).status_code == 422

    def test_finished_job_cannot_be_cancelled(self, client):
        _submit(client)
        r = client.post("/jobs/trip/cancel")
        assert r.status_code == 409 and "already succeeded" in r.text
        assert client.post("/jobs/nope/cancel").status_code == 404

    def test_rerun_uses_the_same_photos_and_records_settings(self, client):
        _submit(client)
        r = client.post("/jobs", json={"rerun_of": "trip", "job_id": "trip-again", "runner": "local"})
        assert r.status_code == 202, r.text
        svc = client.svc
        assert svc.store.input_keys("trip-again") == svc.store.input_keys("trip")
        job = client.get("/jobs/trip-again").json()
        assert job["status"] == "succeeded" and job["source"] == str(client.src)
        assert any(e["message"] == "re-run of trip" for e in job["events"])
        assert job["settings"]["pipeline"] == "minimal" and job["settings"]["steps"]

    def test_negative_start_is_refused(self, client):
        """review L2."""
        assert client.post("/jobs", json={"source": str(client.src), "start": -3}).status_code == 422
        assert client.get("/jobs/estimate", params={"source": str(client.src), "start": -1}).status_code == 422

    def test_submit_needs_a_source_or_a_job(self, client):
        assert client.post("/jobs", json={}).status_code == 422
        assert client.post("/jobs", json={"rerun_of": "ghost"}).status_code == 422
