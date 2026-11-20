"""spec-116 R1 / R7: the pipeline API chooses the machine, refuses over the cap, cancels, and lists runs."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from sim_bench.api.database.models import Album, Base, PipelineRun
from sim_bench.api.database.session import get_session
from sim_bench.api.routers import jobs as jobs_router
from sim_bench.api.routers import pipeline as pipeline_router

from tests.api.test_pipeline_service_runners import STEPS, world  # noqa: F401  (the fixture)


@pytest.fixture
def client(world):
    app = FastAPI()
    app.include_router(pipeline_router.router)
    app.dependency_overrides[get_session] = lambda: world.session
    app.dependency_overrides[jobs_router.get_job_service] = lambda: world.jobs
    c = TestClient(app)
    c.world = world
    return c


def _start(client, **extra):
    return client.post("/api/v1/pipeline/run", json={"album_id": "alb", "steps": STEPS, **extra})


class ut_PipelineRouter:
    def test_the_default_is_this_pc(self, client):
        r = _start(client)
        assert r.status_code == 200 and r.json()["runner"] == "local"
        job_id = r.json()["job_id"]
        status = client.get(f"/api/v1/pipeline/{job_id}").json()
        assert status["status"] == "completed" and status["runner"] == "local"
        assert [s["step"] for s in status["completed_steps"]] == ["select_best"]

    def test_l4_estimate_comes_from_the_album(self, client):
        est = client.get("/api/v1/pipeline/estimate", params={"album_id": "alb", "runner": "modal_gpu"}).json()
        assert est["n_photos"] == 6 and est["gpu_minutes"] > 0 and est["dollars"] > 0 and est["allowed"]

    def test_l4_over_cap_is_refused(self, client):
        for i in range(100):
            Image.new("RGB", (8, 8)).save(client.world.src / f"x{i}.jpg")
        est = client.get("/api/v1/pipeline/estimate", params={"album_id": "alb", "runner": "modal_gpu"}).json()
        assert not est["allowed"]
        r = _start(client, runner="modal_gpu")
        assert r.status_code == 422 and "over the L4 limit" in r.json()["detail"]
        assert client.world.session.query(PipelineRun).count() == 0
        assert client.world.jobs.repo.list() == []

    def test_an_unknown_runner_is_refused(self, client):
        assert _start(client, runner="moon").status_code == 422
        assert client.get("/api/v1/pipeline/estimate", params={"album_id": "alb", "runner": "moon"}).status_code == 422

    def test_runs_route_is_not_taken_for_a_job_id(self, client):
        _start(client)
        rows = client.get("/api/v1/pipeline/runs").json()
        assert len(rows) == 1 and rows[0]["ran_on"] == "This PC"

    def test_cancel_a_finished_run_is_refused(self, client):
        job_id = _start(client).json()["job_id"]
        assert client.post(f"/api/v1/pipeline/{job_id}/cancel").status_code == 409

    def test_cancel_a_run_without_a_job(self, client):
        client.world.session.add(PipelineRun(id="old", album_id="alb", steps=[], status="running"))
        client.world.session.commit()
        assert client.post("/api/v1/pipeline/old/cancel").status_code == 404

    def test_a_bad_album_is_a_422(self, client):
        assert client.post("/api/v1/pipeline/run", json={"album_id": "nope", "steps": STEPS}).status_code == 422
