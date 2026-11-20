"""spec-085 §5 follow-up: HTTP round-trip through the FastAPI response_model.

The unit tests hit the service functions directly. This hits the actual route
(`GET /api/v1/results/{job}/images`) so the `response_model=list[ImageMetrics]`
coercion runs for real — proving the previously-dropped fields now survive all
the way to the JSON the browser receives.
"""

import uuid

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests._db import fresh_api_engine
from sim_bench.api.main import app
from sim_bench.api.database.models import Album, Base, PipelineResult, PipelineRun
from sim_bench.api.database.session import get_session


def test_images_endpoint_exposes_previously_dropped_fields(tmp_path):
    # a file (or the parity run's Postgres schema), so the route's worker thread sees the same tables
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    # the run the result belongs to: Postgres enforces pipeline_results.run_id -> pipeline_runs.id
    session.add(Album(id="alb", name="a", source_path="D:/x"))
    session.add(PipelineRun(id="job1", album_id="alb", steps=[], status="completed"))
    session.flush()
    session.add(PipelineResult(
        id=str(uuid.uuid4()),
        run_id="job1",
        image_metrics={"a.jpg": {
            "person_detected": True,
            "best_frontal_score": 0.8,
            "roll_angles": [3.2],
            "filter_reason": "Best in cluster (0.8)",
            "quality_score": 0.8,
            "person_penalty": -0.15,
        }},
        selected_images=[],
        scene_clusters={},
    ))
    session.commit()

    app.dependency_overrides[get_session] = lambda: session
    try:
        resp = TestClient(app).get("/api/v1/results/job1/images")
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        assert len(rows) == 1
        img = rows[0]
        # These all used to be dropped by the response_model -> blank columns.
        assert img["person_detected"] is True
        assert img["best_frontal_score"] == 0.8
        assert img["roll_angles"] == [3.2]
        assert img["filter_reason"] == "Best in cluster (0.8)"
        assert img["quality_score"] == 0.8
        assert img["person_penalty"] == -0.15
    finally:
        app.dependency_overrides.clear()
        session.close()
