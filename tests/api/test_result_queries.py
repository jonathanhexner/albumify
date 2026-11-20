"""SIGHTING-152: list pages read summary columns in a fixed number of queries; finished results are read once.

Counted with a SQLAlchemy hook, so the regression shows on SQLite too, where every query is fast and nobody notices.
"""
import uuid
from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import Album, PipelineResult, PipelineRun, Person
from sim_bench.api.database.session import get_session
from sim_bench.api.main import app
from sim_bench.api.services.album_service import AlbumService
from sim_bench.api.services.result_service import ResultService, result_cache
from tests._db import fresh_api_engine


@contextmanager
def counted(engine):
    statements = []

    def _count(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", _count)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", _count)


def _album_with_runs(session, n_runs: int, album_id: str = "alb") -> list:
    session.add(Album(id=album_id, name="Trip", source_path="D:/trip"))
    session.flush()
    run_ids = []
    for i in range(n_runs):
        run_id = f"{album_id}-run{i}"
        session.add(PipelineRun(id=run_id, album_id=album_id, steps=[], status="completed"))
        session.flush()
        metrics = {f"D:/trip/{k}.jpg": {"composite_score": 0.5} for k in range(50)}
        session.add(PipelineResult(
            id=str(uuid.uuid4()), run_id=run_id, total_images=50, filtered_images=40, num_clusters=5,
            num_selected=10, image_metrics=metrics, selected_images=list(metrics)[:10], scene_clusters={"0": list(metrics)},
            step_decisions=[{"item_id": p, "step": "select_best", "decision": "selected"} for p in metrics]))
        session.add(Person(id=f"{run_id}-p0", album_id=album_id, run_id=run_id, person_index=0, face_instances=[]))
        run_ids.append(run_id)
    session.commit()
    return run_ids


def test_result_list_is_summary_only_in_fixed_queries(tmp_path):
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    _album_with_runs(session, 2)
    with counted(engine) as few:
        small = ResultService(session).list_results("alb")
    _album_with_runs(session, 6, album_id="alb2")
    with counted(engine) as many:
        big = ResultService(session).list_results("alb2")
    assert len(small) == 2 and len(big) == 6
    assert len(few) == len(many) <= 3, (len(few), len(many))
    row = big[0]
    assert row["num_selected"] == 10 and row["num_people"] == 1 and row["album_name"] == "Trip"
    assert "step_decisions" not in row and "image_metrics" not in row
    assert not any("image_metrics" in s or "step_decisions" in s for s in many), "list read per-photo columns"
    session.close()


def test_runs_table_query_count_does_not_grow_with_runs(tmp_path):
    from sim_bench.api.services.pipeline_service import PipelineService
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    _album_with_runs(session, 2)
    with counted(engine) as few:
        PipelineService(session).list_runs()
    _album_with_runs(session, 8, album_id="alb2")
    with counted(engine) as many:
        rows = PipelineService(session).list_runs()
    assert len(rows) == 10 and {r.album_name for r in rows} == {"Trip"}
    assert len(many) == len(few) == 1, (len(few), len(many))    # runs, albums, counts and people in one query
    assert {r.photos for r in rows} == {50} and {r.picks for r in rows} == {10} and {r.people for r in rows} == {1}
    session.close()


def test_a_finished_result_is_read_from_the_database_once(tmp_path):
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    run_id = _album_with_runs(session, 1)[0]
    service = ResultService(session)
    with counted(engine) as first:
        service.get_images(run_id, selected_only=True)
    with counted(engine) as again:
        images = service.get_images(run_id, selected_only=True)
        service.get_images(run_id)
        service.get_metrics(run_id)
        decisions = service.get_step_decisions(run_id)
    assert len(images) == 10 and len(decisions) == 50
    assert any("pipeline_results" in s for s in first)
    assert not any("pipeline_results" in s for s in again), again
    session.close()


def test_an_unfinished_run_is_not_cached(tmp_path):
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    session.add(Album(id="alb", name="Trip", source_path="D:/trip"))
    session.add(PipelineRun(id="running", album_id="alb", steps=[], status="running"))
    session.commit()
    assert ResultService(session).get_images("running") == []
    assert result_cache.get("running") is None
    session.close()


def test_deleting_the_album_evicts_its_results(tmp_path):
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    run_id = _album_with_runs(session, 1)[0]
    ResultService(session).get_images(run_id)
    assert result_cache.get(run_id) is not None
    assert AlbumService(session).delete("alb")
    assert result_cache.get(run_id) is None
    assert ResultService(session).get_images(run_id) == []
    session.close()


def test_decisions_route(tmp_path):
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    run_id = _album_with_runs(session, 1)[0]
    app.dependency_overrides[get_session] = lambda: session
    try:
        client = TestClient(app)
        resp = client.get(f"/api/v1/results/{run_id}/decisions")
        assert resp.status_code == 200 and len(resp.json()) == 50
        assert client.get("/api/v1/results/nope/decisions").status_code == 404
        listed = client.get("/api/v1/results/", params={"album_id": "alb"}).json()
        assert listed[0]["job_id"] == run_id and listed[0].get("step_decisions") is None
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_images_and_clusters_carry_the_runs_768px_copy(tmp_path, monkeypatch):
    """C5: thumbnails come from the staged copy the run already has, not from the full-size original."""
    from PIL import Image
    from sim_bench.api.services import run_display_copies
    from sim_bench.serving.job_store import JobStore
    from sim_bench.serving.serving_config import ServingConfig

    jobs = tmp_path / "jobs"
    monkeypatch.setattr(run_display_copies, "load_serving_config", lambda: ServingConfig(jobs_root=jobs))
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    run_id = _album_with_runs(session, 1)[0]
    store = JobStore(jobs)
    key = "0123456789abcdef/0.jpg"
    copy = store.inputs.path(key)
    copy.parent.mkdir(parents=True)
    Image.new("RGB", (768, 512)).save(copy)
    # the manifest spells the original with backslashes, the result with forward slashes: same photo
    store.write_input_manifest(run_id, [key], originals={key: r"D:\trip\0.jpg"})

    images = {i["path"]: i for i in ResultService(session).get_images(run_id)}
    assert images["D:/trip/0.jpg"]["display_path"] == str(copy)
    assert images["D:/trip/1.jpg"]["display_path"] is None        # not staged: the page falls back to the original
    clustered = {i["path"]: i for c in ResultService(session).get_clusters(run_id) for i in c["images"]}
    assert clustered["D:/trip/0.jpg"]["display_path"] == str(copy)
    session.close()


def test_people_list_is_one_query_without_face_instances(tmp_path):
    from sim_bench.api.services.people_service import PeopleService
    engine = fresh_api_engine(tmp_path)
    session = sessionmaker(bind=engine)()
    run_id = _album_with_runs(session, 2)[-1]
    session.query(PipelineRun).update({"completed_at": PipelineRun.created_at})
    session.commit()
    session.expunge_all()
    with counted(engine) as latest:
        people = PeopleService(session).list_people("alb")
    with counted(engine) as picked:
        PeopleService(session).list_people("alb", run_id=run_id)
    assert len(people) == 1 and len(latest) == 1 and len(picked) == 1, (latest, picked)
    assert not any("face_instances" in s for s in latest + picked)
    session.close()
