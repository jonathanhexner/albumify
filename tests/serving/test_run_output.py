"""spec-116 R3 + R4: one output file, one writer, the same rows in-process and read back from JSON."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from face_cluster.types import FaceRecord
from sim_bench.api.database.models import (Album, Base, FaceMetricRow, ImageMetricRow, Person, PipelineResult,
                                           PipelineRun)
from sim_bench.api.services.run_persister import RunPersister
from sim_bench.pipeline.context import PipelineContext
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.run_output import RunOutput

A, B = "/inputs/00000000000000aa/a.jpg", "/inputs/00000000000000bb/b.jpg"


def finished_context(a: str = A, b: str = B) -> PipelineContext:
    ctx = PipelineContext()
    ctx.image_paths = [Path(a), Path(b)]
    a, b = str(ctx.image_paths[0]), str(ctx.image_paths[1])
    ctx.active_images = {a}
    ctx.scene_clusters = {0: [a], 1: [b]}
    ctx.scene_cluster_labels = {a: 0, b: 1}
    ctx.selected_images = [a]
    ctx.iqa_scores = {a: np.float32(0.5)}
    ctx.composite_scores = {a: 0.8, b: 0.2}
    bbox = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "x_px": 10.0, "y_px": 20.0, "w_px": 30.0, "h_px": 40.0}
    ctx.insightface_faces = {a.replace("\\", "/"): {"faces": [
        {"face_index": 0, "bbox": bbox, "filter_passed": True, "frontal_score": 0.9,
         "filter_scores": {"confidence": 0.9}}]}}
    ctx.people_clusters = {0: [FaceRecord(face_id=0, image_id="a", bbox=(10, 20, 40, 60), image_path=a,
                                          face_index=0)]}
    ctx.step_decisions = [
        SimpleNamespace(item_id=b, item_type="image", step="filter_quality", decision="rejected",
                        reason="IQA too low", config_used={"min": 0.2}, metrics={"iqa": 0.1}),
        SimpleNamespace(item_id=a, item_type="image", step="select_best", decision="selected",
                        reason="Best in cluster", config_used={}, metrics={}),
    ]
    ctx.siamese_comparisons = [{"img1": a, "img2": b}]
    return ctx


RESULT = SimpleNamespace(success=True, total_duration_ms=1500,
                         step_results=[SimpleNamespace(step_name="select_best", duration_ms=7)])


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    s.add(Album(id="alb", name="Trip", source_path="D:/Trip"))
    s.add(PipelineRun(id="run", album_id="alb", steps=[], status="running"))
    s.commit()
    return s


def _rows(session) -> dict:
    """Every column but the generated ids, so two separate writes can be compared."""
    out = {}
    for model in (PipelineResult, Person, ImageMetricRow, FaceMetricRow):
        cols = [c.key for c in inspect(model).mapper.column_attrs if c.key not in ("id", "person_id", "created_at")]
        out[model.__tablename__] = sorted(
            ({c: getattr(row, c) for c in cols} for row in session.query(model).all()), key=repr)
    return out


class ut_RoundTrip:
    def test_round_trip_writes_the_same_rows(self):
        direct = RunOutput.from_context(finished_context(), RESULT)
        back = RunOutput.from_json(direct.to_json())
        s1, s2 = _session(), _session()
        RunPersister(s1).persist("alb", "run", direct)
        RunPersister(s2).persist("alb", "run", back)
        s1.commit(), s2.commit()
        assert _rows(s1) == _rows(s2)
        assert s1.query(FaceMetricRow).one().person_id == s1.query(Person).one().id

    def test_rows_carry_what_the_pages_read(self):
        s = _session()
        RunPersister(s).persist("alb", "run", RunOutput.from_context(finished_context(), RESULT))
        s.commit()
        result = s.query(PipelineResult).one()
        assert (result.total_images, result.filtered_images, result.num_clusters, result.num_selected) == (2, 1, 2, 1)
        assert result.step_timings == {"select_best": 7} and result.total_duration_ms == 1500
        rows = {Path(r.image_path).stem: r for r in s.query(ImageMetricRow).all()}
        assert rows["a"].is_selected and rows["a"].filter_reason == "Best in cluster"
        assert rows["b"].filter_reason == "IQA too low" and not rows["b"].is_selected
        person = s.query(Person).one()
        assert (person.face_count, person.image_count, person.person_index) == (1, 1, 0)

    def test_a_failed_write_leaves_no_rows(self):
        s = _session()
        output = RunOutput.from_context(finished_context(), RESULT)
        output.people = [{"person_index": 0}]              # malformed: the person cannot be written
        with pytest.raises(KeyError):
            RunPersister(s).persist("alb", "run", output)
        s.commit()
        assert s.query(PipelineResult).count() == 0 and s.query(ImageMetricRow).count() == 0

    def test_a_straightened_pick_is_named_by_its_input_photo(self):
        """The derived file stays on the worker (and the old code lost is_selected for such picks)."""
        ctx = finished_context()
        a = str(ctx.image_paths[0])
        ctx.selected_images = ["/root/.sim_bench/straightened/a_123.jpg"]
        ctx.straightened_from = {ctx.selected_images[0]: a}
        out = RunOutput.from_context(ctx, RESULT)
        assert out.selected_images == [a] and out.image_metrics[a]["is_selected"] is True


class ut_Worker:
    def test_local_run_reads_staged_copies(self, tmp_path):
        """The pipeline gets the shared store's 768 px copies and returns results/run_output.json."""
        store = JobStore(tmp_path)
        keys = ["00000000000000aa/a.jpg", "00000000000000bb/b.jpg"]
        for k in keys:
            store.inputs.put_bytes(k, b"jpeg")
        store.write_input_manifest("j1", keys, {k: f"D:/Trip/{Path(k).name}" for k in keys})
        seen = []

        def runner(images, plan, **_):
            seen.extend(images)
            return finished_context(str(images[0]), str(images[1])), RESULT

        res = JobCurator(store, runner=runner).calc(CurateJobInputs(
            job_id="j1", run_spec={"steps": ["discover_images"], "step_configs": {}}))
        assert res.ok, res.error
        assert seen == [store.inputs.path(k) for k in keys]
        assert "D:/Trip" not in str(seen)
        output = RunOutput.from_json(store.read_run_output("j1"))
        assert set(output.image_metrics) == {str(p) for p in seen}
        assert "results/run_output.json" in store.results_files("j1")

    def test_a_failed_run_returns_no_output(self, tmp_path):
        store = JobStore(tmp_path)
        store.inputs.put_bytes("00000000000000aa/a.jpg", b"x")
        store.write_input_manifest("j1", ["00000000000000aa/a.jpg"])

        def failing(images, plan, **_):
            return finished_context(), SimpleNamespace(success=False, failed_step="score_ava", error_message="boom",
                                                       total_duration_ms=1, step_results=[])

        res = JobCurator(store, runner=failing).calc(CurateJobInputs(job_id="j1"))
        assert not res.ok and store.read_run_output("j1") is None

    def test_progress_file_follows_the_steps(self, tmp_path):
        store = JobStore(tmp_path)
        store.inputs.put_bytes("00000000000000aa/a.jpg", b"x")
        store.write_input_manifest("j1", ["00000000000000aa/a.jpg"])
        snapshots = []

        def runner(images, plan, on_step, **_):
            snapshots.append(store.read_progress("j1"))
            on_step(SimpleNamespace(step_name="discover_images", duration_ms=5, success=True, error_message=None))
            snapshots.append(store.read_progress("j1"))
            return finished_context(), RESULT

        JobCurator(store, runner=runner).calc(CurateJobInputs(job_id="j1", run_spec={
            "steps": ["discover_images", "select_best"], "step_configs": {}}))
        assert snapshots[0] == {"stage": "running", "current_step": "discover_images", "completed_steps": []}
        assert snapshots[1]["current_step"] == "select_best"
        assert snapshots[1]["completed_steps"][0]["step"] == "discover_images"
