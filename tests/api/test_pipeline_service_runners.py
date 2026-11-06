"""spec-116 W2: every album run goes through the job service, on this PC or on a faked Modal."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sim_bench.api.database.models import (Album, Base, FaceMetricRow, ImageMetricRow, Person, PipelineResult,
                                           PipelineRun)
from sim_bench.api.services.people_service import PeopleService
from sim_bench.api.services.pipeline_service import PipelineService
from sim_bench.control.db import ControlDB
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
from sim_bench.serving.job_service import JobService
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder
from sim_bench.serving.run_output import RunOutput
from sim_bench.serving.runners import CANCELLED, ModalRunner, RunOutcome
from sim_bench.serving.serving_config import ServingConfig

from tests.control.fakes import NAMES, fake_pipeline
from tests.serving.test_run_output import RESULT, finished_context

STEPS = ["discover_images", "select_best"]


def _output_for(images):
    """A real RunOutput over the two first staged photos, whatever root they sit under."""
    return lambda context, result: RunOutput.from_context(finished_context(str(images[0]), str(images[1])), RESULT)


def _curate(store, job_id, spec, images_root=None):
    images = store.input_images(job_id)
    if images_root:      # paths as the L4 sees them
        images = [Path(images_root) / p.parent.name / p.name for p in images]
    return JobCurator(store, spec_builder=SpecBuilder(), runner=fake_pipeline, output_builder=_output_for(images)).calc(
        CurateJobInputs(job_id=job_id, pipeline="minimal", run_spec=spec))


class LocalFake:
    """Stands in for the child process: the real JobCurator with the fake pipeline."""

    def __init__(self, store, name="local"):
        self.store, self.name = store, name

    def run(self, job_id):
        res = _curate(self.store, job_id, self.store.read_run_spec(job_id))
        return RunOutcome(ok=res.ok, error=res.error, failed_step=res.failed_step, wall_s=1.0)

    def cancel(self, job_id):
        pass


class FakeModal:
    """The Modal side of ``ModalRunner``: never touches the network."""

    def __init__(self, store, fail=None):
        self.store, self.fail, self.uploaded, self.spec = store, fail, [], None
        self.present = set()

    def missing(self, keys):
        if self.fail == "unreachable":
            raise ConnectionError("Modal is unreachable")
        return [k for k in keys if k not in self.present]

    def upload(self, keys, store):
        self.uploaded.append(list(keys))
        self.present |= set(keys)

    def run_job(self, job_id, keys, pipeline, spec=None):
        self.spec = spec
        if self.fail == "step":
            self.store.write_run_record(job_id, {"ok": False, "error": "CUDA out of memory", "failed_step": "score_ava"})
            return {"ok": False}
        _curate(self.store, job_id, spec, images_root="/inputs")
        return {"ok": True}

    def download_results(self, job_id, store):
        return 3

    def cancel(self):
        pass


@pytest.fixture
def world(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    src = tmp_path / "trip"
    src.mkdir()
    for n in NAMES:
        Image.new("RGB", (1536, 1024), "gray").save(src / n)
    session.add(Album(id="alb", name="Trip", source_path=str(src)))
    session.commit()

    class World:
        pass

    w = World()
    w.session, w.src, w.modal = session, src, None

    def factory(name, store, pipeline, device):
        if name == "local":
            return LocalFake(store)
        return ModalRunner(store, pipeline=pipeline, remote=w.modal)

    w.jobs = JobService(ControlDB(f"sqlite:///{(tmp_path / 'c.db').as_posix()}"),
                        ServingConfig(jobs_root=tmp_path / "jobs", max_photos_l4=100), runner_factory=factory)
    w.modal = FakeModal(w.jobs.store)
    w.svc = PipelineService(session, job_service=w.jobs)
    return w


def _run(w, runner="local", config=None):
    run_id = w.svc.start_pipeline("alb", steps=STEPS, step_configs=config or {}, runner=runner)
    w.svc.execute_pipeline(run_id)
    return run_id


def _no_rows(s) -> bool:
    return not any(s.query(m).count() for m in (PipelineResult, Person, ImageMetricRow, FaceMetricRow))


class ut_EitherMachine:
    def test_run_is_a_job(self, world):
        run_id = _run(world)
        run, job = world.svc.get_status(run_id), world.jobs.repo.get(run_id)
        assert run.status == "completed" and run.progress == 1.0
        assert job.status == "succeeded" and job.runner == "local" and job.source == str(world.src)
        assert world.jobs.run_settings(run_id)["steps"][0]["step"] == "discover_images"

    @pytest.mark.parametrize("runner", ["local", "modal_gpu"])
    def test_rows_name_the_originals_on_both_machines(self, world, runner):
        run_id = _run(world, runner)
        assert world.svc.get_status(run_id).status == "completed"
        s = world.session
        paths = {r.image_path for r in s.query(ImageMetricRow).all()}
        assert paths == {str(world.src / "a.jpg"), str(world.src / "b.jpg")}
        assert s.query(Person).one().face_instances[0]["image_path"] == str(world.src / "a.jpg")
        blob = json.dumps(s.query(PipelineResult).one().image_metrics)
        assert "_inputs" not in blob and "/inputs/" not in blob

    def test_pixel_boxes_land_in_original_pixels(self, world):
        _run(world, "modal_gpu")
        face = world.session.query(FaceMetricRow).one()
        assert face.bbox_x_px == pytest.approx(20.0) and face.bbox_x == 0.1      # 1536 px staged at 768

    def test_the_l4_gets_the_spec_and_a_rerun_uploads_nothing(self, world):
        _run(world, "modal_gpu", {"select_best": {"max_images_per_cluster": 3}})
        assert world.modal.spec["step_configs"]["select_best"] == {"max_images_per_cluster": 3}
        _run(world, "modal_gpu")
        assert [len(u) for u in world.modal.uploaded] == [len(NAMES), 0]

    def test_an_l4_run_keeps_no_worker_export_folder(self, world):
        run_id = _run(world, "modal_gpu")
        assert world.session.query(PipelineResult).filter_by(run_id=run_id).one().fc_export_dir is None

    def test_status_says_where_it_ran(self, world):
        run_id = _run(world, "modal_gpu")
        assert world.svc.live(world.svc.get_status(run_id)) == {"runner": "modal_gpu"}

    def test_stage_while_it_runs(self, world):
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        seen = []
        original = world.modal.upload

        def upload(keys, store):
            seen.append(world.svc.live(world.svc.get_status(run_id))["stage"])
            original(keys, store)

        world.modal.upload = upload
        world.svc.execute_pipeline(run_id)
        assert seen == ["uploading"]

    def test_an_old_run_has_no_job_and_still_reads(self, world):
        world.session.add(PipelineRun(id="old", album_id="alb", steps=["discover_images"], status="completed"))
        world.session.commit()
        run = world.svc.get_status("old")
        assert run.status == "completed" and world.svc.live(run) == {}
        assert [r.ran_on for r in world.svc.list_runs()] == ["This PC"]

    def test_the_albums_table_lists_both_machines(self, world):
        run_id = _run(world, "modal_gpu")
        row = world.svc.list_runs()[0]
        assert (row.run_id, row.ran_on, row.status, row.comparable) == (run_id, "L4 (GPU)", "completed", True)
        assert (row.album_name, row.folder, row.photos, row.picks, row.people) == \
            ("Trip", str(world.src), 2, 1, 1)


class ut_FailureStates:
    def _assert_failed(self, world, run_id, text):
        run = world.svc.get_status(run_id)
        assert run.status == "failed" and text in run.error_message and run.completed_at
        assert world.jobs.repo.get(run_id).status == "failed"
        assert _no_rows(world.session)

    def test_a_step_error(self, world):
        world.modal.fail = "step"
        self._assert_failed(world, _run(world, "modal_gpu"), "score_ava: CUDA out of memory")

    def test_modal_unreachable(self, world):
        world.modal.fail = "unreachable"
        self._assert_failed(world, _run(world, "modal_gpu"), "Modal is unreachable")

    def test_cancel_before_it_starts(self, world):
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        world.svc.cancel(run_id)
        world.svc.execute_pipeline(run_id)
        self._assert_failed(world, run_id, CANCELLED)

    def test_cancel_while_it_runs(self, world):
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        original = world.modal.run_job

        def run_job(*args, **kwargs):
            world.svc.cancel(run_id)
            return original(*args, **kwargs)

        world.modal.run_job = run_job
        world.svc.execute_pipeline(run_id)
        self._assert_failed(world, run_id, CANCELLED)

    def test_no_run_output_is_a_failure(self, world):
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        world.modal.run_job = lambda *a, **k: (_curate(world.jobs.store, run_id, None),
                                               world.jobs.store.paths(run_id).run_output_path.unlink(), {"ok": True})[-1]
        world.svc.execute_pipeline(run_id)
        self._assert_failed(world, run_id, "run_output.json")

    def test_a_write_that_fails_leaves_no_rows(self, world, monkeypatch):
        monkeypatch.setattr(PeopleService, "add_from_payload",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("disk full")))
        self._assert_failed(world, _run(world), "disk full")

    def test_a_process_that_died_ends_failed(self, world):
        """The API was killed mid-run: the next status read ends it instead of showing running forever."""
        run_id = world.svc.start_pipeline("alb", steps=STEPS)
        world.jobs.repo.transition(run_id, "staging", worker="otherhost:1:1")
        world.jobs.repo.transition(run_id, "running")
        world.jobs.repo.reap_orphans(alive=lambda w: False)
        world.session.query(PipelineRun).filter_by(id=run_id).one().status = "running"
        world.session.commit()
        self._assert_failed(world, run_id, "interrupted")

    def test_a_bad_spec_leaves_nothing_behind(self, world):
        with pytest.raises(ValueError, match="Pipeline spec invalid"):
            world.svc.start_pipeline("alb", steps=["not_a_step"])
        assert world.session.query(PipelineRun).count() == 0 and world.jobs.repo.list() == []
