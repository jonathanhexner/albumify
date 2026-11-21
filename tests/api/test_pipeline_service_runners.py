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

from tests._db import fresh_api_engine, fresh_database_url
from sim_bench.api.database.models import (Album, Base, FaceMetricRow, ImageMetricRow, Person, PipelineResult,
                                           PipelineRun)
from sim_bench.api.services.people_service import PeopleService
from sim_bench.api.services.pipeline_service import PipelineService
from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import this_worker
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
from sim_bench.serving.job_service import JobService
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder
from sim_bench.serving.run_output import RunOutput
from sim_bench.serving.runners import CANCELLED, ModalRunner, RunOutcome
from sim_bench.serving.serving_config import ServingConfig
from sim_bench.serving.worker_persister import WorkerPersister

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
    """The Modal side of ``ModalRunner``, never touching the network. ``start`` is a detached call: the L4 worker
    runs in its own job store (the inputs Volume holds the uploaded photos, no originals), sees /inputs/... paths,
    and writes its rows and files itself (spec-117 W4/W5); nothing comes back to the PC's job folder."""

    def __init__(self, world, fail=None):
        self.w, self.fail, self.uploaded, self.spec, self.cancelled = world, fail, [], None, []
        self.present = set()

    def missing(self, keys):
        if self.fail == "unreachable":
            raise ConnectionError("Modal is unreachable")
        return [k for k in keys if k not in self.present]

    def upload(self, keys, store):
        self.uploaded.append(list(keys))
        self.present |= set(keys)

    def start(self, job_id, keys, spec, manifest, pipeline):
        self.spec = spec
        if self.fail == "cancel":                          # the owner cancels while the call is queued
            self.w.svc.cancel(job_id)
            return f"call-{job_id}"
        pc = self.w.jobs.store
        l4 = JobStore(self.w.tmp / "l4", inputs_root=pc.inputs.root, objects=pc.objects)   # objects = "R2"
        l4.write_input_manifest(job_id, keys, manifest.get("originals"), manifest.get("sizes"))
        if self.fail == "no output":
            l4.write_run_output = lambda *_: None
        images = [Path("/inputs") / p.parent.name / p.name for p in l4.input_images(job_id)]
        runner = fake_pipeline
        if self.fail == "step":
            from types import SimpleNamespace
            runner = lambda *a, **k: (None, SimpleNamespace(success=False, failed_step="score_ava",  # noqa: E731
                                                            error_message="CUDA out of memory",
                                                            step_results=[], total_duration_ms=1))
        db = self.w.jobs.db
        JobCurator(l4, spec_builder=SpecBuilder(), runner=runner, output_builder=_output_for(images), db=db,
                   persister=WorkerPersister(db, l4, api_sessions=self.w.sessions)).calc(
            CurateJobInputs(job_id=job_id, pipeline="minimal", run_spec=spec, runner="modal_gpu"))
        return f"call-{job_id}"

    def status(self, call_id):
        return "done"

    def cancel(self, call_id):
        self.cancelled.append(call_id)


@pytest.fixture
def world(tmp_path):
    engine = fresh_api_engine(tmp_path)          # spec-117 R3: Postgres in the needs_neon parity run
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
    w.session, w.src, w.modal, w.tmp = session, src, None, tmp_path
    w.sessions = sessionmaker(bind=engine)          # the L4 worker's own sessions on the API tables

    def factory(name, store, pipeline, device):
        if name == "local":
            return LocalFake(store)
        return ModalRunner(store, pipeline=pipeline, remote=w.modal, db=w.jobs.db, poll_s=0.01)

    w.jobs = JobService(ControlDB(fresh_database_url(tmp_path, "c.db")),
                        ServingConfig(jobs_root=tmp_path / "jobs", max_photos_l4=100), runner_factory=factory)
    w.modal = FakeModal(w)
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

    def test_an_api_restart_mid_run_loses_nothing(self, world):
        """spec-117 R8: the API that started an L4 run dies; the worker finishes with nobody watching; a new API
        process finds the run completed with its rows."""
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        store, repo = world.jobs.store, world.jobs.repo
        world.jobs._stage(run_id, world.src, 0, None)
        store.write_run_spec(run_id, world.modal.spec or {"steps": STEPS, "step_configs": {}})
        repo.transition(run_id, "staging", worker="the-old-api:1:1")
        repo.transition(run_id, "running")
        repo.set_progress(run_id, {"stage": "waiting for gpu"}, heartbeat=False, call_id=f"call-{run_id}")
        # the old API process is gone; the L4 worker runs to the end on its own
        world.modal.start(run_id, store.input_keys(run_id), store.read_run_spec(run_id), store.manifest(run_id),
                          "minimal")
        # a new API process: new services over the same databases
        fresh = PipelineService(world.sessions(), job_service=JobService(world.jobs.db, world.jobs.config,
                                                                         store=store))
        fresh._service._detached = world.modal
        assert fresh._service.repo.reap_orphans(alive=lambda w: False) == []
        run = fresh.get_status(run_id)
        assert run.status == "completed" and fresh._service.repo.get(run_id).status == "succeeded"
        assert {r.image_path for r in fresh._session.query(ImageMetricRow).all()} == \
            {str(world.src / "a.jpg"), str(world.src / "b.jpg")}

    def test_nothing_is_copied_back_from_the_l4(self, world):
        """spec-117 R6: no album.json, run.json or results/ come back; the rows came from the worker. (display/ and
        faces/ here ARE the object store: the fake's "R2" is the PC's job folder.)"""
        run_id = _run(world, "modal_gpu")
        root = world.jobs.store.paths(run_id).root
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        assert [f for f in files if not f.startswith(("display/", "faces/"))] == \
            ["input/manifest.json", "run_spec.json"]
        assert world.svc.get_status(run_id).status == "completed"
        assert world.jobs.repo.get(run_id).call_id == f"call-{run_id}"

    @pytest.mark.parametrize("runner", ["local", "modal_gpu"])
    def test_in_step_progress_reads_the_same_on_both_machines(self, world, runner):
        """spec-117 R5: the job row is the only source; "triage_content 43% (307/715)" on either machine."""
        from app.streamlit.pages.configure import step_detail
        from app.streamlit.api_client import PipelineProgress, PipelineStatus as UiStatus

        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner=runner)
        world.jobs.repo.transition(run_id, "running", worker=this_worker())
        world.jobs.repo.set_progress(run_id, {"stage": "running", "current_step": "triage_content",
                                              "step_fraction": 0.43, "photos_done": 307, "photos_total": 715,
                                              "completed_steps": [{"step": "discover_images"}]})
        live = world.svc.live(world.svc.get_status(run_id))
        assert (live["runner"], live["stage"], live["current_step"]) == (runner, "running", "triage_content")
        assert (live["step_fraction"], live["photos_done"], live["photos_total"]) == (0.43, 307, 715)
        assert live["heartbeat_at"] is not None
        ui = PipelineProgress(status=UiStatus.RUNNING, step_fraction=live["step_fraction"],
                              photos_done=live["photos_done"], photos_total=live["photos_total"])
        assert step_detail(ui) == "43% (307/715)"
        assert not list(world.jobs.store.root.rglob("progress.json"))

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
        """The owner cancels while the L4 call is queued: the call is cancelled by id, the run ends failed."""
        world.modal.fail = "cancel"
        run_id = _run(world, "modal_gpu")
        self._assert_failed(world, run_id, CANCELLED)
        assert world.modal.cancelled == [f"call-{run_id}"]

    def test_no_run_output_is_a_failure(self, world):
        world.modal.fail = "no output"
        self._assert_failed(world, _run(world, "modal_gpu"), "run_output.json")

    def test_a_write_that_fails_leaves_no_rows(self, world, monkeypatch):
        monkeypatch.setattr(PeopleService, "add_from_payload",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("disk full")))
        self._assert_failed(world, _run(world), "disk full")

    def test_a_process_that_died_ends_failed(self, world):
        """The API was killed mid-run: the next status read ends it instead of showing running forever."""
        run_id = world.svc.start_pipeline("alb", steps=STEPS)
        world.jobs.repo.transition(run_id, "staging", worker="otherhost:1:1")
        world.jobs.repo.transition(run_id, "running", worker=this_worker())
        world.jobs.repo.reap_orphans(alive=lambda w: False)
        world.session.query(PipelineRun).filter_by(id=run_id).one().status = "running"
        world.session.commit()
        self._assert_failed(world, run_id, "interrupted")

    def _ghost(self, world):
        """A GPU run whose job the background watchdog failed (Modal timeout) while its run row says running."""
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        world.jobs.repo.transition(run_id, "running")
        world.jobs.repo.fail(run_id, "the L4 worker ended without saving its results (failed: FunctionTimeoutError)")
        world.session.query(PipelineRun).filter_by(id=run_id).one().status = "running"
        world.session.commit()
        return run_id

    def test_a_run_failed_in_the_background_shows_failed_in_albums(self, world):
        """SIGHTING-157: no page asked for its status, so Albums showed it running forever."""
        run_id = self._ghost(world)
        assert world.svc.fail_runs_whose_job_failed() == 1
        row = next(r for r in world.svc.list_runs() if r.run_id == run_id)
        run = world.session.query(PipelineRun).filter_by(id=run_id).one()
        assert row.status == "failed" and "FunctionTimeoutError" in run.error_message and run.completed_at
        assert world.svc.fail_runs_whose_job_failed() == 0          # once

    def test_the_watchdog_tick_ends_the_run_too(self, world):
        from sim_bench.api.main import watchdog_tick
        run_id = self._ghost(world)
        assert watchdog_tick(job_service=world.jobs, session_factory=world.sessions) == 1
        world.session.expire_all()
        assert world.session.query(PipelineRun).filter_by(id=run_id).one().status == "failed"

    def test_a_live_run_is_left_alone(self, world):
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        world.jobs.repo.transition(run_id, "running")
        world.session.query(PipelineRun).filter_by(id=run_id).one().status = "running"
        world.session.commit()
        assert world.svc.fail_runs_whose_job_failed() == 0
        assert world.session.query(PipelineRun).filter_by(id=run_id).one().status == "running"

    def test_deleting_a_run_removes_its_job_files_and_every_row(self, world):
        """SIGHTING-158: deleting the job left the run (and its rows) in Albums."""
        run_id = _run(world)
        assert world.session.query(PipelineResult).filter_by(run_id=run_id).count() == 1
        out = world.svc.delete_run(run_id)
        assert out["deleted_run"] and world.jobs.repo.get(run_id) is None
        assert run_id not in [r.run_id for r in world.svc.list_runs()]
        assert world.session.query(PipelineRun).filter_by(id=run_id).count() == 0
        for model in (PipelineResult, Person, ImageMetricRow, FaceMetricRow):
            assert world.session.query(model).filter_by(run_id=run_id).count() == 0, model.__name__

    def test_a_ghost_run_whose_job_is_gone_can_be_deleted(self, world):
        run_id = self._ghost(world)
        world.jobs.delete_job(run_id)                     # what the old /jobs delete did: the run stayed
        assert run_id in [r.run_id for r in world.svc.list_runs()]
        assert world.svc.delete_run(run_id)["deleted_files"] == 0
        assert run_id not in [r.run_id for r in world.svc.list_runs()]

    def test_a_live_run_is_not_deleted(self, world):
        run_id = world.svc.start_pipeline("alb", steps=STEPS, runner="modal_gpu")
        world.jobs.repo.transition(run_id, "running")
        with pytest.raises(ValueError):
            world.svc.delete_run(run_id)
        assert world.session.query(PipelineRun).filter_by(id=run_id).count() == 1
        with pytest.raises(KeyError):
            world.svc.delete_run("no-such-run")

    def test_a_bad_spec_leaves_nothing_behind(self, world):
        with pytest.raises(ValueError, match="Pipeline spec invalid"):
            world.svc.start_pipeline("alb", steps=["not_a_step"])
        assert world.session.query(PipelineRun).count() == 0 and world.jobs.repo.list() == []
