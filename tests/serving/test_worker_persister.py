"""spec-117 R6: the worker writes the results itself -- rows with ORIGINAL photo paths, files in the object store --
on this PC and on the L4 alike; nothing is copied back.

The "L4" here is a worker job store under its own root whose originals are not readable (deleted after staging),
with display images, crops and result files going to a fake R2 bucket (moto).
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from PIL import Image
from sqlalchemy.orm import sessionmaker

from sim_bench.api.database.models import (Album, FaceMetricRow, ImageMetricRow, Person, PipelineResult,
                                           PipelineRun)
from sim_bench.api.services.people_service import PeopleService
from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import FAILED, RUNNING, STAGING, SUCCEEDED, JobRepo
from sim_bench.control.models import Face, Photo
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder
from sim_bench.serving.run_output import RunOutput
from sim_bench.serving.staging import downsample
from sim_bench.serving.worker_persister import WorkerPersister
from sim_bench.storage.object_store import LocalObjectStore, R2ObjectStore

from tests._db import fresh_api_engine, fresh_database_url
from tests.control.fakes import NAMES, fake_pipeline
from tests.serving.test_run_output import RESULT, finished_context

RUN = "trip-run"


@pytest.fixture(params=["pc", "l4"])
def world(request, tmp_path):
    """A staged job and its album run; ``l4``: worker store elsewhere, originals gone, files to fake R2."""
    l4 = request.param == "l4"
    engine = fresh_api_engine(tmp_path)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    src = tmp_path / "trip"
    src.mkdir()
    for n in NAMES:
        Image.new("RGB", (1536, 1024), "gray").save(src / n)
    with sessions() as s:
        s.add(Album(id="alb", name="Trip", source_path=str(src)))
        s.add(PipelineRun(id=RUN, album_id="alb", steps=["discover_images"], status="running"))
        s.commit()

    objects = LocalObjectStore(tmp_path / "jobs")
    if l4:
        aws = mock_aws()
        aws.start()
        request.addfinalizer(aws.stop)
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="albumify-test")
        objects = R2ObjectStore("albumify-test", s3)
    store = JobStore(tmp_path / ("worker" if l4 else "jobs"), objects=objects)
    keys, originals, sizes = [], {}, {}
    for n in NAMES:
        key = store.inputs.key_for(src / n)
        store.inputs.put_bytes(key, downsample(src / n))
        keys.append(key)
        originals[key], sizes[key] = str(src / n), [1536, 1024]
    store.write_input_manifest(RUN, keys, originals, sizes)
    if l4:
        for n in NAMES:
            (src / n).unlink()                       # the L4 has no originals: nothing may read them

    db = ControlDB(fresh_database_url(tmp_path, "c.db"))
    repo = JobRepo(db)
    repo.create(RUN, runner="modal_gpu" if l4 else "local")
    repo.transition(RUN, STAGING)
    repo.transition(RUN, RUNNING)

    images = store.input_images(RUN)

    def output_for(context, result):
        return RunOutput.from_context(finished_context(str(images[0]), str(images[1])), RESULT)

    curator = JobCurator(store, spec_builder=SpecBuilder(), runner=fake_pipeline, output_builder=output_for,
                         db=db, persister=WorkerPersister(db, store, api_sessions=sessions))

    class W:
        pass

    w = W()
    w.l4, w.src, w.store, w.db, w.repo, w.sessions, w.curator, w.objects = \
        l4, src, store, db, repo, sessions, curator, objects
    w.run = lambda: curator.calc(CurateJobInputs(job_id=RUN, pipeline="minimal",
                                                 runner="modal_gpu" if l4 else "local"))
    return w


class ut_WorkerPersister:
    def test_the_worker_ends_the_job_succeeded_with_every_row(self, world):
        res = world.run()
        assert res.ok
        job = world.repo.get(RUN)
        assert job.status == SUCCEEDED and job.finished_at is not None
        with world.db.session() as s:
            assert s.query(Photo).filter_by(job_id=RUN).count() == len(NAMES)
            assert s.query(Face).filter_by(job_id=RUN).count() == 3
        with world.sessions() as s:
            run = s.get(PipelineRun, RUN)
            assert run.status == "completed" and run.progress == 1.0 and run.completed_steps
            assert s.query(PipelineResult).filter_by(run_id=RUN).count() == 1
            assert s.query(Person).filter_by(run_id=RUN).count() == 1

    def test_rows_name_the_originals_and_no_staged_path_is_stored(self, world):
        world.run()
        with world.sessions() as s:
            paths = {r.image_path for r in s.query(ImageMetricRow).all()}
            assert paths == {str(world.src / "a.jpg"), str(world.src / "b.jpg")}
            blob = json.dumps(s.query(PipelineResult).one().image_metrics)
            assert "_inputs" not in blob and str(world.store.inputs.root) not in blob
            face = s.query(FaceMetricRow).one()
            # pixel boxes in original pixels, from the manifest's sizes (the L4 cannot open the originals)
            assert face.bbox_x_px == pytest.approx(20.0)
            if world.l4:
                assert s.query(PipelineResult).one().fc_export_dir is None

    def test_display_crops_and_result_files_are_in_the_object_store(self, world):
        world.run()
        keys = set(world.objects.list(f"{RUN}/"))
        assert {f"{RUN}/display/a.jpg", f"{RUN}/faces/0.jpg", f"{RUN}/results/bundle.json",
                f"{RUN}/results/embeddings.npy", f"{RUN}/run.json"} <= keys
        with world.db.session() as s:
            assert s.get(Photo, (RUN, "a")).display_key == "display/a.jpg"
            assert s.get(Face, (RUN, 0)).crop_key == "faces/0.jpg"

    def test_the_pc_reads_an_l4_album_from_the_object_store(self, world, tmp_path):
        """Scripts that read album.json (regression) find an L4 run's album although nothing was copied back."""
        world.run()
        pc = JobStore(tmp_path / "pc_jobs", objects=world.objects)
        assert not pc.paths(RUN).album_path.exists()
        album = pc.read_album(RUN)
        if world.l4:
            assert {p["id"] for p in album["picks"]} == {"a", "c"}
        else:
            assert album is None            # the local store is the job folder itself: no second copy

    def test_a_failed_run_write_fails_the_job_and_leaves_no_run_rows(self, world, monkeypatch):
        monkeypatch.setattr(PeopleService, "add_from_payload",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("disk full")))
        res = world.run()
        assert not res.ok and "disk full" in res.error
        job = world.repo.get(RUN)
        assert job.status == FAILED and "disk full" in job.error
        with world.sessions() as s:
            run = s.get(PipelineRun, RUN)
            assert run.status == "failed" and "disk full" in run.error_message
            assert s.query(PipelineResult).count() == 0 and s.query(Person).count() == 0

    def test_a_failed_pipeline_is_recorded_by_the_worker(self, world):
        from types import SimpleNamespace
        world.curator._runner = lambda images, plan, **_: (None, SimpleNamespace(
            success=False, failed_step="score_ava", error_message="CUDA out of memory", step_results=[],
            total_duration_ms=1))
        res = world.run()
        assert not res.ok
        job = world.repo.get(RUN)
        assert (job.status, job.failed_step, job.error) == (FAILED, "score_ava", "CUDA out of memory")
        with world.sessions() as s:
            assert s.get(PipelineRun, RUN).error_message == "score_ava: CUDA out of memory"


def test_job_service_reads_the_row_the_worker_wrote(tmp_path):
    """JobService does not load a job twice: when its worker ended it succeeded, the service only reads that."""
    from sim_bench.serving.job_service import JobService, SubmitInputs
    from sim_bench.serving.runners import RunOutcome
    from sim_bench.serving.serving_config import ServingConfig

    src = tmp_path / "trip"
    src.mkdir()
    for n in NAMES:
        Image.new("RGB", (900, 600), "gray").save(src / n)
    db = ControlDB(fresh_database_url(tmp_path, "c.db"))
    calls = []

    class PersistingWorker:
        name = "local"

        def __init__(self, store):
            self.store = store

        def run(self, job_id):
            curator = JobCurator(self.store, spec_builder=SpecBuilder(), runner=fake_pipeline,
                                 output_builder=lambda c, r: RunOutput(), db=db,
                                 persister=WorkerPersister(db, self.store))
            res = curator.calc(CurateJobInputs(job_id=job_id, pipeline="minimal"))
            calls.append(res.ok)
            return RunOutcome(ok=res.ok)

    svc = JobService(db, ServingConfig(jobs_root=tmp_path / "jobs"),
                     runner_factory=lambda name, store, p, d: PersistingWorker(store))
    out = svc.submit(SubmitInputs(source=src, job_id="trip"))
    assert out.status == SUCCEEDED and calls == [True]
    statuses = [e.message.split(":")[0] for e in svc.repo.events("trip") if e.kind == "status"]
    assert statuses.count("loading") == 1 and statuses[-1].startswith("succeeded")


def test_the_local_worker_gets_the_job_database_by_env_not_argv(tmp_path):
    """The child process writes to the API's job database; a password never lands on a command line."""
    from sim_bench.serving.runners import CONTROL_DB_ENV, LocalRunner

    class FakeDB:
        url = "postgresql://u:secret@host/db"

    runner = LocalRunner(JobStore(tmp_path), db=FakeDB())
    assert "secret" not in " ".join(runner.command("j1"))
    assert runner.child_env()[CONTROL_DB_ENV] == FakeDB.url
