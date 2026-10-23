"""spec-112 TA4 - JobService: one flow for every runner (stage -> run -> load), AC4 failures.

Runners are faked at the boundary (a runner that curates in-process with a fake pipeline, one that
fails, one that crashes); the LocalRunner child-process path is covered by its command contract and
by the real-data proof (TA6), not here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import FAILED, SUCCEEDED
from sim_bench.serving.curate_job import CurateJobInputs, JobCurator
from sim_bench.serving.job_service import JobService, SubmitInputs, new_job_id
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.pipeline_runner import SpecBuilder
from sim_bench.serving.runners import LocalRunner, ModalRunner, RunOutcome
from sim_bench.serving.serving_config import ServingConfig, load_serving_config
from sim_bench.serving.staging import collect_images

from tests.control.fakes import NAMES, fake_pipeline


class InProcessRunner:
    """Curates in-process with the fake pipeline -- stands in for local / Modal alike."""

    def __init__(self, name, store):
        self.name, self.store = name, store

    def run(self, job_id):
        res = JobCurator(self.store, spec_builder=SpecBuilder(), runner=fake_pipeline).calc(
            CurateJobInputs(job_id=job_id, pipeline="minimal"))
        return RunOutcome(ok=res.ok, error=res.error, wall_s=1.0)


class FailingRunner:
    name = "local"

    def run(self, job_id):
        return RunOutcome(ok=False, error="score_ava exploded", failed_step="score_ava")


class CrashingRunner:
    name = "local"

    def run(self, job_id):
        raise RuntimeError("container lost")


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "trip"
    d.mkdir()
    for n in NAMES:
        Image.new("RGB", (1600, 1200), "gray").save(d / n)
    (d / "notes.txt").write_text("not a photo")
    return d


def _service(tmp_path, factory):
    db = ControlDB(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    cfg = ServingConfig(runner="local", pipeline="minimal", jobs_root=tmp_path / "jobs")
    return JobService(db, cfg, runner_factory=factory)


def _in_process(name, store, pipeline, device):
    return InProcessRunner(name, store)


class ut_JobService:
    @pytest.mark.parametrize("runner", ["local", "modal_gpu"])
    def test_same_flow_for_every_runner(self, tmp_path, src, runner):
        """AC1 shape: only the runner setting changes; the job lands in the same DB + store."""
        svc = _service(tmp_path, _in_process)
        res = svc.submit(SubmitInputs(source=src, job_id="trip", runner=runner))
        assert res.status == SUCCEEDED, res.error
        job = svc.repo.get("trip")
        assert job.runner == runner and job.n_inputs == 6
        assert [e.message.split(":")[0] for e in svc.repo.events("trip")
                if e.kind == "status"] == ["queued", "staging", "running", "loading", "succeeded"]

    def test_inputs_are_staged_at_analysis_size(self, tmp_path, src):
        svc = _service(tmp_path, _in_process)
        svc.submit(SubmitInputs(source=src, job_id="trip"))
        staged = svc.store.input_images("trip")
        assert [p.name for p in staged] == list(NAMES)
        assert max(Image.open(staged[0]).size) == 768

    def test_a_second_job_on_the_same_photos_stages_nothing_new(self, tmp_path, src):
        """spec-112 TW1.2/TW1.3: photos are staged once; both jobs see the same paths (cache key)."""
        svc = _service(tmp_path, _in_process)
        svc.submit(SubmitInputs(source=src, job_id="first"))
        mtimes = {p: p.stat().st_mtime_ns for p in svc.store.input_images("first")}
        svc.submit(SubmitInputs(source=src, job_id="second"))
        assert svc.store.input_images("second") == svc.store.input_images("first")
        assert {p: p.stat().st_mtime_ns for p in svc.store.input_images("second")} == mtimes
        staged = [e.message for e in svc.repo.events("second") if e.message.startswith("staged")]
        assert staged == ["staged 6 photos (0 new, 6 already staged)"]

    def test_runner_failure_ends_failed_with_error(self, tmp_path, src):
        svc = _service(tmp_path, lambda *a: FailingRunner())
        res = svc.submit(SubmitInputs(source=src, job_id="trip"))
        job = svc.repo.get("trip")
        assert res.status == FAILED and job.status == FAILED
        assert job.error == "score_ava exploded" and job.failed_step == "score_ava"

    def test_runner_crash_ends_failed_not_running(self, tmp_path, src):
        """AC4."""
        svc = _service(tmp_path, lambda *a: CrashingRunner())
        res = svc.submit(SubmitInputs(source=src, job_id="trip"))
        assert res.status == FAILED and "container lost" in svc.repo.get("trip").error
        assert svc.repo.events("trip")[-1].kind == "error"

    def test_empty_folder_fails_cleanly(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        svc = _service(tmp_path, _in_process)
        res = svc.submit(SubmitInputs(source=empty, job_id="e1"))
        assert res.status == FAILED and "no readable images" in res.error

    def test_missing_folder_is_refused_before_a_job_exists(self, tmp_path):
        svc = _service(tmp_path, _in_process)
        with pytest.raises(ValueError):
            svc.create(SubmitInputs(source=tmp_path / "nope"))
        assert svc.repo.list() == []

    def test_load_existing_job_files(self, tmp_path):
        svc = _service(tmp_path, _in_process)
        svc.store.stage_inputs("old", [(n, b"x") for n in NAMES])
        InProcessRunner("modal_gpu", svc.store).run("old")
        res = svc.load_existing("old", runner="modal_gpu")
        assert res.status == SUCCEEDED and svc.repo.get("old").n_inputs == 6

    def test_new_job_id_is_a_valid_store_id(self, tmp_path):
        jid = new_job_id(Path("D:/Slovakia 26 (best)"))
        JobStore(tmp_path).paths(jid)
        assert jid.startswith("Slovakia-26-best-")


class ut_Runners:
    def test_local_runner_runs_the_curator_as_a_child_process(self, tmp_path):
        cmd = LocalRunner(JobStore(tmp_path), pipeline="minimal").command("j1")
        assert cmd[1:3] == ["-m", "sim_bench.serving.runners"]
        assert cmd[cmd.index("--job-id") + 1] == "j1" and "minimal" in cmd

    def test_local_runner_without_record_reports_the_exit(self, tmp_path, monkeypatch):
        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a, 3, "", "Traceback\nMemoryError"))
        out = LocalRunner(JobStore(tmp_path)).run("j1")
        assert not out.ok and "exited 3" in out.error and "MemoryError" in out.error

    def test_local_runner_keeps_the_error_tail_when_output_is_not_in_the_locale_code_page(
            self, tmp_path, monkeypatch):
        """A real child writing non-cp1255 text: the error tail must survive (Windows, Hebrew locale)."""
        import sys
        # 0x81 is undefined in cp1255; the child writes raw bytes so its own encoding is irrelevant
        script = ("import sys; sys.stdout.buffer.write(bytes([0x81, 10])); sys.stdout.flush(); "
                  "sys.stderr.buffer.write(b'boom ' + bytes([0x81, 10])); sys.exit(4)")
        monkeypatch.setattr(LocalRunner, "command", lambda self, job_id: [sys.executable, "-c", script])
        out = LocalRunner(JobStore(tmp_path)).run("j1")
        assert not out.ok and "exited 4" in out.error and "boom" in out.error

    def test_modal_runner_stages_curates_and_copies_back(self, tmp_path):
        store = JobStore(tmp_path)
        store.stage_inputs("j1", [(n, b"x") for n in NAMES])
        calls = []

        class FakeRemote:
            def stage(self, job_id, images):
                calls.append(("stage", len(images)))

            def curate(self, job_id, pipeline):
                calls.append(("curate", pipeline))
                return {"ok": True}

            def download(self, job_id, dest):
                InProcessRunner("modal_gpu", dest).run(job_id)   # files appear locally
                calls.append(("download",))
                return 1

        out = ModalRunner(store, pipeline="default", remote=FakeRemote()).run("j1")
        assert out.ok and calls == [("stage", 6), ("curate", "default"), ("download",)]
        assert store.read_album("j1") is not None


class ut_ServingConfig:
    def test_env_overrides_file(self, tmp_path, monkeypatch):
        f = tmp_path / "s.yaml"
        f.write_text("runner: local\npipeline: default\njobs_root: ~/x\n")
        monkeypatch.setenv("ALBUMIFY_RUNNER", "modal_gpu")
        cfg = load_serving_config(f)
        assert cfg.runner == "modal_gpu" and cfg.jobs_root.name == "x"

    def test_unknown_runner_and_keys_are_refused(self, tmp_path):
        with pytest.raises(ValueError):
            ServingConfig(runner="lambda")
        f = tmp_path / "s.yaml"
        f.write_text("runer: local\n")
        with pytest.raises(ValueError):
            load_serving_config(f)

    def test_repo_config_loads(self):
        assert load_serving_config().runner in ("local", "modal_gpu", "modal_cpu")


def test_collect_images_skips_non_images_and_slices(src):
    imgs = collect_images(src, start=1, n=2)
    assert [n for n, _ in imgs] == ["b.jpg", "c.jpg"]
