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
from sim_bench.serving.runners import CANCELLED, LocalRunner, ModalRunner, RunOutcome
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


class ResultsOnlyRunner:
    """A Modal stand-in: curates in its own store (the worker's Volume), then copies back only
    ``results_files`` (spec-115 R8), exactly what the real Modal download will do."""

    def __init__(self, name, store, worker_root):
        self.name, self.store, self.worker = name, store, JobStore(worker_root)

    def run(self, job_id):
        self.worker.write_input_manifest(job_id, self.store.input_keys(job_id))
        self.worker.inputs = self.store.inputs              # the worker's own copy of the same photos
        res = JobCurator(self.worker, spec_builder=SpecBuilder(), runner=fake_pipeline).calc(
            CurateJobInputs(job_id=job_id, pipeline="minimal"))
        for rel in self.worker.results_files(job_id):
            dest = self.store.paths(job_id).root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((self.worker.paths(job_id).root / rel).read_bytes())
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

    def test_local_and_results_only_jobs_leave_the_same_files(self, tmp_path, src):
        """spec-115 R10: the worker returns results only; the server renders the rest, so both runners
        leave identical job folders."""
        svc = _service(tmp_path, lambda name, store, *a: InProcessRunner(name, store) if name == "local"
                       else ResultsOnlyRunner(name, store, tmp_path / "modal_volume"))
        svc.submit(SubmitInputs(source=src, job_id="here", runner="local"))
        svc.submit(SubmitInputs(source=src, job_id="there", runner="modal_gpu"))

        def files(job):
            root = svc.store.paths(job).root
            return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        assert svc.repo.get("there").status == SUCCEEDED
        assert files("here") == files("there")
        assert any(f.startswith("faces/") for f in files("there"))
        assert any(f.startswith("display/") for f in files("there"))

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
        import sys
        script = "import sys; sys.stderr.write('Traceback' + chr(10) + 'MemoryError' + chr(10)); sys.exit(3)"
        monkeypatch.setattr(LocalRunner, "command", lambda self, job_id: [sys.executable, "-c", script])
        out = LocalRunner(JobStore(tmp_path)).run("j1")
        assert not out.ok and "exited 3" in out.error and "MemoryError" in out.error

    def test_local_runner_cancel_kills_the_child(self, tmp_path, monkeypatch):
        """spec-115 R4: a cancelled local job ends cancelled, in seconds."""
        import sys
        import threading
        import time
        monkeypatch.setattr(LocalRunner, "command",
                            lambda self, job_id: [sys.executable, "-c", "import time; time.sleep(60)"])
        runner, out = LocalRunner(JobStore(tmp_path)), {}
        t = threading.Thread(target=lambda: out.setdefault("r", runner.run("j1")))
        t0 = time.time()
        t.start()
        while "j1" not in runner._procs and time.time() - t0 < 10:
            time.sleep(0.05)
        runner.cancel("j1")
        t.join(timeout=30)
        assert out["r"].error == CANCELLED and not out["r"].ok
        assert time.time() - t0 < 30

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


class ut_Estimate:
    """spec-115 R2: L4 estimate and spend cap (1,000 photos / 30 GPU-minutes by default)."""

    def test_l4_estimate_has_minutes_and_dollars(self, tmp_path, src):
        est = _service(tmp_path, _in_process).estimate(src, runner="modal_gpu")
        assert est.n_photos == 6 and est.allowed
        assert est.gpu_minutes == round((150 + 6 * 0.6) / 60, 1) and est.dollars is not None

    def test_local_estimate_has_no_gpu_figures(self, tmp_path, src):
        est = _service(tmp_path, _in_process).estimate(src, runner="local")
        assert est.n_photos == 6 and est.gpu_minutes is None and est.allowed

    @pytest.mark.parametrize("limits, phrase", [({"max_photos_l4": 5}, "over the L4 limit of 5"),
                                                ({"max_gpu_minutes": 1.0}, "GPU minutes is over the limit")])
    def test_over_a_limit_the_l4_job_is_refused(self, tmp_path, src, limits, phrase):
        import dataclasses
        svc = _service(tmp_path, _in_process)
        svc.config = dataclasses.replace(svc.config, **limits)
        assert phrase in svc.estimate(src, runner="modal_gpu").reason
        with pytest.raises(ValueError, match=phrase):
            svc.create(SubmitInputs(source=src, job_id="big", runner="modal_gpu"))
        assert svc.repo.get("big") is None


class BlockingRunner:
    """Runs until cancelled, like a long L4 call; writes a partial result first."""

    def __init__(self, name, store):
        import threading
        self.name, self.store, self.started, self._stop = name, store, threading.Event(), threading.Event()

    def run(self, job_id):
        (self.store.paths(job_id).results_dir).mkdir(parents=True, exist_ok=True)
        (self.store.paths(job_id).results_dir / "bundle.json").write_text("{}")
        self.started.set()
        self._stop.wait(30)
        return RunOutcome(ok=False, error=CANCELLED)

    def cancel(self, job_id):
        self._stop.set()


class ut_Cancel:
    """spec-115 R4: end state of a cancelled job."""

    def test_running_job_ends_cancelled_and_drops_partial_results(self, tmp_path, src):
        import threading
        svc = _service(tmp_path, lambda name, store, *a: runner)
        runner = BlockingRunner("local", svc.store)
        job_id = svc.create(SubmitInputs(source=src, job_id="long"))
        t = threading.Thread(target=svc.run, args=(job_id, SubmitInputs(source=src, job_id="long")))
        t.start()
        assert runner.started.wait(10)
        svc.cancel("long")
        t.join(10)
        job = svc.repo.get("long")
        assert job.status == FAILED and job.error == CANCELLED
        assert not svc.store.paths("long").results_dir.exists()
        assert svc.store.input_keys("long")                       # staged photos stay

    def test_queued_job_is_cancelled_before_it_starts(self, tmp_path, src):
        svc = _service(tmp_path, _in_process)
        job_id = svc.create(SubmitInputs(source=src, job_id="q"))
        svc.cancel(job_id)
        assert svc.repo.get("q").status == FAILED
        res = svc.run(job_id, SubmitInputs(source=src, job_id="q"))   # the background task still fires
        assert res.error == CANCELLED and svc.repo.get("q").error == CANCELLED

    def test_finished_job_cannot_be_cancelled(self, tmp_path, src):
        svc = _service(tmp_path, _in_process)
        svc.submit(SubmitInputs(source=src, job_id="done"))
        with pytest.raises(ValueError, match="already succeeded"):
            svc.cancel("done")


class ut_DeleteRemoteInputs:

    def test_delete_calls_the_remote_with_the_jobs_photos_and_logs_it(self, tmp_path, src):
        svc = _service(tmp_path, _in_process)
        svc.submit(SubmitInputs(source=src, job_id="j"))
        seen = {}

        class Remote:
            def delete_inputs(self, job_id, keys):
                seen["args"] = (job_id, list(keys))
                return {"deleted": keys[:4], "kept": 2}

        res = svc.delete_remote_inputs("j", remote=Remote())
        assert seen["args"] == ("j", svc.store.input_keys("j")) and res["kept"] == 2
        assert any("deleted 4 photos from Modal (2 kept" in e.message for e in svc.repo.events("j"))


class ut_RunSettingsRead:

    def test_half_written_run_record_reads_as_no_settings(self, tmp_path):
        svc = _service(tmp_path, _in_process)
        path = svc.store.paths("j").run_path
        path.parent.mkdir(parents=True)
        path.write_text('{"settings": {"pipel')          # mid copy-back
        assert svc.run_settings("j") is None
        assert svc.run_settings("missing") is None


class ut_ReviewFixes:
    """spec-115 code review (2026-10-23): M1, M3, M7, L1, L4."""

    def test_cancel_of_a_job_waiting_for_the_modal_slot_ends_it_at_once(self, tmp_path, src):
        """M1: the second Modal job waits for the first; cancelling it must not wait for the first to end."""
        import threading
        import time
        svc = _service(tmp_path, lambda name, store, *a: runners.pop(0))
        blocking = BlockingRunner("modal_cpu", svc.store)
        runners = [blocking, InProcessRunner("modal_cpu", svc.store)]
        jobs = {}
        for jid in ("first", "second"):
            inputs = SubmitInputs(source=src, job_id=jid, runner="modal_cpu")
            svc.create(inputs)
            jobs[jid] = threading.Thread(target=svc.run, args=(jid, inputs))
        jobs["first"].start()
        assert blocking.started.wait(10)
        jobs["second"].start()
        deadline = time.time() + 10
        while not any("waiting for the other Modal job" in e.message for e in svc.repo.events("second")):
            assert time.time() < deadline
            time.sleep(0.05)
        t0 = time.time()
        svc.cancel("second")
        jobs["second"].join(10)
        assert svc.repo.get("second").error == CANCELLED and time.time() - t0 < 5
        assert svc.repo.get("first").status == "running"          # the first job is untouched
        svc.cancel("first")
        jobs["first"].join(10)
        assert svc.repo.get("first").error == CANCELLED

    def test_local_cancel_before_the_child_starts_still_stops_it(self, tmp_path, monkeypatch):
        """M3: a cancel that lands before the process exists."""
        import sys
        monkeypatch.setattr(LocalRunner, "command",
                            lambda self, job_id: [sys.executable, "-c", "import time; time.sleep(60)"])
        runner = LocalRunner(JobStore(tmp_path))
        runner.cancel("j1")
        out = runner.run("j1")
        assert out.error == CANCELLED

    def test_delete_photos_of_a_live_job_is_refused_and_remote_errors_are_reported(self, tmp_path, src):
        """M7."""
        from sim_bench.serving.job_service import RemoteError
        svc = _service(tmp_path, _in_process)
        svc.create(SubmitInputs(source=src, job_id="live"))
        with pytest.raises(ValueError, match="once it has finished"):
            svc.delete_remote_inputs("live", remote=object())
        svc.submit(SubmitInputs(source=src, job_id="done"))

        class Broken:
            def delete_inputs(self, job_id, keys):
                raise ConnectionError("modal down")

        with pytest.raises(RemoteError, match="modal down"):
            svc.delete_remote_inputs("done", remote=Broken())

    def test_cancel_while_loading_is_refused(self, tmp_path, src):
        """L1."""
        from sim_bench.control.jobs import LOADING, RUNNING, STAGING
        svc = _service(tmp_path, _in_process)
        svc.create(SubmitInputs(source=src, job_id="l"))
        for st in (STAGING, RUNNING, LOADING):
            svc.repo.transition("l", st)
        with pytest.raises(ValueError, match="loading its results"):
            svc.cancel("l")

    def test_job_ids_of_one_folder_in_one_second_differ(self, tmp_path):
        """L4."""
        assert len({new_job_id(tmp_path / "trip") for _ in range(50)}) == 50
