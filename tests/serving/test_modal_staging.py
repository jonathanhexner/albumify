"""spec-115 W3 (R8, R9, R13) + spec-117 W5 (R8): the Modal runner uploads only missing photos, starts a detached
call, follows the job row and copies nothing back; photo retention keeps what another job used within 7 days.
A fake stands in for Modal: its "worker" ends the job in the database itself, as the real one does.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import FAILED, RUNNING, STAGING, SUCCEEDED, JobRepo
from sim_bench.serving.input_retention import RETENTION_DAYS, mark_used, purge, release
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.runners import CANCELLED, ModalRunner

from tests._db import fresh_database_url

DAY = 86400.0
KEYS = ["0000000000000001/a.jpg", "0000000000000002/b.jpg"]


class FakeModal:
    """The Modal side: an input Volume on 'remote' disk; ``start`` = a detached call whose worker writes its own
    files on the remote side and ends the job row (``outcome``: succeeded | failed | none = never ends)."""

    def __init__(self, root: Path, db: ControlDB, outcome: str = "succeeded"):
        self.inputs, self.jobs, self.db, self.outcome = root / "inputs", root / "jobs", db, outcome
        self.uploads, self.calls, self.cancelled, self.status_calls = [], [], [], 0

    def missing(self, keys):
        return [k for k in keys if not (self.inputs / k).is_file()]

    def upload(self, keys, store):
        for k in keys:
            (self.inputs / k).parent.mkdir(parents=True, exist_ok=True)
            (self.inputs / k).write_bytes(store.inputs.path(k).read_bytes())
        self.uploads.append(len(keys))

    def start(self, job_id, keys, spec, manifest, pipeline):
        self.calls.append((job_id, tuple(keys), pipeline, manifest))
        root = self.jobs / job_id
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "album.json").write_text("{}")
        (root / "run.json").write_text(json.dumps({"ok": True}))
        repo = JobRepo(self.db)
        if self.outcome == "succeeded":
            repo.transition(job_id, "loading")
            repo.transition(job_id, SUCCEEDED, "worker saved its results")
        elif self.outcome == "failed":
            repo.fail(job_id, "CUDA out of memory", "score_ava")
        return f"call-{job_id}"

    def status(self, call_id):
        self.status_calls += 1
        return "running" if self.outcome == "none" else "done"

    def cancel(self, call_id):
        self.cancelled.append(call_id)


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "pc")
    for k in KEYS:
        s.inputs.put_bytes(k, b"jpeg")
    s.write_input_manifest("j1", KEYS, {k: f"D:/trip/{Path(k).name}" for k in KEYS}, {k: [1536, 1024] for k in KEYS})
    s.write_input_manifest("j2", KEYS)
    return s


@pytest.fixture
def db(tmp_path):
    d = ControlDB(fresh_database_url(tmp_path, "c.db"))
    repo = JobRepo(d)
    for job_id in ("j1", "j2"):
        repo.create(job_id, runner="modal_gpu")
        repo.transition(job_id, STAGING)
        repo.transition(job_id, RUNNING)
    return d


def _runner(store, fake, db):
    return ModalRunner(store, remote=fake, db=db, poll_s=0.01)


class ut_ModalStaging:

    def test_second_job_on_the_same_photos_uploads_nothing(self, store, db, tmp_path):
        fake = FakeModal(tmp_path / "modal", db)
        first = _runner(store, fake, db).run("j1")
        second = _runner(store, fake, db).run("j2")
        assert first.ok and second.ok
        assert fake.uploads == [2, 0]
        assert second.notes[0] == "uploaded 0 new photos to Modal (2 already there)"

    def test_nothing_comes_back_and_the_job_row_is_the_outcome(self, store, db, tmp_path):
        """spec-117 R6/R8: the worker wrote the database; the PC's job folder holds its manifest only."""
        out = _runner(store, FakeModal(tmp_path / "modal", db), db).run("j1")
        root = store.paths("j1").root
        got = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        assert got == ["input/manifest.json"]
        assert out.ok and JobRepo(db).get("j1").status == SUCCEEDED

    def test_the_call_id_is_stored_and_the_manifest_travels(self, store, db, tmp_path):
        fake = FakeModal(tmp_path / "modal", db)
        _runner(store, fake, db).run("j1")
        assert JobRepo(db).get("j1").call_id == "call-j1"
        manifest = fake.calls[0][3]
        assert manifest["originals"]["0000000000000001/a.jpg"] == "D:/trip/a.jpg"
        assert manifest["sizes"]["0000000000000001/a.jpg"] == [1536, 1024]

    def test_a_worker_failure_is_read_from_the_job_row(self, store, db, tmp_path):
        out = _runner(store, FakeModal(tmp_path / "modal", db, outcome="failed"), db).run("j1")
        assert not out.ok and (out.error, out.failed_step) == ("CUDA out of memory", "score_ava")

    def test_a_call_that_ends_without_results_fails(self, store, db, tmp_path):
        fake = FakeModal(tmp_path / "modal", db, outcome="crash")      # call done, job row never ended
        out = _runner(store, fake, db).run("j1")
        assert not out.ok and "ended without saving its results" in out.error

    def test_a_queued_run_reads_waiting_for_a_gpu(self, store, db, tmp_path):
        """R9: between the spawn and the worker's first write the stage is "waiting for gpu", never "running"."""
        fake = FakeModal(tmp_path / "modal", db, outcome="none")
        runner = _runner(store, fake, db)
        seen = []
        original = fake.status

        def status(call_id):
            seen.append(JobRepo(db).get("j1").progress["stage"])
            runner.cancel("j1")
            return original(call_id)

        fake.status = status
        out = runner.run("j1")
        assert seen == ["waiting for gpu"]
        assert out.error == CANCELLED and fake.cancelled == ["call-j1"]     # cancel goes to the call by id

    def test_cancelled_before_the_call_starts_no_gpu(self, store, db, tmp_path):
        fake = FakeModal(tmp_path / "modal", db)
        runner = _runner(store, fake, db)
        runner.cancel("j1")
        out = runner.run("j1")
        assert not out.ok and out.error == CANCELLED and fake.calls == []

    def test_cancel_during_upload_starts_no_gpu_call(self, store, db, tmp_path):
        fake = FakeModal(tmp_path / "modal", db)
        runner = _runner(store, fake, db)
        original = fake.upload
        fake.upload = lambda keys, st: (original(keys, st), runner.cancel("j1"))   # cancel lands mid-upload
        out = runner.run("j1")
        assert out.error == CANCELLED and fake.calls == []

    def test_job_without_a_manifest_is_refused(self, tmp_path, db):
        s = JobStore(tmp_path / "pc")
        s.stage_inputs("old", [("a.jpg", b"x")])
        out = _runner(s, FakeModal(tmp_path / "modal", db), db).run("old")
        assert not out.ok and "no input manifest" in out.error

    def test_the_real_remote_refuses_a_local_sqlite_database(self, store, tmp_path):
        """The L4 worker writes the database: a SQLite file on this PC cannot be it. Refused before any upload."""
        from sim_bench.serving.detached_modal import DetachedModalRemote
        local = ControlDB(f"sqlite:///{(tmp_path / 'local.db').as_posix()}")
        out = ModalRunner(store, remote=DetachedModalRemote(), volumes=object(), db=local).run("j1")
        assert not out.ok and "DATABASE_URL" in out.error


class ut_InputRetention:

    def _root(self, tmp_path):
        root = tmp_path / "inputs"
        for k in KEYS:
            (root / k).parent.mkdir(parents=True)
            (root / k).write_bytes(b"jpeg")
        return root

    def test_purge_deletes_photos_unused_for_the_window(self, tmp_path):
        root, now = self._root(tmp_path), 100 * DAY
        mark_used(root, "old", KEYS[:1], now - (RETENTION_DAYS + 1) * DAY)
        mark_used(root, "recent", KEYS[1:], now - 1 * DAY)
        res = purge(root, now)
        assert res.deleted == ["0000000000000001"] and res.kept == 1
        assert (root / KEYS[1]).is_file() and not (root / KEYS[0]).exists()

    def test_an_unmarked_photo_falls_back_to_its_folder_time(self, tmp_path):
        """review M2: a photo whose record is missing (just uploaded) or damaged is not purged on sight."""
        root, now = self._root(tmp_path), 100 * DAY
        old, fresh = (root / k.split("/")[0] for k in KEYS)
        os.utime(old, (now - 30 * DAY, now - 30 * DAY))
        os.utime(fresh, (now - 1 * DAY, now - 1 * DAY))
        (fresh / ".used.json").write_text("{not json")
        res = purge(root, now)
        assert res.deleted == ["0000000000000001"] and res.kept == 1

    def test_records_are_written_whole(self, tmp_path):
        root = self._root(tmp_path)
        mark_used(root, "j", KEYS, 50 * DAY)
        folder = root / "0000000000000001"
        assert json.loads((folder / ".used.json").read_text()) == {"j": 50 * DAY}
        assert not list(folder.glob("*.part"))

    def test_release_keeps_photos_another_job_used_recently(self, tmp_path):
        root, now = self._root(tmp_path), 100 * DAY
        mark_used(root, "mine", KEYS, now)
        mark_used(root, "other", KEYS[1:], now - 2 * DAY)
        res = release(root, "mine", KEYS, now)
        assert res.deleted == ["0000000000000001"] and res.kept == 1
        record = json.loads((root / "0000000000000002" / ".used.json").read_text())
        assert "mine" not in record and "other" in record

    def test_purge_removes_old_per_job_copies_on_the_jobs_volume(self, tmp_path):
        root, jobs, now = self._root(tmp_path), tmp_path / "jobs", 100 * DAY
        for job, age in (("ancient", 30), ("fresh", 1)):
            d = jobs / job / "input"
            d.mkdir(parents=True)
            (d / "x.jpg").write_bytes(b"jpeg")
            (jobs / job / "album.json").write_text("{}")
            os.utime(d / "x.jpg", (now - age * DAY, now - age * DAY))
        res = purge(root, now, legacy_jobs_root=jobs)
        assert "ancient/input" in res.deleted and "fresh/input" not in res.deleted
        assert not (jobs / "ancient" / "input" / "x.jpg").exists()
        assert (jobs / "ancient" / "album.json").exists()          # results stay
        assert (jobs / "fresh" / "input" / "x.jpg").exists()
