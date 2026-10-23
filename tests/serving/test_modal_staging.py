"""spec-115 W3 (R8, R9, R13): Modal runner uploads only missing photos and copies back results only;
photo retention keeps what another job used within 7 days. A fake Volume stands in for Modal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sim_bench.serving.input_retention import RETENTION_DAYS, mark_used, purge, release
from sim_bench.serving.job_store import JobStore
from sim_bench.serving.runners import CANCELLED, ModalRunner

DAY = 86400.0
KEYS = ["0000000000000001/a.jpg", "0000000000000002/b.jpg"]


class FakeVolume:
    """The Modal side: an input store + a job store on 'remote' disk, the worker's JobCurator stand-in."""

    def __init__(self, root: Path):
        self.inputs, self.jobs = root / "inputs", root / "jobs"
        self.uploads, self.calls, self.cancelled = [], [], False

    def missing(self, keys):
        return [k for k in keys if not (self.inputs / k).is_file()]

    def upload(self, keys, store):
        for k in keys:
            (self.inputs / k).parent.mkdir(parents=True, exist_ok=True)
            (self.inputs / k).write_bytes(store.inputs.path(k).read_bytes())
        self.uploads.append(len(keys))

    def run_job(self, job_id, keys, pipeline):
        self.calls.append((job_id, tuple(keys), pipeline))
        if self.cancelled:
            return {"ok": False, "error": CANCELLED}
        root = self.jobs / job_id
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "display").mkdir(exist_ok=True)
        (root / "display" / "a.jpg").write_bytes(b"picture")   # must never come back
        (root / "album.json").write_text("{}")
        (root / "run.json").write_text(json.dumps({"ok": True, "timing": {"total_ms": 1000}}))
        (root / "results" / "bundle.json").write_text("{}")
        return {"ok": True}

    def download_results(self, job_id, store):
        worker = JobStore(self.jobs)
        worker_root, n = worker.paths(job_id).root, 0
        for rel in worker.results_files(job_id):
            dest = store.paths(job_id).root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((worker_root / rel).read_bytes())
            n += 1
        return n

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "pc")
    for k in KEYS:
        s.inputs.put_bytes(k, b"jpeg")
    s.write_input_manifest("j1", KEYS)
    s.write_input_manifest("j2", KEYS)
    return s


class ut_ModalStaging:

    def test_second_job_on_the_same_photos_uploads_nothing(self, store, tmp_path):
        vol = FakeVolume(tmp_path / "modal")
        first = ModalRunner(store, remote=vol).run("j1")
        second = ModalRunner(store, remote=vol).run("j2")
        assert first.ok and second.ok
        assert vol.uploads == [2, 0]
        assert second.notes[0] == "uploaded 0 new photos to Modal (2 already there)"

    def test_only_results_come_back(self, store, tmp_path):
        out = ModalRunner(store, remote=FakeVolume(tmp_path / "modal")).run("j1")
        root = store.paths("j1").root
        got = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        assert got == ["album.json", "input/manifest.json", "results/bundle.json", "run.json"]
        assert out.notes[1] == "copied back 3 result files"

    def test_cancelled_call_ends_cancelled_and_copies_nothing(self, store, tmp_path):
        vol = FakeVolume(tmp_path / "modal")
        runner = ModalRunner(store, remote=vol)
        runner.cancel("j1")
        out = runner.run("j1")
        assert not out.ok and out.error == CANCELLED
        assert not store.paths("j1").album_path.exists()

    def test_cancel_during_upload_starts_no_gpu_call(self, store, tmp_path):
        vol = FakeVolume(tmp_path / "modal")
        runner = ModalRunner(store, remote=vol)
        original = vol.upload
        vol.upload = lambda keys, st: (original(keys, st), runner.cancel("j1"))   # cancel lands mid-upload
        out = runner.run("j1")
        assert out.error == CANCELLED and vol.calls == []

    def test_job_without_a_manifest_is_refused(self, tmp_path):
        s = JobStore(tmp_path / "pc")
        s.stage_inputs("old", [("a.jpg", b"x")])
        out = ModalRunner(s, remote=FakeVolume(tmp_path / "modal")).run("old")
        assert not out.ok and "no input manifest" in out.error


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


class ut_ModalVolumeRemoteDownload:

    def test_missing_results_folder_does_not_hide_the_run_error(self, tmp_path, monkeypatch):
        """review M5: the worker failed before writing results; only run.json comes back.

        A stand-in ``modal`` module: importing the real one switches Windows to the selector event loop, which
        breaks every later Playwright test in the same run (it cannot start a browser process)."""
        import sys
        import types
        from sim_bench.serving.runners import ModalVolumeRemote

        class Vol:
            def listdir(self, path, recursive=False):
                raise FileNotFoundError(path)

            def read_file(self, path):
                if path.endswith("run.json"):
                    yield b'{"ok": false, "error": "boom", "failed_step": "score_ava"}'
                else:
                    raise FileNotFoundError(path)

        fake = types.ModuleType("modal")
        fake.Volume = types.SimpleNamespace(from_name=lambda *a, **k: Vol())
        fake_volume = types.ModuleType("modal.volume")
        fake_volume.FileEntryType = types.SimpleNamespace(FILE=1)
        monkeypatch.setitem(sys.modules, "modal", fake)
        monkeypatch.setitem(sys.modules, "modal.volume", fake_volume)
        store = JobStore(tmp_path)
        n = ModalVolumeRemote().download_results("j1", store)
        assert n == 1 and not store.paths("j1").album_path.exists()
        assert json.loads(store.paths("j1").run_path.read_text())["failed_step"] == "score_ava"
        assert not list(store.paths("j1").root.rglob("*.part"))
