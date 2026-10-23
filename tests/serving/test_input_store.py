"""spec-112 TW1.2 - shared, content-addressed input store + job manifests.

A photo is staged once, at a path fixed by its content and the staging version. Jobs point at those
paths through ``input/manifest.json``; the API resolves file keys inside the job folder or the store
and nowhere else.
"""
from __future__ import annotations

import os

import pytest
from PIL import Image

import sim_bench.serving.input_store as input_store_mod
from sim_bench.serving.input_store import InputStore
from sim_bench.serving.job_store import JobStore


@pytest.fixture
def photo(tmp_path):
    d = tmp_path / "trip"
    d.mkdir()
    p = d / "IMG_1.jpg"
    Image.new("RGB", (1600, 1200), (90, 120, 150)).save(p)
    return p


class ut_InputStore:

    def test_key_is_content_hash_plus_original_name(self, tmp_path, photo):
        store = InputStore(tmp_path / "inputs")
        key = store.key_for(photo)
        sha, name = key.split("/")
        assert len(sha) == 16 and name == "IMG_1.jpg"
        # same bytes elsewhere -> same hash; the name stays the photo id
        other = tmp_path / "IMG_1_copy.jpg"
        other.write_bytes(photo.read_bytes())
        assert store.key_for(other) == f"{sha}/IMG_1_copy.jpg"

    def test_a_new_staging_version_gives_a_new_key(self, tmp_path, photo, monkeypatch):
        store = InputStore(tmp_path / "inputs")
        before = store.key_for(photo)
        monkeypatch.setattr(input_store_mod, "STAGING_VERSION", "1024px-q90-v2")
        assert store.key_for(photo) != before

    def test_put_resizes_once_and_never_rewrites(self, tmp_path, photo):
        store = InputStore(tmp_path / "inputs")
        key = store.put(photo)
        path = store.path(key)
        assert max(Image.open(path).size) == 768
        os.utime(path, (1_000_000, 1_000_000))   # the cache invalidates on mtime: it must not move
        assert store.put(photo) == key
        assert path.stat().st_mtime == 1_000_000
        assert not list(path.parent.glob("*.part"))

    @pytest.mark.parametrize("key", ["../x.jpg", "abc/x.jpg", "0123456789abcdef/../x.jpg",
                                     "0123456789abcdef/x.png", ""])
    def test_malformed_keys_are_refused(self, tmp_path, key):
        with pytest.raises(ValueError, match="invalid input key"):
            InputStore(tmp_path).path(key)


class ut_JobManifest:

    KEYS = ["0000000000000001/a.jpg", "0000000000000002/b.jpg"]

    def _store(self, tmp_path):
        s = JobStore(tmp_path / "jobs")
        for k in self.KEYS:
            s.inputs.put_bytes(k, b"jpeg")
        return s

    def test_manifest_points_the_job_at_the_shared_store(self, tmp_path):
        s = self._store(tmp_path)
        s.write_input_manifest("j1", self.KEYS)
        assert s.input_keys("j1") == self.KEYS
        assert [p.name for p in s.input_images("j1")] == ["a.jpg", "b.jpg"]
        assert all(s.inputs.root in p.parents for p in s.input_images("j1"))
        assert s.input_file_keys("j1") == {"a": "inputs/" + self.KEYS[0], "b": "inputs/" + self.KEYS[1]}

    def test_default_store_sits_beside_the_jobs_and_cannot_be_a_job(self, tmp_path):
        s = JobStore(tmp_path / "jobs")
        assert s.inputs.root == tmp_path / "jobs" / "_inputs"
        with pytest.raises(ValueError, match="invalid job_id"):
            s.paths("_inputs")

    def test_two_inputs_with_one_name_are_refused(self, tmp_path):
        s = self._store(tmp_path)
        s.inputs.put_bytes("0000000000000003/a.jpg", b"jpeg")
        with pytest.raises(ValueError, match="share a file name"):
            s.write_input_manifest("j1", [self.KEYS[0], "0000000000000003/a.jpg"])

    def test_an_older_job_keeps_its_own_input_folder(self, tmp_path):
        s = JobStore(tmp_path / "jobs")
        s.stage_inputs("old", [("a.jpg", b"jpeg")])
        assert s.input_keys("old") is None
        assert s.input_file_keys("old") == {"a": "input/a.jpg"}

    def test_resolve_stays_inside_the_job_or_the_store(self, tmp_path):
        s = self._store(tmp_path)
        s.write_input_manifest("j1", self.KEYS)
        s.write_album("j1", {"picks": []})
        assert s.resolve("j1", "inputs/" + self.KEYS[0]) == s.inputs.path(self.KEYS[0]).resolve()
        assert s.resolve("j1", "album.json") == s.paths("j1").album_path.resolve()
        (tmp_path / "secret.jpg").write_bytes(b"x")
        for bad in ("inputs/../../secret.jpg", "../../secret.jpg", "inputs/0000000000000009/zz.jpg", ""):
            assert s.resolve("j1", bad) is None
