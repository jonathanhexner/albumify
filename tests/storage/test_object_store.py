"""spec-117 R7 / R14: files by key, in a local folder or in R2 (moto: a fake S3 in-process, no network)."""
from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from sim_bench.storage.object_store import (
    LocalObjectStore, R2ObjectStore, check_key, job_key, make_object_store,
)

BUCKET = "albumify-test"


@pytest.fixture
def r2():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield R2ObjectStore(BUCKET, s3), s3


@pytest.fixture(params=["local", "r2"])
def store(request, tmp_path):
    if request.param == "local":
        yield LocalObjectStore(tmp_path / "jobs")
        return
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        yield R2ObjectStore(BUCKET, s3)


class ut_ObjectStore:
    def test_put_get_exists_list(self, store):
        store.put("j1/display/a.jpg", b"A")
        store.put("j1/faces/0.jpg", b"F0")
        store.put("j2/display/a.jpg", b"other job")
        assert store.get("j1/display/a.jpg") == b"A"
        assert store.exists("j1/faces/0.jpg") and not store.exists("j1/faces/9.jpg")
        assert store.list("j1/") == ["j1/display/a.jpg", "j1/faces/0.jpg"]
        assert store.list("j1/faces/") == ["j1/faces/0.jpg"]
        with pytest.raises(KeyError):
            store.get("j1/display/missing.jpg")
        assert store.get_or_none("j1/display/missing.jpg") is None

    def test_overwrite_replaces(self, store):
        store.put("j1/results/bundle.json", b"v1")
        store.put("j1/results/bundle.json", b"v2")
        assert store.get("j1/results/bundle.json") == b"v2"

    def test_delete_prefix(self, store):
        """R14: deleting a job's prefix removes its files only, and says how many."""
        for k in ("j1/display/a.jpg", "j1/faces/0.jpg", "j1/results/embeddings.npy", "j10/display/a.jpg"):
            store.put(k, b"x")
        assert store.delete_prefix("j1/") == 3
        assert store.list("j1/") == [] and store.list("j10/") == ["j10/display/a.jpg"]
        assert store.delete_prefix("nothing/") == 0
        with pytest.raises(ValueError):
            store.delete_prefix("j1")          # a bare name could also match j10

    @pytest.mark.parametrize("bad", ["../x.jpg", "j1/../j2/a.jpg", "/abs.jpg", "C:/x.jpg", "j1\\a.jpg",
                                     "j1//a.jpg", "", "j1/./a.jpg"])
    def test_unsafe_keys_are_refused(self, store, bad):
        with pytest.raises(ValueError):
            store.put(bad, b"x")
        assert store.get_or_none(bad) is None

    def test_names_with_spaces_and_unicode_are_fine(self, store):
        key = job_key("j1", "display/IMG 0001 (1)-\u05d0.jpg")
        store.put(key, b"x")
        assert store.list("j1/display/") == [key]


class ut_R2ObjectStore:
    def test_objects_live_under_jobs_prefix(self, r2):
        store, s3 = r2
        store.put("j1/display/a.jpg", b"A")
        names = [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]]
        assert names == ["jobs/j1/display/a.jpg"]

    def test_delete_prefix_beyond_one_page(self, r2):
        store, _ = r2
        for i in range(1005):
            store.put(f"big/faces/{i}.jpg", b"x")
        assert store.delete_prefix("big/") == 1005 and store.list("big/") == []

    def test_from_env_needs_every_variable(self, monkeypatch):
        for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
            monkeypatch.delenv(v, raising=False)
        with pytest.raises(RuntimeError, match="R2_BUCKET"):
            R2ObjectStore.from_env()


def test_make_object_store_is_local_without_r2(tmp_path):
    """R15: no R2_* variables -> the job folders on disk."""
    store = make_object_store(tmp_path)
    assert isinstance(store, LocalObjectStore) and store.root == tmp_path and store.is_local


def test_make_object_store_is_r2_with_every_variable(monkeypatch):
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "albumify")
    store = make_object_store("/unused")
    assert isinstance(store, R2ObjectStore) and store.bucket == "albumify" and not store.is_local
    assert store._s3.meta.endpoint_url == "https://acct.r2.cloudflarestorage.com"


def test_local_store_layout_is_the_job_folder(tmp_path):
    """The local store keeps the pre-spec-117 layout: <jobs_root>/<job_id>/display/<name>."""
    LocalObjectStore(tmp_path).put("j1/display/a.jpg", b"A")
    assert (tmp_path / "j1" / "display" / "a.jpg").read_bytes() == b"A"
    assert check_key("j1/", prefix=True) == "j1/"
