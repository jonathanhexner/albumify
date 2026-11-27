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


# --- spec-120 R5: signed upload URLs ---------------------------------------------------------------------------

def test_sizes_lists_every_object_with_its_size(store):
    store.put("u1/0000.jpg", b"abc")
    store.put("u1/0001.png", b"abcdef")
    store.put("u10/0000.jpg", b"x")
    assert store.sizes("u1/") == {"u1/0000.jpg": 3, "u1/0001.png": 6}
    assert store.sizes("nothing/") == {}


def test_signed_urls_need_r2(tmp_path):
    with pytest.raises(NotImplementedError):
        LocalObjectStore(tmp_path).presigned_put("u1/0000.jpg", 10, "image/jpeg")


@pytest.fixture
def r2v4():
    """The fake bucket with a SigV4 client, as ``R2ObjectStore.from_env`` makes it."""
    from botocore.config import Config
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
        s3.create_bucket(Bucket=BUCKET)
        yield R2ObjectStore(BUCKET, s3), s3


class ut_R2SignedUrls:
    def test_under_shares_the_bucket_with_another_prefix(self, r2):
        store, s3 = r2
        uploads = store.under("uploads/")
        uploads.put("u1/0000.jpg", b"A")
        assert [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]] == ["uploads/u1/0000.jpg"]
        assert store.list("u1/") == [] and uploads.sizes("u1/") == {"u1/0000.jpg": 1}
        with pytest.raises(ValueError):
            store.under("../")

    def test_presigned_put_signs_size_and_type(self, r2v4):
        store, _ = r2v4
        url = store.under("uploads/").presigned_put("u1/0000.jpg", 1234, "image/jpeg", expires_s=600)
        assert "/uploads/u1/0000.jpg?" in url and "X-Amz-Expires=600" in url
        assert "X-Amz-SignedHeaders=content-length%3Bcontent-type%3Bhost" in url

    def test_a_client_that_cannot_sign_the_size_gives_no_url(self, r2):
        store, _ = r2                       # default client: older-style signature, size not signed
        with pytest.raises(RuntimeError, match="s3v4"):
            store.presigned_put("u1/0000.jpg", 1234, "image/jpeg")

    def test_from_env_signs_with_sigv4(self, monkeypatch):
        for v, val in (("R2_ACCOUNT_ID", "acct"), ("R2_ACCESS_KEY_ID", "AKIA"), ("R2_SECRET_ACCESS_KEY", "s"),
                       ("R2_BUCKET", "albumify")):
            monkeypatch.setenv(v, val)
        assert R2ObjectStore.from_env()._s3.meta.config.signature_version == "s3v4"

    def test_presigned_get(self, r2v4):
        store, _ = r2v4
        url = store.presigned_get("j1/display/a.jpg", expires_s=300)
        assert "/jobs/j1/display/a.jpg?" in url and "X-Amz-Expires=300" in url


@pytest.mark.needs_r2
def test_real_r2_signed_put():
    """spec-120 R5 on the real bucket: the signed PUT takes the declared size and type and refuses any other."""
    import os
    from pathlib import Path
    import secrets

    import requests
    from dotenv import dotenv_values

    env_file = Path(os.environ.get("ALBUMIFY_ENV_FILE") or Path(__file__).resolve().parents[2] / ".env")
    env = dotenv_values(env_file) if env_file.is_file() else {}
    if not all(env.get(v) for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")):
        pytest.skip("needs_r2: R2_* not in .env")
    saved = {v: os.environ.get(v) for v in env}
    os.environ.update({v: env[v] for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")})
    try:
        uploads = R2ObjectStore.from_env().under("uploads/")
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
    prefix = f"_test-{secrets.token_hex(4)}/"
    key, body = prefix + "0000.jpg", b"\xff\xd8" + b"0" * 98
    try:
        url = uploads.presigned_put(key, len(body), "image/jpeg", expires_s=120)
        put = lambda data, ctype: requests.put(url, data=data, headers={"Content-Type": ctype}, timeout=30).status_code
        assert put(body + b"more", "image/jpeg") == 403           # bigger than signed
        assert put(body[:50], "image/jpeg") == 403                # smaller than signed
        assert put(body, "application/x-msdownload") == 403       # another type
        assert uploads.sizes(prefix) == {}
        assert put(body, "image/jpeg") == 200
        assert uploads.sizes(prefix) == {key: len(body)}
        assert requests.get(uploads.presigned_get(key, 60), timeout=30).content == body
    finally:
        uploads.delete_prefix(prefix)
    assert uploads.list(prefix) == []
