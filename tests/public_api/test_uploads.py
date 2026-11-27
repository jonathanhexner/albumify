"""spec-120 R5 (limits before anything is stored, signed URLs straight to R2) and R6 (expected wait, where it runs).

The store is the local folder store with signed URLs faked: a "browser PUT" is ``store.put`` of the signed key. The
real R2 check (size and type in the signature) is ``tests/storage/test_object_store.py::test_real_r2_signed_put``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from sim_bench.control.db import ControlDB
from sim_bench.control.models import Job, Upload
from sim_bench.public_api.accounts import AccountRepo, ConsentRequired
from sim_bench.public_api.app import create_app
from sim_bench.public_api.auth import AuthVerifier, Claims
from sim_bench.public_api.limits import load_limits
from sim_bench.public_api.uploads import FileIn, UploadIncomplete, UploadRefused, UploadService
from sim_bench.storage.object_store import LocalObjectStore
from tests.public_api.conftest import ISSUER

MB = 1024 ** 2
NOW = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)


class SigningStore(LocalObjectStore):
    """The local store, plus fake signed URLs that record what was signed."""

    def __init__(self, root):
        super().__init__(root)
        self.signed = []

    def presigned_put(self, key, size, content_type, expires_s=3600):
        self.signed.append((key, size, content_type, expires_s))
        return f"https://r2.test/uploads/{key}?size={size}&type={content_type}"


class World:
    def __init__(self, url, tmp_path):
        self.db = ControlDB(url)
        self.accounts = AccountRepo(self.db)
        self.store = SigningStore(tmp_path / "uploads")
        self.clock = NOW
        self.uploads = UploadService(self.db, self.store, load_limits(), clock=lambda: self.clock)

    def account(self, sub="user_1", email="ann@example.com", consent=True):
        self.accounts.invite(email)
        acc = self.accounts.get_or_create(Claims(sub, email))
        return self.accounts.record_consent(acc.id) if consent else acc

    def send(self, ticket, sizes=None):
        """The browser: PUT each file to its signed key (``sizes`` overrides what is actually sent)."""
        for i, (key, size, _, _) in enumerate(k for k in self.store.signed if k[0].startswith(ticket.upload_id)):
            self.store.put(key, b"x" * (sizes[i] if sizes else size))

    def n_uploads(self):
        with self.db.read_session() as s:
            return s.scalar(select(func.count(Upload.id)))


@pytest.fixture
def world(control_db_url, tmp_path):
    return World(control_db_url, tmp_path)


def photos(n, size=4 * MB, ext=".jpg"):
    return [FileIn(f"IMG_{i:04d}{ext}", size) for i in range(n)]


class ut_UploadLimits:
    def test_1001_photos_refused_before_anything_stored(self, world):
        with pytest.raises(UploadRefused, match="1001 photos is over the limit of 1000") as e:
            world.uploads.create(world.account(), photos(1001, size=MB))
        assert e.value.status == 400 and world.n_uploads() == 0 and world.store.signed == []

    def test_a_30_mb_photo_is_refused(self, world):
        files = photos(3) + [FileIn("big.jpg", 30 * MB)]
        with pytest.raises(UploadRefused, match="big.jpg is 30.0 MB, over the limit of 25 MB") as e:
            world.uploads.create(world.account(), files)
        assert e.value.status == 413 and world.n_uploads() == 0 and world.store.signed == []

    def test_25_mb_exactly_is_accepted(self, world):
        assert world.uploads.create(world.account(), [FileIn("a.jpg", 25 * MB)]).n_photos == 1

    @pytest.mark.parametrize("name,ctype", [("setup.exe", None), ("clip.mp4", "video/mp4"), ("a.gif", "image/gif"),
                                            ("a.jpg", "application/x-msdownload"), ("noext", None)])
    def test_wrong_types_are_refused(self, world, name, ctype):
        with pytest.raises(UploadRefused) as e:
            world.uploads.create(world.account(), [FileIn(name, MB, ctype)])
        assert e.value.status == 415 and world.n_uploads() == 0

    def test_accepted_types_get_their_own_content_type(self, world):
        files = [FileIn("a.JPG", MB, "image/jpeg"), FileIn("b.jpeg", MB), FileIn("c.heic", MB, ""),
                 FileIn("d.HEIF", MB), FileIn("e.png", MB, "image/png"), FileIn("f.webp", MB, "image/webp")]
        ticket = world.uploads.create(world.account(), files)
        assert [f.content_type for f in ticket.files] == ["image/jpeg", "image/jpeg", "image/heic", "image/heif",
                                                          "image/png", "image/webp"]

    def test_over_6_gb_in_total_is_refused(self, world):
        with pytest.raises(UploadRefused, match="over the limit of 6 GB") as e:
            world.uploads.create(world.account(), photos(250, size=25 * MB))      # 6.1 GB
        assert e.value.status == 413 and world.n_uploads() == 0

    def test_empty_upload_and_empty_file_are_refused(self, world):
        with pytest.raises(UploadRefused, match="no photos"):
            world.uploads.create(world.account(), [])
        with pytest.raises(UploadRefused, match="empty"):
            world.uploads.create(world.account(), [FileIn("a.jpg", 0)])

    def test_no_upload_without_face_consent(self, world):
        with pytest.raises(ConsentRequired):
            world.uploads.create(world.account(consent=False), photos(3))
        assert world.n_uploads() == 0


class ut_UploadFlow:
    def test_signed_urls_then_complete_queues_an_owned_job(self, world):
        acc = world.account()
        ticket = world.uploads.create(acc, photos(12))
        assert len(ticket.files) == 12 and ticket.total_bytes == 12 * 4 * MB
        key, size, ctype, expires_s = world.store.signed[0]
        assert key == f"{ticket.upload_id}/0000.jpg" and size == 4 * MB and ctype == "image/jpeg"
        assert expires_s == 180 * 60 and ticket.expires_at == NOW.replace(hour=15)
        with world.db.read_session() as s:
            assert s.get(Job, ticket.upload_id) is None          # no job before every file is there
        world.send(ticket)
        job_id = world.uploads.complete(acc, ticket.upload_id)
        with world.db.read_session() as s:
            job = s.get(Job, job_id)
            assert (job.status, job.owner_id, job.n_inputs, job.runner) == ("queued", acc.id, 12, "modal_cpu")
            assert job.source == f"upload:{job_id}"
            assert s.get(Upload, job_id).completed_at is not None

    def test_user_file_names_are_never_object_keys(self, world):
        ticket = world.uploads.create(world.account(), [FileIn("../../etc/passwd.jpg", MB),
                                                        FileIn("Trip day 1 (א).JPG", MB)])
        assert [k for k, *_ in world.store.signed] == [f"{ticket.upload_id}/0000.jpg", f"{ticket.upload_id}/0001.jpg"]
        assert [f.name for f in ticket.files] == ["../../etc/passwd.jpg", "Trip day 1 (א).JPG"]

    def test_complete_before_every_file_arrived_names_the_missing(self, world):
        acc = world.account()
        ticket = world.uploads.create(acc, photos(3))
        world.store.put(f"{ticket.upload_id}/0000.jpg", b"x" * 4 * MB)
        with pytest.raises(UploadIncomplete) as e:
            world.uploads.complete(acc, ticket.upload_id)
        assert e.value.missing == ["IMG_0001.jpg", "IMG_0002.jpg"]
        world.send(ticket)
        assert world.uploads.complete(acc, ticket.upload_id) == ticket.upload_id

    def test_a_file_of_another_size_drops_the_upload(self, world):
        acc = world.account()
        ticket = world.uploads.create(acc, photos(2))
        world.send(ticket, sizes=[4 * MB, 26 * MB])
        with pytest.raises(UploadRefused, match="another size"):
            world.uploads.complete(acc, ticket.upload_id)
        assert world.store.list(f"{ticket.upload_id}/") == []
        with world.db.read_session() as s:
            assert s.get(Job, ticket.upload_id) is None

    def test_complete_twice_gives_the_same_job(self, world):
        acc = world.account()
        ticket = world.uploads.create(acc, photos(2))
        world.send(ticket)
        assert world.uploads.complete(acc, ticket.upload_id) == world.uploads.complete(acc, ticket.upload_id)

    def test_another_account_cannot_complete_my_upload(self, world):
        ticket = world.uploads.create(world.account(), photos(2))
        world.send(ticket)
        with pytest.raises(KeyError):
            world.uploads.complete(world.account("user_2", "bob@example.com"), ticket.upload_id)
        with pytest.raises(KeyError):
            world.uploads.complete(world.account(), "up-nonexistent")


class ut_UploadEstimate:
    def test_estimate_and_runner_by_size(self, world):
        """R6: >= 300 photos run on the GPU, fewer on Modal's CPU; the wait = upload + processing."""
        small = world.uploads.estimate(299, 299 * 4 * MB)
        large = world.uploads.estimate(300, 300 * 4 * MB)
        assert small.runner == "modal_cpu" and large.runner == "modal_gpu"
        # GPU: (150 s + 300 x 0.6 s) x 1.7 (slowest GPU in the list) = 9.35 min, at $1.10/h (dearest)
        assert large.processing_minutes == round(9.35, 1)
        assert large.dollars == round(9.35 / 60 * 1.10, 2)
        # CPU: 120 s + 299 x 3 s = 16.95 min
        assert small.processing_minutes == round(16.95, 1)
        # 1,200 MB at 20 Mbit/s = 8.4 min
        assert large.upload_minutes == round(300 * 4 * MB * 8 / 20e6 / 60, 1)

    def test_the_ticket_carries_the_estimate(self, world):
        ticket = world.uploads.create(world.account(), photos(400))
        assert ticket.estimate.runner == "modal_gpu" and ticket.estimate.processing_minutes > 0


class ut_UploadRoutes:
    @pytest.fixture
    def client(self, world, signer):
        app = create_app(verifier=AuthVerifier(ISSUER, keys=signer.keys), accounts=world.accounts,
                         uploads=world.uploads)
        return TestClient(app)

    def _auth(self, signer, sub="user_1", email="ann@example.com"):
        return {"Authorization": f"Bearer {signer.token(sub, email)}"}

    def test_upload_then_complete_over_http(self, world, client, signer):
        world.account()
        body = {"files": [{"name": "a.jpg", "size": MB, "content_type": "image/jpeg"},
                          {"name": "b.heic", "size": 2 * MB}]}
        r = client.post("/v1/uploads", json=body, headers=self._auth(signer))
        assert r.status_code == 201, r.text
        t = r.json()
        assert t["n_photos"] == 2 and t["estimate"]["runner"] == "modal_cpu"
        assert [f["content_type"] for f in t["files"]] == ["image/jpeg", "image/heic"]
        assert t["files"][0]["url"].startswith("https://r2.test/uploads/")
        assert "dollars" not in t["estimate"]                        # what we pay is not the user's business
        r = client.post(f"/v1/uploads/{t['upload_id']}/complete", headers=self._auth(signer))
        assert r.status_code == 409 and r.json()["detail"]["missing"] == ["a.jpg", "b.heic"]
        for key, size, *_ in world.store.signed:
            world.store.put(key, b"x" * size)
        r = client.post(f"/v1/uploads/{t['upload_id']}/complete", headers=self._auth(signer))
        assert r.status_code == 200 and r.json() == {"job_id": t["upload_id"], "status": "queued"}

    @pytest.mark.parametrize("files,status", [
        ([{"name": "a.exe", "size": 10}], 415),
        ([{"name": "a.jpg", "size": 30 * MB}], 413),
        ([{"name": f"{i}.jpg", "size": 10} for i in range(1001)], 400),
        ([{"name": "a.jpg", "size": -1}], 422),
    ])
    def test_refusals_over_http(self, world, client, signer, files, status):
        world.account()
        r = client.post("/v1/uploads", json={"files": files}, headers=self._auth(signer))
        assert r.status_code == status and world.n_uploads() == 0

    def test_no_consent_is_403_and_no_token_is_401(self, world, client, signer):
        world.account(consent=False)
        body = {"files": [{"name": "a.jpg", "size": 10}]}
        r = client.post("/v1/uploads", json=body, headers=self._auth(signer))
        assert r.status_code == 403 and "face grouping" in r.json()["detail"]
        assert client.post("/v1/uploads", json=body).status_code == 401
        assert client.post("/v1/uploads/x/complete").status_code == 401

    def test_another_users_upload_is_404(self, world, client, signer):
        world.account()
        world.account("user_2", "bob@example.com")
        r = client.post("/v1/uploads", json={"files": [{"name": "a.jpg", "size": 10}]}, headers=self._auth(signer))
        r = client.post(f"/v1/uploads/{r.json()['upload_id']}/complete",
                        headers=self._auth(signer, "user_2", "bob@example.com"))
        assert r.status_code == 404
