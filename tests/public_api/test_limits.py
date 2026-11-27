"""spec-120 R7 (monthly quota per account) and R8 (monthly spend cap for all accounts, alert at 80 %).

Production limits from ``configs/public.yaml`` throughout, except one test that lowers the photo quota: with the
defaults (1,000 photos an upload, 3 albums, 3,000 photos) the album limit always bites first, so the photo limit
cannot be reached without lowering it.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient

from sim_bench.control.db import ControlDB
from sim_bench.control.models import Upload
from sim_bench.public_api.accounts import AccountRepo
from sim_bench.public_api.app import create_app
from sim_bench.public_api.auth import AuthVerifier, Claims
from sim_bench.public_api.limits import (Limits, QuotaExceeded, SpendCapReached, load_limits, month_start,
                                         next_month)
from sim_bench.public_api.uploads import FileIn, UploadService
from tests.public_api.conftest import ISSUER
from tests.public_api.test_uploads import SigningStore

MB = 1024 ** 2
NOV = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)


class World:
    def __init__(self, url, tmp_path, limits=None):
        self.db = ControlDB(url)
        self.accounts = AccountRepo(self.db)
        self.store = SigningStore(tmp_path / "uploads")
        self.clock = NOV
        self.uploads = UploadService(self.db, self.store, limits or load_limits(), clock=lambda: self.clock)

    def account(self, sub="user_1", email="ann@example.com"):
        self.accounts.invite(email)
        return self.accounts.record_consent(self.accounts.get_or_create(Claims(sub, email)).id)

    def album(self, account, n=10):
        """Create, send every file, complete: one album's worth of quota."""
        ticket = self.uploads.create(account, [FileIn(f"{i}.jpg", MB) for i in range(n)])
        for key, size, *_ in self.store.signed:
            if key.startswith(ticket.upload_id) and not self.store.exists(key):
                self.store.put(key, b"x" * size)
        return self.uploads.complete(account, ticket.upload_id)

    def spent_elsewhere(self, dollars, at=None):
        """Another account's completed upload worth ``dollars`` this month."""
        other = self.account("user_x", "x@example.com")
        with self.db.session() as s:
            s.add(Upload(id=f"up-other-{dollars}", account_id=other.id, files=[], n_photos=0, total_bytes=0,
                         runner="modal_gpu", estimated_dollars=dollars, created_at=at or self.clock,
                         completed_at=at or self.clock))


@pytest.fixture
def world(control_db_url, tmp_path):
    return World(control_db_url, tmp_path)


def test_the_config_holds_the_owner_decisions():
    lim = load_limits()
    assert (lim.max_photos, lim.max_file_mb, lim.max_total_gb) == (1000, 25, 6)
    assert (lim.albums_per_month, lim.photos_per_month) == (3, 3000)
    assert (lim.spend_cap_dollars, lim.alert_fraction, lim.gpu_min_photos) == (50, 0.8, 300)
    assert set(lim.types) == {"jpeg", "heic", "png", "webp"}


def test_unknown_config_keys_are_an_error(tmp_path):
    bad = tmp_path / "public.yaml"
    bad.write_text("max_photos: 10\nmax_fotos: 20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_fotos"):
        load_limits(bad)
    with pytest.raises(ValueError, match="gif"):
        Limits(types=("jpeg", "gif"))


def test_months_are_calendar_months_in_utc():
    assert month_start(NOV) == datetime(2026, 11, 1, tzinfo=timezone.utc)
    assert next_month(NOV) == date(2026, 12, 1)
    assert next_month(datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)) == date(2027, 1, 1)


class ut_MonthlyQuota:
    def test_monthly_quota(self, world):
        """R7: the 4th album of the month gets QuotaExceeded with the reset date; nothing is stored for it."""
        acc = world.account()
        for _ in range(3):
            world.album(acc)
        before = len(world.store.signed)
        with pytest.raises(QuotaExceeded, match="3 albums this month, the limit is 3") as e:
            world.uploads.create(acc, [FileIn("a.jpg", MB)])
        assert e.value.resets_on == date(2026, 12, 1) and len(world.store.signed) == before

    def test_the_quota_resets_next_month(self, world):
        acc = world.account()
        for _ in range(3):
            world.album(acc)
        world.clock = datetime(2026, 12, 1, 0, 0, 1, tzinfo=timezone.utc)
        assert world.album(acc)

    def test_quota_is_per_account(self, world):
        ann, bob = world.account(), world.account("user_2", "bob@example.com")
        for _ in range(3):
            world.album(ann)
        assert world.album(bob)

    def test_unfinished_uploads_cost_no_quota(self, world):
        acc = world.account()
        for _ in range(5):
            world.uploads.create(acc, [FileIn("a.jpg", MB)])       # never sent, never completed
        for _ in range(3):
            world.album(acc)

    def test_quota_is_checked_again_at_complete(self, world):
        """Four uploads started together: the one completed 4th is refused, not queued."""
        acc = world.account()
        tickets = [world.uploads.create(acc, [FileIn("a.jpg", MB)]) for _ in range(4)]
        for key, size, *_ in world.store.signed:
            world.store.put(key, b"x" * size)
        for t in tickets[:3]:
            world.uploads.complete(acc, t.upload_id)
        with pytest.raises(QuotaExceeded):
            world.uploads.complete(acc, tickets[3].upload_id)

    def test_photo_quota(self, control_db_url, tmp_path):
        world = World(control_db_url, tmp_path, replace(load_limits(), photos_per_month=1500))
        acc = world.account()
        world.album(acc, n=1000)
        with pytest.raises(QuotaExceeded, match="600 photos is more than the 500 left this month"):
            world.uploads.create(acc, [FileIn(f"{i}.jpg", MB) for i in range(600)])
        assert world.album(acc, n=500)

    def test_over_quota_is_429_with_the_reset_date(self, world, signer):
        acc = world.account()
        for _ in range(3):
            world.album(acc)
        client = TestClient(create_app(verifier=AuthVerifier(ISSUER, keys=signer.keys), accounts=world.accounts,
                                       uploads=world.uploads))
        r = client.post("/v1/uploads", json={"files": [{"name": "a.jpg", "size": 10}]},
                        headers={"Authorization": f"Bearer {signer.token()}"})
        assert r.status_code == 429 and r.json()["detail"]["resets_on"] == "2026-12-01"


class ut_SpendCap:
    def test_spend_cap_and_alert(self, world, caplog):
        """R8: an upload that would pass $50 this month is refused; passing $40 (80 %) logs one alert."""
        acc = world.account()
        world.spent_elsewhere(39.95)
        caplog.set_level(logging.WARNING, logger="sim_bench.public_api.limits")
        ticket = world.uploads.create(acc, [FileIn(f"{i}.jpg", MB) for i in range(400)])   # GPU, ~$0.2
        assert not [r for r in caplog.records if "spend alert" in r.message]   # creating is only a look ahead
        for key, size, *_ in world.store.signed:
            world.store.put(key, b"x" * size)
        world.uploads.complete(acc, ticket.upload_id)
        alerts = [r for r in caplog.records if "spend alert" in r.message]
        assert len(alerts) == 1 and "of the $50.00 cap" in alerts[0].message

        world.spent_elsewhere(9.7)                                  # now ~$49.85: $0.2 more would pass $50
        with pytest.raises(SpendCapReached) as e:
            world.uploads.create(world.account("user_2", "bob@example.com"),
                                 [FileIn(f"{i}.jpg", MB) for i in range(400)])
        assert e.value.resets_on == date(2026, 12, 1)

    def test_spend_is_checked_again_at_complete(self, world):
        acc = world.account()
        ticket = world.uploads.create(acc, [FileIn(f"{i}.jpg", MB) for i in range(400)])
        world.spent_elsewhere(49.99)                                 # someone else finished meanwhile
        for key, size, *_ in world.store.signed:
            world.store.put(key, b"x" * size)
        with pytest.raises(SpendCapReached):
            world.uploads.complete(acc, ticket.upload_id)

    def test_last_months_spend_does_not_count(self, world):
        world.spent_elsewhere(50.0, at=datetime(2026, 10, 31, 23, 0, tzinfo=timezone.utc))
        assert world.uploads.spend.spent_this_month() == 0.0
        assert world.album(world.account(), n=400)

    def test_over_the_cap_is_503(self, world, signer):
        world.account()
        world.spent_elsewhere(50.0)
        client = TestClient(create_app(verifier=AuthVerifier(ISSUER, keys=signer.keys), accounts=world.accounts,
                                       uploads=world.uploads))
        r = client.post("/v1/uploads", json={"files": [{"name": "a.jpg", "size": 10}]},
                        headers={"Authorization": f"Bearer {signer.token()}"})
        assert r.status_code == 503 and "processing limit" in r.json()["detail"]["message"]
