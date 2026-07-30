"""Unit tests for the Library upload client + album export (spec-092). No network."""
from __future__ import annotations

from pathlib import Path

import pytest

from gphotos.export_album import export_album_to_google_photos
from gphotos.uploader import BATCH_LIMIT, LibraryClient


class FakeResponse:
    def __init__(self, *, status=200, text="", json_data=None, headers=None):
        self.status_code = status
        self.text = text
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Returns queued responses per URL substring (FIFO)."""

    def __init__(self):
        self.calls = []
        self.queues = {}  # substring -> [responses]

    def queue(self, substr, *responses):
        self.queues.setdefault(substr, []).extend(responses)

    def post(self, url, **kw):
        self.calls.append((url, kw))
        for substr, q in self.queues.items():
            if substr in url and q:
                return q.pop(0)
        return FakeResponse()


class ut_GphotosUploader:
    def test_upload_returns_token(self, tmp_path):
        f = tmp_path / "a.jpg"
        f.write_bytes(b"JPEG")
        s = FakeSession()
        s.queue("/uploads", FakeResponse(text="TOKEN123"))
        assert LibraryClient(session=s).upload_bytes(f) == "TOKEN123"
        # raw-upload headers were sent
        assert s.calls[0][1]["headers"]["X-Goog-Upload-Protocol"] == "raw"

    def test_create_album_returns_id(self):
        s = FakeSession()
        s.queue("/albums", FakeResponse(json_data={"id": "ALB1", "title": "T"}))
        assert LibraryClient(session=s).create_album("T") == "ALB1"

    def test_batch_create_counts_successes(self):
        s = FakeSession()
        s.queue(
            ":batchCreate",
            FakeResponse(json_data={"newMediaItemResults": [
                {"mediaItem": {"id": "1"}},
                {"status": {"message": "fail"}},  # no mediaItem -> not counted
                {"mediaItem": {"id": "3"}},
            ]}),
        )
        assert LibraryClient(session=s).batch_create("ALB1", [("t", "n")]) == 2

    def test_retries_on_429_then_succeeds(self):
        s = FakeSession()
        s.queue("/albums", FakeResponse(status=429), FakeResponse(json_data={"id": "ALB1"}))
        slept = []
        client = LibraryClient(session=s, sleep=slept.append)
        assert client.create_album("T") == "ALB1"
        assert len(slept) == 1  # backed off once


class FakeClient:
    """Stands in for LibraryClient through the export flow."""

    def __init__(self):
        self.batches = []
        self.uploaded = []
        self.created_albums = []

    def create_album(self, title):
        self.created_albums.append(title)
        return "ALB1"

    def upload_bytes(self, path):
        self.uploaded.append(str(path))
        return f"tok::{Path(path).name}"

    def batch_create(self, album_id, items):
        self.batches.append(len(items))
        return len(items)


def _imgs(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"img_{i}.jpg"
        p.write_bytes(b"x")
        out.append(p)
    return out


class ut_GphotosExport:
    def test_uploads_and_creates(self, tmp_path):
        fake = FakeClient()
        res = export_album_to_google_photos(
            _imgs(tmp_path, 3), "Trip", client=fake,
            manifest_path=tmp_path / "m.json",
        )
        assert res.album_id == "ALB1"
        assert res.uploaded == 3 and res.created == 3 and res.skipped == 0
        assert fake.created_albums == ["Trip"]

    def test_batches_in_chunks_of_50(self, tmp_path):
        fake = FakeClient()
        export_album_to_google_photos(
            _imgs(tmp_path, BATCH_LIMIT + 10), "Big", client=fake,
            manifest_path=tmp_path / "m.json",
        )
        assert fake.batches == [BATCH_LIMIT, 10]

    def test_idempotent_skips_uploaded(self, tmp_path):
        imgs = _imgs(tmp_path, 2)
        man = tmp_path / "m.json"
        export_album_to_google_photos(imgs, "Trip", client=FakeClient(), manifest_path=man)
        fake2 = FakeClient()
        res = export_album_to_google_photos(imgs, "Trip", client=fake2, manifest_path=man)
        assert res.skipped == 2 and res.uploaded == 0
        assert fake2.uploaded == []          # nothing re-uploaded
        assert fake2.created_albums == []     # album reused, not recreated

    def test_missing_file_recorded_as_failed(self, tmp_path):
        fake = FakeClient()
        res = export_album_to_google_photos(
            [tmp_path / "gone.jpg"], "Trip", client=fake, manifest_path=tmp_path / "m.json"
        )
        assert res.failed == [str(tmp_path / "gone.jpg")]
        assert res.uploaded == 0
