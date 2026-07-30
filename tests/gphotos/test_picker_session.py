"""Unit tests for the Picker client and types (spec-092). No network."""
from __future__ import annotations

import pytest

from gphotos.picker import DOWNLOAD_SUFFIX, PickerClient
from gphotos.types import PickedItem, PickerSession, PollingConfig, _parse_duration


class FakeResponse:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data or {}
        self.content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


class FakeSession:
    """Records requests and returns queued responses (FIFO)."""

    def __init__(self):
        self.calls = []
        self.get_responses = []
        self.post_responses = []

    def post(self, url, json=None, **kw):
        self.calls.append(("POST", url, json))
        return self.post_responses.pop(0)

    def get(self, url, params=None, **kw):
        self.calls.append(("GET", url, params))
        return self.get_responses.pop(0)

    def delete(self, url, **kw):
        self.calls.append(("DELETE", url, None))
        return FakeResponse()


class ut_PickerSession:
    def test_create_session_parses_polling(self):
        fake = FakeSession()
        fake.post_responses = [
            FakeResponse(
                {
                    "id": "S1",
                    "pickerUri": "https://picker/x",
                    "pollingConfig": {"pollInterval": "3.5s", "timeoutIn": "60s"},
                    "mediaItemsSet": False,
                }
            )
        ]
        sess = PickerClient(session=fake).create_session()
        assert sess.id == "S1"
        assert sess.picker_uri == "https://picker/x"
        assert sess.polling.poll_interval_s == 3.5
        assert sess.polling.timeout_s == 60.0
        assert fake.calls[0][0] == "POST"

    def test_poll_until_ready_loops_until_set(self):
        fake = FakeSession()
        fake.get_responses = [
            FakeResponse({"id": "S1", "mediaItemsSet": False}),
            FakeResponse({"id": "S1", "mediaItemsSet": True}),
        ]
        start = PickerSession(id="S1", picker_uri="", media_items_set=False)
        slept = []
        ready = PickerClient(session=fake).poll_until_ready(start, sleep=slept.append)
        assert ready.media_items_set is True
        assert len(slept) == 2  # slept before each re-fetch

    def test_poll_times_out(self):
        fake = FakeSession()
        # session never becomes ready; loop must give up once the budget is spent
        fake.get_responses = [
            FakeResponse({"id": "S1", "mediaItemsSet": False}) for _ in range(5)
        ]
        start = PickerSession(
            id="S1",
            picker_uri="",
            media_items_set=False,
            polling=PollingConfig(poll_interval_s=1, timeout_s=2),
        )
        with pytest.raises(TimeoutError):
            PickerClient(session=fake).poll_until_ready(start, sleep=lambda s: None)

    def test_list_media_items_paginates(self):
        fake = FakeSession()
        fake.get_responses = [
            FakeResponse(
                {
                    "mediaItems": [
                        {
                            "id": "a",
                            "type": "PHOTO",
                            "mediaFile": {
                                "baseUrl": "https://b/a",
                                "filename": "a.jpg",
                                "mimeType": "image/jpeg",
                            },
                        }
                    ],
                    "nextPageToken": "t2",
                }
            ),
            FakeResponse(
                {
                    "mediaItems": [
                        {
                            "id": "b",
                            "type": "PHOTO",
                            "mediaFile": {
                                "baseUrl": "https://b/b",
                                "filename": "b.jpg",
                                "mimeType": "image/jpeg",
                            },
                        }
                    ]
                }
            ),
        ]
        items = PickerClient(session=fake).list_media_items("S1")
        assert [i.id for i in items] == ["a", "b"]
        assert items[0].filename == "a.jpg"
        assert fake.calls[1][2].get("pageToken") == "t2"  # token carried forward

    def test_download_uses_d_suffix_and_writes_bytes(self, tmp_path):
        fake = FakeSession()
        fake.get_responses = [FakeResponse(content=b"JPEGDATA")]
        item = PickedItem(
            id="a", filename="a.jpg", base_url="https://b/a", mime_type="image/jpeg"
        )
        dest = tmp_path / "a.jpg"
        PickerClient(session=fake).download_item(item, dest)
        assert dest.read_bytes() == b"JPEGDATA"
        assert fake.calls[0][1].endswith(DOWNLOAD_SUFFIX)


class ut_DurationParse:
    def test_parses_seconds(self):
        assert _parse_duration("3.5s", 1.0) == 3.5

    def test_default_on_missing(self):
        assert _parse_duration(None, 7.0) == 7.0

    def test_default_on_garbage(self):
        assert _parse_duration("bogus", 9.0) == 9.0
