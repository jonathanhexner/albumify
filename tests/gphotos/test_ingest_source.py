"""Unit tests for the ingest adapter (spec-092). No network, no browser."""
from __future__ import annotations

from pathlib import Path

from gphotos.ingest_source import import_from_google_photos
from gphotos.types import PickedItem, PickerSession


class FakeClient:
    """Stands in for PickerClient through the whole import flow."""

    def __init__(self, n_items=1):
        self.n_items = n_items
        self.deleted = None
        self.waited = False

    def create_session(self):
        return PickerSession(id="S1", picker_uri="https://pick/x", media_items_set=True)

    def poll_until_ready(self, session, on_wait=None):
        self.waited = True
        return session

    def list_media_items(self, session_id, page_size=100):
        return [
            PickedItem(
                id=str(i),
                filename=f"img_{i}.jpg",
                base_url=f"https://b/{i}",
                mime_type="image/jpeg",
            )
            for i in range(self.n_items)
        ]

    def download_item(self, item, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    def delete_session(self, session_id):
        self.deleted = session_id


class ut_IngestSource:
    def test_import_returns_source_dir_and_items(self, tmp_path):
        fake = FakeClient(n_items=3)
        res = import_from_google_photos(
            out_dir=tmp_path, client=fake, on_picker_url=lambda u: None
        )
        assert res.count == 3
        assert res.source_directory == Path(tmp_path)
        assert sorted(p.name for p in Path(tmp_path).glob("*.jpg")) == [
            "img_0.jpg",
            "img_1.jpg",
            "img_2.jpg",
        ]

    def test_cleans_up_session(self, tmp_path):
        fake = FakeClient()
        import_from_google_photos(
            out_dir=tmp_path, client=fake, on_picker_url=lambda u: None
        )
        assert fake.deleted == "S1"
        assert fake.waited is True

    def test_picker_url_callback_used_instead_of_browser(self, tmp_path):
        seen = []
        import_from_google_photos(
            out_dir=tmp_path, client=FakeClient(), on_picker_url=seen.append
        )
        assert seen == ["https://pick/x"]
