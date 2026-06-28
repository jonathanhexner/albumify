"""Unit tests for the download cache (spec-092). No network."""
from __future__ import annotations

from pathlib import Path

from gphotos.cache import DownloadCache
from gphotos.types import PickedItem


class FakeClient:
    """Stands in for PickerClient.download_item; writes a marker file."""

    def __init__(self):
        self.downloads = []

    def download_item(self, item, dest: Path) -> Path:
        self.downloads.append(item.id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest


def _item(i, name):
    return PickedItem(
        id=i, filename=name, base_url=f"https://b/{i}", mime_type="image/jpeg"
    )


class ut_GphotosCache:
    def test_materialize_downloads_all(self, tmp_path):
        client = FakeClient()
        items = [_item("1", "a.jpg"), _item("2", "b.jpg")]
        out = DownloadCache(tmp_path).materialize(client, items)
        assert len(out) == 2
        assert client.downloads == ["1", "2"]
        assert all(Path(d.local_path).exists() for d in out)

    def test_idempotent_skips_existing(self, tmp_path):
        cache = DownloadCache(tmp_path)
        items = [_item("1", "a.jpg")]
        cache.materialize(FakeClient(), items)
        client2 = FakeClient()
        cache.materialize(client2, items)
        assert client2.downloads == []  # second run hit the cache

    def test_filename_collision_disambiguated(self, tmp_path):
        client = FakeClient()
        items = [_item("1", "same.jpg"), _item("2", "same.jpg")]
        out = DownloadCache(tmp_path).materialize(client, items)
        names = {Path(d.local_path).name for d in out}
        assert len(names) == 2  # distinct files on disk
