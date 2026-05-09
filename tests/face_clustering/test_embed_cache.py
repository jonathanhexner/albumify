"""Unit tests for face_cluster.cache (embed cache infrastructure)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

from face_cluster.cache import (
    EMBED_CACHE_VERSION,
    clear_embed_cache,
    compute_image_fingerprint,
    get_cache_info,
    load_embed_cache,
    save_embed_cache,
    validate_cache,
)
from face_cluster.types import FaceRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _make_face(face_id: int) -> FaceRecord:
    return FaceRecord(
        face_id=face_id,
        image_id="img",
        bbox=(0.0, 0.0, 100.0, 100.0),
        area=10000.0,
        blur_score=80.0,
        image_path="img.jpg",
        is_core=False,
        embedding_normalized=np.random.rand(512).astype(np.float32),
    )


def _write_dummy_image(path: Path, size_bytes: int = 100) -> None:
    path.write_bytes(b"\xff\xd8\xff" + b"\x00" * (size_bytes - 3))


# ---------------------------------------------------------------------------
# ut_Fingerprint
# ---------------------------------------------------------------------------


class ut_Fingerprint:

    def test_stable_for_same_images(self, tmp_path):
        """Same image set returns the same fingerprint on two consecutive calls."""
        _write_dummy_image(tmp_path / "photo.jpg")
        fp1 = compute_image_fingerprint(tmp_path, _EXTENSIONS)
        fp2 = compute_image_fingerprint(tmp_path, _EXTENSIONS)
        assert fp1 == fp2
        assert fp1.startswith("sha256:")

    def test_changes_on_image_add(self, tmp_path):
        """Adding a new image changes the fingerprint."""
        _write_dummy_image(tmp_path / "photo1.jpg")
        fp_before = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        _write_dummy_image(tmp_path / "photo2.jpg")
        fp_after = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        assert fp_before != fp_after

    def test_changes_on_image_remove(self, tmp_path):
        """Removing an image changes the fingerprint."""
        _write_dummy_image(tmp_path / "photo1.jpg")
        _write_dummy_image(tmp_path / "photo2.jpg")
        fp_before = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        (tmp_path / "photo2.jpg").unlink()
        fp_after = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        assert fp_before != fp_after

    def test_changes_on_size_change(self, tmp_path):
        """Modifying an image file (size change) changes the fingerprint."""
        img = tmp_path / "photo.jpg"
        _write_dummy_image(img, size_bytes=100)
        fp_before = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        _write_dummy_image(img, size_bytes=200)
        fp_after = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        assert fp_before != fp_after

    def test_ignores_non_image_files(self, tmp_path):
        """Non-image files (.txt etc.) are excluded from the fingerprint."""
        _write_dummy_image(tmp_path / "photo.jpg")
        fp_before = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        fp_after = compute_image_fingerprint(tmp_path, _EXTENSIONS)

        assert fp_before == fp_after

    def test_empty_dir_is_stable(self, tmp_path):
        """Empty directory returns a stable, non-crashing fingerprint."""
        fp1 = compute_image_fingerprint(tmp_path, _EXTENSIONS)
        fp2 = compute_image_fingerprint(tmp_path, _EXTENSIONS)
        assert fp1 == fp2
        assert fp1.startswith("sha256:")


# ---------------------------------------------------------------------------
# ut_CacheSaveLoad
# ---------------------------------------------------------------------------


class ut_CacheSaveLoad:

    def test_round_trip(self, tmp_path):
        """save_embed_cache + load_embed_cache returns the same faces and meta."""
        faces = [_make_face(i) for i in range(5)]
        fp = "sha256:abc123"
        save_embed_cache(tmp_path, faces, fp, tmp_path, n_images=2, embed_time_s=10.0)

        result = load_embed_cache(tmp_path)
        assert result is not None
        loaded_faces, meta = result

        assert len(loaded_faces) == 5
        assert meta["n_faces"] == 5
        assert meta["image_fingerprint"] == fp
        assert meta["embed_cache_version"] == EMBED_CACHE_VERSION
        assert "created_at" in meta
        assert meta["embed_time_seconds"] == 10.0

    def test_embeddings_preserved_exactly(self, tmp_path):
        """Embedding values survive the round trip without modification."""
        faces = [_make_face(0)]
        expected = faces[0].embedding_normalized.copy()
        save_embed_cache(tmp_path, faces, "sha256:x", tmp_path, n_images=1, embed_time_s=5.0)

        loaded_faces, _ = load_embed_cache(tmp_path)
        np.testing.assert_array_equal(loaded_faces[0].embedding_normalized, expected)

    def test_load_missing_returns_none(self, tmp_path):
        """load_embed_cache returns None when no cache exists."""
        assert load_embed_cache(tmp_path) is None

    def test_second_save_replaces_first(self, tmp_path):
        """A second save atomically replaces the first cache."""
        faces_v1 = [_make_face(i) for i in range(3)]
        faces_v2 = [_make_face(i) for i in range(7)]
        fp = "sha256:test"

        save_embed_cache(tmp_path, faces_v1, fp, tmp_path, n_images=1, embed_time_s=5.0)
        save_embed_cache(tmp_path, faces_v2, fp, tmp_path, n_images=1, embed_time_s=5.0)

        loaded_faces, meta = load_embed_cache(tmp_path)
        assert len(loaded_faces) == 7
        assert meta["n_faces"] == 7

    def test_corrupt_meta_returns_none(self, tmp_path):
        """Corrupt cache_meta.json causes load_embed_cache to return None (not raise)."""
        cache_dir = tmp_path / ".embed_cache"
        cache_dir.mkdir()
        (cache_dir / "cache_meta.json").write_text("not json", encoding="utf-8")
        (cache_dir / "faces_cache.pkl").write_bytes(b"")

        result = load_embed_cache(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# ut_CacheValidation
# ---------------------------------------------------------------------------


class ut_CacheValidation:

    def _valid_faces(self) -> List[FaceRecord]:
        return [_make_face(0)]

    def _valid_meta(self, fp: str) -> dict:
        return {
            "embed_cache_version": EMBED_CACHE_VERSION,
            "image_fingerprint": fp,
        }

    def test_valid_cache_passes(self):
        fp = "sha256:abc"
        valid, reason = validate_cache(self._valid_meta(fp), self._valid_faces(), fp)
        assert valid, reason
        assert reason == "ok"

    def test_rejects_version_mismatch(self):
        fp = "sha256:abc"
        meta = {"embed_cache_version": "0", "image_fingerprint": fp}
        valid, reason = validate_cache(meta, self._valid_faces(), fp)
        assert not valid
        assert "version mismatch" in reason

    def test_rejects_fingerprint_mismatch(self):
        meta = {
            "embed_cache_version": EMBED_CACHE_VERSION,
            "image_fingerprint": "sha256:old",
        }
        valid, reason = validate_cache(meta, self._valid_faces(), "sha256:new")
        assert not valid
        assert "fingerprint" in reason

    def test_rejects_empty_faces(self):
        fp = "sha256:abc"
        valid, reason = validate_cache(self._valid_meta(fp), [], fp)
        assert not valid
        assert "empty" in reason

    def test_rejects_missing_embedding(self):
        fp = "sha256:abc"
        face = FaceRecord(
            face_id=0, image_id="img", bbox=(0, 0, 1, 1), area=1.0,
            blur_score=0.0, image_path="img.jpg", is_core=False,
            embedding_normalized=None,
        )
        valid, reason = validate_cache(self._valid_meta(fp), [face], fp)
        assert not valid
        assert "embedding" in reason

    def test_rejects_bad_embedding_shape(self):
        fp = "sha256:abc"
        face = FaceRecord(
            face_id=0, image_id="img", bbox=(0, 0, 1, 1), area=1.0,
            blur_score=0.0, image_path="img.jpg", is_core=False,
            embedding_normalized=np.zeros(128, dtype=np.float32),
        )
        valid, reason = validate_cache(self._valid_meta(fp), [face], fp)
        assert not valid
        assert "shape" in reason


# ---------------------------------------------------------------------------
# ut_CacheManagement
# ---------------------------------------------------------------------------


class ut_CacheManagement:

    def test_clear_existing_cache_returns_true(self, tmp_path):
        """clear_embed_cache returns True and removes the cache directory."""
        save_embed_cache(
            tmp_path, [_make_face(0)], "sha256:x", tmp_path, n_images=1, embed_time_s=1.0
        )
        assert clear_embed_cache(tmp_path) is True
        assert load_embed_cache(tmp_path) is None

    def test_clear_missing_cache_returns_false(self, tmp_path):
        """clear_embed_cache returns False when no cache directory exists."""
        assert clear_embed_cache(tmp_path) is False

    def test_get_cache_info_returns_meta(self, tmp_path):
        """get_cache_info returns the meta dict when a cache exists."""
        save_embed_cache(
            tmp_path, [_make_face(0)], "sha256:x", tmp_path, n_images=1, embed_time_s=3.5
        )
        info = get_cache_info(tmp_path)
        assert info is not None
        assert info["n_faces"] == 1
        assert info["embed_time_seconds"] == 3.5

    def test_get_cache_info_returns_none_when_absent(self, tmp_path):
        """get_cache_info returns None when no cache exists."""
        assert get_cache_info(tmp_path) is None
