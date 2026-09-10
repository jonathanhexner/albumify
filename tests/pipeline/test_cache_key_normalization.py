"""SIGHTING-112: CacheKey normalizes path separators so one image → one key.

Windows-specific: normpath converts '/' → '\\' on Windows, so a folder typed with
forward slashes (studio) and one with backslashes (batch/os.path.join) collapse to
the same key. Guarded off non-Windows where normpath does not convert '\\'.
"""

import os

import pytest

pytestmark = pytest.mark.needs_data  # needs the Budapest album (private, not in CI)

from sim_bench.pipeline.cache_handler import CacheKey

_win = pytest.mark.skipif(os.name != "nt", reason="path-separator normalization is Windows-specific")


@_win
def test_forward_and_back_slash_yield_same_key():
    fwd = CacheKey("D:/album/a.jpg", "quality_niqe", "niqe")
    bwd = CacheKey("D:\\album\\a.jpg", "quality_niqe", "niqe")
    assert fwd.image_path == bwd.image_path
    assert fwd.to_string() == bwd.to_string()


@_win
def test_mixed_separator_from_os_path_join_normalizes():
    # what the studio actually produced: forward-slash folder + backslash filename
    mixed = CacheKey("D:/Budapest2025_Google\\20250822_112331.jpg", "quality_musiq", "musiq")
    batch = CacheKey("D:\\Budapest2025_Google\\20250822_112331.jpg", "quality_musiq", "musiq")
    assert mixed.image_path == batch.image_path  # studio now matches batch's stored key


def test_normalization_is_idempotent():
    k = CacheKey(os.path.join("x", "y", "a.jpg"), "f", "m")
    assert k.image_path == os.path.normpath(k.image_path)


def test_validation_still_enforced():
    with pytest.raises(ValueError):
        CacheKey("", "f", "m")
    with pytest.raises(ValueError):
        CacheKey("a.jpg", "", "m")
