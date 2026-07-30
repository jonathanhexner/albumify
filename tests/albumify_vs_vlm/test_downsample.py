"""Spec-102 T1.1 — downsample working-set builder (pure, no real datasets)."""

from pathlib import Path

import numpy as np
from PIL import Image

from sim_bench.albumify_vs_vlm.downsample import (
    DownsampleConfig,
    downsample_trip,
    input_set_hash,
)


def _make_trip(root: Path, sizes: dict[str, tuple[int, int]]) -> Path:
    src = root / "src"
    src.mkdir(parents=True)
    for stem, (w, h) in sizes.items():
        arr = (np.arange(w * h * 3, dtype=np.uint8) % 251).reshape(h, w, 3)
        Image.fromarray(arr).save(src / f"{stem}.jpg", "JPEG", quality=95)
    return src


def test_longest_edge_capped_and_ratio_preserved(tmp_path):
    src = _make_trip(tmp_path, {"20250822_112331": (1600, 1200)})
    res = downsample_trip(src, tmp_path / "out", "t", DownsampleConfig(max_edge=768))
    assert res.n_images == 1
    w, h = res.records[0].out_wh
    assert max(w, h) == 768
    assert (w, h) == (768, 576)  # 4:3 preserved


def test_small_image_not_upscaled(tmp_path):
    src = _make_trip(tmp_path, {"20250822_112331": (400, 300)})
    res = downsample_trip(src, tmp_path / "out", "t", DownsampleConfig(max_edge=768))
    assert res.records[0].out_wh == (400, 300)


def test_input_set_hash_is_deterministic_and_order_independent(tmp_path):
    src = _make_trip(
        tmp_path,
        {"20250822_112331": (200, 200), "20250822_112359": (200, 200)},
    )
    r1 = downsample_trip(src, tmp_path / "o1", "t", DownsampleConfig(max_edge=768))
    r2 = downsample_trip(src, tmp_path / "o2", "t", DownsampleConfig(max_edge=768))
    assert r1.input_set_hash == r2.input_set_hash
    # order-independent: reversed record list hashes the same
    assert input_set_hash(list(reversed(r1.records))) == r1.input_set_hash


def test_hash_changes_when_content_changes(tmp_path):
    src_a = _make_trip(tmp_path / "a", {"20250822_112331": (200, 200)})
    src_b = _make_trip(tmp_path / "b", {"20250822_112331": (200, 201)})
    ha = downsample_trip(src_a, tmp_path / "oa", "t").input_set_hash
    hb = downsample_trip(src_b, tmp_path / "ob", "t").input_set_hash
    assert ha != hb


def test_outputs_written_as_jpg(tmp_path):
    src = _make_trip(tmp_path, {"20250822_112331": (300, 300)})
    res = downsample_trip(src, tmp_path / "out", "t")
    assert Path(res.records[0].dst).exists()
    assert res.records[0].dst.endswith(".jpg")
