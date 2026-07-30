"""ut for occlusion_bench.dataset (spec-096 T1.3)."""

import os
import shutil

from PIL import Image

from sim_bench.occlusion_bench import dataset as ds


def _img(path, color):
    Image.new("RGB", (24, 24), color).save(path, "JPEG")


def _setup(tmp_path):
    pos = tmp_path / "pos"; neg = tmp_path / "neg"
    pos.mkdir(); neg.mkdir()
    _img(str(pos / "a.jpg"), (200, 60, 60))
    _img(str(pos / "b.jpg"), (60, 200, 60))
    _img(str(neg / "c.jpg"), (60, 60, 200))
    # duplicate of a positive placed among negatives (the finger-in-album case)
    shutil.copy2(str(pos / "a.jpg"), str(neg / "a_copy.jpg"))
    return [ds.SourceDir("occl", str(pos), 1), ds.SourceDir("alb", str(neg), 0)]


def test_build_prefix_dedupe_manifest(tmp_path):
    out = str(tmp_path / "out")
    res = ds.build(_setup(tmp_path), out, hard_negatives={"c.jpg"})

    assert res.n_positives == 2 and res.n_negatives == 1
    assert res.n_duplicates_skipped == 1  # a_copy deduped, stayed a positive

    rows = ds.load_manifest(out)
    ids = {r["id"] for r in rows}
    assert ids == {"occl__a.jpg", "occl__b.jpg", "alb__c.jpg"}
    assert os.path.isfile(os.path.join(out, "positives", "occl__a.jpg"))
    assert os.path.isfile(os.path.join(out, "negatives", "alb__c.jpg"))

    by_id = {r["id"]: r for r in rows}
    assert by_id["alb__c.jpg"]["hard_negative"] == "True"
    assert by_id["occl__a.jpg"]["source_path"].endswith("a.jpg")
    assert all(r["split"] in ("train", "test") for r in rows)


def test_split_is_deterministic(tmp_path):
    out1 = str(tmp_path / "o1"); out2 = str(tmp_path / "o2")
    srcs = _setup(tmp_path)
    s1 = {r["id"]: r["split"] for r in (ds.build(srcs, out1), ds.load_manifest(out1))[1]}
    s2 = {r["id"]: r["split"] for r in (ds.build(srcs, out2), ds.load_manifest(out2))[1]}
    assert s1 == s2  # sha1-derived, no RNG


def test_group_split_keeps_bursts_together(tmp_path):
    """Near-duplicates (burst frames) must land in the SAME split (T1.4)."""
    from sim_bench.occlusion_bench.dataset import apply_group_split, load_manifest
    pos = tmp_path / "pos"; neg = tmp_path / "neg"
    pos.mkdir(); neg.mkdir()
    # a "burst": same scene, filename timestamps 3s apart (near-dupe pixels too)
    _img(str(pos / "20260705_215104.jpg"), (200, 60, 60))
    _img(str(pos / "20260705_215107.jpg"), (201, 61, 61))
    _img(str(neg / "20250822_120000.jpg"), (60, 60, 200))
    out = str(tmp_path / "out")
    ds.build([ds.SourceDir("occl", str(pos), 1), ds.SourceDir("alb", str(neg), 0)], out)
    apply_group_split(out)
    rows = {r["id"]: r for r in load_manifest(out)}
    a = rows["occl__20260705_215104.jpg"]; b = rows["occl__20260705_215107.jpg"]
    assert a["group_id"] == b["group_id"]          # burst grouped
    assert a["split"] == b["split"]                # -> same split, no leakage
    assert rows["alb__20250822_120000.jpg"]["group_id"] != a["group_id"]
