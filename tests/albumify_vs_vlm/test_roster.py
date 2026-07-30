"""Spec-102 T1.2 — coverage-roster template (day segmentation + validation)."""

from sim_bench.albumify_vs_vlm.downsample import DownsampleResult, ImageRecord
from sim_bench.albumify_vs_vlm.roster import build_roster_template, validate_roster


def _rec(stem: str) -> ImageRecord:
    return ImageRecord(
        src=f"{stem}.jpg", dst=f"out/{stem}.jpg", stem=stem,
        orig_wh=(100, 100), out_wh=(100, 100), content_sha256="x",
    )


def _result(stems: list[str]) -> DownsampleResult:
    return DownsampleResult(
        trip="t", out_dir="out", config=None,
        records=[_rec(s) for s in stems], input_set_hash="HASH",
    )


def test_days_segmented_from_filename_timestamps():
    res = _result(["20250822_112331", "20250822_190000", "20250823_090000"])
    roster = build_roster_template(res)
    days = {d["date"]: d["images"] for d in roster["days"]}
    assert set(days) == {"2025-08-22", "2025-08-23"}
    assert len(days["2025-08-22"]) == 2
    assert days["2025-08-23"] == ["20250823_090000"]


def test_carries_input_set_hash_and_count():
    roster = build_roster_template(_result(["20250822_112331"]))
    assert roster["input_set_hash"] == "HASH"
    assert roster["n_images"] == 1


def test_untimestamped_file_bucketed_unknown():
    roster = build_roster_template(_result(["random_name"]))
    dates = {d["date"] for d in roster["days"]}
    assert "unknown" in dates


def test_validate_flags_unlabelled_roster():
    roster = build_roster_template(_result(["20250822_112331"]))
    errors = validate_roster(roster, expected_hash="HASH")
    # template has empty persons/scenes -> both coverage checks fire
    assert any("person" in e for e in errors)
    assert any("scene" in e for e in errors)


def test_validate_flags_hash_mismatch():
    roster = build_roster_template(_result(["20250822_112331"]))
    errors = validate_roster(roster, expected_hash="DIFFERENT")
    assert any("input_set_hash" in e for e in errors)


def test_validate_passes_when_labelled():
    roster = build_roster_template(_result(["20250822_112331"]))
    roster["persons"] = [{"id": "P1", "name": "Ana", "images": ["20250822_112331"]}]
    roster["scenes"] = [{"id": "S1", "label": "Castle", "day": "2025-08-22",
                         "images": ["20250822_112331"]}]
    assert validate_roster(roster, expected_hash="HASH") == []
