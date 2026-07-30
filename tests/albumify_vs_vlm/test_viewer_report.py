"""Spec-102 T4/T7 — pure bits of the blind viewer and report (no image I/O)."""

import json

from sim_bench.albumify_vs_vlm.ab_viewer import _seeded_swap, build_ab_html
from sim_bench.albumify_vs_vlm.report import _annotation_section


def test_seeded_swap_is_deterministic_per_trip():
    assert _seeded_swap("budapest") == _seeded_swap("budapest")
    assert _seeded_swap("austria") == _seeded_swap("austria")


def test_build_ab_html_preserves_order_and_labels():
    html = build_ab_html("t", __import__("pathlib").Path("."),
                         "A", ["a1", "a2"], "B", ["b1"])
    # index labels present and album headers rendered
    assert "Album A" in html and "Album B" in html
    assert html.count('class="idx"') == 3


def test_annotation_section_placeholder_when_missing():
    out = _annotation_section(None)
    assert "Not yet run" in out


def test_annotation_section_renders_when_present(tmp_path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({
        "album_type": "trip", "trip_subtype": "city", "narrative": "A family city trip.",
        "n_groups": 1,
        "groups": [{"day": "2025-08-22", "label": "Basilica", "moment_type": "landmark",
                    "reason": "clean facade"}],
    }), encoding="utf-8")
    out = _annotation_section(p)
    assert "trip" in out and "city" in out
    assert "Basilica" in out and "clean facade" in out
