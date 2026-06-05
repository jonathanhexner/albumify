"""spec-072 — the face-metric display registry (single source of truth).

Verifies FACE_METRIC_COLUMNS reads both row shapes via canonical attribute
names, that Area % is derived from area_ratio, and that the strip-style
display tolerates missing/None metrics (renders empty -> the component
shows the em-dash).
"""
from __future__ import annotations

from face_cluster.views.face_metrics import FACE_METRIC_COLUMNS, FaceMetricRow


def _row(**over):
    base = dict(
        face_id=1, status="assigned", disposition="clustered", cluster_id=1, blur=1204.0, area=345290.0,
        area_ratio=0.052, det_score=0.93, yaw=5.0, pitch=-3.0, roll=1.0,
        rejection_reason=None, crop_path=None,
    )
    base.update(over)
    return FaceMetricRow(**base)


def test_registry_labels_are_the_expected_metrics():
    labels = [c.label for c in FACE_METRIC_COLUMNS]
    assert labels == ["Blur", "Area (px)", "Area %", "Det score", "Yaw", "Pitch", "Roll"]


def test_area_pct_is_derived_from_area_ratio():
    r = _row(area_ratio=0.052)
    pct = next(c for c in FACE_METRIC_COLUMNS if c.label == "Area %")
    assert pct.read(r) == 5.2                 # numeric -> table sorts correctly
    assert pct.display(r) == "5.2%"           # formatted -> strip


def test_read_returns_raw_numeric_for_table_sorting():
    r = _row(blur=1204.0)
    blur = next(c for c in FACE_METRIC_COLUMNS if c.label == "Blur")
    assert blur.read(r) == 1204.0 and isinstance(blur.read(r), float)


def test_none_metric_displays_empty():
    r = _row(det_score=None, area_ratio=None)
    det = next(c for c in FACE_METRIC_COLUMNS if c.label == "Det score")
    pct = next(c for c in FACE_METRIC_COLUMNS if c.label == "Area %")
    assert det.read(r) is None and det.display(r) == ""   # component renders "-"
    assert pct.read(r) is None and pct.display(r) == ""


def test_zero_value_is_not_dropped():
    # blur of exactly 0.0 must still read as 0.0, not None (falsy-but-present).
    r = _row(blur=0.0)
    blur = next(c for c in FACE_METRIC_COLUMNS if c.label == "Blur")
    assert blur.read(r) == 0.0


def test_specs_carry_help_tooltips():
    assert all(c.help for c in FACE_METRIC_COLUMNS)
