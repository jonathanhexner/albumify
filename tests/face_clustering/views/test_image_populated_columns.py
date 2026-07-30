"""spec-082 — Images tab only offers columns that actually have data.

Face-clustering runs have no IQA / AVA / composite / sharpness (those are
Albumify image-scoring outputs), so the Images table was showing a wall of
None. ``populated_columns`` is the rule that drops them; tested here without
Streamlit.
"""
from __future__ import annotations

from types import SimpleNamespace

from face_cluster.views.image_metrics import (
    IMAGE_METRIC_COLUMNS,
    populated_columns,
)


def _row(**kw):
    base = dict(
        n_faces=2, filter_passed=True, composite_score=None, iqa_score=None,
        ava_score=None, sharpness_score=None, width_px=4000, height_px=3000,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_drops_all_none_columns():
    rows = [_row(), _row(n_faces=0)]
    labels = [c.label for c in populated_columns(rows)]
    # present (non-None across the rows)
    assert {"Faces", "Gate passed", "Width", "Height"} <= set(labels)
    # dropped (None for every row in a face-clustering run)
    assert "Composite" not in labels
    assert "IQA" not in labels
    assert "AVA" not in labels
    assert "Sharpness" not in labels


def test_keeps_a_column_with_any_value():
    rows = [_row(composite_score=None), _row(composite_score=0.42)]
    labels = [c.label for c in populated_columns(rows)]
    assert "Composite" in labels  # one non-None value is enough


def test_empty_rows_yields_no_columns():
    assert populated_columns([]) == []


def test_default_columns_are_a_subset_of_the_registry():
    # guards against a default label that no ColumnSpec can supply
    registry = {c.label for c in IMAGE_METRIC_COLUMNS}
    from face_cluster.views.image_metrics import DEFAULT_IMAGE_COLUMNS
    assert set(DEFAULT_IMAGE_COLUMNS) <= registry
