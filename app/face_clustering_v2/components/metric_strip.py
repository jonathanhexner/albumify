"""spec-072 — declarative metric strip.

Renders a row of ``st.metric`` widgets from a ``List[ColumnSpec]`` — the
same registry the tabular renderers use (``rows_to_records``). One
declaration (e.g. ``FACE_METRIC_COLUMNS``), two renderers: this strip and
the sortable table. Replaces the hand-written ``c1.metric(...)`` blocks.
"""
from __future__ import annotations

from typing import Any, Sequence

import streamlit as st

from face_cluster.views._specs import ColumnSpec

_MISSING = "—"


def render_metric_strip(
    obj: Any,
    columns: Sequence[ColumnSpec],
    *,
    n_cols: int | None = None,
) -> None:
    """Render each ColumnSpec as an ``st.metric`` against ``obj``.

    A column whose attribute is absent / None on ``obj`` shows ``—`` (so the
    same list can drive objects that expose only a subset — e.g. ``FaceView``
    has no ``det_score``). ``columns`` is read via ``ColumnSpec.display`` /
    ``.help`` exactly like the table path.

    Args:
        obj: the row/view object to read attributes from.
        columns: the metric specs to render, in order.
        n_cols: columns per row; defaults to ``len(columns)`` (single row).
    """
    specs = list(columns)
    if not specs:
        return
    per_row = n_cols or len(specs)
    cols = st.columns(per_row)
    for i, spec in enumerate(specs):
        value = spec.display(obj)
        cols[i % per_row].metric(spec.label, value or _MISSING, help=spec.help)
