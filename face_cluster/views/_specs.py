"""Shared declarative-spec types for the v2 view layer.

These types are the "vocabulary" every view service speaks. When a view
needs to display a tabular list, it declares a list of ``ColumnSpec``
and one renderer iterates over it — no per-column literal code. When a
view needs to format an action-type-specific summary, it dispatches via
``ActionTypeFormat`` rather than an inline dict-of-lambdas.

Pattern is the same as spec-041's ``UI_SPEC``/widget_factory for params:
one declarative list, one renderer, zero duplication.

This module is **Streamlit-free** — it's pure typed metadata. Streamlit
rendering happens in ``app/face_clustering_v2/components/`` against
these specs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column in a tabular display.

    Used by:
    - ``render_run_table`` (History, generic-table renderer)
    - any future tabular view (Cluster Analysis, Merged Clusters, …)

    Reads from a row object via ``field`` (an attribute name); falls
    back through ``fallback_fields`` in order if the primary attribute
    is missing or falsy. The optional ``formatter`` post-processes the
    value to a display string.

    Args:
        field: primary attribute name on the row object.
        label: human-readable column header.
        fallback_fields: try these attribute names in order if ``field``
            is missing or falsy. Empty by default — no fallback.
        formatter: optional ``Callable[[Any], str]`` that turns the
            attribute value into a display string. Default is ``str()``;
            ``None`` values render as empty string.
        help: optional tooltip text. Ignored by table renderers; used by
            ``render_metric_strip`` when the same spec drives a metric
            strip (spec-072).
        getter: optional ``Callable[[row], value]`` — when set, the value is
            COMPUTED from the row (e.g. ``len(x)``, a PASS/FAIL string) instead
            of read from an attribute. ``field``/``fallback_fields`` are then
            ignored (spec-078).
        delta: optional ``Callable[[row], str]`` — a metric-strip delta line
            (the small text under an ``st.metric`` value). Strips only.
    """
    field: str
    label: str
    fallback_fields: Tuple[str, ...] = ()
    formatter: Optional[Callable[[Any], str]] = None
    help: Optional[str] = None
    getter: Optional[Callable[[Any], Any]] = None
    delta: Optional[Callable[[Any], str]] = None

    def read(self, row: Any) -> Any:
        """Return the raw value for this column from a row object.

        When ``getter`` is set, returns ``getter(row)`` directly. Otherwise
        tries ``field`` first; on None / missing / falsy, tries each
        ``fallback_fields`` entry in order. Returns the first non-None
        non-empty value, or None if all fields are absent.
        """
        if self.getter is not None:
            return self.getter(row)
        for name in (self.field, *self.fallback_fields):
            if hasattr(row, name):
                v = getattr(row, name)
                if v is not None and v != "":
                    return v
        return None

    def display(self, row: Any) -> str:
        """Return the formatted display string for this column."""
        value = self.read(row)
        if value is None:
            return ""
        if self.formatter is not None:
            return self.formatter(value)
        return str(value)


def rows_to_records(
    rows: Iterable[Any],
    columns: Iterable[ColumnSpec],
) -> list[dict[str, str]]:
    """Render a list of row objects through a column spec into display dicts.

    Each output dict maps ``column.label -> column.display(row)``. Used
    by tabular components to construct the DataFrame Streamlit displays.
    """
    cols = list(columns)
    return [{c.label: c.display(row) for c in cols} for row in rows]
