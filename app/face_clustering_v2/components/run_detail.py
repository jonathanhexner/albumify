"""spec-042 H2 — run detail panel for the History tab.

Renders the selected run's header, config view (read-only widgets via
``widget_factory.render_field(readonly=True)`` — zero ``cfg.get('field')``
literals), summary, log viewer, payload viewer, and data-files panel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from app.face_clustering_v2.components.metric_strip import render_metric_strip
from app.face_clustering_v2.ui_spec import UI_SPEC, fields_by_group
from app.face_clustering_v2.widget_factory import render_field
from face_cluster.views.metric_specs import RUN_SUMMARY_STRIP
from face_cluster.views.history import HistoryService, RunDetail, RunSummary


def render_run_detail(detail: RunDetail, service: HistoryService) -> None:
    """Render the full detail panel for one selected run.

    Layout:
        - Header (album, run name, parent link, optional config-delta
          expander, comment input);
        - Run Summary expander (config read-only + funnel + cluster
          counts + stage timeline);
        - Two-column row: Log viewer expander | Payload viewer expander;
        - Data Files expander.

    Reads from ``detail`` (already populated by the service).
    Side effects: writes through ``service.update_comment(...)`` when the
    comment text input value changes.
    """
    _render_header(detail, service)
    with st.expander("Run Summary", expanded=True):
        _render_config_section(detail.config)
        if detail.summary is not None:
            _render_summary_metrics(detail.summary)

    col_log, col_payload = st.columns(2)
    with col_log:
        with st.expander("View Log", expanded=False):
            _render_log_viewer(detail.row.log_file)
    with col_payload:
        with st.expander("Full Payload (debug)", expanded=False):
            payload = service.get_action_payload(detail.row.id)
            st.json(payload)

    if detail.row.output_dir:
        with st.expander("Data Files", expanded=False):
            _render_files_panel(Path(detail.row.output_dir))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header(detail: RunDetail, service: HistoryService) -> None:
    """Album/run name + parent link + comment input."""
    album = detail.row.display_album
    name = detail.row.run_name or detail.row.run_id or f"#{detail.row.id}"
    st.markdown(f"**Album:** `{album}` &nbsp;|&nbsp; **Run:** `{name}`")

    if detail.parent_row is not None:
        parent = detail.parent_row
        parent_label = parent.run_name or parent.run_id or f"#{parent.id}"
        st.caption(f"Parent: {parent_label}")
        if detail.config_delta:
            with st.expander("Config Delta (vs parent)", expanded=False):
                df = pd.DataFrame([
                    {"field": d.field, "parent": d.parent_value, "child": d.child_value}
                    for d in detail.config_delta
                ])
                st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.caption("Base run (no parent)")

    current_comment = detail.row.comment or ""
    new_comment = st.text_input(
        "Comment",
        value=current_comment,
        max_chars=2048,
        placeholder="Add a note about this run...",
        key=f"hist_comment_{detail.row.id}",
    )
    if new_comment != current_comment:
        service.update_comment(detail.row.id, new_comment)


# ---------------------------------------------------------------------------
# Config section — zero hardcoded field names
# ---------------------------------------------------------------------------

_CONFIG_DISPLAY_GROUPS = ("cluster", "quality", "merge", "cap")


def _render_config_section(config: Dict[str, Any]) -> None:
    """Render the run's saved FCParams config as read-only widgets,
    grouped by ``UI_SPEC.ui_group``.

    Replaces 16 lines of ``st.text(f"K: {cfg.get('K', '?')}")`` literals
    in the legacy tab. Iterates over ``UI_SPEC`` instead — zero
    field-name string literals in this code.
    """
    if not config:
        st.caption("No config recorded for this run.")
        return
    grouped = fields_by_group()
    for group_name in _CONFIG_DISPLAY_GROUPS:
        field_names = grouped.get(group_name, [])
        present = [f for f in field_names if f in config]
        if not present:
            continue
        st.markdown(f"**{group_name.title()}**")
        for field_name in present:
            render_field(
                field_name,
                readonly=True,
                value_override=config[field_name],
            )


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def _render_summary_metrics(summary: RunSummary) -> None:
    """Quality funnel, cluster counts, and stage durations from RunSummary."""
    render_metric_strip(summary, RUN_SUMMARY_STRIP, n_cols=5)

    if summary.stage_durations:
        st.markdown("**Stage durations**")
        timing_df = pd.DataFrame([
            {"Stage": name, "Duration (s)": f"{value:.1f}"}
            for name, value in summary.stage_durations.items()
        ])
        st.dataframe(timing_df, hide_index=True, use_container_width=False)


# ---------------------------------------------------------------------------
# Log viewer
# ---------------------------------------------------------------------------

def _render_log_viewer(log_file: Optional[str]) -> None:
    """Tail of the run's log (last 200 lines)."""
    if not log_file:
        st.caption("No log file path recorded.")
        return
    log_path = Path(log_file)
    if not log_path.exists():
        st.warning(f"Log file not found: `{log_file}`")
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    st.code("\n".join(lines[-200:]), language=None)


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

def _render_files_panel(output_dir: Path) -> None:
    """List files in the run's output dir with sizes."""
    if not output_dir.exists():
        st.warning(f"Output directory not found: `{output_dir}`")
        return
    entries = sorted(output_dir.iterdir(), key=lambda p: p.name)
    if not entries:
        st.caption("Output directory is empty.")
        return
    rows = []
    for p in entries:
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        rows.append({
            "Name": p.name,
            "Kind": "dir" if p.is_dir() else "file",
            "Size (bytes)": size,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
