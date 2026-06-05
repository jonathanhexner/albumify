"""spec-042 H3 — History tab orchestrator.

Pure orchestration: query the service, hand its typed output to
components for rendering. No SQL, no JSON parsing, no field-name string
literals — all of that is in ``face_cluster.views.history`` and the
declarative ``ColumnSpec`` lists.

Target: <= 50 LOC. Legacy ``app/face_clustering/tabs/history_tab.py``
is 365 LOC mixing 7 responsibilities; this rebuild splits each
responsibility into its own component (filter bar, table, detail,
load button, actions table) so the tab itself is a 6-call sequence.
"""
from __future__ import annotations

import streamlit as st

from app.face_clustering_v2._telemetry import tab_done, tab_start
from app.face_clustering_v2.components.actions_table import render_actions_table
from app.face_clustering_v2.components.load_button import render_load_button
from app.face_clustering_v2.components.run_detail import render_run_detail
from app.face_clustering_v2.components.run_filter_bar import render_filter_bar
from app.face_clustering_v2.components.run_table import render_run_table
from face_cluster.views.history import (
    RUN_COLUMNS,
    HistoryService,
)


def render_history_tab() -> None:
    """Render the History tab.

    Layout:
        - Filter bar (album / date range / free-text)
        - Pipeline runs table (with selection)
        - Selected-run detail panel (config / summary / log / files)
        - Load button (writes to st.session_state)
        - Recent Actions sub-table + payload inspector

    Reads from: ``HistoryService`` (queries the global action_log DB).
    Writes to: ``st.session_state`` when the user clicks Load or edits
    a comment.
    """
    st.header("History")
    tab_start("history", None)
    service = HistoryService()

    query = render_filter_bar(service.list_albums())
    runs = service.list_runs(query)
    tab_done("history", n_runs=len(runs))

    st.subheader(f"Pipeline Runs ({len(runs)})")
    selected_id = render_run_table(runs, RUN_COLUMNS, key="hist_run_table")
    if selected_id is not None:
        detail = service.get_run_detail(selected_id)
        st.divider()
        render_run_detail(detail, service)
        st.divider()
        render_load_button(detail, service)

    st.divider()
    st.subheader("Recent Actions")
    render_actions_table(service.list_other_actions(), service)
