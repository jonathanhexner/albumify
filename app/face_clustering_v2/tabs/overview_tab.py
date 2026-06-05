"""spec-066 — v2 Overview tab. Run-level dashboard across all fc_app_v2 runs.

Pure orchestration: build the service, fetch one ``DashboardMetrics``,
render the 4-metric strip + charts. The only wall-clock lives here (the
"last run age"); the service stays clock-free for testability. No run
needs to be loaded — this reads global history, not a single run.
"""
from __future__ import annotations

import logging

import streamlit as st

from app.face_clustering_v2._telemetry import tab_done, tab_start
from app.face_clustering_v2.components.dashboard_charts import render_bar, render_timeseries
from app.face_clustering_v2.components.metric_strip import render_metric_strip
from face_cluster.views.metric_specs import OVERVIEW_STRIP
from face_cluster.views.overview import DashboardMetrics, OverviewService

logger = logging.getLogger(__name__)


def render_overview_tab() -> None:
    """Render the Overview dashboard."""
    st.header("Overview")
    tab_start("overview", None)
    service = _get_service()
    metrics = service.compute_dashboard(limit=50)

    render_metric_strip(metrics, OVERVIEW_STRIP, n_cols=4)

    if metrics.gate_pass_rate is not None:
        st.caption(f"Gate pass rate: {metrics.gate_pass_rate:.0%} of runs completed.")

    st.divider()
    render_bar([s.album for s in metrics.per_album], [s.n_runs for s in metrics.per_album],
               title="Runs per album", key="ov_album")
    render_bar([s.status for s in metrics.per_status], [s.n_runs for s in metrics.per_status],
               title="Runs per status", key="ov_status")
    if service.has_real_profiles(metrics):
        render_bar([s.profile for s in metrics.per_profile], [s.n_runs for s in metrics.per_profile],
                   title="Runs per profile", key="ov_profile")
    render_timeseries(metrics.timeseries, key="ov_timeseries")

    tab_done("overview", total_runs=metrics.total_runs,
             albums=len(metrics.per_album), profiles=len(metrics.per_profile))


def _get_service() -> OverviewService:
    if "_overview_service" not in st.session_state:
        st.session_state["_overview_service"] = OverviewService()
    return st.session_state["_overview_service"]