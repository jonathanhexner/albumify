"""spec-066 — v2 Overview tab. Run-level dashboard across all fc_app_v2 runs.

Pure orchestration: build the service, fetch one ``DashboardMetrics``,
render the 4-metric strip + charts. The only wall-clock lives here (the
"last run age"); the service stays clock-free for testability. No run
needs to be loaded — this reads global history, not a single run.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

from app.face_clustering_v2._telemetry import tab_done, tab_start
from app.face_clustering_v2.components.dashboard_charts import render_bar, render_timeseries
from face_cluster.views.overview import DashboardMetrics, OverviewService

logger = logging.getLogger(__name__)


def render_overview_tab() -> None:
    """Render the Overview dashboard."""
    st.header("Overview")
    tab_start("overview", None)
    service = _get_service()
    metrics = service.compute_dashboard(limit=50)

    c = st.columns(4)
    c[0].metric("Total runs", metrics.total_runs)
    c[1].metric("Total faces ever", f"{metrics.total_faces_ever:,}")
    c[2].metric(
        "Avg n_clusters",
        "-" if metrics.avg_n_clusters is None else f"{metrics.avg_n_clusters:.1f}",
        help=None if metrics.median_n_clusters is None else f"median {metrics.median_n_clusters:g}",
    )
    c[3].metric("Last run", _age(metrics.last_run_at))

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


def _age(iso: Optional[str]) -> str:
    """Human 'time since' for the last run. None -> 'never'."""
    if not iso:
        return "never"
    try:
        ts = datetime.fromisoformat(iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
    except Exception:  # noqa: BLE001
        return "?"
    secs = int(delta.total_seconds())
    if secs < 3600:
        return f"{max(0, secs // 60)}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"