"""spec-066 — reusable Plotly chart helpers for the Overview tab.

Render-only: each helper takes a list of typed stat dataclasses (from
``face_cluster.views.overview``) and draws one chart. No compute, no SQL.
Empty input renders a caption, never an empty axis box.
"""
from __future__ import annotations

from typing import List, Sequence

import streamlit as st

from face_cluster.views.overview import RunPoint


def render_bar(
    labels: Sequence[str],
    values: Sequence[int],
    *,
    title: str,
    key: str,
    empty_msg: str = "No data yet.",
) -> None:
    """Horizontal bar chart of ``labels`` -> ``values``.

    Caller flattens its stat dataclasses into parallel label/value lists so
    this helper stays agnostic to AlbumStat vs StatusStat vs ProfileStat.
    """
    st.markdown(f"**{title}**")
    if not labels:
        st.caption(empty_msg)
        return
    import plotly.graph_objects as go

    # Reverse so the largest bar sits at the top (Plotly draws y bottom-up).
    fig = go.Figure(
        go.Bar(x=list(values)[::-1], y=list(labels)[::-1], orientation="h")
    )
    fig.update_layout(
        height=max(140, 34 * len(labels)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="runs",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_timeseries(points: List[RunPoint], *, key: str) -> None:
    """Line chart of n_clusters over the run window (oldest -> newest)."""
    st.markdown("**n_clusters over recent runs**")
    usable = [p for p in points if p.n_clusters is not None]
    if not usable:
        st.caption("No completed runs with a cluster count yet.")
        return
    import plotly.graph_objects as go

    x = [p.started_at or str(p.run_id) for p in usable]
    y = [p.n_clusters for p in usable]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="n_clusters",
        xaxis_title="run time",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
