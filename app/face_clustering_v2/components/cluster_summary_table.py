"""spec-074 — all-clusters summary table.

One sortable row per cluster (size / diameter / spread / nearest other
cluster + its size + distance). Columns come from the CLUSTER_SUMMARY_COLUMNS
registry; raw values are read so the table sorts numerically. Clicking a row
returns that cluster_id (the tab drives the drill-in).
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import streamlit as st

from face_cluster.views.cluster_analysis import CLUSTER_SUMMARY_COLUMNS


def render_cluster_summary(rows: Sequence) -> Optional[int]:
    """Render the summary table; return the clicked cluster_id (or None).

    Numeric columns (raw ``ColumnSpec.read``) keep the table sortable — click
    a header to find the biggest cluster, the closest pair, etc.
    """
    if not rows:
        st.caption("No clusters to summarise.")
        return None
    df = pd.DataFrame([{c.label: c.read(r) for c in CLUSTER_SUMMARY_COLUMNS} for r in rows])
    event = st.dataframe(
        df, hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="v2_cluster_summary",
    )
    sel = getattr(getattr(event, "selection", None), "rows", []) or []
    if sel:
        return int(df.iloc[sel[0]]["Cluster"])
    return None
