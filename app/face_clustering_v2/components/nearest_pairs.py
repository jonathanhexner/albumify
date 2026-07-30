"""spec-075 / spec-082 — Nearest cluster-pairs panel for Merged Clusters.

Render-only. The tab resolves a ``ClusterAnalysisService`` and hands it in;
this draws the "what was almost merged" expander (the N closest cluster pairs +
their merge verdict, sortable). Cached per run dir so the DB read happens once.
Extracted from ``merged_clusters_tab`` to keep that tab a thin orchestrator
(spec-053 LOC budget).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from face_cluster.views.cluster_analysis import NEAREST_PAIR_COLUMNS, ClusterAnalysisService


def render_nearest_pairs(service: ClusterAnalysisService, run_dir: Path, *, top: int = 20) -> None:
    """Draw the closest ``top`` cluster pairs + their merge verdict."""
    pk = f"_nearest_pairs::{run_dir}"
    if pk not in st.session_state:
        st.session_state[pk] = service.nearest_cluster_pairs(top)
    pairs = st.session_state[pk]
    with st.expander(
        f"Nearest cluster pairs (closest {len(pairs)}) — what was almost merged",
        expanded=True,
    ):
        if pairs:
            st.dataframe(
                pd.DataFrame([{c.label: c.display(p) for c in NEAREST_PAIR_COLUMNS} for p in pairs]),
                hide_index=True, width="stretch",
            )
        else:
            st.caption("Need >= 2 clusters to compute pairs.")
