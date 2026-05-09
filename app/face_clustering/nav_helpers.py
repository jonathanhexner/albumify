"""Navigation helpers: breadcrumb and no-result placeholder."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def _breadcrumb():
    result = st.session_state.pipeline_result
    parts  = []
    if result:
        parts.append(f"Run: **{Path(result.output_dir).name}**")
    if st.session_state.selected_cluster is not None:
        parts.append(f"Cluster **{st.session_state.selected_cluster}**")
    if st.session_state.selected_face is not None:
        parts.append(f"Face **{st.session_state.selected_face}**")
    if parts:
        st.caption(" > ".join(parts))


def _no_result():
    st.info("No run loaded. Use **Run** to start a new pipeline run, **Recluster** to re-cluster an existing run, or **History** to load a past run.")
