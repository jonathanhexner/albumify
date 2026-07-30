"""Face Clustering Streamlit app — entry point.

Usage:
    .venv/Scripts/streamlit run app/face_clustering/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure `app/face_clustering/` siblings (constants, state, shared, …) are importable
# without a full package install.  Streamlit runs scripts as __main__ so the package
# root (app/face_clustering/) is not automatically on sys.path.
_pkg_dir = Path(__file__).parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

# Repo root for sim_bench.* imports.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# spec-041 follow-up: align logging with Albumify / v2. Writes to
# logs/<timestamp>/fc_app_legacy.log.
from sim_bench.logging_setup import setup_logging  # noqa: E402
setup_logging("fc_app_legacy")

import streamlit as st

from state import _init_state
from session_helpers import _try_restore_session, _render_session_panel
from tabs.run_tab import render_run_tab
from tabs.recluster_tab import render_recluster_tab
from tabs.history_tab import render_history_tab
from tabs.overview_tab import render_run_overview_tab
from tabs.cluster_analysis_tab import render_cluster_analysis_tab
from tabs.merged_clusters_tab import render_merged_clusters_tab
from tabs.merge_analysis_tab import render_merge_analysis_tab
from tabs.face_analysis_tab import render_face_analysis_tab
from tabs.labeling_review_tab import render_labeling_review_tab
from tabs.ml_training_tab import render_ml_training_tab
from tabs.label_tab import render_label_verification_tab
from face_popup import maybe_show_face_popup
from cluster_popup import maybe_show_cluster_popup

st.set_page_config(page_title="Face Clustering", layout="wide")
st.title("Face Clustering")

_init_state()
_try_restore_session()

# Deep link support: ?load_run=<path>&cluster=<id>
_qp = st.query_params
if "load_run" in _qp and st.session_state.pipeline_result is None:
    from face_cluster.loader import load_pipeline_result
    _dl_dir = Path(_qp["load_run"])
    if _dl_dir.exists() and (_dl_dir / "faces.csv").exists():
        st.session_state.pipeline_result = load_pipeline_result(_dl_dir)
        st.session_state.active_run_dir = str(_dl_dir)
if "cluster" in _qp:
    try:
        st.session_state.selected_cluster = int(_qp["cluster"])
    except (ValueError, TypeError):
        pass

_render_session_panel()

(tab_run, tab_recluster, tab_hist, tab_overview, tab_cluster,
 tab_merged, tab_merge_analysis, tab_face, tab_labeling, tab_ml,
 tab_label_verify) = st.tabs([
    "Run", "Recluster", "History", "Clusters (Base)", "Cluster Analysis",
    "Clusters (Merged)", "Merge Analysis", "Face Analysis",
    "Labeling Review", "ML Training", "Label Verification",
])

with tab_run:            render_run_tab()
with tab_recluster:      render_recluster_tab()
with tab_hist:           render_history_tab()
with tab_overview:       render_run_overview_tab()
with tab_cluster:        render_cluster_analysis_tab()
with tab_merged:         render_merged_clusters_tab()
with tab_merge_analysis: render_merge_analysis_tab()
with tab_face:           render_face_analysis_tab()
with tab_labeling:       render_labeling_review_tab()
with tab_ml:             render_ml_training_tab()
with tab_label_verify:   render_label_verification_tab()

maybe_show_face_popup(st.session_state.pipeline_result)
maybe_show_cluster_popup(st.session_state.pipeline_result)
