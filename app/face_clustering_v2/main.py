"""spec-040 Phase 5b — Face Clustering v2 Streamlit entry.

Run with:
    .venv/Scripts/streamlit run app/face_clustering_v2/main.py

This is the new FC App that drives ``FCAppRunner`` (the unified pipeline
framework) instead of the legacy ``FaceClusteringPipeline``. Coexists with
the original ``app/face_clustering/`` (which stays as-is during the
strangler-fig migration). Producer tag in the global DB: ``fc_app_v2``.

Today it ships with just two tabs (Run, Clusters) — the minimum surface
needed to validate the v2 pipeline against real albums and inspect the
output. Other tabs (Recluster, Merge Analysis, History, etc.) are
follow-up work tracked in ``specs/040-unified-pipeline-framework/CONCRETE_PLAN.md``
Phase 5b.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `app.face_clustering_v2` is importable
# when Streamlit runs this file as __main__.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# spec-041 follow-up: align logging with Albumify / CLI. Writes to
# logs/<timestamp>/fc_app_v2.log; module loggers in face_cluster/* and
# sim_bench/pipeline/* inherit handlers automatically.
from sim_bench.logging_setup import setup_logging
setup_logging("fc_app_v2")

import streamlit as st

from app.face_clustering_v2.tabs.run_tab import render_run_tab
from app.face_clustering_v2.tabs.cluster_analysis_tab import render_cluster_analysis_tab
from app.face_clustering_v2.tabs.face_analysis_tab import render_face_analysis_tab
from app.face_clustering_v2.tabs.face_metrics_tab import render_face_metrics_tab
from app.face_clustering_v2.tabs.gallery_tab import render_gallery_tab
from app.face_clustering_v2.tabs.history_tab import render_history_tab
from app.face_clustering_v2.tabs.images_tab import render_images_tab
from app.face_clustering_v2.tabs.merged_clusters_tab import render_merged_clusters_tab
from app.face_clustering_v2.tabs.overview_tab import render_overview_tab
from app.face_clustering_v2.tabs.quality_tab import render_quality_tab
from app.face_clustering_v2.tabs.recluster_tab import render_recluster_tab


st.set_page_config(page_title="Face Clustering v2", layout="wide")


def _seed_session_state_from_query_params() -> None:
    """Map a small allowlist of ``?key=value`` query params to session_state.

    Runs once per session (idempotent via the ``_qp_seeded`` sentinel) so
    later reruns / user navigation are never overridden.

    Allowlist:
      - ``?current_run_dir=<path>`` -> ``current_run_dir`` + ``active_run_dir``
        (mirrors what the History tab's Load button writes; every analysis
        tab resolves its data from these dir keys — see
        ``cluster_analysis_tab._resolve_current_run_dir`` et al).
      - ``?selected_face_id=<int>`` -> ``selected_face_id`` (Face Analysis
        tab default; ignored if not an int).

    Opt-in: with no query params this writes nothing and the app behaves
    exactly as before. Spec-067 / SIGHTING-091: lets the budapest e2e suite
    bypass the canvas-rendered ``st.dataframe`` row-pick, and gives real
    users a shareable deep-link to a run.
    """
    if st.session_state.get("_qp_seeded"):
        return
    st.session_state["_qp_seeded"] = True

    run_dir = st.query_params.get("current_run_dir")
    if run_dir:
        st.session_state["current_run_dir"] = run_dir
        st.session_state["active_run_dir"] = run_dir

    face_id = st.query_params.get("selected_face_id")
    if face_id is not None:
        try:
            st.session_state["selected_face_id"] = int(face_id)
        except (TypeError, ValueError):
            pass

    # spec-071: ?selected_merge_pair=a,b drives the Merged Clusters detail panel
    # directly, so e2e can bypass the un-clickable canvas dataframe row-pick.
    pair = st.query_params.get("selected_merge_pair")
    if pair:
        try:
            a, b = (int(x) for x in pair.split(","))
            st.session_state["selected_merge_pair"] = (a, b)
        except (TypeError, ValueError):
            pass


_seed_session_state_from_query_params()

st.title("Face Clustering — v2 (spec-040)")
st.caption(
    "New FC App on the unified pipeline framework. The original app at "
    "`app/face_clustering/` stays in place during the strangler-fig migration."
)

(tab_run, tab_analysis, tab_face, tab_metrics, tab_images, tab_gallery, tab_merged,
 tab_quality, tab_recluster, tab_overview, tab_history) = st.tabs(
    ["Run", "Cluster Analysis", "Face Analysis", "Face Metrics", "Images", "Gallery",
     "Merged Clusters", "Quality", "Recluster", "Overview", "History"]
)

with tab_run:
    render_run_tab()
with tab_analysis:
    render_cluster_analysis_tab()
with tab_face:
    render_face_analysis_tab()
with tab_metrics:
    render_face_metrics_tab()
with tab_images:
    render_images_tab()
with tab_gallery:
    render_gallery_tab()
with tab_merged:
    render_merged_clusters_tab()
with tab_quality:
    render_quality_tab()
with tab_recluster:
    render_recluster_tab()
with tab_overview:
    render_overview_tab()
with tab_history:
    render_history_tab()
