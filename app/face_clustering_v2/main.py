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
from app.face_clustering_v2.tabs.history_tab import render_history_tab
from app.face_clustering_v2.tabs.recluster_tab import render_recluster_tab


st.set_page_config(page_title="Face Clustering v2", layout="wide")
st.title("Face Clustering — v2 (spec-040)")
st.caption(
    "New FC App on the unified pipeline framework. The original app at "
    "`app/face_clustering/` stays in place during the strangler-fig migration."
)

tab_run, tab_analysis, tab_face, tab_recluster, tab_history = st.tabs(
    ["Run", "Cluster Analysis", "Face Analysis", "Recluster", "History"]
)

with tab_run:
    render_run_tab()
with tab_analysis:
    render_cluster_analysis_tab()
with tab_face:
    render_face_analysis_tab()
with tab_recluster:
    render_recluster_tab()
with tab_history:
    render_history_tab()
