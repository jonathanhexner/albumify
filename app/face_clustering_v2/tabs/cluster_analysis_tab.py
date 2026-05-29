"""spec-045 Phase 6 — Cluster Analysis tab orchestrator.

Pure orchestration: gather state, build the Service, dispatch to render
components. No SQL, no FS, no ``cfg.get`` literals. See spec §7.1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from app.face_clustering_v2.components.cluster_debug import render_cluster_debug
from app.face_clustering_v2.components.cluster_metrics import render_cluster_metrics
from app.face_clustering_v2.components.cluster_picker import render_cluster_picker
from app.face_clustering_v2.components.face_grid import render_face_grid
from app.face_clustering_v2.components.force_merge import render_force_merge
from app.face_clustering_v2.components.nearest_clusters import render_nearest_clusters
from face_cluster.repositories.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from face_cluster.views.cluster_analysis import ClusterAnalysisService


def render_cluster_analysis_tab() -> None:
    """Render the Cluster Analysis tab.

    Layout: cluster picker → metrics → faces → nearest → force merge → graph debug.
    Reads:  ClusterAnalysisService over a per-run face_clustering.db.
    Writes: ``selected_cluster``, ``current_run_dir`` (on force-merge apply).
    """
    st.header("Cluster Analysis")
    run_dir = _resolve_current_run_dir()
    if run_dir is None:
        st.info("No run loaded. Open the **History** tab and click *Load into analysis tabs*.")
        return

    service = _get_service(run_dir)
    rows = service.list_clusters()
    cluster_id = render_cluster_picker(rows)
    if cluster_id is None:
        return

    detail_handle = service.compute_detail_async(cluster_id)
    render_cluster_metrics(detail_handle)
    render_face_grid(detail_handle, run_dir=run_dir)
    render_nearest_clusters(detail_handle)

    render_force_merge(service, cluster_ids=[r.cluster_id for r in rows], current_cluster=cluster_id)

    debug_handle = service.compute_debug_async(cluster_id)
    render_cluster_debug(debug_handle)


def _resolve_current_run_dir() -> Optional[Path]:
    """Single source of truth for which run this tab is looking at.

    Priority (spec §7.2):
      1. ``current_run_dir`` (set by History → Load Run via load_button T050,
         or by force-merge apply).
      2. ``v2_last_run_dir`` (set by spec-050 run_tab on every fresh run).
      3. ``active_run_dir`` (legacy History session-state key, still emitted
         by load_button alongside current_run_dir for backward compat).
      4. None.
    """
    for key in ("current_run_dir", "v2_last_run_dir", "active_run_dir"):
        value = st.session_state.get(key)
        if value:
            path = Path(value)
            if path.is_dir():
                return path
    return None


def _get_service(run_dir: Path) -> ClusterAnalysisService:
    """Build a Service for ``run_dir``. Lazily caches on the run_dir key so
    repeated reruns within the same dir reuse the same Repository instance
    (RunStore opens its own sqlite connections per call — caching the
    Repository is cheap and avoids re-validating the run dir on every poll).
    """
    cache_key = f"_cluster_analysis_service::{run_dir}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached
    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    service = ClusterAnalysisService(repo)
    st.session_state[cache_key] = service
    return service
