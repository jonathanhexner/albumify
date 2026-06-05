"""spec-045 Phase 6 — Cluster Analysis tab orchestrator.

Pure orchestration: gather state, build the Service, dispatch to render
components. No SQL, no FS, no ``cfg.get`` literals. See spec §7.1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from app.face_clustering_v2.components.cluster_debug import render_cluster_debug
from app.face_clustering_v2.components.cluster_metrics import render_cluster_metrics
from app.face_clustering_v2.components.cluster_picker import render_cluster_picker
from app.face_clustering_v2.components.cluster_summary_table import render_cluster_summary
from app.face_clustering_v2.components.face_grid import render_face_grid
from app.face_clustering_v2.components.force_merge import render_force_merge
from app.face_clustering_v2.components.nearest_clusters import render_nearest_clusters
from sim_bench.db.face_clustering.cluster_analysis_repo import (
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
        tab_skipped("cluster_analysis", "no_run_loaded")
        st.info(
            "No completed run available yet. "
            "Either run a fresh pipeline from the **Run** tab, or open the "
            "**History** tab and click *Load into analysis tabs* on a "
            "completed run. (A run that's still in progress or failed before "
            "writing its DB is skipped here.)"
        )
        return

    tab_start("cluster_analysis", run_dir)
    service = _get_service(run_dir)
    if service is None:
        # _get_service already showed an st.error; bail out gracefully.
        tab_skipped("cluster_analysis", "repo_failed")
        return
    rows = service.list_clusters()

    # spec-074: all-clusters summary (restores the V1 overview). Cached per run
    # dir so it doesn't recompute nearest-cluster distances on every rerun.
    summary_key = f"_cluster_summary::{run_dir}"
    if summary_key not in st.session_state:
        st.session_state[summary_key] = service.cluster_summary()
    summary = st.session_state[summary_key]
    with st.expander(f"All clusters ({len(summary)}) — click a row to open", expanded=True):
        picked = render_cluster_summary(summary)
    # Loop-safe: only navigate when the summary selection actually changes.
    if picked is not None and picked != st.session_state.get("_summary_last_pick"):
        st.session_state["_summary_last_pick"] = picked
        st.session_state["_goto_cluster"] = picked
        st.rerun()

    cluster_id = render_cluster_picker(rows)
    if cluster_id is None:
        tab_skipped("cluster_analysis", "no_cluster_selected")
        return
    tab_done("cluster_analysis", n_clusters=len(rows), selected_cluster=cluster_id)

    # SIGHTING-079 fix: sync compute + st.spinner. AsyncHandle never advanced
    # past "Analysing cluster…" because Streamlit doesn't poll background
    # threads. For ≤100-face clusters, sync compute is sub-second.
    try:
        with st.spinner("Analysing cluster…"):
            detail = service.compute_detail(cluster_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_detail failed for cluster %s in %s", cluster_id, run_dir)
        st.error(f"Could not analyse cluster {cluster_id}: {exc}")
        return

    render_cluster_metrics(detail)
    render_face_grid(detail, run_dir=run_dir)
    render_nearest_clusters(detail)

    render_force_merge(service, cluster_ids=[r.cluster_id for r in rows], current_cluster=cluster_id)

    try:
        with st.spinner("Computing graph diagnostics…"):
            debug = service.compute_debug(cluster_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_debug failed for cluster %s in %s", cluster_id, run_dir)
        st.error(f"Could not compute graph diagnostics for cluster {cluster_id}: {exc}")
        return
    render_cluster_debug(debug)


def _run_dir_is_loadable(path: Path) -> bool:
    """True iff ``path`` is a complete run dir the Repository can open.

    spec-050's run_tab writes ``v2_last_run_dir`` BEFORE the pipeline runs
    (deliberate — failures leave a recoverable pointer). If the pipeline
    crashed mid-run or hasn't finished, the dir exists but the DB doesn't.
    The resolver must skip those and fall through to the next session key
    rather than handing the Repository a half-baked run dir.
    """
    return path.is_dir() and (path / "face_clustering.db").is_file()


def _resolve_current_run_dir() -> Optional[Path]:
    """Single source of truth for which run this tab is looking at.

    Priority (spec §7.2):
      1. ``current_run_dir`` (set by History → Load Run via load_button T050,
         or by force-merge apply).
      2. ``v2_last_run_dir`` (set by spec-050 run_tab on every fresh run).
      3. ``active_run_dir`` (legacy History session-state key, still emitted
         by load_button alongside current_run_dir for backward compat).
      4. None.

    Skips keys whose value points at an in-progress / failed run dir
    (no face_clustering.db yet) — falls through to the next key instead.
    """
    for key in ("current_run_dir", "v2_last_run_dir", "active_run_dir"):
        value = st.session_state.get(key)
        if value and _run_dir_is_loadable(Path(value)):
            return Path(value)
    return None


def _get_service(run_dir: Path) -> Optional[ClusterAnalysisService]:
    """Build a Service for ``run_dir``. Lazily caches on the run_dir key.

    Returns None if construction fails (e.g., dir was deleted or the
    RunStore validation rejects the schema). Surfaces the underlying
    error via ``st.error`` so the user sees what went wrong instead of
    a Streamlit traceback overlay.

    The resolver pre-filters dirs without a face_clustering.db; this
    try/except is belt-and-braces for race conditions (dir disappears
    between resolver check and Repository construction) and for the
    deeper RunStore validation (schema version mismatch, missing
    pipeline_run.json, etc.).
    """
    cache_key = f"_cluster_analysis_service::{run_dir}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Cluster Analysis tab: Repository construction failed for %s", run_dir)
        st.error(
            f"Cannot open run at `{run_dir}`: {exc}. "
            "If the run failed mid-pipeline, the dir is left empty for "
            "debugging — load a completed run from the History tab instead."
        )
        return None
    service = ClusterAnalysisService(repo)
    st.session_state[cache_key] = service
    return service
