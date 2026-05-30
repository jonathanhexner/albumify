"""spec-065 — v2 Merged Clusters tab. Read-only viewer over
``merge_decisions``. Sync only (SIGHTING-079). No SQL / FS / cfg.get
here — arch tests enforce.
"""
from __future__ import annotations
import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import streamlit as st
from app.face_clustering_v2.components.run_table import render_run_table
from face_cluster.views._specs import ColumnSpec
from face_cluster.views.merged_clusters import (
    MergeDecisionCriteria, MergedClustersService,
)
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig, ClusterAnalysisRepository,
)
logger = logging.getLogger(__name__)

_COLS = (
    ColumnSpec("iteration", "iter"),
    ColumnSpec("cluster_a", "cluster_a"),
    ColumnSpec("cluster_b", "cluster_b"),
    ColumnSpec("actually_merged", "merged?", formatter=lambda v: "yes" if v else "no"),
    ColumnSpec("exemplar_dist", "exemplar_dist", formatter=lambda v: f"{float(v):.3f}"),
    ColumnSpec("support", "support"),
    ColumnSpec("action", "action"),
    ColumnSpec("rejection_reason", "reason"),
)
_FILTERS = {"all": None, "only merged": True, "only rejected": False}


def render_merged_clusters_tab() -> None:
    """Render the v2 Merged Clusters tab — filter + table + detail panel."""
    st.header("Merged Clusters")
    run_dir = _resolve_run_dir()
    if run_dir is None:
        st.info("No run loaded. Open a run from the History tab first.")
        return
    service = _get_or_build_service(run_dir)
    if service is None:
        return
    choice = st.selectbox("Show", list(_FILTERS), key="v2_mc_filter")
    rows = service.list_merge_decisions(MergeDecisionCriteria(actually_merged=_FILTERS[choice]))
    if not rows:
        st.info("No merge_decisions match the current filter.")
        return
    indexed = [SimpleNamespace(id=i, **asdict(r)) for i, r in enumerate(rows)]
    sel = render_run_table(indexed, _COLS, key="v2_mc_table")
    if sel is None:
        return
    row = rows[sel]
    st.subheader(f"Pair (cluster_a={row.cluster_a}, cluster_b={row.cluster_b})")
    st.json(asdict(row))


def _resolve_run_dir() -> Optional[Path]:
    for key in ("current_run_dir", "v2_last_run_dir", "active_run_dir"):
        v = st.session_state.get(key)
        if v and Path(v).is_dir() and (Path(v) / "face_clustering.db").is_file():
            return Path(v)
    return None


def _get_or_build_service(run_dir: Path) -> Optional[MergedClustersService]:
    key = f"_merged_clusters_service::{run_dir}"
    if key in st.session_state:
        return st.session_state[key]
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Merged Clusters: repo construction failed for %s", run_dir)
        st.error(f"Cannot open run: {exc}")
        return None
    svc = MergedClustersService(repo)
    st.session_state[key] = svc
    return svc
