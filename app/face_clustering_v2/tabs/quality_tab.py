"""spec-065 — v2 Quality tab. Aggregated per-gate pass / fail counts.
Sync only (SIGHTING-079). No SQL / FS / cfg.get here — arch tests enforce.
"""
from __future__ import annotations
import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import streamlit as st
from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from app.face_clustering_v2.components.metric_strip import render_metric_strip
from app.face_clustering_v2.components.quality_bar_chart import render_quality_bar_chart
from app.face_clustering_v2.components.run_table import render_run_table
from face_cluster.views.metric_specs import QUALITY_SUMMARY_STRIP
from face_cluster.views._specs import ColumnSpec
from face_cluster.views.quality import QualityService
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig, ClusterAnalysisRepository, FilterDecisionCriteria,
)
logger = logging.getLogger(__name__)
_COLS = (
    ColumnSpec("item_id", "item_id"),
    ColumnSpec("item_type", "type"),
    ColumnSpec("filter_name", "gate"),
    ColumnSpec("rejected", "rejected?", formatter=lambda v: "yes" if v else "no"),
    ColumnSpec("reason", "reason"),
)


def render_quality_tab() -> None:
    """Render the v2 Quality tab — summary strip + per-gate chart + table."""
    st.header("Quality")
    run_dir = _resolve_run_dir()
    if run_dir is None:
        tab_skipped("quality", "no_run_loaded")
        st.info("No run loaded. Open a run from the History tab first.")
        return
    tab_start("quality", run_dir)
    service = _get_or_build_service(run_dir)
    if service is None:
        tab_skipped("quality", "repo_failed")
        return
    summary = service.summary()
    tab_done("quality", n_items=summary.n_items, n_decisions=summary.n_decisions,
             n_rejected=summary.n_rejected)
    render_metric_strip(summary, QUALITY_SUMMARY_STRIP, n_cols=5)
    render_quality_bar_chart(summary.gates, key="v2_q_chart")
    gate_names = ["(all)"] + [g.gate_name for g in summary.gates]
    gate = st.selectbox("Filter by gate", gate_names, key="v2_q_gate")
    rejected_only = st.checkbox("Rejected only", value=True, key="v2_q_rej")
    crit = FilterDecisionCriteria(filter_name=None if gate == "(all)" else gate, rejected=True if rejected_only else None)
    rows = service.list_filter_decisions(crit)
    if not rows:
        st.info("No filter_decisions match the current filter.")
        return
    indexed = [SimpleNamespace(id=i, **{k: v for k, v in asdict(r).items() if k != "measured"}) for i, r in enumerate(rows)]
    render_run_table(indexed, _COLS, key="v2_q_table")


def _resolve_run_dir() -> Optional[Path]:
    for key in ("current_run_dir", "v2_last_run_dir", "active_run_dir"):
        v = st.session_state.get(key)
        if v and Path(v).is_dir() and (Path(v) / "face_clustering.db").is_file():
            return Path(v)
    return None


def _get_or_build_service(run_dir: Path) -> Optional[QualityService]:
    key = f"_quality_service::{run_dir}"
    if key in st.session_state:
        return st.session_state[key]
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Quality: repo construction failed for %s", run_dir)
        st.error(f"Cannot open run: {exc}")
        return None
    svc = QualityService(repo)
    st.session_state[key] = svc
    return svc
