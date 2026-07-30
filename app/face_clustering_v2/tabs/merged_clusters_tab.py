"""spec-065 — v2 Merged Clusters tab. Read-only viewer over
``merge_decisions``. Sync only (SIGHTING-079). No SQL / FS / cfg.get
here — arch tests enforce.
"""
from __future__ import annotations
import logging
from dataclasses import asdict
from types import SimpleNamespace
import streamlit as st
from app.face_clustering_v2._run_context import (
    cached_cluster_service, cached_service, resolve_run_dir,
)
from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from app.face_clustering_v2.components.cluster_pair_crops import render_cluster_pair_crops
from app.face_clustering_v2.components.merge_gate_badges import render_merge_gate_badges
from app.face_clustering_v2.components.nearest_pairs import render_nearest_pairs
from app.face_clustering_v2.components.run_table import render_run_table
from face_cluster.views._specs import ColumnSpec
from face_cluster.views.merged_clusters import (
    MergeDecisionCriteria, MergedClustersService,
)
logger = logging.getLogger(__name__)

_COLS = (
    ColumnSpec("iteration", "iter"), ColumnSpec("cluster_a", "cluster_a"), ColumnSpec("cluster_b", "cluster_b"),
    ColumnSpec("actually_merged", "merged?", formatter=lambda v: "yes" if v else "no"),
    ColumnSpec("exemplar_dist", "exemplar_dist", formatter=lambda v: f"{float(v):.3f}"),
    ColumnSpec("support", "support"), ColumnSpec("action", "action"),
    ColumnSpec("rejection_reason", "reason"),
)
_FILTERS = {"all": None, "only merged": True, "only rejected": False}


def render_merged_clusters_tab() -> None:
    """Render the v2 Merged Clusters tab — filter + table + detail panel."""
    st.header("Merged Clusters")
    run_dir = resolve_run_dir()
    if run_dir is None:
        tab_skipped("merged_clusters", "no_run_loaded")
        st.info("No run loaded. Open a run from the History tab first.")
        return
    tab_start("merged_clusters", run_dir)
    service = cached_service(run_dir, MergedClustersService, cache_prefix="_merged_clusters_service")
    if service is None:
        tab_skipped("merged_clusters", "repo_failed")
        return
    s = service.summary()
    st.caption(f"{s.n_merged} merged | {s.n_rejected} rejected | top reject gate: {s.top_rejection_gate or '-'}")

    # spec-075: the N closest cluster pairs (what was *almost* merged + why),
    # shown even when zero pairs crossed the candidate threshold. Cached per run.
    ca = cached_cluster_service(run_dir, cache_prefix="_mc_ca_service")
    if ca is not None:
        render_nearest_pairs(ca, run_dir)

    choice = st.selectbox("Show", list(_FILTERS), key="v2_mc_filter")
    rows = service.list_merge_decisions(MergeDecisionCriteria(actually_merged=_FILTERS[choice]))
    tab_done("merged_clusters", n_rows=len(rows), filter=choice)
    if not rows:
        st.info("No merge_decisions match the current filter.")
        return
    indexed = [SimpleNamespace(id=i, **asdict(r)) for i, r in enumerate(rows)]
    sel = render_run_table(indexed, _COLS, key="v2_mc_table")
    # spec-071: fall back to the ?selected_merge_pair seed when no canvas row-pick (SIGHTING-091).
    pair = st.session_state.get("selected_merge_pair")
    row = rows[sel] if sel is not None else next(
        (r for r in rows if (r.cluster_a, r.cluster_b) == pair), None)
    if row is None:
        return
    st.subheader(f"Pair (cluster_a={row.cluster_a}, cluster_b={row.cluster_b}) - {row.action}")
    render_merge_gate_badges(service.gate_badges(row))
    st.caption(f"exemplar_dist={row.exemplar_dist:.3f} | threshold={row.threshold_used:.3f} | support={row.support}/{row.required_support} | post_diameter={row.post_diameter:.3f}")
    render_cluster_pair_crops(service.pair_faces(row))
