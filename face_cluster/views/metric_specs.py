"""spec-078 — declarative metric-strip registries.

One ``List[ColumnSpec]`` per metric strip in the v2 app. Each replaces a
hand-written block of ``cN.metric(...)`` calls; the component renders them via
``render_metric_strip``. Getters read attributes off the strip's source object
(ClusterView / RunSummary / DashboardMetrics / …) by duck typing — no heavy
imports here, and this module stays Streamlit-free.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from face_cluster.views._specs import ColumnSpec


def _age(iso: Optional[str]) -> str:
    """Human 'time since' for the Overview 'Last run' metric. None -> 'never'."""
    if not iso:
        return "never"
    try:
        ts = datetime.fromisoformat(iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - ts).total_seconds())
    except Exception:  # noqa: BLE001
        return "?"
    if secs < 3600:
        return f"{max(0, secs // 60)}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def _avg_clusters(m) -> str:
    if m.avg_n_clusters is None:
        return "-"
    s = f"{m.avg_n_clusters:.1f}"
    return s if m.median_n_clusters is None else f"{s} (med {m.median_n_clusters:g})"


# ClusterView (Cluster Analysis metric strip)
CLUSTER_METRIC_STRIP: List[ColumnSpec] = [
    ColumnSpec("size", "Faces"),
    ColumnSpec("diameter", "Diameter", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("avg_intra_dist", "Avg intra-dist", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("", "Exemplars", getter=lambda v: len(v.exemplar_face_ids)),
    ColumnSpec("", "Outliers", getter=lambda v: len(v.outlier_face_ids)),
]

# ClusterDebugView (graph diagnostics strip)
CLUSTER_DEBUG_STRIP: List[ColumnSpec] = [
    ColumnSpec("", "Edges", getter=lambda d: f"{d.n_edges} / {d.max_possible_edges}"),
    ColumnSpec("edge_density", "Density", formatter=lambda v: f"{v:.1%}"),
    ColumnSpec("chain_score", "Chain score", formatter=lambda v: f"{v:.2f}",
               help="diameter / (2 x median dist). > 1.5 suggests chain."),
    ColumnSpec("", "Bridge faces", getter=lambda d: len(d.bridge_face_ids)),
]

# RunSummary (History run-detail strip) — None renders as the em-dash
RUN_SUMMARY_STRIP: List[ColumnSpec] = [
    ColumnSpec("n_faces", "Faces"),
    ColumnSpec("n_core", "Core"),
    ColumnSpec("n_noise", "Noise"),
    ColumnSpec("n_clusters_base", "Clusters (base)"),
    ColumnSpec("n_clusters_merged", "Clusters (merged)"),
]

# ForceMergePreview (3 gate PASS/FAIL badges + delta detail)
FORCE_MERGE_STRIP: List[ColumnSpec] = [
    ColumnSpec("", "Exemplar gate",
               getter=lambda p: "PASS" if p.passes_exemplar else "FAIL",
               delta=lambda p: f"d={p.exemplar_dist:.3f} <= {p.threshold:.3f}"),
    ColumnSpec("", "Support gate",
               getter=lambda p: "PASS" if p.passes_support else "FAIL",
               delta=lambda p: f"support={p.support}"),
    ColumnSpec("", "Diameter gate",
               getter=lambda p: "PASS" if p.passes_diameter else "FAIL",
               delta=lambda p: f"post={p.post_diameter:.3f}"),
]

# QualitySummary (Quality tab summary strip)
QUALITY_SUMMARY_STRIP: List[ColumnSpec] = [
    ColumnSpec("n_items", "Items"),
    ColumnSpec("n_decisions", "Decisions"),
    ColumnSpec("n_rejected", "Rejected"),
    ColumnSpec("", "Top reject gate", getter=lambda s: s.top_rejection_gate or "-"),
    ColumnSpec("", "Pass rate", getter=lambda s: f"{s.pass_rate * 100:.1f}%"),
]

# DashboardMetrics (Overview headline strip)
OVERVIEW_STRIP: List[ColumnSpec] = [
    ColumnSpec("total_runs", "Total runs"),
    ColumnSpec("", "Total faces ever", getter=lambda m: f"{m.total_faces_ever:,}"),
    ColumnSpec("", "Avg n_clusters", getter=_avg_clusters),
    ColumnSpec("", "Last run", getter=lambda m: _age(m.last_run_at)),
]

# Small count strips (tab-computed SimpleNamespace)
FACE_COUNT_STRIP: List[ColumnSpec] = [
    ColumnSpec("n_faces", "Faces"),
    ColumnSpec("n_assigned", "Assigned"),
    ColumnSpec("n_unassigned", "Unassigned"),
]
IMAGE_COUNT_STRIP: List[ColumnSpec] = [
    ColumnSpec("n_images", "Images"),
    ColumnSpec("n_pass", "Gate passed"),
]

__all__ = [
    "CLUSTER_METRIC_STRIP", "CLUSTER_DEBUG_STRIP", "RUN_SUMMARY_STRIP",
    "FORCE_MERGE_STRIP", "QUALITY_SUMMARY_STRIP", "OVERVIEW_STRIP",
    "FACE_COUNT_STRIP", "IMAGE_COUNT_STRIP",
]
