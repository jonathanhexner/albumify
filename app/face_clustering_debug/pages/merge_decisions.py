"""Merge decisions page — cluster distances vs thresholds, and decision explorer."""

import numpy as np
import pandas as pd
import streamlit as st

from app.face_clustering_debug.components.algorithm_explanation import render_algorithm_explanation
from app.face_clustering_debug.components.decision_card import render_merge_decision
from app.face_clustering_debug.components.distance_heatmap import render_distance_heatmap
from app.face_clustering_debug.services.protocols import DataLoaderProtocol


def render_merge_decisions_page(loader: DataLoaderProtocol, method: str) -> None:
    st.header("🔗 Merge Decisions")

    result = loader.load_clustering_result(method)
    if result is None:
        st.error(f"No results for method **{method}**")
        return

    # Show dynamic algorithm explanation with current params
    render_algorithm_explanation(algorithm=result.algorithm, params=result.params)

    if not result.merge_decisions:
        st.warning("⚠️ No merge decisions in this result file (generated without debug data).")
        st.info("Re-run the benchmark to get full debug data:")
        st.code("python scripts/benchmark_face_clustering.py --album-path <path/to/album>")
        return

    decisions = result.merge_decisions
    n_merged = sum(1 for d in decisions if d.merged)

    m1, m2, m3 = st.columns(3)
    m1.metric("Pairs evaluated", len(decisions))
    m2.metric("✅ Merged", n_merged)
    m3.metric("❌ Rejected", len(decisions) - n_merged)

    st.subheader("📋 Cluster Distances vs Threshold")
    st.caption(
        "**T_A / T_B** = each cluster's own threshold (Q_p of exemplar pairwise dists, clamped to [floor, ceiling]).  "
        "**T_used** = threshold that blocked/triggered the decision (from bidirectional check).  "
        "**Gap** = min_dist − T_used.  Sorted by gap ascending."
    )
    st.dataframe(_build_distance_table(decisions, result.params), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔍 Inspect a Decision")

    filter_opt = st.radio("Show", ["All", "Merged ✅", "Rejected ❌"], horizontal=True)
    filtered = _filter(decisions, filter_opt)

    if not filtered:
        st.info("No decisions match the filter.")
        return

    selected_idx = st.selectbox(
        "Select pair",
        range(len(filtered)),
        format_func=lambda i: (
            f"{'✅' if filtered[i].merged else '❌'}  "
            f"C{filtered[i].cluster_a} ↔ C{filtered[i].cluster_b}  "
            f"dist={filtered[i].min_distance:.3f}  "
            f"T_A={filtered[i].threshold_a:.3f}  T_B={filtered[i].threshold_b:.3f}  "
            f"({filtered[i].reason})"
        ),
    )

    decision = filtered[selected_idx]
    render_merge_decision(decision)
    _render_exemplar_analysis(decision, result.params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_exemplar_analysis(decision, params: dict) -> None:
    """Show per-exemplar min distances and cross-distance heatmap."""
    floor = params.get("threshold_floor", "?")
    ceiling = params.get("threshold_ceiling", "?")
    percentile = params.get("threshold_percentile", "?")

    st.markdown(
        f"**Params:** T = Q{percentile} of exemplar pairwise dists, "
        f"floor={floor}, ceiling={ceiling}  |  "
        f"merge_min_pairs={params.get('merge_min_pairs','?')}  "
        f"merge_min_distinct={params.get('merge_min_distinct','?')}"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        _render_exemplar_min_table(
            decision.min_dists_a, decision.threshold_a, "A", decision.threshold_b
        )
    with col_b:
        _render_exemplar_min_table(
            decision.min_dists_b, decision.threshold_b, "B", decision.threshold_a
        )

    if decision.cross_distances is not None:
        mat = np.array(decision.cross_distances)
        row_labels = [f"A{i}" for i in range(mat.shape[0])]
        col_labels = [f"B{j}" for j in range(mat.shape[1])]
        render_distance_heatmap(
            mat, row_labels=row_labels, col_labels=col_labels,
            title=(
                f"Cross-exemplar distances  "
                f"T_A={decision.threshold_a:.3f}  T_B={decision.threshold_b:.3f}"
            ),
        )


def _render_exemplar_min_table(
    min_dists, own_threshold: float, side: str, other_threshold: float
) -> None:
    """Table: for each exemplar on one side, its closest distance to the other cluster."""
    if not min_dists:
        st.caption(f"No per-exemplar data for cluster {side}.")
        return

    rows = []
    for i, d in enumerate(min_dists):
        within_own = d <= own_threshold
        within_other = d <= other_threshold
        rows.append({
            f"Exemplar {side}": i,
            "Min dist to other": round(d, 3),
            f"≤ T_{side} ({own_threshold:.3f})": "✅" if within_own else "❌",
            f"≤ T_other ({other_threshold:.3f})": "✅" if within_other else "❌",
        })

    st.caption(f"**Cluster {side}** — per-exemplar closest distance to cluster {'B' if side == 'A' else 'A'}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_distance_table(decisions: list, params: dict) -> pd.DataFrame:
    floor = params.get("threshold_floor", float("nan"))
    ceiling = params.get("threshold_ceiling", float("nan"))
    rows = [
        {
            "C_A": d.cluster_a,
            "C_B": d.cluster_b,
            "Min Dist": round(d.min_distance, 3),
            "T_A": round(d.threshold_a, 3),
            "T_B": round(d.threshold_b, 3),
            "T_used": round(d.threshold_used, 3),
            "Gap": round(d.min_distance - d.threshold_used, 3),
            "Floor": round(floor, 3) if isinstance(floor, float) else floor,
            "Ceil": round(ceiling, 3) if isinstance(ceiling, float) else ceiling,
            "Result": "✅" if d.merged else "❌",
            "Reason": d.reason,
        }
        for d in decisions
    ]
    return pd.DataFrame(sorted(rows, key=lambda r: r["Gap"]))


def _filter(decisions: list, opt: str) -> list:
    if opt == "Merged ✅":
        return [d for d in decisions if d.merged]
    if opt == "Rejected ❌":
        return [d for d in decisions if not d.merged]
    return decisions
