"""Merge and attach decision card components."""

import pandas as pd
import streamlit as st

from app.face_clustering_debug.models.schemas import AttachDecision, MergeDecision


def render_merge_decision(decision: MergeDecision) -> None:
    """Render a merge decision card - handles both hybrid_knn and hybrid_closest_face."""
    icon = "✅" if decision.merged else "❌"
    color = "green" if decision.merged else "red"
    st.markdown(
        f":{color}[{icon} **Cluster {decision.cluster_a} ↔ Cluster {decision.cluster_b}**] "
        f"— `{decision.reason}`"
    )

    # Detect algorithm type by checking for hybrid_closest_face specific fields
    is_closest_face = decision.d3_cross_a is not None or decision.fits_a > 0

    if is_closest_face:
        _render_closest_face_metrics(decision)
    else:
        _render_hdbscan_knn_metrics(decision)


def _render_hdbscan_knn_metrics(decision: MergeDecision) -> None:
    """Render metrics for hybrid_hdbscan_knn (exemplar-based)."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Threshold T", f"{decision.threshold_used:.3f}")
    c2.metric("Pairs ≤ T", decision.pairs_within_threshold)
    c3.metric("Min distance", f"{decision.min_distance:.3f}")
    c4.metric("Distinct A", decision.exemplars_a_involved)
    c5.metric("Distinct B", decision.exemplars_b_involved)


def _render_closest_face_metrics(decision: MergeDecision) -> None:
    """Render metrics for hybrid_closest_face (face-based d3_cross)."""
    # Show multiplier prominently
    multiplier = decision.merge_threshold_multiplier if decision.merge_threshold_multiplier else 1.5
    st.info(f"**Merge threshold multiplier: {multiplier}×** — Effective thresholds: "
            f"T_A×{multiplier} = {decision.threshold_a * multiplier:.3f}, "
            f"T_B×{multiplier} = {decision.threshold_b * multiplier:.3f}")

    # Top row: thresholds and results
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("T_A (raw)", f"{decision.threshold_a:.3f}")
    c2.metric("T_B (raw)", f"{decision.threshold_b:.3f}")
    c3.metric("Total Fits", f"{decision.n_fits_total}")
    c4.metric("Required", f"{decision.merge_min_faces}")
    c5.metric("Early Exit Dist", f"{decision.min_distance:.3f}")

    # Second row: fits breakdown with all distances
    st.markdown("---")
    st.markdown("**Per-Face d3_cross Distances:**")

    n_faces_a = len(decision.d3_cross_a) if decision.d3_cross_a else 0
    n_faces_b = len(decision.d3_cross_b) if decision.d3_cross_b else 0
    eff_t_a = decision.threshold_a * multiplier
    eff_t_b = decision.threshold_b * multiplier

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Cluster A → B** ({decision.fits_a}/{n_faces_a} fit, threshold={eff_t_a:.3f})")
        if decision.d3_cross_a:
            _render_distances_inline(decision.d3_cross_a, eff_t_a, "A")
        else:
            st.caption("No d3_cross data")

    with col_b:
        st.markdown(f"**Cluster B → A** ({decision.fits_b}/{n_faces_b} fit, threshold={eff_t_b:.3f})")
        if decision.d3_cross_b:
            _render_distances_inline(decision.d3_cross_b, eff_t_b, "B")
        else:
            st.caption("No d3_cross data")


def _render_distances_inline(distances: list, threshold: float, label: str) -> None:
    """Render all distances inline with fit/not-fit indicators."""
    rows = []
    for i, d3 in enumerate(distances):
        fits = d3 <= threshold
        rows.append({
            "Face": f"{label}{i}",
            "d3_cross": f"{d3:.4f}",
            "≤ T": "✅" if fits else "❌",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=min(200, 35 * len(rows) + 38))


def render_attach_decision(decision: AttachDecision, get_crop_fn=None) -> None:
    """Render an attachment decision card with face crop and candidate table."""
    attached = decision.attached_to is not None
    icon, color = ("✅", "green") if attached else ("🔴", "red")
    target = f"cluster **{decision.attached_to}**" if attached else "**noise**"
    st.markdown(f":{color}[{icon} **Face #{decision.face_index}**] → {target}")

    if not decision.candidates:
        return

    col_face, col_table = st.columns([1, 4])

    if get_crop_fn is not None:
        with col_face:
            crop = get_crop_fn(decision.face_index)
            if crop:
                st.image(crop, caption=f"Face #{decision.face_index}", use_container_width=True)
            if attached:
                st.success(f"→ C{decision.attached_to}")
            else:
                st.error("→ noise")

    with col_table:
        rows = []
        for c in decision.candidates:
            # JSON stores: cluster, threshold, matches, required, min_dist, qualifies
            rows.append({
                "Cluster": c.get("cluster", c.get("cluster_id", "?")),
                "T": f"{c.get('threshold', 0.0):.3f}",
                "Matches": c.get("matches", c.get("n_within", "?")),
                "Required": c.get("required", "?"),
                "Min Dist": f"{c.get('min_dist', c.get('distance', 0.0)):.3f}",
                "Qualifies": "✅" if c.get("qualifies", c.get("qualified", False)) else "❌",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
