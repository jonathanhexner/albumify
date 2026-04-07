"""Dynamic algorithm explanation component using clustering method metadata."""

from typing import Any, Dict, Optional

import streamlit as st

from sim_bench.clustering import load_clustering_method


def render_algorithm_explanation(
    algorithm: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Render algorithm explanation from the clustering method's doc_explanation.

    Args:
        algorithm: Algorithm name (e.g., 'hybrid_hdbscan_knn'). If None, shows general guide.
        params: Current parameter values from the clustering result.
    """
    with st.expander("📖 Algorithm Explanation", expanded=False):
        if algorithm is None:
            _render_general_guide()
            return

        # Try to load the clustering method to get its documentation
        try:
            config = {"algorithm": algorithm, "params": params or {}}
            clusterer = load_clustering_method(config)
            decision_info = clusterer.get_decision_info()
            _render_method_explanation(decision_info, params or {})
        except Exception as e:
            st.warning(f"Could not load documentation for '{algorithm}': {e}")
            _render_general_guide()


def _render_method_explanation(
    decision_info: Dict[str, Any],
    current_params: Dict[str, Any],
) -> None:
    """Render explanation from clustering method's decision_info."""
    algorithm = decision_info.get("algorithm", "Unknown")
    doc_explanation = decision_info.get("doc_explanation", "No documentation available.")
    decision_parameters = decision_info.get("decision_parameters", {})

    st.markdown(f"### {algorithm}")
    st.markdown(doc_explanation)

    if decision_parameters:
        st.markdown("---")
        st.markdown("### Decision Parameters")

        # Build parameter table
        rows = []
        for param_name, param_meta in decision_parameters.items():
            current_value = current_params.get(param_name, param_meta.get("current_value", param_meta.get("default")))
            default = param_meta.get("default", "—")
            description = param_meta.get("description", "")
            decision_role = param_meta.get("decision_role", "")

            # Format current value
            if isinstance(current_value, float):
                current_str = f"{current_value:.4f}"
            elif current_value is None:
                current_str = "auto"
            else:
                current_str = str(current_value)

            # Format default
            if isinstance(default, float):
                default_str = f"{default:.4f}"
            elif default is None:
                default_str = "auto"
            else:
                default_str = str(default)

            rows.append({
                "Parameter": f"`{param_name}`",
                "Current": current_str,
                "Default": default_str,
                "Decision Role": decision_role,
            })

        # Display as markdown table
        st.markdown("| Parameter | Current | Default | Decision Role |")
        st.markdown("|-----------|---------|---------|---------------|")
        for row in rows:
            st.markdown(
                f"| {row['Parameter']} | **{row['Current']}** | {row['Default']} | {row['Decision Role']} |"
            )

        # Show detailed descriptions in expander
        with st.expander("📝 Parameter Details"):
            for param_name, param_meta in decision_parameters.items():
                description = param_meta.get("description", "No description")
                decision_role = param_meta.get("decision_role", "")
                st.markdown(f"**{param_name}**: {description}")
                if decision_role:
                    st.caption(f"↳ {decision_role}")


def _render_general_guide() -> None:
    """Render general clustering algorithm guide (fallback)."""
    st.markdown("""
## Clustering Algorithm Guide

All methods use **cosine distance** (1 - cosine_similarity, range [0, 2]).
Typical thresholds: same person ~0.1-0.3, different people ~0.5+.

| Method | Threshold Source | Merge Criteria | Best For |
|--------|-----------------|----------------|----------|
| **hdbscan** | None (density-based) | Automatic | Simple datasets |
| **hybrid_hdbscan_knn** | Exemplar↔Exemplar pairwise | ≥3 exemplar pairs within T | General use |
| **hybrid_closest_face** | All faces d3 (k-th neighbor) | ≥2 faces fit into other cluster | Pose variation |
| **Tcore2all** | Exemplar→All-faces | Same as hybrid_knn | Wider pose spread |
| **merge_twotier** | Exemplar↔Exemplar | Primary + secondary rule | Conservative merging |
| **attach_strong1** | Exemplar↔Exemplar | + single exemplar at 0.8×T | Reduce noise fragments |

---

## Debugging Tips

- **Clusters not merging?** → Check d3_cross values vs thresholds
- **Over-merging?** → Raise `threshold_floor` or `merge_min_faces`
- **Pose splits?** → Try `hybrid_closest_face` with `merge_threshold_multiplier: 1.5`
- **Too much noise?** → Lower `attach_min_exemplars` or try `attach_strong1`

---

*Select a specific clustering result to see its algorithm's documentation and parameters.*
    """)


def render_decision_summary(
    algorithm: str,
    params: Dict[str, Any],
    threshold_a: float,
    threshold_b: float,
    outcome: str,
    outcome_details: Dict[str, Any],
) -> None:
    """Render a compact decision summary showing threshold vs actual values.

    Args:
        algorithm: Algorithm name
        params: Algorithm parameters
        threshold_a: Threshold for cluster A
        threshold_b: Threshold for cluster B
        outcome: 'merged' or 'rejected'
        outcome_details: Dict with algorithm-specific outcome data
    """
    is_merged = outcome.lower() == "merged"
    icon = "✅" if is_merged else "❌"
    color = "green" if is_merged else "red"

    st.markdown(f"### {icon} Decision: {outcome.upper()}")

    # Show thresholds
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Threshold A (T_A)", f"{threshold_a:.4f}")
    with col2:
        st.metric("Threshold B (T_B)", f"{threshold_b:.4f}")

    # Algorithm-specific details
    if "hybrid_closest" in algorithm:
        _render_closest_face_decision_summary(params, outcome_details)
    elif "hybrid" in algorithm or "hdbscan_knn" in algorithm:
        _render_hdbscan_knn_decision_summary(params, outcome_details)


def _render_hdbscan_knn_decision_summary(
    params: Dict[str, Any],
    details: Dict[str, Any],
) -> None:
    """Render decision summary for hybrid_hdbscan_knn."""
    merge_min_pairs = params.get("merge_min_pairs", 3)
    merge_min_distinct = params.get("merge_min_distinct", 2)
    pairs_within = details.get("pairs_within_threshold", 0)
    distinct_a = details.get("exemplars_a_involved", 0)
    distinct_b = details.get("exemplars_b_involved", 0)

    st.markdown("**Merge Criteria (Exemplar-Based):**")
    st.markdown(f"""
```
Cross-exemplar pairs within T: {pairs_within} (need ≥{merge_min_pairs}) {'✓' if pairs_within >= merge_min_pairs else '✗'}
Distinct exemplars from A: {distinct_a} (need ≥{merge_min_distinct}) {'✓' if distinct_a >= merge_min_distinct else '✗'}
Distinct exemplars from B: {distinct_b} (need ≥{merge_min_distinct}) {'✓' if distinct_b >= merge_min_distinct else '✗'}
```
    """)


def _render_closest_face_decision_summary(
    params: Dict[str, Any],
    details: Dict[str, Any],
) -> None:
    """Render decision summary for hybrid_closest_face."""
    merge_min_faces = params.get("merge_min_faces", 2)
    multiplier = params.get("merge_threshold_multiplier", 1.5)
    fits_a = details.get("fits_a", 0)
    fits_b = details.get("fits_b", 0)
    n_fits_total = fits_a + fits_b

    st.markdown("**Merge Criteria (Face-Based d3_cross):**")
    st.markdown(f"""
```
Merge threshold multiplier: {multiplier}×
Faces from A that fit into B: {fits_a}
Faces from B that fit into A: {fits_b}
Total fits: {n_fits_total} (need ≥{merge_min_faces}) {'✓' if n_fits_total >= merge_min_faces else '✗'}
```
    """)
