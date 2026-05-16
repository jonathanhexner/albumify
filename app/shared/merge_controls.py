"""Shared Streamlit UI controls for face clustering merge parameters.

Used by both the main Album App (pipeline_runner.py) and the standalone
Face Clustering App (config_controls.py).  Single source of truth —
add new merge parameters here and both apps pick them up.
"""
from __future__ import annotations

import streamlit as st


def render_merge_params(key_prefix: str = "rc_") -> dict:
    """Render merge parameter controls. Returns dict of PipelineConfig-valid params.

    Args:
        key_prefix: Prefix for Streamlit widget keys to avoid collisions
                    when the same controls appear in multiple tabs/pages.
    """
    st.markdown("**Candidate & Exemplar Thresholds**")
    c1, c2 = st.columns(2)
    with c1:
        merge_candidate_thresh = st.slider(
            "merge_candidate_threshold", 0.01, 1.5, 0.45, step=0.01, key=f"{key_prefix}merge_candidate",
            help="Loose threshold for proposing merge candidates.",
        )
    with c2:
        merge_exemplar_thresh = st.slider(
            "merge_exemplar_threshold", 0.01, 1.5, 0.45, step=0.01, key=f"{key_prefix}merge_exemplar_thresh",
            help="Fixed exemplar threshold used when adaptive is disabled.",
        )
    st.markdown("**Adaptive Merge Threshold**")
    use_adaptive = st.checkbox(
        "use_adaptive_merge_threshold", value=True, key=f"{key_prefix}merge_use_adaptive",
        help="Use adaptive per-cluster thresholds instead of a fixed value.",
    )
    c1, c2 = st.columns(2)
    with c1:
        exemplar_pct = st.slider(
            "exemplar_percentile", 0, 100, 90, key=f"{key_prefix}merge_exemplar_pct",
            help="Percentile of intra-cluster exemplar distances used as T_local.",
        )
    with c2:
        global_pct = st.slider(
            "global_percentile", 0, 100, 75, key=f"{key_prefix}merge_global_pct",
            help="Percentile across all cluster thresholds used as T_global.",
        )
    c1, c2 = st.columns(2)
    with c1:
        alpha = st.slider(
            "alpha", 0.0, 3.0, 1.0, step=0.05, key=f"{key_prefix}merge_alpha",
            help="Weight of T_local in adaptive formula.",
        )
    with c2:
        beta = st.slider(
            "beta", 0.0, 3.0, 0.5, step=0.05, key=f"{key_prefix}merge_beta",
            help="Weight of T_global in adaptive formula.",
        )
    st.markdown("**Cross-Distance OR Gate (small clusters)**")
    use_cross_gate = st.checkbox(
        "merge_use_cross_gate", value=True, key=f"{key_prefix}merge_use_cross_gate",
        help="Allow Gate A to pass via p25 all-pairs cross-distance (OR path). Only applies to small clusters.",
    )
    c1, c2 = st.columns(2)
    with c1:
        cross_thresh = st.slider(
            "merge_cross_threshold", 0.01, 1.5, 0.40, step=0.01, key=f"{key_prefix}merge_cross_thresh",
            help="Threshold for p25 of ALL cross-cluster node-pair distances.",
            disabled=not use_cross_gate,
        )
    with c2:
        cross_max_size = st.slider(
            "merge_cross_max_size", 1, 20, 5, key=f"{key_prefix}merge_cross_max_size",
            help="OR path only when min(|A|,|B|) <= this. Prevents loosening for large-vs-large merges.",
            disabled=not use_cross_gate,
        )
    st.markdown("**Support Count Gate**")
    c1, c2 = st.columns(2)
    with c1:
        support_frac = st.slider(
            "merge_support_frac", 0.0, 1.0, 0.3, step=0.05, key=f"{key_prefix}merge_support_frac",
            help="Minimum fraction of the smaller cluster that must be in the cross-cluster K-NN.",
        )
    with c2:
        support_min = st.slider(
            "merge_support_min", 1, 20, 2, key=f"{key_prefix}merge_support_min",
            help="Minimum absolute number of supporting cross-cluster pairs.",
        )
    support_unique = st.checkbox(
        "merge_support_unique", value=False, key=f"{key_prefix}merge_support_unique",
        help="Use greedy bipartite matching: each node used at most once. More conservative than raw count.",
    )
    st.markdown("**Margin & Diameter Gates**")
    c1, c2 = st.columns(2)
    with c1:
        merge_margin = st.slider(
            "merge_margin", 0.0, 0.5, 0.0, step=0.01, key=f"{key_prefix}merge_margin",
            help="Min gap between best and second-best candidate distances. 0 = disabled.",
        )
    with c2:
        diameter_factor = st.slider(
            "diameter_expansion_factor", 1.0, 5.0, 1.5, step=0.05, key=f"{key_prefix}merge_diameter",
            help="Max allowed post-merge diameter relative to larger input cluster.",
        )
    st.markdown("**Absolute Diameter Cap (spec-031, runs after merge)**")
    cap_enabled = st.checkbox(
        "cluster_diameter_cap_enabled", value=False,
        key=f"{key_prefix}cap_enabled",
        help="Post-merge safety net: reject any merged cluster whose internal "
             "diameter exceeds an absolute ceiling. Reverts violators to their "
             "pre-merge component clusters.",
    )
    c1, c2 = st.columns(2)
    with c1:
        max_full_diameter = st.slider(
            "max_full_diameter", 0.3, 2.0, 1.2, step=0.05,
            key=f"{key_prefix}cap_max_full",
            help="Max pairwise cosine distance across ALL faces in the cluster.",
            disabled=not cap_enabled,
        )
    with c2:
        max_exemplar_diameter = st.slider(
            "max_exemplar_diameter", 0.2, 2.0, 0.8, step=0.05,
            key=f"{key_prefix}cap_max_exemplar",
            help="Max pairwise cosine distance restricted to the cluster's exemplars.",
            disabled=not cap_enabled,
        )
    return {
        "merge_candidate_threshold":      merge_candidate_thresh,
        "merge_exemplar_threshold":       merge_exemplar_thresh,
        "use_adaptive_merge_threshold":   use_adaptive,
        "merge_exemplar_percentile":      exemplar_pct,
        "merge_global_percentile":        global_pct,
        "merge_threshold_alpha":          alpha,
        "merge_threshold_beta":           beta,
        "merge_use_cross_gate":           use_cross_gate,
        "merge_cross_threshold":          cross_thresh,
        "merge_cross_max_size":           cross_max_size,
        "merge_support_frac":             support_frac,
        "merge_support_min":              support_min,
        "merge_support_unique":           support_unique,
        "merge_margin":                   merge_margin,
        "merge_diameter_expansion_factor": diameter_factor,
        "cluster_diameter_cap_enabled":   cap_enabled,
        "max_full_diameter":              max_full_diameter,
        "max_exemplar_diameter":          max_exemplar_diameter,
    }
