"""Attach decisions page — why noise points attached or stayed noise."""

import pandas as pd
import streamlit as st

from app.face_clustering_debug.components.algorithm_explanation import render_algorithm_explanation
from app.face_clustering_debug.components.decision_card import render_attach_decision
from app.face_clustering_debug.services.protocols import DataLoaderProtocol


def render_attach_decisions_page(loader: DataLoaderProtocol, method: str) -> None:
    st.header("📎 Attachment Decisions")
    st.caption(
        "For each HDBSCAN noise point, shows which clusters were considered "
        "and why it attached (or didn't). "
        "**Matches** = exemplars within threshold T.  **Required** = attach_min_exemplars."
    )

    result = loader.load_clustering_result(method)
    if result is None:
        st.error(f"No results for method **{method}**")
        return

    # Show dynamic algorithm explanation with current params
    render_algorithm_explanation(algorithm=result.algorithm, params=result.params)

    if not result.attach_decisions:
        st.warning("⚠️ No attachment decisions — file was generated without debug data.")
        st.info("Select **hybrid_knn** (it has debug data), or re-run the benchmark.")
        return

    decisions = result.attach_decisions
    n_attached = sum(1 for d in decisions if d.attached_to is not None)

    m1, m2, m3 = st.columns(3)
    m1.metric("Noise points evaluated", len(decisions))
    m2.metric("✅ Attached", n_attached)
    m3.metric("🔴 Stayed noise", len(decisions) - n_attached)

    # Summary table
    rows = [
        {
            "Face": d.face_index,
            "Result": f"→ cluster {d.attached_to}" if d.attached_to is not None else "noise",
            "Candidates checked": len(d.candidates),
            "Qualifying clusters": sum(1 for c in d.candidates
                                       if c.get("qualifies", c.get("qualified", False))),
        }
        for d in decisions
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.divider()

    # Detail explorer
    st.subheader("🔍 Inspect a Decision")
    status_filter = st.radio("Show", ["All", "Attached ✅", "Stayed Noise 🔴"], horizontal=True)
    filtered = _filter(decisions, status_filter)

    if not filtered:
        st.info("No decisions match the filter.")
        return

    selected_idx = st.selectbox(
        "Select face",
        range(len(filtered)),
        format_func=lambda i: (
            f"Face #{filtered[i].face_index}  "
            f"{'→ C' + str(filtered[i].attached_to) if filtered[i].attached_to is not None else '→ noise'}"
        ),
        key="attach_select",
    )

    render_attach_decision(filtered[selected_idx], get_crop_fn=loader.get_face_crop)


def _filter(decisions: list, status: str) -> list:
    if status == "Attached ✅":
        return [d for d in decisions if d.attached_to is not None]
    if status == "Stayed Noise 🔴":
        return [d for d in decisions if d.attached_to is None]
    return decisions
