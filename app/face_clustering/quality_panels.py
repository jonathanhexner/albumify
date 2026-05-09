"""Quality report and cluster provenance rendering helpers."""
from __future__ import annotations

import json
from typing import Optional

import streamlit as st


def _render_quality_report(face_row: "pd.Series") -> None:
    gate_names  = ["blur", "pose_yaw", "pose_pitch", "area"]
    gate_labels = {"blur": "Blur", "pose_yaw": "Pose Yaw", "pose_pitch": "Pose Pitch", "area": "Area"}
    rows_present = [g for g in gate_names if f"quality_{g}_value" in face_row.index]
    if not rows_present:
        st.caption("Quality verdict: N/A (run predates spec 012)")
        return
    det = face_row.get("det_score")
    d10 = face_row.get("d10_score")
    with st.expander("Quality Report", expanded=False):
        for gate in rows_present:
            val    = face_row.get(f"quality_{gate}_value")
            passed = face_row.get(f"quality_{gate}_pass")
            if val is None or (hasattr(val, "__class__") and str(val) == "nan"):
                st.markdown(f"**{gate_labels[gate]}**: N/A")
                continue
            badge = ":green[PASS]" if passed else ":red[FAIL]"
            st.markdown(f"**{gate_labels[gate]}**: {float(val):.1f}  {badge}")
        if det is not None and str(det) != "nan":
            st.markdown(f"**Detection confidence**: {float(det):.3f}")
        if d10 is not None and str(d10) != "nan":
            st.markdown(f"**D10 score**: {float(d10):.3f}")


def _render_cluster_provenance(cluster_row: "pd.Series", merge_log: Optional[list]) -> None:
    origin = cluster_row.get("origin") if "origin" in cluster_row.index else None
    if origin is None or str(origin) == "nan":
        return
    parent_ids_raw = cluster_row.get("parent_cluster_ids", "[]")
    parents = json.loads(str(parent_ids_raw)) if parent_ids_raw else []
    with st.expander("Provenance", expanded=False):
        origin_colors = {
            "base": "blue", "auto_merge": "orange",
            "manual_merge": "violet", "remerge": "orange", "unknown": "gray",
        }
        color = origin_colors.get(str(origin), "gray")
        st.markdown(f"**Origin**: :{color}[{origin}]")
        if parents:
            st.markdown(f"**Merged from clusters**: {', '.join(str(p) for p in parents)}")
        if str(origin) == "auto_merge" and merge_log:
            relevant = [e for e in merge_log
                        if e.get("actually_merged") and
                        (e.get("cluster_a") in parents or e.get("cluster_b") in parents)]
            if relevant:
                st.caption(f"Merge log entries: {len(relevant)} decision(s) involved")
