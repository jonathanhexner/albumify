"""spec-040 Phase 5b — Run tab for FC App v2.

Thin UI wrapper around ``app.face_clustering_v2.pipeline.run_v2_pipeline``.
Collects source dir, output dir, and a few core config knobs; renders
progress; reports back the result. Anything more involved (history,
re-cluster, merge analysis) is deferred to follow-up tabs.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.face_clustering_v2.pipeline import run_v2_pipeline


def render_run_tab() -> None:
    st.subheader("Run face clustering — v2")
    st.caption(
        "Runs the unified spec-040 pipeline (producer chain + FCAppRunner). "
        "Writes a schema v5 face_clustering.db and a row in the global "
        "action_log with `producer='fc_app_v2'`."
    )

    src = st.text_input(
        "Source image directory",
        value=str(st.session_state.get("v2_src_dir", "")),
        help="Directory of JPG/PNG images. Subdirectories are not scanned.",
        key="v2_src_input",
    )
    out = st.text_input(
        "Output directory",
        value=str(st.session_state.get("v2_out_dir", str(Path.home() / ".sim_bench" / "runs" / "v2_latest"))),
        help="Destination for the v5 run artifacts.",
        key="v2_out_input",
    )

    st.markdown("**Core clustering config** (other knobs use defaults — see CONCRETE_PLAN Phase 5b).")
    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.number_input("K (kNN neighbors)", min_value=1, max_value=50, value=5, step=1, key="v2_K")
    with c2:
        thr = st.number_input("distance_threshold", min_value=0.05, max_value=1.0, value=0.35, step=0.05, key="v2_dt")
    with c3:
        min_cs = st.number_input("min_cluster_size", min_value=1, max_value=10, value=2, step=1, key="v2_mcs")

    merge_on = st.checkbox("merge_enabled (run conservative merger)", value=True, key="v2_merge")
    cap_on = st.checkbox(
        "cluster_diameter_cap_enabled (spec-031 safety rail)",
        value=True, key="v2_cap",
        help="Rejects merges whose combined cluster diameter exceeds the absolute ceiling.",
    )

    if st.button("Run", type="primary", key="v2_run_btn"):
        if not src or not Path(src).exists():
            st.error(f"Source directory does not exist: {src}")
            return
        st.session_state.v2_src_dir = src
        st.session_state.v2_out_dir = out

        cfg = {
            "K": int(k), "distance_threshold": float(thr),
            "min_cluster_size": int(min_cs),
            "merge_enabled": bool(merge_on),
            "cluster_diameter_cap_enabled": bool(cap_on),
            # Permissive defaults for gates the InsightFace pipeline doesn't
            # currently populate — matches face_cluster_bridge.build_fc_config.
            "blur_min": 0.0,
            "yaw_max": 999.0, "pitch_max": 999.0, "roll_max": 999.0,
        }
        # Apply same cfg to every clustering step (each step picks only its
        # relevant fields via _build_fc_config).
        from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
        step_configs = {name: cfg for name in UNIFIED_CLUSTERING_STEPS}

        progress = st.progress(0.0)
        status = st.empty()

        def _cb(step: str, fraction: float, msg: str) -> None:
            try:
                progress.progress(min(1.0, max(0.0, float(fraction))))
            except Exception:
                pass
            status.text(f"{step}: {msg}")

        with st.spinner("Running v2 pipeline..."):
            result = run_v2_pipeline(
                src_dir=Path(src),
                output_dir=Path(out),
                step_configs=step_configs,
                progress_cb=_cb,
            )

        progress.progress(1.0)
        if result.success:
            st.success(
                f"Run complete — {result.n_clusters} clusters from {result.n_faces} "
                f"faces across {result.n_images} images "
                f"(noise={result.n_noise}). DB: {result.db_path}"
            )
            st.session_state.v2_last_run_dir = str(result.output_dir)
            st.session_state.v2_last_result = result
        else:
            st.error(f"Run failed: {result.error_message}")
