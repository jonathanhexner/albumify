"""Tab 1: Run — full pipeline execution."""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import streamlit as st

from face_cluster import FaceClusteringPipeline, PipelineConfig
from face_cluster.cache import clear_embed_cache, get_cache_info

from state import _AsyncState, _invalidate_run_caches
from session_helpers import _create_session_from_result
from run_panels import _render_live_log, _render_log_expander, _render_stage_plan, _render_run_files_panel
from config_controls import _render_merge_params
from constants import _LOG_MAX_STORED


def _render_embed_cache_status(output_dir_str: str) -> None:
    if not output_dir_str:
        return
    output_dir = Path(output_dir_str)
    info = get_cache_info(output_dir)
    if info is None:
        st.caption("Embed cache: none — first run will embed all faces")
        return
    created  = info.get("created_at", "")[:16].replace("T", " ")
    n_faces  = info.get("n_faces", "?")
    saved_s  = info.get("embed_time_seconds", 0)
    col_info, col_btn = st.columns([4, 1])
    col_info.caption(
        f"Embed cache: {n_faces} faces, created {created} (saves ~{saved_s:.0f}s on next run)"
    )
    if col_btn.button("Clear cache", key="clear_embed_cache_btn"):
        cleared = clear_embed_cache(output_dir)
        st.success("Embed cache cleared — next run will re-embed all faces.") if cleared else st.info("No embed cache found.")
        st.rerun()


def render_run_tab():
    st.header("Run Pipeline")
    col1, col2 = st.columns(2)
    with col1:
        image_dir = st.text_input(
            "Image directory",
            value=st.session_state.last_image_dir,
            placeholder=r"D:\Google_Germany",
        )
    with col2:
        output_dir_str = st.text_input(
            "Session directory",
            value=st.session_state.last_output_dir,
            placeholder=r"results\my_album",
            help="Session root folder. Pipeline output goes to <session>/base/.",
        )
    with st.expander("How the clustering algorithm works"):
        st.markdown("""
**Mutual kNN Graph + Connected Components**

1. **Build a mutual kNN graph.** An edge between A and B only if both are in each other's top-K
   **and** their cosine distance is below `distance_threshold`.
2. **Find connected components.** Each component becomes a cluster.
   Components smaller than `min_cluster_size` become noise.

| Parameter | Lower value | Higher value |
|-----------|------------|--------------|
| `K` | Fewer edges, more noise | More edges, risk of chain connections |
| `distance_threshold` | Strict separation | Looser, more false edges |
| `min_cluster_size` | Keeps small clusters | More noise |

**Optional stages**: Split, Merge, Attach
""")
    with st.expander("Pipeline Config", expanded=True):
        st.markdown("**Stage 1 · Discover** — scan image dir for .jpg/.jpeg/.png/.heic")
        st.markdown("**Stage 2 · Embed** — InsightFace buffalo_l: detect faces, ArcFace 512-d embeddings")
        _render_embed_cache_status(output_dir_str)
        st.divider()
        st.markdown("**Stage 3 · Quality Gate**")
        c1, c2, c3 = st.columns(3)
        with c1:
            blur_min = st.slider("blur_min", 0.0, 500.0, 50.0, 5.0,
                help="Min Laplacian variance.")
        with c2:
            max_faces = st.slider("max_faces_per_image_core", 1, 50, 3,
                help="Keep only the N largest faces per source image.")
        with c3:
            min_face_area = st.number_input("min_face_area px (0=off)", min_value=0, value=0, step=500)
        c4 = st.columns(1)[0]
        with c4:
            det_score_min_val = st.number_input(
                "det_score_min (0=off)", min_value=0.0, max_value=1.0,
                value=0.0, step=0.05, format="%.2f",
                help="Min InsightFace detection confidence (0–1). 0 = disabled. Recommended: 0.7",
            )
        st.markdown("*Pose filter*")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            yaw_max = st.slider("yaw_max °", 5.0, 90.0, 30.0, 1.0)
        with c2:
            pitch_max = st.slider("pitch_max °", 5.0, 90.0, 25.0, 1.0)
        with c3:
            roll_max = st.slider("roll_max °", 5.0, 90.0, 25.0, 1.0)
        with c4:
            require_pose = st.checkbox("require_pose")
        st.divider()
        st.markdown("**Stage 4 · Crops** — save aligned 112x112 crops + crop_manifest.json")
        st.divider()
        st.markdown("**Stage 5 · Cluster** — mutual kNN graph + connected components")
        c1, c2, c3 = st.columns(3)
        with c1:
            K = st.slider("K (kNN neighbours)", 1, 100, 5)
        with c2:
            distance_threshold = st.slider("distance_threshold", 0.01, 1.0, 0.35, 0.01)
        with c3:
            min_cluster_size = st.slider("min_cluster_size", 1, 50, 2)
        st.divider()
        st.markdown("**Stage 6 · Exemplars**")
        c1, c2, c3 = st.columns(3)
        with c1:
            N_exemplars_max = st.slider("N_exemplars_max", 1, 100, 10)
        with c2:
            exemplars_d10_threshold = st.slider("exemplars_d10_threshold", 0.01, 1.0, 0.35, 0.01)
        with c3:
            exemplar_suppression_radius = st.slider("exemplar_suppression_radius", 0.01, 1.0, 0.2, 0.01)
        st.divider()
        st.markdown("**Stage 7 · Export** — write faces.csv, clusters.csv, embeddings.npy")
        st.divider()
        st.markdown("**Optional Stages**")
        c1, c2, c3 = st.columns(3)
        with c1:
            split_enabled  = st.checkbox("split_enabled")
        with c2:
            merge_enabled  = st.checkbox("merge_enabled")
        with c3:
            attach_enabled = st.checkbox("attach_enabled")

    run_merge_params: dict = {}
    if merge_enabled:
        with st.expander("Merge Parameters", expanded=True):
            run_merge_params = _render_merge_params(key_prefix="run_")

    worker: _AsyncState = st.session_state.pipeline_worker

    if worker is not None and worker.is_running:
        new_lines = worker.drain_logs()
        if new_lines:
            log = st.session_state.pipeline_log
            log.extend(new_lines)
            if len(log) > _LOG_MAX_STORED:
                st.session_state.pipeline_log = log[-_LOG_MAX_STORED:]
        st.info("Pipeline running — switch to other tabs freely.")
        st.subheader("Stage Execution Plan")
        _render_stage_plan(st.session_state.active_run_dir)
        st.subheader("Live Log")
        _render_live_log(st.session_state.pipeline_log)
        time.sleep(0.5)
        st.rerun()
        return

    if worker is not None and worker.is_done:
        if worker.result is not None and st.session_state.pipeline_result is None:
            st.session_state.pipeline_result = worker.result
            st.session_state.current_source_album = Path(
                worker.result.summary.get("source_album", "")
            ).name
            _invalidate_run_caches()
            _create_session_from_result(worker.result)
        result = st.session_state.pipeline_result
        if result is not None:
            st.success("Pipeline complete")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faces",    result.summary["n_faces"])
            m2.metric("Core",     result.summary["n_core"])
            m3.metric("Clusters", result.summary["n_clusters"])
            m4.metric("Noise",    result.summary["n_noise"])
            st.subheader("Stage Execution Plan")
            _render_stage_plan(st.session_state.active_run_dir)
            st.info("Switch to **Clusters (Base)** to analyse.")
            _render_log_expander(st.session_state.pipeline_log, label="Run log")
            with st.expander("Output Files", expanded=False):
                _render_run_files_panel(Path(result.output_dir), key_prefix="run_tab_")

    if worker is not None and worker.has_error:
        st.error(f"Pipeline failed: {worker.error}")
        _render_log_expander(st.session_state.pipeline_log, label="Error log")

    run_disabled = not (image_dir and output_dir_str)
    if st.button("Run Pipeline", type="primary", disabled=run_disabled):
        config = PipelineConfig(
            K=K, distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size, blur_min=blur_min,
            max_faces_per_image_core=max_faces,
            min_face_area=min_face_area if min_face_area > 0 else None,
            yaw_max=yaw_max, pitch_max=pitch_max, roll_max=roll_max,
            require_pose=require_pose,
            det_score_min=det_score_min_val if det_score_min_val > 0 else None,
            N_exemplars_max=N_exemplars_max,
            exemplars_d10_threshold=exemplars_d10_threshold,
            exemplar_suppression_radius=exemplar_suppression_radius,
            split_enabled=split_enabled, merge_enabled=merge_enabled,
            attach_enabled=attach_enabled, **run_merge_params,
        )
        _session_root = Path(output_dir_str)
        _base_dir     = _session_root / "base"

        def _run_pipeline():
            run_config = dataclasses.replace(
                config,
                stages=["discover", "embed", "quality", "crops",
                        "cluster", "exemplars", "merge", "export"],
                source_dir=str(image_dir),
                output_dir=str(_base_dir),
            )
            return FaceClusteringPipeline().run(run_config)

        st.session_state.last_image_dir  = image_dir
        st.session_state.last_output_dir = output_dir_str
        st.session_state.session_root    = _session_root
        new_worker = _AsyncState()
        st.session_state.pipeline_worker = new_worker
        st.session_state.pipeline_log    = []
        st.session_state.active_run_dir  = _base_dir
        st.session_state.pipeline_result = None
        st.session_state.active_session  = None
        new_worker.start(_run_pipeline)
        st.rerun()
