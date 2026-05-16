"""Tab 2: Recluster — re-run cluster/merge/export on existing embeddings."""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import streamlit as st

from face_cluster import FaceClusteringPipeline, PipelineConfig
from face_cluster.run_naming import RunDirSpec, allocate_run_dir

from state import _AsyncState, _invalidate_run_caches, _RC_PARAM_KEYS
from session_helpers import _append_recluster_step_to_session
from run_panels import (
    _list_available_runs, _latest_log_file,
    _render_live_log, _render_log_expander, _render_stage_plan,
)
from config_controls import _render_merge_params
from _profile_bar import render_profile_bar, render_profile_save_bar


def _render_merge_criteria_reference():
    """Static explanation of the 4 merge criteria with formulas and tuning tips."""
    st.markdown("""
Two clusters **Ci** and **Cj** are merged only when **all four** checks pass:

---

#### (A) Exemplar Agreement
**Check:** `min_exemplar_dist(Ci, Cj) <= merge_threshold`

When adaptive thresholds are on:
```
T_local  = max(T_Ci, T_Cj)   where T_Ck = P{exemplar_percentile} of intra-cluster exemplar distances
T_global = P{global_percentile} across all cluster thresholds
merge_threshold = alpha * T_local + beta * T_global
```
When adaptive is off: `merge_threshold = merge_exemplar_threshold` (fixed)

---

#### (B) Support Count
**Check:** `cross_pairs_below_threshold >= max(support_frac * min(|Ci|,|Cj|), support_min)`

---

#### (C) Margin vs Next Best
**Check:** For each exemplar in Ci, Cj must be the nearest other cluster by at least `merge_margin`.

---

#### (D) Post-Merge Diameter
**Check:** `diameter(Ci + Cj) <= diameter_expansion_factor * max(diameter(Ci), diameter(Cj))`
""")


def _render_profile_bar() -> None:
    render_profile_bar(_RC_PARAM_KEYS, widget_prefix="rc_")


def _render_profile_save_bar() -> None:
    render_profile_save_bar(_RC_PARAM_KEYS, widget_prefix="rc_")


def render_recluster_tab():
    st.header("Recluster")
    st.caption(
        "Re-run **cluster -> exemplars -> merge -> export** on an existing run's embeddings. "
        "Skips detect/embed/quality — reuses the original core/holdout split."
    )
    _render_profile_bar()

    runs = _list_available_runs(complete_only=True)
    if not runs:
        st.info("No completed runs found. Run a full pipeline first.")
        return

    run_labels = {
        r["run_id"]: f"{r['output_folder']}  |  {r['album']}  |  {r['faces']} faces  |  {r['clusters']} clusters"
        for r in runs
    }
    selected_run_id = st.selectbox(
        "Source run", list(run_labels.keys()),
        format_func=lambda rid: run_labels[rid],
        key="rc_source_run_id",
    )
    selected_run   = next(r for r in runs if r["run_id"] == selected_run_id)
    source_dir_str = selected_run["_dir"]

    # Derive source_album from session state or source dir name
    source_album = (
        st.session_state.get("current_source_album")
        or selected_run.get("album")
        or Path(source_dir_str).parent.parent.name
        or "unknown"
    )
    results_root = Path("results")
    st.caption(f"Album: `{source_album}`")
    output_dir_str = st.text_input(
        "Output directory (auto-allocated on run)",
        value=str(results_root / source_album / "recluster_N"),
        disabled=True,
        key="rc_output_dir_preview",
        help="Unique directory allocated automatically to prevent overwrites.",
    )

    st.divider()

    with st.expander("Base Clustering Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            rc_K = st.slider(
                "K", 1, 100, 5, key="rc_K",
                help="PipelineConfig.K. Mutual kNN edge requires both nodes in each other's top-K.",
            )
        with c2:
            rc_dist = st.slider(
                "distance_threshold", 0.01, 1.0, 0.35, step=0.01, key="rc_dist",
                help=(
                    "PipelineConfig.distance_threshold. Max cosine distance for kNN edge. "
                    "Lower = stricter."
                ),
            )
        with c3:
            rc_min_cluster = st.slider(
                "min_cluster_size", 1, 50, 2, key="rc_min_cluster",
                help="PipelineConfig.min_cluster_size. Components below this become noise.",
            )
        c1, c2, c3 = st.columns(3)
        with c1:
            rc_N_exemplars = st.slider(
                "N_exemplars_max", 1, 100, 10, key="rc_N_exemplars",
                help="PipelineConfig.N_exemplars_max. Max exemplars selected per cluster.",
            )
        with c2:
            rc_d10_thresh = st.slider(
                "exemplars_d10_threshold", 0.01, 1.0, 0.35, step=0.01, key="rc_d10_thresh",
                help=(
                    "PipelineConfig.exemplars_d10_threshold. Max d10 (10th-NN cosine distance) "
                    "for exemplar eligibility."
                ),
            )
        with c3:
            rc_suppression = st.slider(
                "exemplar_suppression_radius", 0.01, 1.0, 0.2, step=0.01, key="rc_suppression",
                help="PipelineConfig.exemplar_suppression_radius. Min cosine distance between selected exemplars.",
            )

    st.markdown("**Optional stages**")
    c1, c2, c3 = st.columns(3)
    with c1:
        rc_split = st.checkbox(
            "split_enabled", key="rc_split",
            help="PipelineConfig.split_enabled. Run cluster split safeguard.",
        )
    with c2:
        rc_merge = st.checkbox(
            "merge_enabled", key="rc_merge",
            help="PipelineConfig.merge_enabled. Run conservative merge after exemplars.",
        )
    with c3:
        rc_attach = st.checkbox(
            "attach_enabled", key="rc_attach",
            help="PipelineConfig.attach_enabled. Attach holdout faces to clusters.",
        )

    with st.expander("Merge Parameters (applied when merge is enabled)", expanded=rc_merge):
        merge_params = _render_merge_params()
    with st.expander("Merge Criteria Reference", expanded=False):
        _render_merge_criteria_reference()

    st.divider()

    recluster_worker: _AsyncState = st.session_state.recluster_worker

    if recluster_worker is not None and recluster_worker.is_running:
        new_lines = recluster_worker.drain_logs()
        if new_lines:
            st.session_state.pipeline_log.extend(new_lines)
        elapsed     = recluster_worker.elapsed_s()
        elapsed_str = f"  ({elapsed:.0f}s elapsed)" if elapsed else ""
        st.info(f"Reclustering in progress...{elapsed_str}  |  Output: `{output_dir_str}`")
        _render_stage_plan(Path(output_dir_str) if output_dir_str else None)
        _render_live_log(st.session_state.pipeline_log)
        time.sleep(0.5)
        st.rerun()
        return

    if recluster_worker is not None and recluster_worker.is_done:
        result_dir = str(recluster_worker.result.output_dir) if recluster_worker.result else None
        if result_dir and result_dir != st.session_state.recluster_promoted_dir:
            st.session_state.pipeline_result        = recluster_worker.result
            st.session_state.recluster_promoted_dir = result_dir
            st.session_state.current_source_album   = source_album
            _invalidate_run_caches()
            _append_recluster_step_to_session(recluster_worker.result)
        result      = recluster_worker.result
        elapsed     = recluster_worker.elapsed_s()
        elapsed_str = f" in {elapsed:.1f}s" if elapsed else ""
        st.success(f"Recluster complete{elapsed_str}. Switch to **Clusters (Base)** to analyse.")
        if result is not None:
            cr  = result.cluster_result
            mcr = result.merged_cluster_result
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clusters", mcr.n_clusters if mcr else cr.n_clusters)
            c2.metric("Noise",    mcr.n_noise    if mcr else cr.n_noise)
            if mcr:
                c3.metric("Merges performed", cr.n_clusters - mcr.n_clusters)
            c4.metric("Faces", len(result.faces))
            st.caption(f"Output: `{result.output_dir}`")
            log_file = _latest_log_file(Path(result.output_dir))
            if log_file:
                st.caption(f"Log: `{log_file}`")
        _render_log_expander(st.session_state.pipeline_log)

    if recluster_worker is not None and recluster_worker.has_error:
        elapsed     = recluster_worker.elapsed_s()
        elapsed_str = f" after {elapsed:.1f}s" if elapsed else ""
        st.error(f"Recluster failed{elapsed_str}: {recluster_worker.error}")
        log_file = _latest_log_file(Path(output_dir_str)) if output_dir_str else None
        if log_file:
            st.caption(f"Log: `{log_file}`")
        _render_log_expander(st.session_state.pipeline_log)

    _render_profile_save_bar()
    st.divider()

    if st.button("Recluster", type="primary", key="rc_run_button"):
        config = PipelineConfig(
            K=rc_K, distance_threshold=rc_dist, min_cluster_size=rc_min_cluster,
            N_exemplars_max=rc_N_exemplars, exemplars_d10_threshold=rc_d10_thresh,
            exemplar_suppression_radius=rc_suppression,
            split_enabled=rc_split, merge_enabled=rc_merge, attach_enabled=rc_attach,
            **merge_params,
        )
        _source = source_dir_str
        spec    = RunDirSpec(source_album=source_album, kind="recluster")
        _output = allocate_run_dir(spec, Path("results"))
        st.info(f"Output allocated: `{_output}`")

        def _run_recluster():
            run_config = dataclasses.replace(
                config,
                stages=["cluster", "exemplars", "merge", "export"],
                source_dir=str(_source),
                output_dir=str(_output),
            )
            return FaceClusteringPipeline().run(run_config)

        w = _AsyncState()
        st.session_state.recluster_worker       = w
        st.session_state.pipeline_log           = []
        st.session_state.active_run_dir         = _output
        st.session_state.pipeline_result        = None
        st.session_state.recluster_promoted_dir = None
        w.start(_run_recluster)
        st.rerun()
