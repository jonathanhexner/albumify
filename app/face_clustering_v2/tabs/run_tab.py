"""spec-041 — Run tab driven by FCParams metadata, no widget literals.

Every UI-bound knob is declared in ``face_cluster.fc_params.FCParams``
with a ``json_schema_extra`` block. The widget factory at
``app/face_clustering_v2/widget_factory.py`` reads those hints and
renders the right Streamlit control. This tab is just the layout —
which groups to show in which expanders, in what order.

Adding a new knob: add a Field to FCParams. The widget appears
automatically in the group it declares.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.face_clustering_v2._profile_bar import render_profile_bar
from app.face_clustering_v2.pipeline import run_v2_pipeline
from app.face_clustering_v2.widget_factory import (
    build_params_from_state,
    render_group,
)


# Display order of the expanders. Groups not listed here are skipped
# (the merge / cap groups are gated by the merge_enabled checkbox).
_GROUP_ORDER = ["cluster", "quality", "exemplars", "optional"]
_GROUP_TITLES = {
    "cluster":   "Stage 5 · Cluster",
    "quality":   "Stage 3 · Quality Gate",
    "exemplars": "Stage 6 · Exemplars",
    "optional":  "Optional Stages",
    "merge":     "Merge Parameters",
    "cap":       "Diameter Cap (spec-031)",
}
_GROUP_EXPANDED = {
    "cluster":   True,
    "quality":   True,
    "exemplars": False,
    "optional":  True,
}
_GROUP_COLUMNS = {
    "cluster":   3,
    "quality":   3,
    "exemplars": 3,
    "optional":  3,
    "merge":     2,
    "cap":       2,
}


def render_run_tab() -> None:
    st.subheader("Run face clustering — v2")
    st.caption(
        "Runs the unified spec-040 pipeline (producer chain + FCAppRunner). "
        "Writes a schema v5 face_clustering.db and a row in the global "
        "action_log with `producer='fc_app_v2'`. Knobs are declared in "
        "`face_cluster/fc_params.py` — this tab renders them via the widget factory."
    )

    render_profile_bar()

    # --- I/O paths -------------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        src = st.text_input(
            "Source image directory",
            value=str(st.session_state.get("v2_src_dir", "")),
            help="Directory of JPG/PNG images. Subdirectories are not scanned.",
            key="v2_src_input",
        )
    with c2:
        out = st.text_input(
            "Output directory",
            value=str(st.session_state.get(
                "v2_out_dir",
                str(Path.home() / ".sim_bench" / "runs" / "v2_latest"),
            )),
            help="Destination for the v5 run artifacts.",
            key="v2_out_input",
        )

    # --- Top-level knob groups ------------------------------------------
    for group in _GROUP_ORDER:
        with st.expander(_GROUP_TITLES[group], expanded=_GROUP_EXPANDED.get(group, False)):
            render_group(group, columns=_GROUP_COLUMNS.get(group, 1))

    # --- Merge sub-panel (gated by merge_enabled) ------------------------
    if st.session_state.get("v2_merge_enabled", False):
        with st.expander(_GROUP_TITLES["merge"], expanded=True):
            render_group("merge", columns=_GROUP_COLUMNS["merge"])
        with st.expander(_GROUP_TITLES["cap"], expanded=False):
            render_group("cap", columns=_GROUP_COLUMNS["cap"])

    # --- Run -------------------------------------------------------------
    run_disabled = not (src and out)
    if st.button("Run", type="primary", key="v2_run_btn", disabled=run_disabled):
        if not Path(src).exists():
            st.error(f"Source directory does not exist: {src}")
            return
        st.session_state.v2_src_dir = src
        st.session_state.v2_out_dir = out

        params = build_params_from_state()
        if params is None:
            return  # build_params_from_state already emitted st.error

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
                params=params,
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
