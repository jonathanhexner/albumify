"""spec-041 — Run tab driven by FCParams metadata, no widget literals.

Every UI-bound knob is declared in ``face_cluster.fc_params.FCParams``
with a ``json_schema_extra`` block. The widget factory at
``app/face_clustering_v2/widget_factory.py`` reads those hints and
renders the right Streamlit control. This tab is just the layout —
which groups to show in which expanders, in what order.

spec-050: I/O paths reworked. The user now supplies a **source dir**
and an **album name**; the per-run output dir is allocated as a fresh
``<base>/<uuid>/`` and the UUID is also the ``run_id`` recorded in
action_log. ``v2_last_run_dir`` is written to session state BEFORE the
pipeline runs so failed runs (or mid-run tab switches) still leave a
recoverable pointer.

Adding a new knob: add a Field to FCParams. The widget appears
automatically in the group it declares.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from app.face_clustering_v2._profile_bar import render_profile_bar
from app.face_clustering_v2.pipeline import run_v2_pipeline
from app.face_clustering_v2.widget_factory import (
    build_params_from_state,
    render_group,
)
from face_cluster.run_layout import allocate_run_dir


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

_DEFAULT_RUNS_BASE = Path.home() / ".sim_bench" / "runs"


def render_run_tab() -> None:
    st.subheader("Run face clustering — v2")
    st.caption(
        "Runs the unified spec-040 pipeline (producer chain + FCAppRunner). "
        "Writes a schema v5 face_clustering.db and a row in the global "
        "action_log with `producer='fc_app_v2'`. Knobs are declared in "
        "`face_cluster/fc_params.py` — this tab renders them via the widget factory."
    )

    render_profile_bar()

    # --- Required inputs -------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        src = st.text_input(
            "Source image directory",
            value=str(st.session_state.get("v2_src_dir", "")),
            help="Directory of JPG/PNG images. Subdirectories are not scanned.",
            key="v2_src_input",
        )
    with c2:
        album = st.text_input(
            "Album name (required)",
            value=str(st.session_state.get("v2_album", "")),
            help=(
                "Free-form label persisted to action_log.source_album. "
                "Used by the History tab and the Clusters tab's run picker "
                "to identify this run later."
            ),
            key="v2_album_input",
        )

    with st.expander("Advanced: base runs directory", expanded=False):
        base_dir_str = st.text_input(
            "Base directory for per-run output",
            value=str(st.session_state.get("v2_runs_base", _DEFAULT_RUNS_BASE)),
            help=(
                "Each run is allocated a fresh subdirectory "
                "`<base>/<uuid>/` containing face_clustering.db, crops, "
                "and exports. Default: ~/.sim_bench/runs."
            ),
            key="v2_runs_base_input",
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
    # Button stays enabled regardless of input state. Validation runs on
    # click so the user sees an explicit error instead of a silently-greyed
    # button (which was hard to read: text_input values commit only on
    # blur/Enter, so "looks filled in but button greyed" was confusing).
    if st.button("Run", type="primary", key="v2_run_btn"):
        if not src or not album.strip():
            st.error("Source directory and album name are both required.")
            return
        if not Path(src).exists():
            st.error(f"Source directory does not exist: {src}")
            return

        # Persist inputs so the next session sees the same values.
        st.session_state.v2_src_dir = src
        st.session_state.v2_album = album.strip()
        st.session_state.v2_runs_base = base_dir_str

        # Allocate the per-run output dir BEFORE doing any work, and write
        # it to session state immediately. If the pipeline raises, the
        # user can still find the run dir (and a `failed` action_log row).
        try:
            run_dir, run_id = allocate_run_dir(Path(base_dir_str), album.strip())
        except Exception as e:
            st.error(f"Could not allocate run directory under {base_dir_str}: {e}")
            return
        st.session_state.v2_last_run_dir = str(run_dir)
        st.session_state.v2_last_run_id = run_id
        tab_start("run", run_dir)

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

        with st.spinner(f"Running v2 pipeline into {run_dir.name[:8]}..."):
            result = run_v2_pipeline(
                src_dir=Path(src),
                run_dir=run_dir,
                run_id=run_id,
                album=album.strip(),
                params=params,
                profile=st.session_state.get("v2_profile_last_loaded"),
                progress_cb=_cb,
            )

        progress.progress(1.0)
        if result.success:
            tab_done("run", n_clusters=result.n_clusters, n_faces=result.n_faces,
                     n_images=result.n_images, n_noise=result.n_noise)
            st.success(
                f"Run complete — {result.n_clusters} clusters from {result.n_faces} "
                f"faces across {result.n_images} images "
                f"(noise={result.n_noise}). Run dir: {result.output_dir}"
            )
            st.session_state.v2_last_result = result
        else:
            tab_skipped("run", "pipeline_failed")
            st.error(f"Run failed: {result.error_message}")
