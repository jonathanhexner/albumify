"""spec-042 H2 — Load button for the History tab.

Renders the 'Load into analysis tabs' button with three states (already
loaded / incomplete / ready) and on click writes the typed
``LoadedRun`` into ``st.session_state``.
"""
from __future__ import annotations

import streamlit as st

from face_cluster.views.history import HistoryService, RunDetail


def render_load_button(detail: RunDetail, service: HistoryService) -> None:
    """Render the Load button with the right state for ``detail.row``.

    States:
        - Already loaded: green success message, no button.
        - Incomplete (status != complete or missing artifacts):
          disabled button + warning naming the missing files.
        - Complete + not loaded: enabled primary button.

    Side effects (on click):
        - Calls ``service.load_run(detail.row.id)`` → ``LoadedRun``.
        - Writes ``st.session_state["pipeline_result"]`` and
          ``st.session_state["current_source_album"]``.
        - Triggers ``st.rerun()`` so analysis tabs pick up the new state.
    """
    if _is_already_loaded(detail):
        st.success(
            f"Run `{detail.row.run_name or detail.row.run_id}` is currently loaded."
        )
        return

    if not detail.has_required_artifacts or detail.row.status != "complete":
        if detail.row.status != "complete":
            reason = f"status is `{detail.row.status}`, not 'complete'"
        else:
            reason = (
                "no loadable artifacts found in the run dir — expected "
                "`face_clustering.db` (v5 / v2 runs), `_v4/face_clustering.db` "
                "(transitional), or the legacy CSV trio "
                "(`faces.csv` + `clusters.csv` + `embeddings.npy`)"
            )
        st.warning(f"Cannot load this run: {reason}.")
        st.button(
            "Load into analysis tabs",
            type="primary",
            disabled=True,
            help="Run incomplete — see warning above",
            key=f"hist_load_disabled_{detail.row.id}",
        )
        return

    if st.button(
        "Load into analysis tabs",
        type="primary",
        key=f"hist_load_{detail.row.id}",
    ):
        loaded = service.load_run(detail.row.id)
        st.session_state["pipeline_result"] = loaded.pipeline_result
        st.session_state["current_source_album"] = loaded.source_album
        st.session_state["active_run_dir"] = str(loaded.output_dir)
        # spec-045 T050: Cluster Analysis tab reads ``current_run_dir`` (see
        # spec §7.2 resolver). Keep it in lockstep with active_run_dir.
        st.session_state["current_run_dir"] = str(loaded.output_dir)
        st.success(
            f"Loaded `{detail.row.run_name or detail.row.run_id}` "
            "— switch to an analysis tab to view it."
        )
        st.rerun()


def _is_already_loaded(detail: RunDetail) -> bool:
    """True iff the currently-loaded run matches this detail's output_dir."""
    loaded = st.session_state.get("pipeline_result")
    active_dir = st.session_state.get("active_run_dir")
    if loaded is None or not active_dir or not detail.row.output_dir:
        return False
    try:
        from pathlib import Path
        return Path(active_dir).resolve() == Path(detail.row.output_dir).resolve()
    except Exception:
        return False
