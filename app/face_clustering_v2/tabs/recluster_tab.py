"""spec-063 — v2 Recluster tab orchestrator.

Pure orchestration. Layout: prior-run picker -> params editor -> button.
SIGHTING-079 lesson: sync compute + ``st.spinner``, never background polling.
"""
from __future__ import annotations

import logging

import streamlit as st

from app.face_clustering_v2.components.run_picker import render_run_picker
from app.face_clustering_v2.ui_spec import UI_SPEC, fields_by_group
from app.face_clustering_v2.widget_factory import (
    build_params_from_state,
    render_field,
)
from face_cluster.repositories import RunHistoryRepository
from face_cluster.views.recluster import ReclusterService

logger = logging.getLogger(__name__)

_GROUP_ORDER = ["cluster", "quality", "exemplars", "optional", "merge", "cap"]
_GROUP_TITLES = {
    "cluster": "Cluster", "quality": "Quality Gate", "exemplars": "Exemplars",
    "optional": "Optional Stages", "merge": "Merge", "cap": "Diameter Cap",
}
_KEY_PREFIX = "recluster_"  # namespace; Run tab uses the empty prefix


def render_recluster_tab() -> None:
    """Render the Recluster tab. Writes ``current_run_dir`` on success."""
    st.header("Recluster")
    st.caption(
        "Re-cluster a prior run with new parameters. Producer steps "
        "(detect / align / embed) are skipped — only the 8-step clustering "
        "chain runs."
    )

    service = ReclusterService(RunHistoryRepository())

    picked = render_run_picker(label="Prior run", key="v2_recluster_picker")
    if picked is None:
        return
    if picked.is_orphan:
        st.warning(
            f"Selected run's directory is missing ({picked.output_dir}). "
            "Pick a non-orphan run to recluster."
        )
        return

    for group in _GROUP_ORDER:
        names = fields_by_group().get(group, [])
        if not names:
            continue
        with st.expander(_GROUP_TITLES[group], expanded=(group == "cluster")):
            for name in names:
                if name in UI_SPEC:
                    render_field(name, key_prefix=_KEY_PREFIX)

    if not st.button("Run recluster", type="primary", key="v2_recluster_btn"):
        return

    params = build_params_from_state(key_prefix=_KEY_PREFIX)
    if params is None:
        return

    try:
        with st.spinner("Reclustering..."):
            result = service.recluster(picked.output_dir, params)
    except Exception as exc:  # noqa: BLE001
        logger.exception("recluster failed for %s", picked.output_dir)
        st.error(f"Recluster failed: {exc}")
        return

    st.session_state["current_run_dir"] = str(result.snapshot_dir)
    st.success(
        f"Recluster complete — {result.n_clusters} clusters from "
        f"{result.n_faces} faces. New run dir: {result.snapshot_dir.name}"
    )
