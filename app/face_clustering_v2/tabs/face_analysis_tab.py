"""spec-064 — v2 Face Analysis tab. Per-face drill-down; reads
``selected_face_id`` from session_state. Sync + ``st.spinner`` (SIGHTING-079).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

from app.face_clustering_v2.components.face_bbox_overlay import render_face_bbox_overlay
from app.face_clustering_v2.components.face_detail_panel import render_face_detail_panel
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from face_cluster.views.face_view import FaceAnalysisService

logger = logging.getLogger(__name__)


def render_face_analysis_tab() -> None:
    """Render the Face Analysis tab."""
    st.header("Face Analysis")
    run_dir = _resolve_run_dir()
    if run_dir is None:
        st.info("No run loaded. Open a run from the History tab first.")
        return
    service = _get_service(run_dir)
    if service is None:
        return
    ids = service.list_face_ids()
    if not ids:
        st.info("No faces in this run.")
        return
    default_id = max(min(int(st.session_state.get("selected_face_id") or ids[0]), ids[-1]), ids[0])
    face_id = int(st.number_input(
        "Face id", min_value=int(ids[0]), max_value=int(ids[-1]),
        value=default_id, step=1, key="v2_fa_face_id",
    ))
    st.session_state["selected_face_id"] = face_id
    try:
        with st.spinner("Loading face..."):
            view = service.compute_face_detail(face_id)
            record = service.get_face_record(face_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_face_detail failed for %s in %s", face_id, run_dir)
        st.error(f"Could not load face {face_id}: {exc}")
        return
    render_face_bbox_overlay(
        source_image_path=view.image_path,
        crop_fallback=run_dir / "crops" / f"face_{face_id:04d}_aligned.jpg",
        bbox=tuple(record.bbox) if record.bbox else None,
        landmarks=record.landmarks,
    )
    render_face_detail_panel(view)


def _resolve_run_dir() -> Optional[Path]:
    for key in ("current_run_dir", "v2_last_run_dir", "active_run_dir"):
        v = st.session_state.get(key)
        if v and Path(v).is_dir() and (Path(v) / "face_clustering.db").is_file():
            return Path(v)
    return None


def _get_service(run_dir: Path) -> Optional[FaceAnalysisService]:
    key = f"_face_analysis_service::{run_dir}"
    if key in st.session_state:
        return st.session_state[key]
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Face Analysis: repo construction failed for %s", run_dir)
        st.error(f"Cannot open run: {exc}")
        return None
    svc = FaceAnalysisService(repo)
    st.session_state[key] = svc
    return svc
