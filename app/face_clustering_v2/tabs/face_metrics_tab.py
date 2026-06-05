"""spec-069 — v2 Face Metrics tab.

One sortable row per face: thumbnail + blur / area / det_score / pose +
clustering status (assigned / unassigned). Native ``st.dataframe`` header
sort, so the operator can find the blurriest face, smallest face, lowest
det_score, etc. Sync only (SIGHTING-079). No SQL / FS / cfg.get here —
the service owns reads.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from face_cluster.views.face_metrics import FaceMetricsService
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)

logger = logging.getLogger(__name__)


def render_face_metrics_tab() -> None:
    """Render the Face Metrics tab — sortable per-face metric table."""
    st.header("Face Metrics")
    run_dir = _resolve_run_dir()
    if run_dir is None:
        tab_skipped("face_metrics", "no_run_loaded")
        st.info("No run loaded. Open a run from the History tab first.")
        return
    tab_start("face_metrics", run_dir)
    service = _get_service(run_dir)
    if service is None:
        tab_skipped("face_metrics", "repo_failed")
        return

    rows = service.list_faces()
    n_assigned = sum(1 for r in rows if r.status == "assigned")
    n_unassigned = len(rows) - n_assigned

    c1, c2, c3 = st.columns(3)
    c1.metric("Faces", len(rows))
    c2.metric("Assigned", n_assigned)
    c3.metric("Unassigned", n_unassigned)

    cc1, cc2 = st.columns(2)
    choice = cc1.selectbox("Show", ["all", "assigned", "unassigned"], key="v2_fm_status")
    page_size = int(cc2.selectbox("Faces per page", [25, 50, 100], key="v2_fm_page_size"))
    matching = rows if choice == "all" else [r for r in rows if r.status == choice]

    # Paginate BEFORE building the frame so only the current page's crops are
    # base64-encoded. Encoding all faces at once produced a multi-MB single
    # dataframe that stalled browser rendering (spec-066 e2e blocker).
    n_pages = max(1, (len(matching) + page_size - 1) // page_size)
    page = int(st.number_input("Page", 1, n_pages, 1, key="v2_fm_page")) if n_pages > 1 else 1
    shown = matching[(page - 1) * page_size: page * page_size]

    st.caption(
        f"Showing {len(shown)} of {len(matching)} faces (page {page}/{n_pages}). "
        "Click a column header to sort. Pose is blank until pose computation "
        "is enabled (SIGHTING-093)."
    )
    df = pd.DataFrame([
        {
            "face": _thumb_uri(r.crop_path),
            "id": r.face_id,
            "status": r.status,
            "cluster": r.cluster_id,
            "blur": round(r.blur, 2),
            "area_px": round(r.area, 0),
            "det_score": None if r.det_score is None else round(r.det_score, 3),
            "yaw": r.yaw, "pitch": r.pitch, "roll": r.roll,
        }
        for r in shown
    ])

    event = st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="v2_fm_table",
        column_config={
            "face": st.column_config.ImageColumn("face", width="small"),
        },
    )
    # spec-070 drill-in: selecting a row seeds ``selected_face_id`` so the
    # Face Analysis tab opens that face (large image + bbox + pose overlay).
    sel_rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if sel_rows:
        picked = int(df.iloc[sel_rows[0]]["id"])
        st.session_state["selected_face_id"] = picked
        st.info(
            f"Selected face {picked} — open the **Face Analysis** tab to see "
            "the large image with the bbox + head-pose overlay."
        )
    tab_done("face_metrics", n_faces=len(rows),
             n_assigned=n_assigned, n_unassigned=n_unassigned, shown=len(shown))


def _thumb_uri(crop_path: Optional[str]) -> Optional[str]:
    """Base64 data-URI for the crop, or None (ImageColumn renders blank)."""
    if not crop_path:
        return None
    p = Path(crop_path)
    if not p.is_file():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _resolve_run_dir() -> Optional[Path]:
    for key in ("current_run_dir", "v2_last_run_dir", "active_run_dir"):
        v = st.session_state.get(key)
        if v and Path(v).is_dir() and (Path(v) / "face_clustering.db").is_file():
            return Path(v)
    return None


def _get_service(run_dir: Path) -> Optional[FaceMetricsService]:
    key = f"_face_metrics_service::{run_dir}"
    if key in st.session_state:
        return st.session_state[key]
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Face Metrics: repo construction failed for %s", run_dir)
        st.error(f"Cannot open run: {exc}")
        return None
    svc = FaceMetricsService(repo)
    st.session_state[key] = svc
    return svc
