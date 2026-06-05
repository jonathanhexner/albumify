"""spec-077 — v2 Images tab. Per-image scores + image-level gate, with a
user-controllable column set. Read-only; sortable. No SQL/FS here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from app.face_clustering_v2._run_context import resolve_run_dir
from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from face_cluster.views.image_metrics import (
    DEFAULT_IMAGE_COLUMNS,
    IMAGE_METRIC_COLUMNS,
    ImageMetricsService,
)
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)

logger = logging.getLogger(__name__)


def render_images_tab() -> None:
    """Render the Images tab — per-image metrics with choosable columns."""
    st.header("Images")
    run_dir = resolve_run_dir()
    if run_dir is None:
        tab_skipped("images", "no_run_loaded")
        st.info("No run loaded. Open a run from the History tab first.")
        return
    tab_start("images", run_dir)
    service = _get_service(run_dir)
    if service is None:
        tab_skipped("images", "repo_failed")
        return
    rows = service.list_images()
    if not rows:
        tab_skipped("images", "no_images")
        st.info("No images recorded for this run.")
        return

    n_pass = sum(1 for r in rows if r.filter_passed)
    c1, c2 = st.columns(2)
    c1.metric("Images", len(rows))
    c2.metric("Gate passed", n_pass)

    labels = [c.label for c in IMAGE_METRIC_COLUMNS]
    chosen = st.multiselect("Columns", labels, default=list(DEFAULT_IMAGE_COLUMNS), key="v2_img_cols")
    cols = [c for c in IMAGE_METRIC_COLUMNS if c.label in chosen]
    st.caption("Raw values shown so the table sorts numerically. Click a header to sort.")
    df = pd.DataFrame([
        {"image": Path(r.image_path).name, **{c.label: c.read(r) for c in cols}}
        for r in rows
    ])
    st.dataframe(df, hide_index=True, width="stretch", key="v2_images_table")
    tab_done("images", n_images=len(rows), n_pass=n_pass, n_cols=len(cols))


def _get_service(run_dir: Path) -> Optional[ImageMetricsService]:
    key = f"_image_metrics_service::{run_dir}"
    if key in st.session_state:
        return st.session_state[key]
    try:
        repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Images: repo construction failed for %s", run_dir)
        st.error(f"Cannot open run: {exc}")
        return None
    svc = ImageMetricsService(repo)
    st.session_state[key] = svc
    return svc
