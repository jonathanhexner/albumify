"""spec-077 / spec-083 — v2 Images tab.

Browse images as a clickable thumbnail **Grid** (default) or a sortable
**Table**; click **Open** on an image to analyse it in place (master-detail:
the source photo with the face boxes that passed filtration). Read-only; no
SQL/FS here — the service owns reads.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from app.face_clustering_v2._run_context import resolve_run_dir
from app.face_clustering_v2._telemetry import tab_done, tab_skipped, tab_start
from app.face_clustering_v2.components.image_analysis import render_image_analysis
from app.face_clustering_v2.components.image_pick_grid import render_image_pick_grid
from app.face_clustering_v2.components.metric_strip import render_metric_strip
from face_cluster.views.image_metrics import (
    DEFAULT_IMAGE_COLUMNS,
    ImageMetricsService,
    populated_columns,
)
from face_cluster.views.metric_specs import IMAGE_COUNT_STRIP
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)

logger = logging.getLogger(__name__)


def render_images_tab() -> None:
    """Render the Images tab — clickable grid / sortable table + in-tab analysis."""
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

    # Master-detail: an image is selected -> show its analysis + a Back button.
    selected = st.session_state.get("selected_image_path")
    if selected and any(r.image_path == selected for r in rows):
        if st.button("← Back to images", key="v2_img_back"):
            del st.session_state["selected_image_path"]
            st.rerun()
        render_image_analysis(service.image_detail(selected))
        tab_done("images", n_images=len(rows), view="detail")
        return

    n_pass = sum(1 for r in rows if r.filter_passed)
    render_metric_strip(
        SimpleNamespace(n_images=len(rows), n_pass=n_pass), IMAGE_COUNT_STRIP, n_cols=2,
    )
    layout = st.radio("Layout", ["Grid (click to open)", "Table (sortable)"],
                      horizontal=True, key="v2_img_layout")
    with st.spinner(f"Building {len(rows)} thumbnails…"):
        thumbs = _thumbs(rows, run_dir)

    if layout.startswith("Grid"):
        st.caption("Click **Open** under an image to analyse it (photo + face boxes).")
        render_image_pick_grid(rows, thumbs)
        tab_done("images", n_images=len(rows), n_pass=n_pass, view="grid")
        return
    _render_table(rows, thumbs, service)
    tab_done("images", n_images=len(rows), n_pass=n_pass, view="table")


def _render_table(rows, thumbs: Dict[str, Optional[bytes]], service) -> None:
    """spec-082 sortable dataframe (Table layout). Only non-None columns offered."""
    populated = populated_columns(rows)
    pop_labels = [c.label for c in populated]
    default = [l for l in DEFAULT_IMAGE_COLUMNS if l in pop_labels] or pop_labels
    chosen = st.multiselect("Columns", pop_labels, default=default, key="v2_img_cols")
    cols = [c for c in populated if c.label in chosen]
    st.caption("Select a row's checkbox to open it (the Grid layout is the easier click).")
    df = pd.DataFrame([
        {"image": _data_uri(thumbs.get(r.image_path)), "name": Path(r.image_path).name,
         **{c.label: c.read(r) for c in cols}}
        for r in rows
    ])
    event = st.dataframe(
        df, hide_index=True, width="stretch", key="v2_images_table",
        on_select="rerun", selection_mode="single-row",
        column_config={"image": st.column_config.ImageColumn("img", width="small")},
    )
    sel = getattr(getattr(event, "selection", None), "rows", []) or []
    if sel:
        st.session_state["selected_image_path"] = rows[sel[0]].image_path
        st.rerun()


def _data_uri(data: Optional[bytes]) -> Optional[str]:
    """base64 data-URL for the ImageColumn (or None -> blank cell)."""
    if not data:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _thumbs(rows, run_dir) -> Dict[str, Optional[bytes]]:
    """EXIF-corrected ~200px JPEG thumbnail bytes per image, cached per run dir.

    PIL releases the GIL during JPEG decode/encode, so the threaded resize is a
    real speedup (~12s -> ~3s for 122 large photos). Bytes (not data-URLs) so the
    grid can ``st.image`` them directly; the table derives a data-URL on demand.
    """
    key = f"_img_thumb_bytes::{run_dir}"
    cached = st.session_state.get(key)
    if cached is not None:
        return cached
    from concurrent.futures import ThreadPoolExecutor

    paths = [r.image_path for r in rows]
    with ThreadPoolExecutor(max_workers=8) as pool:
        encoded = list(pool.map(_encode_thumb, paths))
    out = dict(zip(paths, encoded))
    st.session_state[key] = out
    return out


def _encode_thumb(path: str) -> Optional[bytes]:
    """One EXIF-corrected ~200px JPEG thumbnail as bytes (or None on failure)."""
    from PIL import Image, ImageOps
    try:
        img = ImageOps.exif_transpose(Image.open(path))
        img.thumbnail((200, 200))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=72)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


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
