"""spec-083 — clickable face grid for the Face Metrics tab.

The reliable click-to-open pattern: a real ``st.button`` under each thumbnail
(canvas ``st.dataframe`` row-select only fires on a ~20px checkbox column, which
users can't find — the whole reason "I can't click on faces" kept recurring).
Mirrors ``face_grid.py`` but takes the flat ``FaceMetricRow`` (disposition +
metrics) rather than a ClusterView. Render-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import streamlit as st

from face_cluster.views.face_metrics import FaceMetricRow

GRID_COLS = 6
# disposition -> a coloured tag so the grid carries the same signal as the table
_TAG = {"clustered": ":green[clustered]", "noise": ":orange[noise]", "filtered": ":red[filtered]"}


def render_face_pick_grid(rows: Sequence[FaceMetricRow]) -> None:
    """Grid of face thumbnails; each has an Open button -> Face Analysis."""
    if not rows:
        st.caption("No faces match the current filter.")
        return
    for i in range(0, len(rows), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for col, r in zip(cols, rows[i:i + GRID_COLS]):
            with col:
                crop = r.crop_path
                if crop and Path(crop).is_file():
                    try:
                        st.image(crop, width=104)
                    except Exception:  # noqa: BLE001 — a bad crop must not crash the page
                        pass
                area = "" if r.area_ratio is None else f" · {r.area_ratio * 100:.1f}%"
                st.caption(
                    f"`{r.face_id:04d}` {_TAG.get(r.disposition, r.disposition)}\n\n"
                    f"blur={r.blur:.0f}{area}"
                )
                if st.button("Open", key=f"fm_open_{r.face_id}", width="stretch"):
                    st.session_state["selected_face_id"] = int(r.face_id)
                    from app.face_clustering_v2._nav import navigate_to
                    navigate_to("Face Analysis")
