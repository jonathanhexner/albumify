"""spec-083 — clickable image grid for the Images tab.

Same reliable pattern as ``face_pick_grid``: a real ``st.button`` under each
thumbnail (canvas row-select was un-findable). Clicking sets
``selected_image_path`` so the tab swaps to the in-tab Image Analysis
(master-detail; there is no separate Image-Analysis nav page). Render-only —
the tab prepares the thumbnail bytes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import streamlit as st

from sim_bench.run_db.store import ImageRow

GRID_COLS = 5


def render_image_pick_grid(rows: Sequence[ImageRow], thumbs: Dict[str, Optional[bytes]]) -> None:
    """Grid of image thumbnails; each Open button selects that image."""
    if not rows:
        st.caption("No images match the current filter.")
        return
    for i in range(0, len(rows), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for col, r in zip(cols, rows[i:i + GRID_COLS]):
            with col:
                data = thumbs.get(r.image_path)
                if data:
                    try:
                        st.image(data, width="stretch")
                    except Exception:  # noqa: BLE001
                        pass
                dims = f"{r.width_px}x{r.height_px}" if r.width_px else "?"
                gate = ":green[gate ok]" if r.filter_passed else ":red[gate fail]"
                st.caption(
                    f"**{Path(r.image_path).name}**\n\n"
                    f"{r.n_faces} faces · {r.n_passed} passed · {dims} · {gate}"
                )
                if st.button("Open", key=f"img_open_{r.image_path}", width="stretch"):
                    st.session_state["selected_image_path"] = r.image_path
                    st.rerun()
