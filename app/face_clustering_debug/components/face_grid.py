"""Reusable face thumbnail grid component."""

from pathlib import Path
from typing import Callable, List, Optional

import streamlit as st

from app.face_clustering_debug.models.schemas import FaceInfo


def _truncate_filename(filename: str, max_len: int = 12) -> str:
    """Truncate filename for display, keeping extension hint."""
    if len(filename) <= max_len:
        return filename
    return filename[:max_len - 2] + ".."


def render_face_grid(
    faces: List[FaceInfo],
    get_crop_fn: Callable[[int], Optional[bytes]],
    highlight_indices: Optional[List[int]] = None,
    columns: int = 10,
    key_prefix: str = "",
) -> Optional[int]:
    """Render a clickable grid of face thumbnails.

    Args:
        faces: Faces to display.
        get_crop_fn: Returns JPEG bytes for a face index.
        highlight_indices: Indices to mark as exemplars (⭐).
        columns: Number of grid columns.
        key_prefix: Unique prefix for widget keys to avoid duplicates across multiple grids.

    Returns:
        Selected face index, or None.
    """
    selected: Optional[int] = None
    highlight_set = set(highlight_indices or [])

    for row_start in range(0, len(faces), columns):
        row_faces = faces[row_start:row_start + columns]
        cols = st.columns(len(row_faces))
        for col, face in zip(cols, row_faces):
            crop = get_crop_fn(face.index)
            star = "⭐" if face.index in highlight_set else ""

            # Extract filename from image_path
            filename = ""
            if face.image_path:
                filename = Path(face.image_path).stem
                filename = _truncate_filename(filename)

            caption = f"{star}#{face.index} {filename}"

            with col:
                if crop:
                    st.image(crop, caption=caption, use_container_width=True)
                    btn_key = f"{key_prefix}sel_{face.index}" if key_prefix else f"sel_{face.index}"
                    if st.button("🔍", key=btn_key, help=f"Inspect face #{face.index}"):
                        selected = face.index
                else:
                    st.caption(caption)

    return selected
