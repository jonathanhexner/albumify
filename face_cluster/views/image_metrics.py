"""spec-077 — Service for the per-image metrics table.

Read-only, one typed row per source image: its quality scores
(iqa / ava / sharpness / composite), the image-level gate (``filter_passed``),
dimensions, and face count. Streamlit-free; reads through the repository's
RunStore. Mirrors FaceMetricsService for the image side.
"""
from __future__ import annotations

from typing import List

from face_cluster.views._specs import ColumnSpec
from sim_bench.db.face_clustering.cluster_analysis_repo import ClusterAnalysisRepository
from sim_bench.run_db.store import ImageRow


# spec-077 — declarative columns; the tab lets the user pick which to show.
# Raw values are read for numeric sorting (see images_tab).
IMAGE_METRIC_COLUMNS: List[ColumnSpec] = [
    ColumnSpec("n_faces", "Faces"),
    ColumnSpec("filter_passed", "Gate passed", formatter=lambda v: "yes" if v else "no"),
    ColumnSpec("composite_score", "Composite", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("iqa_score", "IQA", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("ava_score", "AVA", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("sharpness_score", "Sharpness", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("width_px", "Width"),
    ColumnSpec("height_px", "Height"),
]

# columns shown by default (the rest are opt-in via the tab's multiselect)
DEFAULT_IMAGE_COLUMNS = ("Faces", "Gate passed", "Composite", "IQA")


class ImageMetricsService:
    """Typed read API for the v2 Images tab."""

    def __init__(self, repo: ClusterAnalysisRepository) -> None:
        if repo is None:
            raise ValueError("ImageMetricsService requires a non-None ClusterAnalysisRepository.")
        self._repo = repo

    def list_images(self) -> List[ImageRow]:
        """One :class:`ImageRow` per source image (passthrough to RunStore)."""
        return self._repo._run_store.list_images()


__all__ = [
    "ImageMetricsService",
    "ImageRow",
    "IMAGE_METRIC_COLUMNS",
    "DEFAULT_IMAGE_COLUMNS",
]
