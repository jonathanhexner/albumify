"""spec-077 — Service for the per-image metrics table.

Read-only, one typed row per source image: its quality scores
(iqa / ava / sharpness / composite), the image-level gate (``filter_passed``),
dimensions, and face count. Streamlit-free; reads through the repository's
RunStore. Mirrors FaceMetricsService for the image side.
"""
from __future__ import annotations

from typing import List, Sequence

from face_cluster.views._specs import ColumnSpec
from sim_bench.db.face_clustering.cluster_analysis_repo import ClusterAnalysisRepository
from sim_bench.run_db.store import ImageRow


# spec-077 — declarative columns; the tab lets the user pick which to show.
# Raw values are read for numeric sorting (see images_tab).
IMAGE_METRIC_COLUMNS: List[ColumnSpec] = [
    ColumnSpec("n_faces", "Faces"),
    ColumnSpec("n_passed", "Passed"),  # spec-083: faces that passed filtration
    ColumnSpec("filter_passed", "Gate passed", formatter=lambda v: "yes" if v else "no"),
    ColumnSpec("composite_score", "Composite", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("iqa_score", "IQA", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("ava_score", "AVA", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("sharpness_score", "Sharpness", formatter=lambda v: f"{v:.3f}"),
    ColumnSpec("width_px", "Width"),
    ColumnSpec("height_px", "Height"),
]

# columns shown by default (the rest are opt-in via the tab's multiselect)
DEFAULT_IMAGE_COLUMNS = ("Faces", "Passed", "Gate passed", "Composite", "IQA")


def populated_columns(
    rows: Sequence[ImageRow], columns: Sequence[ColumnSpec] = IMAGE_METRIC_COLUMNS,
) -> List[ColumnSpec]:
    """Columns that have at least one non-None value across ``rows``.

    spec-082: face-clustering runs have no IQA / AVA / composite / sharpness
    (those are Albumify image-scoring outputs), so offering them produced a
    useless wall of ``None``. Filtering here keeps the tab thin and the rule
    unit-testable without Streamlit.
    """
    return [c for c in columns if any(c.read(r) is not None for r in rows)]


class ImageMetricsService:
    """Typed read API for the v2 Images tab."""

    def __init__(self, repo: ClusterAnalysisRepository) -> None:
        if repo is None:
            raise ValueError("ImageMetricsService requires a non-None ClusterAnalysisRepository.")
        self._repo = repo

    def list_images(self) -> List[ImageRow]:
        """One :class:`ImageRow` per source image (passthrough to RunStore)."""
        return self._repo._run_store.list_images()

    def image_detail(self, image_path: str):
        """Full per-image detail (scores + every face's bbox / cluster / gate).

        Returns a ``face_cluster.image_detail.ImageDetail`` — drives the Image
        Analysis view (source photo + all face boxes + per-face table)."""
        return self._repo._run_store.image_detail(image_path)


__all__ = [
    "ImageMetricsService",
    "ImageRow",
    "IMAGE_METRIC_COLUMNS",
    "DEFAULT_IMAGE_COLUMNS",
    "populated_columns",
]
