"""spec-069 — Service for the v2 Face Metrics tab.

Read-only, one typed row per face: its per-face quality metrics (blur /
area / det_score / pose) + its clustering status (assigned to a cluster, or
unassigned). Lets an operator sort to find the blurriest face, the smallest
face, the lowest-confidence detection, etc.

Architecture (mirrors quality.py / cluster_analysis.py):
* Streamlit-free — no ``streamlit`` import here.
* Reads through the repository layer — the metrics are produced by the
  pipeline (detection + quality_gate.compute_blur_scores) and persisted to
  the per-run ``faces`` table; this service never recomputes them.
* ``status`` is DERIVED (``all_faces − assigned``) because the pipeline does
  not persist noise rows (SIGHTING-093) — no DB change required.
* Typed I/O — returns ``list[FaceMetricRow]``, never raw dicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from face_cluster.run_layout import crop_path
from face_cluster.views._specs import ColumnSpec
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisCriteria,
    ClusterAnalysisRepository,
)
from sim_bench.pipeline.clustering_labels import is_noise

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceMetricRow:
    """One face with its metrics + clustering status."""
    face_id: int
    status: str                  # "assigned" | "unassigned" (kept for back-compat)
    disposition: str             # "clustered" | "noise" | "filtered" — the clear 3-way
    cluster_id: Optional[int]    # set when assigned
    blur: float
    area: float
    area_ratio: Optional[float]  # face bbox area / image area (0..1)
    det_score: Optional[float]
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    rejection_reason: Optional[str]  # why the face was held out (gate name), if any
    crop_path: Optional[str]     # absolute path to the aligned crop, if present

    @property
    def area_pct(self) -> Optional[float]:
        """Area as a percentage (0..100) — kept numeric so the table sorts."""
        return None if self.area_ratio is None else self.area_ratio * 100.0


# spec-072 — the ONE declaration of the face metrics. Both the Face Metrics
# table (via ``rows_to_records``) and the Face Analysis strip (via
# ``render_metric_strip``) render from this list. Add a metric here once and it
# appears in both. Field names are the canonical attrs exposed by BOTH
# ``FaceMetricRow`` and ``FaceView`` (the latter via properties). A column whose
# attr is absent/None on a given object renders as "—".
FACE_METRIC_COLUMNS: List[ColumnSpec] = [
    ColumnSpec("blur", "Blur", formatter=lambda v: f"{v:.0f}",
               help="Laplacian variance — higher is sharper."),
    ColumnSpec("area", "Area (px)", formatter=lambda v: f"{v:,.0f}",
               help="Face bbox area in source-image pixels."),
    ColumnSpec("area_pct", "Area %", formatter=lambda v: f"{v:.1f}%",
               help="Face bbox area as a percentage of the whole image."),
    ColumnSpec("det_score", "Det score", formatter=lambda v: f"{v:.3f}",
               help="InsightFace detection confidence."),
    ColumnSpec("yaw", "Yaw", formatter=lambda v: f"{v:.1f}", help="Head yaw (deg)."),
    ColumnSpec("pitch", "Pitch", formatter=lambda v: f"{v:.1f}", help="Head pitch (deg)."),
    ColumnSpec("roll", "Roll", formatter=lambda v: f"{v:.1f}", help="Head roll (deg)."),
]


class FaceMetricsService:
    """Typed read API for the v2 Face Metrics tab."""

    def __init__(self, repo: ClusterAnalysisRepository) -> None:
        if repo is None:
            raise ValueError(
                "FaceMetricsService requires a non-None ClusterAnalysisRepository."
            )
        self._repo = repo

    def list_faces(self) -> List[FaceMetricRow]:
        """One row per face in the run, with metrics + derived status.

        ``status`` is derived: a face is ``assigned`` iff it has a
        non-noise ``cluster_assignments`` row; otherwise ``unassigned``
        (the pipeline does not persist noise rows — SIGHTING-093).
        """
        run_dir = self._repo._config.run_dir  # repo owns the run dir
        faces = self._repo._run_store.faces()
        assign = {
            a.face_id: a.cluster_id
            for a in self._repo.find_assignments(
                ClusterAnalysisCriteria(include_noise=True)
            )
        }

        rows: List[FaceMetricRow] = []
        for f in faces:
            cid = assign.get(f.face_id)
            reason = getattr(f, "rejection_reason", None)
            if cid is not None and not is_noise(cid):
                status, cluster_id, disposition = "assigned", int(cid), "clustered"
            elif reason:
                # has a gate verdict -> it was held out before/at the gate
                status, cluster_id, disposition = "unassigned", None, "filtered"
            else:
                # passed gating but matched no cluster
                status, cluster_id, disposition = "unassigned", None, "noise"
            yaw, pitch, roll = f.pose if f.pose else (None, None, None)
            cp = crop_path(run_dir, f.face_id)
            rows.append(FaceMetricRow(
                face_id=f.face_id,
                status=status,
                disposition=disposition,
                cluster_id=cluster_id,
                blur=float(f.blur_score or 0.0),
                area=float(f.area or 0.0),
                area_ratio=(None if getattr(f, "area_ratio", None) is None else float(f.area_ratio)),
                det_score=(None if f.det_score is None else float(f.det_score)),
                yaw=yaw, pitch=pitch, roll=roll,
                rejection_reason=getattr(f, "rejection_reason", None),
                crop_path=(str(cp) if cp.is_file() else None),
            ))
        return rows


__all__ = ["FaceMetricsService", "FaceMetricRow", "FACE_METRIC_COLUMNS"]
