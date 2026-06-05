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
    status: str                  # "assigned" | "unassigned"
    cluster_id: Optional[int]    # set when assigned
    blur: float
    area: float
    det_score: Optional[float]
    yaw: Optional[float]
    pitch: Optional[float]
    roll: Optional[float]
    crop_path: Optional[str]     # absolute path to the aligned crop, if present


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
            if cid is None or is_noise(cid):
                status, cluster_id = "unassigned", None
            else:
                status, cluster_id = "assigned", int(cid)
            yaw, pitch, roll = f.pose if f.pose else (None, None, None)
            cp = crop_path(run_dir, f.face_id)
            rows.append(FaceMetricRow(
                face_id=f.face_id,
                status=status,
                cluster_id=cluster_id,
                blur=float(f.blur_score or 0.0),
                area=float(f.area or 0.0),
                det_score=(None if f.det_score is None else float(f.det_score)),
                yaw=yaw, pitch=pitch, roll=roll,
                crop_path=(str(cp) if cp.is_file() else None),
            ))
        return rows


__all__ = ["FaceMetricsService", "FaceMetricRow"]
