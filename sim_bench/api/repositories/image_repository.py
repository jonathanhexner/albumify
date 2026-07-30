"""spec-086 (slice 1): ImageRepository — read per-image metrics via SQL JOIN.

Replaces the JSON-blob stitch in ``people_service.get_person_images`` with a real
query over the normalized ``image_metric_rows`` / ``face_metric_rows`` tables.
Returns the canonical ``ImageMetrics`` shape (spec-085) so the API/UI is unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from sqlalchemy.orm import Session

from sim_bench.api.database.models import ImageMetricRow, FaceMetricRow
from sim_bench.api.schemas.result import ImageMetrics


class ImageRepository:
    """Reads image/face metrics from the normalized tables."""

    def __init__(self, session: Session):
        self._session = session

    def get_images_for_person(self, run_id: str, person_id: str) -> List[ImageMetrics]:
        """All images containing ``person_id``, with full per-image metrics.

        The person→image link is the JOIN key (``face_metric_rows.person_id``).
        Every face in each matched image is returned (not just the person's), so
        the Image Detail popup can box and score all faces.
        """
        if not run_id or not person_id:
            return []

        # Images where this person appears (the JOIN on person_id).
        paths = [
            row[0]
            for row in (
                self._session.query(FaceMetricRow.image_path)
                .filter(FaceMetricRow.run_id == run_id,
                        FaceMetricRow.person_id == person_id)
                .distinct()
            )
        ]
        if not paths:
            return []

        imgs = {
            r.image_path: r
            for r in self._session.query(ImageMetricRow).filter(
                ImageMetricRow.run_id == run_id,
                ImageMetricRow.image_path.in_(paths),
            )
        }
        faces_by_path: dict = defaultdict(list)
        for f in (
            self._session.query(FaceMetricRow)
            .filter(FaceMetricRow.run_id == run_id,
                    FaceMetricRow.image_path.in_(paths))
            .order_by(FaceMetricRow.image_path, FaceMetricRow.face_index)
        ):
            faces_by_path[f.image_path].append(f)

        # Preserve a stable order (by image path) for deterministic output.
        return [
            self._to_image_metrics(imgs[p], faces_by_path.get(p, []))
            for p in sorted(paths)
            if p in imgs
        ]

    @staticmethod
    def _to_image_metrics(img: ImageMetricRow, faces: List[FaceMetricRow]) -> ImageMetrics:
        """Rebuild the ImageMetrics contract from one image row + its face rows."""
        filter_scores = []
        pose, eyes, smile, roll = [], [], [], []
        for f in faces:
            entry = {
                "face_index": f.face_index,
                "confidence": f.confidence,
                "bbox_ratio": f.bbox_ratio,
                "relative_size": f.relative_size,
                "eye_ratio": f.eye_ratio,
                "filter_passed": f.filter_passed,
            }
            if f.bbox_w is not None or f.bbox_w_px is not None:
                entry["bbox"] = {
                    "x": f.bbox_x, "y": f.bbox_y, "w": f.bbox_w, "h": f.bbox_h,
                    "x_px": f.bbox_x_px, "y_px": f.bbox_y_px,
                    "w_px": f.bbox_w_px, "h_px": f.bbox_h_px,
                }
            filter_scores.append(entry)
            if f.pose_score is not None:
                pose.append(f.pose_score)
            if f.eyes_score is not None:
                eyes.append(f.eyes_score)
            if f.smile_score is not None:
                smile.append(f.smile_score)
            if f.roll_angle is not None:
                roll.append(f.roll_angle)

        return ImageMetrics(
            path=img.image_path,
            iqa_score=img.iqa_score,
            ava_score=img.ava_score,
            sharpness=img.sharpness,
            composite_score=img.composite_score,
            quality_score=img.quality_score,
            person_penalty=img.person_penalty,
            cluster_id=img.cluster_id,
            face_count=img.face_count,
            is_selected=bool(img.is_selected),
            filter_reason=img.filter_reason,
            person_detected=img.person_detected,
            body_facing_score=img.body_facing_score,
            person_confidence=img.person_confidence,
            best_frontal_score=img.best_frontal_score,
            best_centrality=img.best_centrality,
            face_pose_scores=pose or None,
            face_eyes_scores=eyes or None,
            face_smile_scores=smile or None,
            roll_angles=roll or None,
            filter_scores=filter_scores or None,
        )
