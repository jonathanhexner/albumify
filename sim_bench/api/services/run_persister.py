"""One writer for a finished run's rows (spec-116 R4), whichever machine ran it."""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session

from sim_bench.api.database.models import FaceMetricRow, ImageMetricRow, PipelineResult, Person
from sim_bench.api.services.people_service import PeopleService
from sim_bench.serving.run_output import RunOutput

__all__ = ["RunPersister"]


class RunPersister:
    """``RunOutput`` -> ``pipeline_results``, ``people``, ``image_metric_rows``, ``face_metric_rows``."""

    def __init__(self, session: Session, logger: Optional[logging.Logger] = None):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    def persist(self, album_id: str, run_id: str, output: RunOutput) -> List[Person]:
        """Add every row and flush, in one transaction the CALLER commits with the run's status (R8: a failed
        run has no partial rows). On any error it rolls back and raises."""
        try:
            self._session.add(PipelineResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                total_images=output.total_images,
                filtered_images=output.filtered_images,
                num_clusters=len(output.scene_clusters),
                num_selected=len(output.selected_images),
                scene_clusters=output.scene_clusters,
                face_subclusters=output.face_subclusters,
                selected_images=output.selected_images,
                image_metrics=output.image_metrics,
                siamese_comparisons=output.siamese_comparisons,
                step_timings=output.step_timings,
                total_duration_ms=output.total_duration_ms,
                fc_export_dir=output.fc_export_dir,
                step_decisions=output.step_decisions,
            ))
            people = PeopleService(self._session).add_from_payload(album_id, run_id, output.people)
            self._logger.info("run %s: %d people", run_id, len(people))
            # People exist now, so faces can be linked to their person_id (spec-086).
            self._write_metric_tables(run_id, output.image_metrics, people)
            self._session.flush()
            return people
        except Exception:
            self._session.rollback()
            raise

    def _write_metric_tables(self, run_id: str, image_metrics: dict, people: list) -> None:
        """spec-086: persist normalized image/face metric rows from the same
        ``image_metrics`` dict used for the blob column. Faces are linked to the
        Person they were clustered into so ``ImageRepository`` can JOIN on it.
        """
        def _norm(p: str) -> str:
            return str(p).replace("\\", "/")

        # (image_path, face_index) -> Person.id, from the just-created people.
        face_to_person: dict = {}
        for person in people or []:
            for fi in (person.face_instances or []):
                ip = fi.get("image_path")
                if ip is None:
                    continue
                face_to_person[(_norm(ip), fi.get("face_index"))] = person.id

        for path, m in image_metrics.items():
            self._session.add(ImageMetricRow(
                run_id=run_id,
                image_path=path,
                iqa_score=m.get("iqa_score"),
                ava_score=m.get("ava_score"),
                sharpness=m.get("sharpness"),
                composite_score=m.get("composite_score"),
                quality_score=m.get("quality_score"),
                person_penalty=m.get("person_penalty"),
                cluster_id=m.get("cluster_id"),
                face_count=m.get("face_count") or 0,
                is_selected=bool(m.get("is_selected")),
                filter_reason=m.get("filter_reason"),
                person_detected=m.get("person_detected"),
                body_facing_score=m.get("body_facing_score"),
                person_confidence=m.get("person_confidence"),
                best_frontal_score=m.get("best_frontal_score"),
                best_centrality=m.get("best_centrality"),
            ))

            filter_scores = m.get("filter_scores") or []
            pose = m.get("face_pose_scores") or []
            eyes = m.get("face_eyes_scores") or []
            smile = m.get("face_smile_scores") or []
            roll = m.get("roll_angles") or []
            for i, fs in enumerate(filter_scores):
                bbox = fs.get("bbox") or {}
                fidx = fs.get("face_index", i)
                self._session.add(FaceMetricRow(
                    run_id=run_id,
                    image_path=path,
                    face_index=fidx,
                    person_id=face_to_person.get((_norm(path), fidx)),
                    bbox_x=bbox.get("x"), bbox_y=bbox.get("y"),
                    bbox_w=bbox.get("w"), bbox_h=bbox.get("h"),
                    bbox_x_px=bbox.get("x_px"), bbox_y_px=bbox.get("y_px"),
                    bbox_w_px=bbox.get("w_px"), bbox_h_px=bbox.get("h_px"),
                    confidence=fs.get("confidence"),
                    filter_passed=bool(fs.get("filter_passed", True)),
                    bbox_ratio=fs.get("bbox_ratio"),
                    relative_size=fs.get("relative_size"),
                    eye_ratio=fs.get("eye_ratio"),
                    pose_score=pose[i] if i < len(pose) else None,
                    eyes_score=eyes[i] if i < len(eyes) else None,
                    smile_score=smile[i] if i < len(smile) else None,
                    roll_angle=roll[i] if i < len(roll) else None,
                ))
