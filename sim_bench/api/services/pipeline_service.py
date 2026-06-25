"""Pipeline service - orchestrates pipeline execution."""

import logging
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from sim_bench.api.database.models import (
    Album, PipelineRun, PipelineResult, ImageMetricRow, FaceMetricRow,
)
from sim_bench.api.schemas.result import ImageMetrics
from sim_bench.api.services.people_service import PeopleService
from sim_bench.api.services.config_service import ConfigService
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.run import execute_spec
from sim_bench.pipeline.spec import PipelineSpec
from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
from face_cluster.fc_params import FCParams


# No more hardcoded pipeline - loaded from config service


@dataclass
class JobProgress:
    """Progress update for a running job."""
    step: str
    progress: float
    message: str


@dataclass
class JobState:
    """State for a running job."""
    run_id: str
    album_id: str
    context: PipelineContext
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    completed: bool = False


# Shared job storage across all PipelineService instances
_jobs: dict[str, JobState] = {}


def _build_reason_by_path(step_decisions) -> dict:
    """Map each image path to its most informative "why" reason. spec-084.

    Images filtered early (e.g. ``filter_quality``) never reach ``select_best``,
    so a select_best-only reason would be blank for exactly the filtered images
    the user wants explained. We therefore keep the earliest image-level reason
    and let the final ``select_best`` decision override it when present.

    Decisions are appended in step order, so the earlier rejecting step's reason
    is recorded first and ``select_best`` (the last word) overwrites it.
    """
    reason_by_path: dict = {}
    for d in (step_decisions or []):
        if d.item_type != "image":
            continue  # face-level decisions aren't per-image reasons
        if d.step == "select_best" or d.item_id not in reason_by_path:
            reason_by_path[d.item_id] = d.reason
    return reason_by_path


class PipelineService:
    """Service for running pipelines."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    def start_pipeline(
        self,
        album_id: str,
        steps: list[str] = None,
        pipeline_name: str = "default_pipeline",
        step_configs: dict[str, dict] = None,
        fail_fast: bool = True
    ) -> str:
        """
        Start a pipeline run.

        Args:
            album_id: Album to process
            steps: Explicit step list (overrides pipeline_name if provided)
            pipeline_name: Name of pipeline from config (e.g., "default_pipeline", "minimal_pipeline")
            step_configs: Step configuration overrides
            fail_fast: Stop on first error

        Returns job_id for tracking.
        """
        self._logger.info(f"Starting pipeline for album {album_id}")

        album = self._session.query(Album).filter(Album.id == album_id).first()
        if album is None:
            self._logger.error(f"Album not found: {album_id}")
            raise ValueError(f"Album not found: {album_id}")

        # Load steps from config if not explicitly provided
        if steps is None:
            config_service = ConfigService(self._session)
            config = config_service.get_default_profile().config
            steps = config.get(pipeline_name, config.get("default_pipeline", []))
            self._logger.info(f"Using pipeline '{pipeline_name}' with {len(steps)} steps")

        run_id = str(uuid.uuid4())
        self._logger.info(f"Created pipeline run {run_id} with steps: {steps}")

        run = PipelineRun(
            id=run_id,
            album_id=album_id,
            pipeline_name=pipeline_name,
            steps=steps,
            step_configs=step_configs or {},
            fail_fast=fail_fast,
            status="pending"
        )

        self._session.add(run)
        self._session.commit()

        # Create context with cache handler
        cache_handler = UniversalCacheHandler(self._session)
        context = PipelineContext(
            source_directory=Path(album.source_path),
            cache_handler=cache_handler
        )
        # spec-088: name the FC-app export dir after the album (was the "album"
        # fallback in face_cluster_export.py:30 because this was never set).
        context.album_name = album.name

        _jobs[run_id] = JobState(
            run_id=run_id,
            album_id=album_id,
            context=context
        )

        return run_id

    def _broadcast_clustering_config(
        self, steps: list[str], step_configs: dict[str, dict]
    ) -> dict[str, dict]:
        """spec-079 — the ONE config interpreter for identity clustering.

        Albumify keeps a single user-facing clustering config block (still keyed
        ``cluster_people`` for UI/profile back-compat). When the pipeline runs
        App A's unified clustering chain, we translate that block through
        ``FCParams`` (the shared, typed config) and broadcast it to the unified
        steps via ``FCParams.to_step_configs()`` — the SAME interpretation App A
        uses. No second config language; divergence-by-construction is removed.

        No-op when the unified steps aren't in the run (e.g. minimal_pipeline).
        """
        if not any(s in steps for s in UNIFIED_CLUSTERING_STEPS):
            return step_configs

        raw = dict(step_configs.get("cluster_people") or {})
        allowed = set(FCParams.model_fields)
        fcp = FCParams(**{k: v for k, v in raw.items() if k in allowed})
        dropped = sorted(set(raw) - allowed)
        if dropped:
            self._logger.info(
                "clustering config: %d keys not on FCParams ignored: %s",
                len(dropped), dropped,
            )
        # Per-step config already present in step_configs wins over the broadcast
        # (lets a caller override a single unified step explicitly).
        for name, cfg in fcp.to_step_configs().items():
            step_configs[name] = {**cfg, **step_configs.get(name, {})}
        # spec-088 / SIGHTING-107: route the "Export for analysis" toggle (an IO
        # concern, NOT an FCParams clustering knob) to the analysis-export step,
        # along with the clustering params it serializes into the export.
        if raw.get("export_for_analysis"):
            step_configs["face_cluster_analysis_export"] = {
                "export_for_analysis": True,
                **fcp.model_dump(),
            }
        # The clustering block was a CONFIG SOURCE, not a step. Remove it so the
        # spec validator doesn't reject it against ClusterPeopleConfig (extra=forbid)
        # — the full FCParams legitimately carries knobs that subset doesn't have.
        if "cluster_people" not in steps:
            step_configs.pop("cluster_people", None)
        self._logger.info(
            "clustering config: broadcast FCParams to %d unified steps "
            "(K=%s yaw_max=%s blur_min=%s merge_enabled=%s)",
            len(UNIFIED_CLUSTERING_STEPS), fcp.K, fcp.yaw_max, fcp.blur_min, fcp.merge_enabled,
        )
        return step_configs

    def execute_pipeline(self, job_id: str) -> None:
        """Execute a pipeline synchronously."""
        self._logger.info(f"Executing pipeline {job_id}")

        job = _jobs.get(job_id)
        if job is None:
            self._logger.error(f"Job not found: {job_id}")
            raise ValueError(f"Job not found: {job_id}")

        run = self._session.query(PipelineRun).filter(PipelineRun.id == job_id).first()
        if run is None:
            raise ValueError(f"Pipeline run not found: {job_id}")

        run.status = "running"
        run.started_at = datetime.utcnow()
        self._session.commit()

        def progress_callback(step: str, progress: float, message: str) -> None:
            run.current_step = step
            run.progress = progress
            self._session.commit()

            for queue in job.subscribers:
                queue.put_nowait(JobProgress(step=step, progress=progress, message=message))

        # Track completed steps per-step for live progress
        def on_step_complete(step_result) -> None:
            """Called after each step to persist progress to DB."""
            from sqlalchemy.orm.attributes import flag_modified
            steps_done = list(run.completed_steps or [])
            steps_done.append({
                "step": step_result.step_name,
                "duration_ms": step_result.duration_ms,
                "status": "completed" if step_result.success else "failed",
                "error": step_result.error_message,
            })
            run.completed_steps = steps_done
            flag_modified(run, "completed_steps")
            self._session.commit()

        # The pipeline is defined by data: steps + per-step params. Both apps
        # submit a PipelineSpec to the one shared primitive (execute_spec), which
        # validates it (mandatory steps, deps, typed params) then runs it once.
        # Albumify keeps its own persistence below; only execution is shared.
        step_configs = self._broadcast_clustering_config(
            run.steps or [], dict(run.step_configs or {})
        )
        spec = PipelineSpec(steps=run.steps, step_configs=step_configs)
        result = execute_spec(
            spec, job.context,
            fail_fast=run.fail_fast,
            progress_cb=progress_callback,
            on_step_complete=on_step_complete,
        )

        if result.success:
            run.status = "completed"
            self._logger.info(f"Pipeline {job_id} completed successfully in {result.total_duration_ms}ms")

            # Build serializable face_subclusters from context.face_clusters
            face_subclusters = None
            if job.context.face_clusters:
                face_subclusters = {
                    str(scene_id): {
                        str(sub_id): {
                            "face_count": sub.get("face_count", "0"),
                            "images": sub.get("images", []),
                            "has_faces": sub.get("has_faces", False),
                            "identity": sub.get("identity", ""),
                        }
                        for sub_id, sub in subclusters.items()
                    }
                    for scene_id, subclusters in job.context.face_clusters.items()
                }

            # spec-084: per-image "why" reason, keyed by path. Built once.
            reason_by_path = _build_reason_by_path(job.context.step_decisions)

            # Built once and reused: the blob column AND the normalized tables
            # (spec-086) derive from the same dict, so they cannot diverge.
            image_metrics = {
                path: self._build_image_metrics(job.context, path, reason_by_path)
                for path in [str(p) for p in job.context.image_paths]
            }

            pipeline_result = PipelineResult(
                id=str(uuid.uuid4()),
                run_id=job_id,
                total_images=len(job.context.image_paths),
                filtered_images=len(job.context.active_images),
                num_clusters=len(job.context.scene_clusters),
                num_selected=len(job.context.selected_images),
                scene_clusters={k: v for k, v in job.context.scene_clusters.items()},
                face_subclusters=face_subclusters,
                selected_images=job.context.selected_images,
                image_metrics=image_metrics,
                siamese_comparisons=job.context.siamese_comparisons or [],
                step_timings={r.step_name: r.duration_ms for r in result.step_results},
                total_duration_ms=result.total_duration_ms,
                fc_export_dir=job.context.fc_export_dir,
                step_decisions=[
                    {"item_id": d.item_id, "item_type": d.item_type, "step": d.step,
                     "decision": d.decision, "reason": d.reason,
                     "config_used": d.config_used, "metrics": d.metrics}
                    for d in (job.context.step_decisions or [])
                ] or None,
            )

            self._session.add(pipeline_result)

            # Persist people records from face clustering results
            # Prefer refined clusters from identity_refinement step if available
            people_clusters = job.context.refined_people_clusters or job.context.people_clusters
            cluster_source = "refined" if job.context.refined_people_clusters else "original"
            self._logger.info(f"People clusters in context: {len(people_clusters)} clusters (source: {cluster_source})")
            created: list = []
            if people_clusters:
                try:
                    people_service = PeopleService(self._session)
                    created = people_service.create_from_clusters(
                        album_id=job.album_id,
                        run_id=job_id,
                        people_clusters=people_clusters,
                        people_thumbnails=job.context.people_thumbnails or None,
                        attachment_decisions=job.context.attachment_decisions or None
                    )
                    self._logger.info(f"Created {len(created)} Person records")
                except Exception as e:
                    self._logger.warning(f"Failed to persist people records: {e}", exc_info=True)
            else:
                self._logger.warning("No people_clusters found in context - skipping Person creation")

            # spec-086: write the normalized metric tables (dual-write next to the
            # blob). People exist now, so faces can be linked to their person_id.
            try:
                self._write_metric_tables(job_id, image_metrics, created)
            except Exception as e:
                self._logger.warning(f"Failed to write metric tables: {e}", exc_info=True)
        else:
            run.status = "failed"
            run.error_message = result.error_message
            self._logger.error(f"Pipeline {job_id} failed: {result.error_message}")

        run.completed_at = datetime.utcnow()
        run.progress = 1.0
        self._session.commit()

        job.completed = True

    def _build_image_metrics(
        self, context: PipelineContext, path: str, reason_by_path: dict = None
    ) -> dict:
        """Build complete metrics dict for a single image.

        Face scoring steps store scores keyed by cache key
        (``"<path>:face_<index>"``), not by image path.  This helper
        collects per-face values back into a list keyed by the image path
        so that they are persisted correctly in the database.

        ``reason_by_path`` (spec-084): optional {path: select_best reason} map
        so the Results table can show *why* an image was selected/filtered.
        """
        # Normalize path for cache key lookups (steps use forward slashes)
        path_normalized = path.replace('\\', '/')

        # MediaPipe faces (if available)
        faces = context.faces.get(path, [])

        # Aggregate face scores by iterating over detected faces
        pose_scores = []
        eyes_scores = []
        smile_scores = []
        for face in faces:
            face_path = str(face.original_path).replace('\\', '/')
            cache_key = f"{face_path}:face_{face.face_index}"
            pose = context.face_pose_scores.get(cache_key)
            if pose is not None:
                pose_scores.append(pose)
            eyes = context.face_eyes_scores.get(cache_key)
            if eyes is not None:
                eyes_scores.append(eyes)
            smile = context.face_smile_scores.get(cache_key)
            if smile is not None:
                smile_scores.append(smile)

        # InsightFace person detection (if available)
        person_data = context.persons.get(path_normalized, {}) if hasattr(context, 'persons') and context.persons else {}
        # Also try original path format
        if not person_data:
            person_data = context.persons.get(path, {}) if hasattr(context, 'persons') and context.persons else {}

        # InsightFace faces (if available) - try both path formats
        insightface_data = {}
        if hasattr(context, 'insightface_faces') and context.insightface_faces:
            insightface_data = context.insightface_faces.get(path_normalized, {})
            if not insightface_data:
                insightface_data = context.insightface_faces.get(path, {})
        insightface_faces = insightface_data.get('faces', [])

        # Get InsightFace face scores if MediaPipe faces not available
        if not pose_scores and insightface_faces:
            for face_info in insightface_faces:
                face_index = face_info.get('face_index', 0)
                cache_key = f"{path_normalized}:face_{face_index}"
                pose = context.face_pose_scores.get(cache_key)
                if pose is not None:
                    pose_scores.append(pose)
                eyes = context.face_eyes_scores.get(cache_key)
                if eyes is not None:
                    eyes_scores.append(eyes)
                smile = context.face_smile_scores.get(cache_key)
                if smile is not None:
                    smile_scores.append(smile)

        # Extract face filtering and frontal scores from InsightFace faces
        filter_stats = insightface_data.get('filter_stats', {})
        frontal_stats = insightface_data.get('frontal_stats', {})

        # Get best frontal score, roll angle, and centrality from all faces
        best_frontal_score = None
        best_centrality = None
        roll_angles = []
        filter_scores_list = []
        frontal_scores_list = []

        for face_info in insightface_faces:
            # Filter scores
            filter_scores = face_info.get('filter_scores', {})
            face_bbox = face_info.get('bbox')  # {x, y, w, h, x_px, y_px, w_px, h_px}
            if filter_scores or face_bbox:
                entry = {
                    'face_index': face_info.get('face_index', 0),
                    'confidence': filter_scores.get('confidence') or face_info.get('confidence'),
                    'bbox_ratio': filter_scores.get('bbox_ratio'),
                    'relative_size': filter_scores.get('relative_size'),
                    'eye_ratio': filter_scores.get('eye_ratio'),
                    'filter_passed': face_info.get('filter_passed', True),
                }
                if face_bbox:
                    entry['bbox'] = face_bbox  # Include face bounding box for UI overlay
                filter_scores_list.append(entry)

            # Frontal scores (only for faces that passed filtering)
            if face_info.get('filter_passed', True):
                frontal_score = face_info.get('frontal_score')
                if frontal_score is not None:
                    frontal_scores_data = face_info.get('frontal_scores', {})
                    frontal_scores_list.append({
                        'face_index': face_info.get('face_index', 0),
                        'frontal_score': frontal_score,
                        'eye_bbox_ratio': frontal_scores_data.get('eye_bbox_ratio'),
                        'asymmetry': frontal_scores_data.get('asymmetry'),
                        'is_clusterable': face_info.get('is_clusterable', True),
                    })

                    if best_frontal_score is None or frontal_score > best_frontal_score:
                        best_frontal_score = frontal_score

                centrality = face_info.get('centrality')
                if centrality is not None:
                    if best_centrality is None or centrality > best_centrality:
                        best_centrality = centrality

                roll_angle = face_info.get('roll_angle')
                if roll_angle is not None:
                    roll_angles.append(roll_angle)

        # spec-085 (C-lite): build the canonical ImageMetrics directly. The schema
        # is the single definition of the shape; this function only supplies the
        # values (the bespoke extraction from context). Returns a dict for the
        # JSON column. Field names/types are validated against the schema here.
        return ImageMetrics(
            path=path,
            iqa_score=context.iqa_scores.get(path),
            ava_score=context.ava_scores.get(path),
            sharpness=context.sharpness_scores.get(path),
            cluster_id=context.scene_cluster_labels.get(path),
            face_count=len(faces) or len(insightface_faces),
            face_pose_scores=pose_scores or None,
            face_eyes_scores=eyes_scores or None,
            face_smile_scores=smile_scores or None,
            composite_score=context.composite_scores.get(path),
            # spec-084: composite breakdown + the human-readable decision reason.
            quality_score=context.quality_scores.get(path),
            person_penalty=context.person_penalties.get(path),
            filter_reason=(reason_by_path or {}).get(path),
            is_selected=path in context.selected_images,
            # InsightFace-specific metrics
            person_detected=person_data.get('person_detected'),
            body_facing_score=person_data.get('body_facing_score'),
            person_confidence=person_data.get('confidence'),
            # Face filtering metrics
            filter_stats=filter_stats or None,
            filter_scores=filter_scores_list or None,
            # Frontal scoring metrics
            frontal_stats=frontal_stats or None,
            frontal_scores=frontal_scores_list or None,
            best_frontal_score=best_frontal_score,
            best_centrality=best_centrality,
            roll_angles=roll_angles or None,
        ).model_dump()

    def _write_metric_tables(
        self, run_id: str, image_metrics: dict, people: list
    ) -> None:
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

    def get_status(self, job_id: str) -> Optional[PipelineRun]:
        """Get the status of a pipeline run."""
        return self._session.query(PipelineRun).filter(PipelineRun.id == job_id).first()

    def get_result(self, job_id: str) -> Optional[PipelineResult]:
        """Get the result of a completed pipeline run."""
        return self._session.query(PipelineResult).filter(PipelineResult.run_id == job_id).first()

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to progress updates for a job."""
        job = _jobs.get(job_id)
        if job is None:
            self._logger.warning(f"Subscribe failed - job not found: {job_id}")
            raise ValueError(f"Job not found: {job_id}")

        queue = asyncio.Queue()
        job.subscribers.append(queue)
        self._logger.debug(f"Client subscribed to job {job_id}")
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe from progress updates."""
        job = _jobs.get(job_id)
        if job and queue in job.subscribers:
            job.subscribers.remove(queue)
            self._logger.debug(f"Client unsubscribed from job {job_id}")
