"""Pipeline service - orchestrates pipeline execution."""

import logging
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from sim_bench.api.database.models import Album, PipelineRun, PipelineResult
from sim_bench.api.services.people_service import PeopleService
from sim_bench.api.services.config_service import ConfigService
from sim_bench.pipeline.cache_handler import UniversalCacheHandler
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_registry


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

        _jobs[run_id] = JobState(
            run_id=run_id,
            album_id=album_id,
            context=context
        )

        return run_id

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

        import sim_bench.pipeline.steps.all_steps
        registry = get_registry()
        executor = PipelineExecutor(registry)

        def progress_callback(step: str, progress: float, message: str) -> None:
            run.current_step = step
            run.progress = progress
            self._session.commit()

            for queue in job.subscribers:
                queue.put_nowait(JobProgress(step=step, progress=progress, message=message))

        config = PipelineConfig(
            fail_fast=run.fail_fast,
            step_configs=run.step_configs or {},
            progress_callback=progress_callback
        )

        result = executor.execute(job.context, run.steps, config)

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
                image_metrics={
                    path: self._build_image_metrics(job.context, path)
                    for path in [str(p) for p in job.context.image_paths]
                },
                siamese_comparisons=job.context.siamese_comparisons or [],
                step_timings={r.step_name: r.duration_ms for r in result.step_results},
                total_duration_ms=result.total_duration_ms
            )

            self._session.add(pipeline_result)

            # Persist people records from face clustering results
            # Prefer refined clusters from identity_refinement step if available
            people_clusters = job.context.refined_people_clusters or job.context.people_clusters
            cluster_source = "refined" if job.context.refined_people_clusters else "original"
            self._logger.info(f"People clusters in context: {len(people_clusters)} clusters (source: {cluster_source})")
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
        else:
            run.status = "failed"
            run.error_message = result.error_message
            self._logger.error(f"Pipeline {job_id} failed: {result.error_message}")

        run.completed_at = datetime.utcnow()
        run.progress = 1.0
        self._session.commit()

        job.completed = True

    def _build_image_metrics(self, context: PipelineContext, path: str) -> dict:
        """Build complete metrics dict for a single image.

        Face scoring steps store scores keyed by cache key
        (``"<path>:face_<index>"``), not by image path.  This helper
        collects per-face values back into a list keyed by the image path
        so that they are persisted correctly in the database.
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
            if filter_scores:
                filter_scores_list.append({
                    'face_index': face_info.get('face_index', 0),
                    'confidence': filter_scores.get('confidence'),
                    'bbox_ratio': filter_scores.get('bbox_ratio'),
                    'relative_size': filter_scores.get('relative_size'),
                    'eye_ratio': filter_scores.get('eye_ratio'),
                    'filter_passed': face_info.get('filter_passed', True),
                })

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

        return {
            "iqa_score": context.iqa_scores.get(path),
            "ava_score": context.ava_scores.get(path),
            "sharpness": context.sharpness_scores.get(path),
            "cluster_id": context.scene_cluster_labels.get(path),
            "face_count": len(faces) or len(insightface_faces),
            "face_pose_scores": pose_scores or None,
            "face_eyes_scores": eyes_scores or None,
            "face_smile_scores": smile_scores or None,
            "composite_score": context.composite_scores.get(path),
            "is_selected": path in context.selected_images,
            # InsightFace-specific metrics
            "person_detected": person_data.get('person_detected'),
            "body_facing_score": person_data.get('body_facing_score'),
            "person_confidence": person_data.get('confidence'),
            # Face filtering metrics
            "filter_stats": filter_stats or None,
            "filter_scores": filter_scores_list or None,
            # Frontal scoring metrics
            "frontal_stats": frontal_stats or None,
            "frontal_scores": frontal_scores_list or None,
            "best_frontal_score": best_frontal_score,
            "best_centrality": best_centrality,
            "roll_angles": roll_angles or None,
        }

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
