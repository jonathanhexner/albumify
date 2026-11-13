"""Result service - business logic for pipeline results."""

import logging
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from sim_bench.api.database.models import Album, PipelineRun, PipelineResult, Person
from sim_bench.api.schemas.result import ImageMetrics
from sim_bench.api.services.run_display_copies import RunDisplayCopies, normalise


RESULT_COLUMNS = (
    "total_images", "filtered_images", "num_clusters", "num_selected", "scene_clusters", "face_subclusters",
    "selected_images", "image_metrics", "siamese_comparisons", "step_timings", "total_duration_ms",
    "fc_export_dir", "step_decisions",
)


class ResultCache:
    """Finished runs' result rows, kept in this process (SIGHTING-152).

    A ``pipeline_results`` row is written once, when its run finishes, and never changed, so a copy can be served
    instead of re-reading up to 2.8 MB from the database on every request. Only rows that exist are cached; the
    run's writer evicts its own entry, and deleting an album evicts its runs.
    """

    def __init__(self, max_runs: int = 8):
        self._max_runs = max_runs
        self._rows: "OrderedDict[str, SimpleNamespace]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, run_id: str) -> Optional[SimpleNamespace]:
        with self._lock:
            row = self._rows.get(run_id)
            if row is not None:
                self._rows.move_to_end(run_id)
            return row

    def put(self, run_id: str, row: SimpleNamespace) -> None:
        with self._lock:
            self._rows[run_id] = row
            self._rows.move_to_end(run_id)
            while len(self._rows) > self._max_runs:
                self._rows.popitem(last=False)

    def evict(self, run_ids: Optional[Iterable[str]] = None) -> None:
        """Drop the given runs, or every run when ``run_ids`` is None."""
        with self._lock:
            if run_ids is None:
                self._rows.clear()
            for run_id in run_ids or ():
                self._rows.pop(run_id, None)


result_cache = ResultCache()


class ResultService:
    """Service for managing and exporting pipeline results."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    def _result(self, job_id: str) -> Optional[SimpleNamespace]:
        """The run's result row (attributes = ``RESULT_COLUMNS``), from ``result_cache`` after the first read."""
        cached = result_cache.get(job_id)
        if cached is not None:
            return cached
        row = (
            self._session.query(*(getattr(PipelineResult, c) for c in RESULT_COLUMNS))
            .filter(PipelineResult.run_id == job_id)
            .first()
        )
        if row is None:
            return None
        snapshot = SimpleNamespace(**dict(zip(RESULT_COLUMNS, row)))
        snapshot.display_paths = RunDisplayCopies().paths(job_id)   # thumbnails from the 768 px copies (C5)
        result_cache.put(job_id, snapshot)
        return snapshot

    def get_step_decisions(self, job_id: str) -> Optional[list]:
        """The run's per-item step decisions (Explore, image popup); None when the run has no result."""
        result = self._result(job_id)
        if result is None:
            return None
        return result.step_decisions or []

    def get_result(self, job_id: str) -> Optional[dict]:
        """Get full result details for a job."""
        run = (
            self._session.query(PipelineRun)
            .filter(PipelineRun.id == job_id)
            .first()
        )
        if not run:
            return None

        album = run.album
        result = self._result(job_id)

        # Count people if available
        num_people = (
            self._session.query(Person)
            .filter(Person.run_id == job_id)
            .count()
        )

        return {
            'job_id': job_id,
            'album_id': album.id,
            'album_name': album.name,
            'status': run.status,
            'pipeline_name': run.pipeline_name,
            'steps': run.steps or [],
            'total_images': result.total_images if result else 0,
            'filtered_images': result.filtered_images if result else 0,
            'num_clusters': result.num_clusters if result else 0,
            'num_selected': result.num_selected if result else 0,
            'num_people': num_people if num_people > 0 else None,
            'scene_clusters': result.scene_clusters if result else {},
            'selected_images': result.selected_images if result else [],
            'step_timings': result.step_timings if result else {},
            'total_duration_ms': result.total_duration_ms if result else 0,
            'fc_export_dir': result.fc_export_dir if result else None,
            'step_decisions': result.step_decisions if result else None,
            'created_at': run.created_at,
            'started_at': run.started_at,
            'completed_at': run.completed_at
        }

    def _build_image_dict(self, path: str, metrics: dict, is_selected: bool = False,
                          display_paths: Optional[dict] = None) -> dict:
        """Build the API image dict from stored metrics.

        spec-085: derive straight from ``ImageMetrics`` (the single source of
        truth) instead of a hand-copied field list. Adding a per-image metric
        now means adding it to ``ImageMetrics`` only — there is no second list to
        keep in sync, so the field set cannot silently narrow (the spec-084 bug).
        Unknown keys in ``metrics`` are ignored; missing ones default to None.
        """
        data = {**metrics, 'path': path}
        data['is_selected'] = metrics.get('is_selected', is_selected)
        data['display_path'] = (display_paths or {}).get(normalise(path))
        return ImageMetrics(**data).model_dump()

    def get_images(
        self,
        job_id: str,
        cluster_id: Optional[int] = None,
        person_id: Optional[str] = None,
        selected_only: bool = False
    ) -> list[dict]:
        """Get images from a result, optionally filtered.

        Args:
            job_id: Pipeline run ID
            cluster_id: Filter by scene cluster ID
            person_id: Filter by person ID
            selected_only: Only return selected images
        """
        result = self._result(job_id)
        if not result:
            return []

        images = []

        if selected_only:
            # Return only selected images
            for path in (result.selected_images or []):
                metrics = (result.image_metrics or {}).get(path, {})
                images.append(self._build_image_dict(path, metrics, is_selected=True,
                                                     display_paths=result.display_paths))
        elif cluster_id is not None:
            # Return images from specific cluster
            cluster_images = (result.scene_clusters or {}).get(str(cluster_id), [])
            # Also try with int key
            if not cluster_images:
                cluster_images = (result.scene_clusters or {}).get(cluster_id, [])

            for path in cluster_images:
                metrics = (result.image_metrics or {}).get(path, {})
                img = self._build_image_dict(path, metrics, display_paths=result.display_paths)
                img['cluster_id'] = cluster_id
                images.append(img)
        elif person_id:
            # Return images containing a specific person
            person = (
                self._session.query(Person)
                .filter(Person.id == person_id)
                .first()
            )
            if person:
                seen_paths = set()
                for face in (person.face_instances or []):
                    path = face.get('image_path')
                    if path and path not in seen_paths:
                        seen_paths.add(path)
                        metrics = (result.image_metrics or {}).get(path, {})
                        images.append(self._build_image_dict(path, metrics, display_paths=result.display_paths))
        else:
            # Return all images with metrics
            for path, metrics in (result.image_metrics or {}).items():
                images.append(self._build_image_dict(path, metrics, display_paths=result.display_paths))

        return images

    def get_clusters(self, job_id: str) -> list[dict]:
        """Get cluster information for a result."""
        result = self._result(job_id)
        if not result:
            return []

        selected_set = set(result.selected_images or [])
        image_metrics = result.image_metrics or {}

        # Build person_labels: {image_path: [person_display_name, ...]}
        people = (
            self._session.query(Person)
            .filter(Person.run_id == job_id)
            .all()
        )
        image_people: dict[str, list[str]] = {}
        for person in people:
            display_name = person.name or f"Person {person.person_index + 1}"
            for face in (person.face_instances or []):
                path = face.get('image_path')
                if path:
                    image_people.setdefault(path, [])
                    if display_name not in image_people[path]:
                        image_people[path].append(display_name)

        clusters = []

        for cluster_id, images in (result.scene_clusters or {}).items():
            best_image = None
            selected_count = 0
            total_face_count = 0
            has_faces = False
            cluster_images = []
            cluster_person_labels: dict[str, list[str]] = {}

            for path in images:
                metrics = image_metrics.get(path, {})
                is_selected = path in selected_set
                img_dict = self._build_image_dict(path, metrics, is_selected=is_selected,
                                                   display_paths=result.display_paths)
                cid = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
                img_dict['cluster_id'] = cid
                cluster_images.append(img_dict)

                if is_selected:
                    selected_count += 1
                    if best_image is None:
                        best_image = path

                fc = metrics.get('face_count', 0) or 0
                if fc > 0:
                    has_faces = True
                    total_face_count += fc

                if path in image_people:
                    cluster_person_labels[path] = image_people[path]

            clusters.append({
                'cluster_id': int(cluster_id) if isinstance(cluster_id, str) else cluster_id,
                'image_count': len(images),
                'selected_count': selected_count,
                'has_faces': has_faces,
                'face_count': total_face_count,
                'images': cluster_images,
                'best_image': best_image,
                'person_labels': cluster_person_labels,
            })

        # Sort by cluster ID
        clusters.sort(key=lambda c: c['cluster_id'])
        return clusters

    def get_metrics(self, job_id: str) -> Optional[dict]:
        """Get aggregate metrics for a pipeline run."""
        result = self._result(job_id)
        if not result:
            return None

        # Calculate averages
        image_metrics = result.image_metrics or {}
        iqa_scores = [m.get('iqa_score') for m in image_metrics.values() if m.get('iqa_score')]
        ava_scores = [m.get('ava_score') for m in image_metrics.values() if m.get('ava_score')]

        avg_iqa = sum(iqa_scores) / len(iqa_scores) if iqa_scores else None
        avg_ava = sum(ava_scores) / len(ava_scores) if ava_scores else None

        # Count people
        num_people = (
            self._session.query(Person)
            .filter(Person.run_id == job_id)
            .count()
        )

        return {
            'total_images': result.total_images,
            'filtered_images': result.filtered_images,
            'num_clusters': result.num_clusters,
            'num_selected': result.num_selected,
            'num_people': num_people if num_people > 0 else None,
            'avg_iqa_score': round(avg_iqa, 4) if avg_iqa else None,
            'avg_ava_score': round(avg_ava, 4) if avg_ava else None,
            'step_timings': result.step_timings or {},
            'total_duration_ms': result.total_duration_ms or 0
        }

    def get_comparisons(self, job_id: str) -> Optional[list]:
        """Get Siamese/duplicate comparison log for a pipeline run."""
        result = self._result(job_id)
        if not result:
            return None

        return result.siamese_comparisons or []

    def get_subclusters(self, job_id: str) -> Optional[dict]:
        """Get face sub-clusters for a pipeline run."""
        result = self._result(job_id)
        if not result:
            return None

        return result.face_subclusters or {}

    def export_results(
        self,
        job_id: str,
        output_path: str,
        include_selected: bool = True,
        include_all_filtered: bool = False,
        organize_by_cluster: bool = False,
        organize_by_person: bool = False,
        copy_mode: str = "copy"
    ) -> dict:
        """Export pipeline results to a directory.

        Args:
            job_id: Pipeline run ID
            output_path: Destination directory
            include_selected: Include selected images
            include_all_filtered: Include all filtered (not rejected) images
            organize_by_cluster: Create subdirs per cluster
            organize_by_person: Create subdirs per person
            copy_mode: "copy" or "symlink"

        Returns:
            Dict with success status and file count
        """
        result = self._result(job_id)
        if not result:
            return {
                'success': False,
                'output_path': output_path,
                'files_exported': 0,
                'errors': ['Result not found']
            }

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        files_exported = 0
        errors = []

        # Determine which images to export
        images_to_export = []

        if include_selected:
            for path in (result.selected_images or []):
                cluster_id = (result.image_metrics or {}).get(path, {}).get('cluster_id')
                images_to_export.append((path, cluster_id))

        if include_all_filtered:
            selected_set = set(result.selected_images or [])
            for path, metrics in (result.image_metrics or {}).items():
                if path not in selected_set:
                    images_to_export.append((path, metrics.get('cluster_id')))

        # Export images
        for src_path, cluster_id in images_to_export:
            src = Path(src_path)
            if not src.exists():
                errors.append(f"Source not found: {src_path}")
                continue

            # Determine destination
            if organize_by_cluster and cluster_id is not None:
                dest_dir = output_dir / f"cluster_{cluster_id}"
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / src.name
            else:
                dest = output_dir / src.name

            # Handle filename conflicts
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = dest.parent / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                if copy_mode == "symlink":
                    dest.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dest)
                files_exported += 1
            except Exception as e:
                errors.append(f"Failed to export {src_path}: {e}")

        # Export by person if requested
        if organize_by_person:
            people = (
                self._session.query(Person)
                .filter(Person.run_id == job_id)
                .all()
            )

            for person in people:
                person_name = person.name or f"person_{person.person_index}"
                person_dir = output_dir / "people" / person_name
                person_dir.mkdir(parents=True, exist_ok=True)

                seen_paths = set()
                for face in (person.face_instances or []):
                    src_path = face.get('image_path')
                    if not src_path or src_path in seen_paths:
                        continue
                    seen_paths.add(src_path)

                    src = Path(src_path)
                    if not src.exists():
                        continue

                    dest = person_dir / src.name
                    if dest.exists():
                        continue  # Skip duplicates in person folders

                    try:
                        if copy_mode == "symlink":
                            dest.symlink_to(src.resolve())
                        else:
                            shutil.copy2(src, dest)
                        files_exported += 1
                    except Exception as e:
                        errors.append(f"Failed to export {src_path} for person: {e}")

        self._logger.info(
            f"Exported {files_exported} files to {output_path} "
            f"({len(errors)} errors)"
        )

        return {
            'success': len(errors) == 0,
            'output_path': output_path,
            'files_exported': files_exported,
            'errors': errors
        }

    def list_results(self, album_id: Optional[str] = None) -> list[dict]:
        """List completed runs with their summary counts, newest first, optionally for one album.

        Summary columns only, in two queries whatever the number of runs (SIGHTING-152): a result row carries every
        photo's metrics and step decisions; the run's decisions come from :meth:`get_step_decisions`.
        """
        query = (
            self._session.query(
                PipelineRun.id, PipelineRun.album_id, Album.name, PipelineRun.status, PipelineRun.created_at,
                PipelineRun.completed_at, PipelineResult.total_images, PipelineResult.filtered_images,
                PipelineResult.num_clusters, PipelineResult.num_selected, PipelineResult.fc_export_dir,
                PipelineResult.total_duration_ms,
            )
            .join(Album, Album.id == PipelineRun.album_id)
            .join(PipelineResult, PipelineResult.run_id == PipelineRun.id)
            .filter(PipelineRun.status == "completed")
        )
        if album_id:
            query = query.filter(PipelineRun.album_id == album_id)
        rows = query.order_by(PipelineRun.completed_at.desc()).all()

        people_query = self._session.query(Person.run_id, func.count()).group_by(Person.run_id)
        if album_id:
            people_query = people_query.filter(Person.album_id == album_id)
        people = dict(people_query.all())

        return [{
            'job_id': r[0],
            'album_id': r[1],
            'album_name': r[2],
            'status': r[3],
            'total_images': r[6],
            'filtered_images': r[7],
            'num_clusters': r[8],
            'num_selected': r[9],
            'num_people': people.get(r[0]) or None,
            'fc_export_dir': r[10],
            'created_at': r[4],
            'completed_at': r[5],
            'total_duration_ms': r[11],
        } for r in rows]
