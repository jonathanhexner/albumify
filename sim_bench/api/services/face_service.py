"""Face service - business logic for face management operations."""

import base64
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image, ImageOps
from sqlalchemy.orm import Session

from sim_bench.api.database.models import (
    FaceOverride,
    Person,
    PipelineResult,
    PipelineRun,
    UniversalCache,
    UserEvent,
)
from sim_bench.api.schemas.face import (
    FaceInfo,
    PersonDistance,
    BorderlineFace,
    PersonSummary,
    FaceAction,
    BatchChangeResponse,
)


class FaceService:
    """Service for face management operations."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    def get_all_faces(
        self,
        album_id: str,
        run_id: str,
        status_filter: List[str] = None
    ) -> List[FaceInfo]:
        """
        Get all faces for a pipeline run with their current status.

        Reads face data directly from UniversalCache (insightface_detection).
        """
        # Load face overrides for this album
        overrides = self._get_face_overrides(album_id)
        override_map = {o.face_key: o for o in overrides}

        # Load people for assignment lookup
        people = self._session.query(Person).filter(Person.run_id == run_id).all()
        person_map = {p.id: p for p in people}

        # Build face-to-person lookup from person.face_instances
        face_to_person = {}
        for person in people:
            for fi in person.face_instances or []:
                img_path = fi.get('image_path', '').replace('\\', '/')
                face_idx = fi.get('face_index', 0)
                face_key = f"{img_path}:face_{face_idx}"
                face_to_person[face_key] = {
                    'person_id': person.id,
                    'person_name': person.name or f"Person {person.person_index + 1}",
                    'method': fi.get('assignment_method', 'core'),
                    'confidence': fi.get('assignment_confidence'),
                }

        # Get all face detections from cache
        cache_entries = (
            self._session.query(UniversalCache)
            .filter(UniversalCache.feature_type == "insightface_detection")
            .all()
        )

        faces = []
        for entry in cache_entries:
            img_path = entry.image_path.replace('\\', '/')

            # Parse cached face data (stored as JSON bytes in data_blob)
            if not entry.data_blob:
                continue
            face_data = json.loads(entry.data_blob.decode('utf-8'))

            face_list = face_data.get('faces', [])

            for face in face_list:
                face_idx = face.get('face_index', 0)
                face_key = f"{img_path}:face_{face_idx}"
                bbox = face.get('bbox')

                # Determine status
                override = override_map.get(face_key)
                person_info = face_to_person.get(face_key)

                if override:
                    status = override.status
                    person_id = override.person_id
                    person_name = None
                    if person_id and person_id in person_map:
                        p = person_map[person_id]
                        person_name = p.name or f"Person {p.person_index + 1}"
                    method = "user"
                elif person_info:
                    status = "assigned"
                    person_id = person_info['person_id']
                    person_name = person_info['person_name']
                    method = person_info['method']
                else:
                    status = "unassigned"
                    person_id = None
                    person_name = None
                    method = None

                # Apply filter
                if status_filter and status not in status_filter:
                    continue

                face_info = FaceInfo(
                    face_key=face_key,
                    image_path=img_path,
                    face_index=face_idx,
                    thumbnail_base64=self._generate_face_thumbnail(img_path, bbox),
                    bbox=bbox,
                    status=status,
                    person_id=person_id,
                    person_name=person_name,
                    assignment_method=method,
                    assignment_confidence=person_info.get('confidence') if person_info else None,
                    frontal_score=face.get('frontal_score'),
                    centroid_distance=None,
                    exemplar_matches=None,
                )
                faces.append(face_info)

        self._logger.info(f"Retrieved {len(faces)} faces from cache")
        return faces

    def get_face(
        self,
        album_id: str,
        run_id: str,
        face_key: str
    ) -> Optional[FaceInfo]:
        """Get single face with full details."""
        faces = self.get_all_faces(album_id, run_id)
        for face in faces:
            if face.face_key == face_key:
                return face
        return None

    def get_face_distances(
        self,
        album_id: str,
        run_id: str,
        face_key: str
    ) -> List[PersonDistance]:
        """
        Get distances from a face to all people.

        Returns list sorted by centroid_distance ascending.
        """
        # Get face embedding
        face_embedding = self._get_face_embedding(face_key, run_id)
        if face_embedding is None:
            self._logger.warning(f"No embedding found for face {face_key}")
            return []

        # Get all people with their exemplars
        people = self._session.query(Person).filter(Person.run_id == run_id).all()

        # Get attachment thresholds from config
        config = self._get_refinement_config(run_id)
        centroid_threshold = config.get('centroid_threshold', 0.38)
        exemplar_threshold = config.get('exemplar_threshold', 0.40)

        distances = []
        for person in people:
            # Compute centroid distance
            centroid = self._get_person_centroid(person.id, run_id)
            if centroid is None:
                continue

            centroid_dist = self._cosine_distance(face_embedding, centroid)

            # Get exemplar embeddings and compute matches
            exemplar_embeddings = self._get_person_exemplars(person.id, run_id)
            exemplar_distances = [
                self._cosine_distance(face_embedding, ex)
                for ex in exemplar_embeddings
            ]

            matches = sum(1 for d in exemplar_distances if d <= exemplar_threshold)
            min_exemplar_dist = min(exemplar_distances) if exemplar_distances else 999.0

            # Would it attach?
            min_required = max(2, int(np.ceil(0.3 * len(exemplar_embeddings))))
            if len(exemplar_embeddings) <= 3:
                min_required = 1
            would_attach = centroid_dist <= centroid_threshold and matches >= min_required

            distances.append(PersonDistance(
                person_id=person.id,
                person_name=person.name or f"Person {person.person_index + 1}",
                thumbnail_base64=self._get_person_thumbnail(person),
                centroid_distance=round(centroid_dist, 4),
                exemplar_matches=matches,
                min_exemplar_distance=round(min_exemplar_dist, 4),
                would_attach=would_attach,
            ))

        # Sort by distance
        distances.sort(key=lambda d: d.centroid_distance)
        return distances

    def get_borderline_faces(
        self,
        album_id: str,
        run_id: str,
        limit: int = 10
    ) -> List[BorderlineFace]:
        """
        Get faces in the uncertainty zone for user review.

        Returns faces where distance is between attach and reject thresholds,
        sorted by uncertainty (most uncertain first).
        """
        # Get config thresholds
        config = self._get_refinement_config(run_id)
        attach_threshold = config.get('centroid_threshold', 0.38)
        reject_threshold = config.get('reject_threshold', 0.45)

        # Get unassigned faces
        unassigned = self.get_all_faces(album_id, run_id, status_filter=["unassigned"])

        borderline = []
        for face in unassigned:
            # Get distances to all people
            distances = self.get_face_distances(album_id, run_id, face.face_key)
            if not distances:
                continue

            closest = distances[0]

            # Check if in borderline zone
            if attach_threshold < closest.centroid_distance < reject_threshold:
                # Compute uncertainty score (0 = at midpoint, 1 = at edge)
                midpoint = (attach_threshold + reject_threshold) / 2
                range_half = (reject_threshold - attach_threshold) / 2
                uncertainty = 1 - abs(closest.centroid_distance - midpoint) / range_half

                borderline.append(BorderlineFace(
                    face=face,
                    closest_person_id=closest.person_id,
                    closest_person_name=closest.person_name,
                    closest_person_thumbnail=closest.thumbnail_base64,
                    distance=closest.centroid_distance,
                    uncertainty_score=round(uncertainty, 3),
                    attach_threshold=attach_threshold,
                    reject_threshold=reject_threshold,
                ))

        # Sort by uncertainty (most uncertain first = lowest score)
        borderline.sort(key=lambda b: b.uncertainty_score)
        return borderline[:limit]

    def get_people_summary(
        self,
        album_id: str,
        run_id: str
    ) -> List[PersonSummary]:
        """Get summary of all identified people."""
        people = self._session.query(Person).filter(Person.run_id == run_id).all()

        summaries = []
        for person in people:
            summaries.append(PersonSummary(
                person_id=person.id,
                name=person.name or f"Person {person.person_index + 1}",
                thumbnail_base64=self._get_person_thumbnail(person),
                face_count=person.face_count or 0,
                exemplar_count=min(5, person.face_count or 0),
                cluster_tightness=None,
            ))

        # Sort by face count descending
        summaries.sort(key=lambda s: s.face_count, reverse=True)
        return summaries

    def apply_single_change(
        self,
        album_id: str,
        run_id: str,
        change: FaceAction
    ) -> bool:
        """
        Apply a single face change immediately.

        Returns True if successful.
        """
        result = self.apply_batch_changes(album_id, run_id, [change], recluster=False)
        return result.applied_count > 0

    def apply_batch_changes(
        self,
        album_id: str,
        run_id: str,
        changes: List[FaceAction],
        recluster: bool = True
    ) -> BatchChangeResponse:
        """
        Apply multiple changes and optionally recluster.
        """
        applied = 0
        failed = 0
        failures = []

        for change in changes:
            try:
                self._apply_single_change(album_id, run_id, change)
                applied += 1
            except Exception as e:
                self._logger.error(f"Failed to apply change for {change.face_key}: {e}")
                failed += 1
                failures.append({"face_key": change.face_key, "error": str(e)})

        self._session.commit()

        auto_assigned = 0
        new_unassigned = 0

        if recluster and applied > 0:
            recluster_result = self._trigger_recluster(album_id, run_id)
            auto_assigned = recluster_result.get('auto_assigned', 0)
            new_unassigned = recluster_result.get('still_unassigned', 0)

        return BatchChangeResponse(
            applied_count=applied,
            failed_count=failed,
            failures=failures,
            auto_assigned_count=auto_assigned,
            new_unassigned_count=new_unassigned,
        )

    def create_person(
        self,
        album_id: str,
        run_id: str,
        name: str,
        face_keys: List[str]
    ) -> Optional[Person]:
        """Create a new person from selected faces."""
        # Get next person index
        max_index = (
            self._session.query(Person.person_index)
            .filter(Person.run_id == run_id)
            .order_by(Person.person_index.desc())
            .first()
        )
        next_index = (max_index[0] + 1) if max_index else 0

        # Create person
        person = Person(
            id=str(uuid.uuid4()),
            album_id=album_id,
            run_id=run_id,
            person_index=next_index,
            name=name,
            face_count=len(face_keys),
            image_count=len(set(fk.rsplit(':face_', 1)[0] for fk in face_keys)),
            face_instances=[],
        )

        # Add faces to person
        for face_key in face_keys:
            parts = face_key.rsplit(':face_', 1)
            if len(parts) != 2:
                continue

            image_path = parts[0]
            face_index = int(parts[1])

            person.face_instances.append({
                'image_path': image_path,
                'face_index': face_index,
                'bbox': None,
                'assignment_method': 'user',
                'assignment_confidence': 1.0,
            })

            # Create override
            self._create_override(album_id, run_id, face_key, "assigned", person.id)

        # Set thumbnail from first face
        if face_keys:
            first_key = face_keys[0]
            parts = first_key.rsplit(':face_', 1)
            person.thumbnail_image_path = parts[0]
            person.thumbnail_face_index = int(parts[1])

        self._session.add(person)
        self._session.commit()

        return person

    # ========================================================
    # Helper Methods
    # ========================================================

    def _get_pipeline_result(self, run_id: str) -> Optional[PipelineResult]:
        """Get pipeline result for a run."""
        return (
            self._session.query(PipelineResult)
            .filter(PipelineResult.run_id == run_id)
            .first()
        )

    def _get_face_overrides(self, album_id: str) -> List[FaceOverride]:
        """Get all face overrides for an album."""
        return (
            self._session.query(FaceOverride)
            .filter(FaceOverride.album_id == album_id)
            .all()
        )

    def _get_face_embedding(
        self,
        face_key: str,
        run_id: str
    ) -> Optional[np.ndarray]:
        """Load face embedding from cache."""
        from sim_bench.api.database.models import UniversalCache

        cache_entry = (
            self._session.query(UniversalCache)
            .filter(UniversalCache.image_path == face_key)
            .filter(UniversalCache.feature_type == "face_embedding")
            .first()
        )

        if cache_entry and cache_entry.data_blob:
            embedding = np.frombuffer(cache_entry.data_blob, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
        return None

    def _get_person_centroid(self, person_id: str, run_id: str) -> Optional[np.ndarray]:
        """Get centroid embedding for a person."""
        person = self._session.query(Person).filter(Person.id == person_id).first()
        if not person:
            return None

        embeddings = []
        for fi in person.face_instances or []:
            img_path = fi.get('image_path', '').replace('\\', '/')
            face_idx = fi.get('face_index', 0)
            face_key = f"{img_path}:face_{face_idx}"
            emb = self._get_face_embedding(face_key, run_id)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None

        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def _get_person_exemplars(self, person_id: str, run_id: str) -> List[np.ndarray]:
        """Get exemplar embeddings for a person (top 5 by quality)."""
        person = self._session.query(Person).filter(Person.id == person_id).first()
        if not person:
            return []

        # Get all embeddings with their frontal scores
        face_data = []
        for fi in person.face_instances or []:
            img_path = fi.get('image_path', '').replace('\\', '/')
            face_idx = fi.get('face_index', 0)
            face_key = f"{img_path}:face_{face_idx}"
            emb = self._get_face_embedding(face_key, run_id)
            if emb is not None:
                frontal = fi.get('frontal_score', 0.5)
                face_data.append((emb, frontal))

        # Sort by frontal score and take top 5
        face_data.sort(key=lambda x: x[1], reverse=True)
        return [emb for emb, _ in face_data[:5]]

    def _get_person_thumbnail(self, person: Person) -> Optional[str]:
        """Get base64 thumbnail for a person."""
        if not person.thumbnail_image_path:
            return None
        return self._generate_face_thumbnail(
            person.thumbnail_image_path,
            person.thumbnail_bbox
        )

    def _get_refinement_config(self, run_id: str) -> dict:
        """Get identity refinement config for a run."""
        run = self._session.query(PipelineRun).filter(PipelineRun.id == run_id).first()
        if run and run.step_configs:
            return run.step_configs.get('identity_refinement', {})
        return {}

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        return float(1 - np.dot(a, b))

    def _generate_face_thumbnail(
        self,
        image_path: str,
        bbox: dict,
        size: int = 80
    ) -> Optional[str]:
        """Generate base64 thumbnail for a face."""
        try:
            path = Path(image_path)
            if not path.exists():
                return None

            img = Image.open(path)
            img = ImageOps.exif_transpose(img)

            if bbox:
                w, h = img.size
                x = int(bbox.get('x', 0) * w)
                y = int(bbox.get('y', 0) * h)
                bw = int(bbox.get('w', 0.1) * w)
                bh = int(bbox.get('h', 0.1) * h)

                # Add padding
                padding = int(bw * 0.2)
                x = max(0, x - padding)
                y = max(0, y - padding)
                bw = min(w - x, bw + 2 * padding)
                bh = min(h - y, bh + 2 * padding)

                img = img.crop((x, y, x + bw, y + bh))

            # Resize to square
            img.thumbnail((size, size))

            # Convert to base64
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=80)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except Exception as e:
            self._logger.warning(f"Failed to generate thumbnail for {image_path}: {e}")
            return None

    def _apply_single_change(
        self,
        album_id: str,
        run_id: str,
        change: FaceAction
    ) -> None:
        """Apply a single change (internal)."""
        face_key = change.face_key
        action = change.action

        # Parse face_key
        parts = face_key.rsplit(':face_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid face_key format: {face_key}")

        image_path = parts[0]
        face_index = int(parts[1])

        if action == "assign":
            # Get or create target person
            target_person_id = change.target_person_id
            if change.new_person_name:
                person = self.create_person(album_id, run_id, change.new_person_name, [face_key])
                target_person_id = person.id
            elif not target_person_id:
                raise ValueError("assign requires target_person_id or new_person_name")

            # Remove from current person if any
            self._remove_face_from_persons(run_id, face_key)

            # Add to target person
            target = self._session.query(Person).filter(Person.id == target_person_id).first()
            if target:
                face_instances = list(target.face_instances or [])
                face_instances.append({
                    'image_path': image_path,
                    'face_index': face_index,
                    'bbox': None,
                    'assignment_method': 'user',
                    'assignment_confidence': 1.0,
                })
                target.face_instances = face_instances
                target.face_count = len(face_instances)

            # Create override
            self._create_override(album_id, run_id, face_key, "assigned", target_person_id)

        elif action == "unassign":
            self._remove_face_from_persons(run_id, face_key)
            self._delete_override(album_id, face_key)

        elif action == "untag":
            self._remove_face_from_persons(run_id, face_key)
            self._create_override(album_id, run_id, face_key, "untagged", None)

        elif action == "not_a_face":
            self._remove_face_from_persons(run_id, face_key)
            embedding = self._get_face_embedding(face_key, run_id)
            self._create_override(
                album_id, run_id, face_key, "not_a_face", None,
                embedding=embedding.tolist() if embedding is not None else None
            )

        # Record event for undo
        self._record_event(album_id, run_id, f"face.{action}", {
            "face_key": face_key,
            "target_person_id": change.target_person_id,
            "new_person_name": change.new_person_name,
        })

    def _remove_face_from_persons(self, run_id: str, face_key: str) -> None:
        """Remove a face from all persons."""
        parts = face_key.rsplit(':face_', 1)
        if len(parts) != 2:
            return

        image_path = parts[0]
        face_index = int(parts[1])

        people = self._session.query(Person).filter(Person.run_id == run_id).all()
        for person in people:
            updated = [
                fi for fi in (person.face_instances or [])
                if not (
                    fi.get('image_path', '').replace('\\', '/') == image_path.replace('\\', '/')
                    and fi.get('face_index') == face_index
                )
            ]
            if len(updated) != len(person.face_instances or []):
                person.face_instances = updated
                person.face_count = len(updated)

    def _create_override(
        self,
        album_id: str,
        run_id: str,
        face_key: str,
        status: str,
        person_id: Optional[str],
        embedding: Optional[list] = None
    ) -> FaceOverride:
        """Create or update a face override."""
        existing = (
            self._session.query(FaceOverride)
            .filter(FaceOverride.album_id == album_id)
            .filter(FaceOverride.face_key == face_key)
            .first()
        )

        if existing:
            existing.status = status
            existing.person_id = person_id
            existing.run_id = run_id
            existing.embedding = embedding
            return existing

        override = FaceOverride(
            id=str(uuid.uuid4()),
            album_id=album_id,
            run_id=run_id,
            face_key=face_key,
            status=status,
            person_id=person_id,
            embedding=embedding,
        )
        self._session.add(override)
        return override

    def _delete_override(self, album_id: str, face_key: str) -> None:
        """Delete a face override."""
        self._session.query(FaceOverride).filter(
            FaceOverride.album_id == album_id,
            FaceOverride.face_key == face_key
        ).delete()

    def _record_event(
        self,
        album_id: str,
        run_id: str,
        event_type: str,
        event_data: dict
    ) -> None:
        """Record a user event for undo capability."""
        event = UserEvent(
            id=str(uuid.uuid4()),
            album_id=album_id,
            run_id=run_id,
            event_type=event_type,
            event_data=event_data,
        )
        self._session.add(event)

    def _trigger_recluster(
        self,
        album_id: str,
        run_id: str
    ) -> dict:
        """
        Re-run identity refinement with updated overrides.

        Returns dict with auto_assigned and still_unassigned counts.
        """
        # TODO: Implement full reclustering
        # For now, just return zeros
        self._logger.info(f"Reclustering triggered for run {run_id}")
        return {
            'auto_assigned': 0,
            'still_unassigned': 0,
        }
