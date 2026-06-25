"""People service - business logic for managing detected people."""

import logging
import uuid
from collections import defaultdict
from typing import Optional

from sqlalchemy.orm import Session

from sim_bench.api.database.models import Person, Album, PipelineRun


def _bbox_to_xywh(bbox) -> Optional[list]:
    """Reshape a face bbox to ``[x, y, w, h]`` across face/bbox representations.

    UNIT-PRESERVING: pixels in -> pixels out. This only reshapes
    ``(x1, y1, x2, y2)`` -> ``(x, y, w, h)``; it does NOT normalize to ``[0, 1]``.
    Normalization is the caller's job -- see :func:`_normalized_bbox`.

    spec-079 / SIGHTING-100: the unified chain produces ``FaceRecord`` whose
    ``bbox`` is a tuple ``(x1, y1, x2, y2)``; legacy faces used a dict or a
    ``BoundingBox`` object with ``.x/.y/.w/.h``.
    """
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return [bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', 0), bbox.get('h', 0)]
    if isinstance(bbox, (tuple, list)):
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    return [bbox.x, bbox.y, bbox.w, bbox.h]


def _normalized_bbox(face) -> Optional[list]:
    """Return a face bbox as ``[x, y, w, h]`` in normalized ``[0, 1]`` coords.

    SIGHTING-104: the Streamlit consumers (``people_browser._crop_face`` and
    ``draw_face_bboxes``) assume ``thumbnail_bbox`` is normalized, but the
    spec-079 ``FaceRecord.bbox`` is in PIXELS, so they silently rendered the
    gray placeholder instead of the cropped face.

    Resolution order (most authoritative first):
      1. spec-040 canonical ratio fields (``bbox_*_ratio``, already in [0, 1]);
      2. normalize the pixel bbox via the raw image dims (``image_*_px``);
      3. fall back to the raw reshape (the consumer's guard handles pixel-scale).
    """
    if face is None:
        return None

    # 1. spec-040 canonical normalized geometry -- the intended source of truth.
    rx = getattr(face, 'bbox_x_ratio', None)
    ry = getattr(face, 'bbox_y_ratio', None)
    rw = getattr(face, 'bbox_w_ratio', None)
    rh = getattr(face, 'bbox_h_ratio', None)
    if None not in (rx, ry, rw, rh):
        return [rx, ry, rw, rh]

    xywh = _bbox_to_xywh(getattr(face, 'bbox', None))
    if xywh is None:
        return None

    # 2. Normalize pixel coords using the raw image dimensions when available.
    iw = getattr(face, 'image_width_px', None)
    ih = getattr(face, 'image_height_px', None)
    if iw and ih and max(xywh) > 1.5:  # >1.5 => pixel-scale, not already [0, 1]
        x, y, w, h = xywh
        return [x / iw, y / ih, w / iw, h / ih]

    # 3. Last resort: return as-is (already normalized, or no dims to normalize).
    return xywh


class PeopleService:
    """Service for managing detected people (face clusters)."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    def list_people(self, album_id: str, run_id: Optional[str] = None) -> list[Person]:
        """List all people in an album.

        Args:
            album_id: Album ID
            run_id: Optional specific run ID (defaults to latest run)
        """
        query = self._session.query(Person).filter(Person.album_id == album_id)

        if run_id:
            query = query.filter(Person.run_id == run_id)
        else:
            # Get latest run for this album
            latest_run = (
                self._session.query(PipelineRun)
                .filter(PipelineRun.album_id == album_id)
                .filter(PipelineRun.status == "completed")
                .order_by(PipelineRun.completed_at.desc())
                .first()
            )
            if latest_run:
                query = query.filter(Person.run_id == latest_run.id)
            else:
                return []

        return query.order_by(Person.face_count.desc()).all()

    def get_person(self, album_id: str, person_id: str) -> Optional[Person]:
        """Get a specific person by ID."""
        return (
            self._session.query(Person)
            .filter(Person.album_id == album_id)
            .filter(Person.id == person_id)
            .first()
        )

    def get_person_images(self, album_id: str, person_id: str) -> list[dict]:
        """Get all images containing a person.

        Returns list of dicts with image_path and face info.
        """
        person = self.get_person(album_id, person_id)
        if not person:
            return []

        # Debug: log what we have
        self._logger.debug(
            f"Person {person_id}: face_instances={len(person.face_instances or [])}, "
            f"thumbnail_image_path={person.thumbnail_image_path}"
        )

        # Group face instances by image
        images_map = defaultdict(list)
        for face in person.face_instances or []:
            img_path = face.get('image_path', '')
            if img_path:
                images_map[img_path].append(face)
            else:
                self._logger.warning(f"Face instance missing image_path: {face}")

        # If no valid images found but we have thumbnail, use it as fallback
        if not images_map and person.thumbnail_image_path:
            self._logger.warning(
                f"No face_instances with valid image_path for person {person_id}, "
                f"using thumbnail as fallback"
            )
            images_map[person.thumbnail_image_path] = [{
                'image_path': person.thumbnail_image_path,
                'face_index': person.thumbnail_face_index or 0,
                'bbox': person.thumbnail_bbox,
            }]

        result = []
        for image_path, faces in images_map.items():
            result.append({
                'image_path': image_path,
                'face_count': len(faces),
                'faces': faces
            })

        return result

    def rename_person(self, album_id: str, person_id: str, name: str) -> Optional[Person]:
        """Rename a person."""
        person = self.get_person(album_id, person_id)
        if not person:
            return None

        old_name = person.name
        person.name = name
        self._session.commit()
        self._session.refresh(person)

        self._logger.info(
            f"Renamed person {person_id} from '{old_name}' to '{name}'"
        )
        return person

    def merge_people(
        self,
        album_id: str,
        person_ids: list[str]
    ) -> Optional[Person]:
        """Merge multiple people into one.

        The first person in the list becomes the merged person.
        """
        if len(person_ids) < 2:
            self._logger.warning("Merge requires at least 2 people")
            return None

        people = [self.get_person(album_id, pid) for pid in person_ids]
        people = [p for p in people if p is not None]

        if len(people) < 2:
            self._logger.warning("Not enough valid people to merge")
            return None

        # First person becomes the merged person
        primary = people[0]

        # Collect all face instances
        all_faces = list(primary.face_instances or [])
        all_images = set(f['image_path'] for f in all_faces)

        for other in people[1:]:
            for face in (other.face_instances or []):
                all_faces.append(face)
                all_images.add(face['image_path'])

            # Delete the merged person
            self._session.delete(other)

        # Update primary person
        primary.face_instances = all_faces
        primary.face_count = len(all_faces)
        primary.image_count = len(all_images)

        self._session.commit()
        self._session.refresh(primary)

        self._logger.info(
            f"Merged {len(people)} people into {primary.id} "
            f"(now {primary.face_count} faces)"
        )
        return primary

    def split_person(
        self,
        album_id: str,
        person_id: str,
        face_indices: list[int]
    ) -> list[Person]:
        """Split faces out of a person into a new person.

        Args:
            album_id: Album ID
            person_id: Person to split from
            face_indices: Indices of faces (in face_instances) to split out

        Returns:
            List of [original_person, new_person]
        """
        person = self.get_person(album_id, person_id)
        if not person:
            return []

        faces = list(person.face_instances or [])
        if not faces:
            return [person]

        # Validate indices
        valid_indices = set(i for i in face_indices if 0 <= i < len(faces))
        if not valid_indices:
            self._logger.warning("No valid face indices to split")
            return [person]

        # Split faces
        remaining_faces = []
        split_faces = []

        for i, face in enumerate(faces):
            if i in valid_indices:
                split_faces.append(face)
            else:
                remaining_faces.append(face)

        if not split_faces or not remaining_faces:
            self._logger.warning("Split would leave one person empty")
            return [person]

        # Update original person
        person.face_instances = remaining_faces
        person.face_count = len(remaining_faces)
        person.image_count = len(set(f['image_path'] for f in remaining_faces))

        # Create new person
        new_person = Person(
            id=str(uuid.uuid4()),
            album_id=album_id,
            run_id=person.run_id,
            person_index=self._get_next_person_index(album_id, person.run_id),
            name=None,
            face_instances=split_faces,
            face_count=len(split_faces),
            image_count=len(set(f['image_path'] for f in split_faces))
        )

        # Set thumbnail for new person (first face)
        if split_faces:
            new_person.thumbnail_image_path = split_faces[0]['image_path']
            new_person.thumbnail_face_index = split_faces[0]['face_index']
            new_person.thumbnail_bbox = split_faces[0].get('bbox')

        self._session.add(new_person)
        self._session.commit()
        self._session.refresh(person)
        self._session.refresh(new_person)

        self._logger.info(
            f"Split {len(split_faces)} faces from person {person_id} "
            f"into new person {new_person.id}"
        )
        return [person, new_person]

    def _get_next_person_index(self, album_id: str, run_id: str) -> int:
        """Get the next available person index for a run."""
        max_index = (
            self._session.query(Person.person_index)
            .filter(Person.album_id == album_id)
            .filter(Person.run_id == run_id)
            .order_by(Person.person_index.desc())
            .first()
        )
        return (max_index[0] + 1) if max_index else 0

    def _get_thumbnail_info(self, face) -> tuple:
        """Extract thumbnail path and bbox from a face object."""
        if not face:
            return None, None
        # Prefer pre-cropped image if available
        crop_path = getattr(face, 'crop_path', None)
        if crop_path:
            return str(crop_path), None
        # Fall back to original image + bbox (normalized to [0,1] -- SIGHTING-104).
        bbox = _normalized_bbox(face)
        # spec-079 / SIGHTING-100: FaceRecord uses image_path; legacy used original_path.
        return str(getattr(face, 'original_path', None) or getattr(face, 'image_path', '')), bbox

    def create_from_clusters(
        self,
        album_id: str,
        run_id: str,
        people_clusters: dict,
        people_thumbnails: dict = None,
        attachment_decisions: dict = None
    ) -> list[Person]:
        """Create Person records from pipeline clustering results.

        Args:
            album_id: Album ID
            run_id: Pipeline run ID
            people_clusters: Dict mapping cluster_id to list of Face objects
            people_thumbnails: Optional dict mapping cluster_id to best face
            attachment_decisions: Optional dict mapping face_key to attachment info

        Returns:
            List of created Person records
        """
        created = []
        attachment_decisions = attachment_decisions or {}

        for cluster_id, faces in people_clusters.items():
            if not faces:
                continue

            # Build face instances
            face_instances = []
            images = set()

            for face in faces:
                # Get and validate image path. spec-079 / SIGHTING-100: the
                # unified chain produces FaceRecord (image_path); legacy faces
                # used original_path. Support both.
                raw_path = getattr(face, 'original_path', None) or getattr(face, 'image_path', None)
                img_path = str(raw_path) if raw_path else None
                if not img_path or img_path in ('', '.', 'None'):
                    self._logger.warning(
                        f"Skipping face with invalid path: {raw_path} "
                        f"(cluster {cluster_id}, face_index {face.face_index})"
                    )
                    continue

                bbox = _normalized_bbox(face)  # [0,1] -- SIGHTING-104

                # Look up assignment method from attachment_decisions
                face_key = f"{img_path.replace(chr(92), '/')}:face_{face.face_index}"
                decision = attachment_decisions.get(face_key, {})
                assignment_method = decision.get('method', 'core')
                assignment_confidence = decision.get('confidence', 1.0)

                instance = {
                    'image_path': img_path,
                    'face_index': face.face_index,
                    'bbox': bbox,
                    'score': getattr(face, 'quality', None) and face.quality.overall if hasattr(face, 'quality') else None,
                    'assignment_method': assignment_method,
                    'assignment_confidence': assignment_confidence
                }
                face_instances.append(instance)
                images.add(img_path)

            # Determine thumbnail - prefer saved crop_path over original image
            thumbnail_face = (people_thumbnails or {}).get(cluster_id) or (faces[0] if faces else None)
            thumbnail_path, thumbnail_bbox = self._get_thumbnail_info(thumbnail_face)

            person = Person(
                id=str(uuid.uuid4()),
                album_id=album_id,
                run_id=run_id,
                person_index=cluster_id,
                name=None,
                face_instances=face_instances,
                face_count=len(face_instances),
                image_count=len(images),
                thumbnail_image_path=thumbnail_path,
                thumbnail_face_index=thumbnail_face.face_index if thumbnail_face else None,
                thumbnail_bbox=thumbnail_bbox
            )

            self._session.add(person)
            created.append(person)

        self._session.commit()

        # Refresh all created records
        for person in created:
            self._session.refresh(person)

        self._logger.info(
            f"Created {len(created)} people records for album {album_id}, run {run_id}"
        )
        return created

    def delete_for_run(self, run_id: str) -> int:
        """Delete all people records for a pipeline run."""
        count = (
            self._session.query(Person)
            .filter(Person.run_id == run_id)
            .delete()
        )
        self._session.commit()
        self._logger.info(f"Deleted {count} people records for run {run_id}")
        return count
