"""People service - business logic for managing detected people."""

import logging
import uuid
from collections import defaultdict
from typing import Optional

from sqlalchemy.orm import Session

from sim_bench.api.database.models import Person, Album, PipelineRun


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

        # Group face instances by image
        images_map = defaultdict(list)
        for face in person.face_instances or []:
            images_map[face['image_path']].append(face)

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

    def create_from_clusters(
        self,
        album_id: str,
        run_id: str,
        people_clusters: dict,
        people_thumbnails: dict = None
    ) -> list[Person]:
        """Create Person records from pipeline clustering results.

        Args:
            album_id: Album ID
            run_id: Pipeline run ID
            people_clusters: Dict mapping cluster_id to list of Face objects
            people_thumbnails: Optional dict mapping cluster_id to best face

        Returns:
            List of created Person records
        """
        created = []

        for cluster_id, faces in people_clusters.items():
            if not faces:
                continue

            # Build face instances
            face_instances = []
            images = set()

            for face in faces:
                instance = {
                    'image_path': str(face.original_path),
                    'face_index': face.face_index,
                    'bbox': list(face.bbox) if face.bbox is not None else None,
                    'score': getattr(face, 'quality', None) and face.quality.overall
                }
                face_instances.append(instance)
                images.add(str(face.original_path))

            # Determine thumbnail
            thumbnail_face = None
            if people_thumbnails and cluster_id in people_thumbnails:
                thumbnail_face = people_thumbnails[cluster_id]
            elif faces:
                # Default to first face
                thumbnail_face = faces[0]

            person = Person(
                id=str(uuid.uuid4()),
                album_id=album_id,
                run_id=run_id,
                person_index=cluster_id,
                name=None,
                face_instances=face_instances,
                face_count=len(face_instances),
                image_count=len(images),
                thumbnail_image_path=str(thumbnail_face.original_path) if thumbnail_face else None,
                thumbnail_face_index=thumbnail_face.face_index if thumbnail_face else None,
                thumbnail_bbox=list(thumbnail_face.bbox) if thumbnail_face and thumbnail_face.bbox is not None else None
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
