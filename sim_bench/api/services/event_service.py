"""Event service - business logic for managing user events."""

import logging
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import Session

from sim_bench.api.database.models import UserEvent, Person


class EventService:
    """Service for recording and managing user events."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    def record(
        self,
        event_type: str,
        event_data: dict,
        album_id: Optional[str] = None,
        run_id: Optional[str] = None,
        source: str = "user",
        status: str = "completed"
    ) -> UserEvent:
        """Record a new event.

        Args:
            event_type: Type of event (e.g., "face.assign", "feedback.rating")
            event_data: Event payload
            album_id: Optional album ID
            run_id: Optional pipeline run ID
            source: Event source ("user", "ai", "system")
            status: Event status ("pending", "in_progress", "completed", "failed")

        Returns:
            Created UserEvent
        """
        event = UserEvent(
            id=str(uuid.uuid4()),
            album_id=album_id,
            run_id=run_id,
            event_type=event_type,
            event_data=event_data,
            source=source,
            status=status,
            created_at=datetime.utcnow()
        )

        self._session.add(event)
        self._session.commit()
        self._session.refresh(event)

        self._logger.info(f"Recorded event: {event_type} (id={event.id})")
        return event

    def get_event(self, event_id: str) -> Optional[UserEvent]:
        """Get an event by ID."""
        return self._session.query(UserEvent).filter(UserEvent.id == event_id).first()

    def list_events(
        self,
        album_id: Optional[str] = None,
        run_id: Optional[str] = None,
        event_type: Optional[str] = None,
        include_undone: bool = False,
        limit: int = 100
    ) -> List[UserEvent]:
        """List events with optional filtering.

        Args:
            album_id: Filter by album
            run_id: Filter by pipeline run
            event_type: Filter by event type (supports prefix matching with "face.*")
            include_undone: Include undone events
            limit: Maximum number of events to return

        Returns:
            List of UserEvent objects
        """
        query = self._session.query(UserEvent)

        if album_id:
            query = query.filter(UserEvent.album_id == album_id)

        if run_id:
            query = query.filter(UserEvent.run_id == run_id)

        if event_type:
            if event_type.endswith("*"):
                prefix = event_type[:-1]
                query = query.filter(UserEvent.event_type.startswith(prefix))
            else:
                query = query.filter(UserEvent.event_type == event_type)

        if not include_undone:
            query = query.filter(UserEvent.is_undone == False)

        return query.order_by(UserEvent.created_at.desc()).limit(limit).all()

    def undo(self, event_id: str) -> Optional[UserEvent]:
        """Undo an event by reversing its effect.

        Args:
            event_id: ID of event to undo

        Returns:
            The undo event, or None if event not found
        """
        event = self.get_event(event_id)
        if not event:
            self._logger.warning(f"Event not found: {event_id}")
            return None

        if event.is_undone:
            self._logger.warning(f"Event already undone: {event_id}")
            return None

        # Reverse the effect based on event type
        reversed_data = self._reverse_event(event)

        if reversed_data is None:
            self._logger.warning(f"Cannot reverse event type: {event.event_type}")
            return None

        # Mark original event as undone
        event.is_undone = True

        # Create undo event
        undo_event = UserEvent(
            id=str(uuid.uuid4()),
            album_id=event.album_id,
            run_id=event.run_id,
            event_type=f"{event.event_type}.undo",
            event_data=reversed_data,
            source="user",
            status="completed",
            created_at=datetime.utcnow()
        )

        # Link undo event to original
        event.undone_by_id = undo_event.id

        self._session.add(undo_event)
        self._session.commit()

        self._logger.info(f"Undone event: {event_id} with undo event {undo_event.id}")
        return undo_event

    def _reverse_event(self, event: UserEvent) -> Optional[dict]:
        """Compute the reverse of an event.

        Returns:
            Reversed event data, or None if not reversible
        """
        event_data = event.event_data or {}

        if event.event_type == "face.assign":
            # Reverse: split the face back out
            return {
                "face_key": event_data.get("face_key"),
                "from_person_id": event_data.get("to_person_id"),
                "to_person_id": event_data.get("from_person_id"),
                "original_event_id": event.id
            }

        elif event.event_type == "face.split":
            # Reverse: assign the face back
            return {
                "face_key": event_data.get("face_key"),
                "from_person_id": None,
                "to_person_id": event_data.get("from_person_id"),
                "original_event_id": event.id
            }

        elif event.event_type == "face.reassign":
            # Reverse: move back to original cluster
            return {
                "face_key": event_data.get("face_key"),
                "from_person_id": event_data.get("to_person_id"),
                "to_person_id": event_data.get("from_person_id"),
                "original_event_id": event.id
            }

        elif event.event_type == "selection.add":
            # Reverse: remove from selection
            return {
                "image_path": event_data.get("image_path"),
                "cluster_id": event_data.get("cluster_id"),
                "original_event_id": event.id
            }

        elif event.event_type == "selection.remove":
            # Reverse: add back to selection
            return {
                "image_path": event_data.get("image_path"),
                "cluster_id": event_data.get("cluster_id"),
                "original_event_id": event.id
            }

        # Feedback events don't need reversal - just mark as undone
        elif event.event_type.startswith("feedback."):
            return {"original_event_id": event.id}

        return None

    def apply_face_override(
        self,
        album_id: str,
        run_id: str,
        face_key: str,
        override_type: str,
        from_person_id: Optional[str] = None,
        to_person_id: Optional[str] = None
    ) -> Optional[UserEvent]:
        """Apply a face override and update Person records.

        Args:
            album_id: Album ID
            run_id: Pipeline run ID
            face_key: Face identifier (image_path:face_N)
            override_type: Type of override ("assign", "split", "reassign")
            from_person_id: Source person ID (for split/reassign)
            to_person_id: Target person ID (for assign/reassign)

        Returns:
            Created UserEvent, or None if operation failed
        """
        # Validate inputs
        if override_type == "assign" and not to_person_id:
            self._logger.error("assign requires to_person_id")
            return None

        if override_type == "split" and not from_person_id:
            self._logger.error("split requires from_person_id")
            return None

        if override_type == "reassign" and (not from_person_id or not to_person_id):
            self._logger.error("reassign requires both from_person_id and to_person_id")
            return None

        # Parse face_key
        parts = face_key.rsplit(":face_", 1)
        if len(parts) != 2:
            self._logger.error(f"Invalid face_key format: {face_key}")
            return None

        image_path = parts[0]
        face_index = int(parts[1])

        # Update Person records
        if override_type == "split":
            from_person = self._session.query(Person).filter(Person.id == from_person_id).first()
            if from_person:
                # Remove face from person's face_instances
                face_instances = from_person.face_instances or []
                updated_instances = [
                    fi for fi in face_instances
                    if not (fi.get('image_path', '').replace('\\', '/') == image_path.replace('\\', '/')
                            and fi.get('face_index') == face_index)
                ]
                from_person.face_instances = updated_instances
                from_person.face_count = len(updated_instances)
                from_person.image_count = len(set(fi.get('image_path') for fi in updated_instances))

        elif override_type in ("assign", "reassign"):
            if from_person_id:
                from_person = self._session.query(Person).filter(Person.id == from_person_id).first()
                if from_person:
                    face_instances = from_person.face_instances or []
                    updated_instances = [
                        fi for fi in face_instances
                        if not (fi.get('image_path', '').replace('\\', '/') == image_path.replace('\\', '/')
                                and fi.get('face_index') == face_index)
                    ]
                    from_person.face_instances = updated_instances
                    from_person.face_count = len(updated_instances)
                    from_person.image_count = len(set(fi.get('image_path') for fi in updated_instances))

            to_person = self._session.query(Person).filter(Person.id == to_person_id).first()
            if to_person:
                face_instances = to_person.face_instances or []
                face_instances.append({
                    'image_path': image_path,
                    'face_index': face_index,
                    'bbox': None,
                    'assignment_method': 'user_assigned',
                    'assignment_confidence': 1.0
                })
                to_person.face_instances = face_instances
                to_person.face_count = len(face_instances)
                to_person.image_count = len(set(fi.get('image_path') for fi in face_instances))

        # Record event
        event = self.record(
            event_type=f"face.{override_type}",
            event_data={
                "face_key": face_key,
                "from_person_id": from_person_id,
                "to_person_id": to_person_id
            },
            album_id=album_id,
            run_id=run_id
        )

        self._session.commit()
        return event

    def get_face_overrides_for_run(self, run_id: str) -> List[dict]:
        """Get all face overrides for a pipeline run in format for identity_refinement.

        Args:
            run_id: Pipeline run ID

        Returns:
            List of override dicts compatible with identity_refinement step
        """
        events = self.list_events(
            run_id=run_id,
            event_type="face.*",
            include_undone=False
        )

        overrides = []
        for event in events:
            if event.event_type.endswith(".undo"):
                continue

            override_type = event.event_type.replace("face.", "")
            if override_type not in ("assign", "split", "reassign", "create"):
                continue

            overrides.append({
                "override_type": override_type,
                "face_key": event.event_data.get("face_key"),
                "from_cluster": event.event_data.get("from_person_id"),
                "to_cluster": event.event_data.get("to_person_id"),
                "new_cluster_id": event.event_data.get("new_cluster_id"),
                "created_at": event.created_at.isoformat() if event.created_at else ""
            })

        return overrides
