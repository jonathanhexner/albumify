"""Events router - API endpoints for user events and actions."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from sim_bench.api.database.session import get_session
from sim_bench.api.services.event_service import EventService


router = APIRouter(prefix="/api/v1/events", tags=["events"])


class EventResponse(BaseModel):
    """Response model for a user event."""
    id: str
    album_id: Optional[str]
    run_id: Optional[str]
    event_type: str
    event_data: dict
    status: str
    result: Optional[dict]
    source: str
    created_at: datetime
    is_undone: bool

    class Config:
        from_attributes = True


class FaceAssignRequest(BaseModel):
    """Request to assign a face to a person."""
    to_person_id: str


class FaceSplitRequest(BaseModel):
    """Request to split a face from a person."""
    pass  # No additional data needed, face_key is in path


class FaceReassignRequest(BaseModel):
    """Request to reassign a face to a different person."""
    from_person_id: str
    to_person_id: str


@router.get("/{album_id}", response_model=List[EventResponse])
def list_events(
    album_id: str,
    run_id: Optional[str] = Query(None, description="Filter by pipeline run"),
    event_type: Optional[str] = Query(None, description="Filter by event type (supports wildcards like 'face.*')"),
    include_undone: bool = Query(False, description="Include undone events"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    session: Session = Depends(get_session)
):
    """List user events for an album."""
    service = EventService(session)
    return service.list_events(
        album_id=album_id,
        run_id=run_id,
        event_type=event_type,
        include_undone=include_undone,
        limit=limit
    )


@router.get("/{album_id}/{event_id}", response_model=EventResponse)
def get_event(
    album_id: str,
    event_id: str,
    session: Session = Depends(get_session)
):
    """Get a specific event."""
    service = EventService(session)
    event = service.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.album_id != album_id:
        raise HTTPException(status_code=404, detail="Event not found in this album")
    return event


@router.post("/{album_id}/{event_id}/undo", response_model=EventResponse)
def undo_event(
    album_id: str,
    event_id: str,
    session: Session = Depends(get_session)
):
    """Undo a user event."""
    service = EventService(session)

    # Verify event exists and belongs to album
    event = service.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.album_id != album_id:
        raise HTTPException(status_code=404, detail="Event not found in this album")

    undo_event = service.undo(event_id)
    if undo_event is None:
        raise HTTPException(
            status_code=400,
            detail="Could not undo event (already undone or not reversible)"
        )
    return undo_event


@router.post("/{album_id}/runs/{run_id}/faces/{face_key}/assign", response_model=EventResponse)
def assign_face(
    album_id: str,
    run_id: str,
    face_key: str,
    request: FaceAssignRequest,
    session: Session = Depends(get_session)
):
    """Assign an unassigned face to a person.

    The face_key format is: image_path:face_N (URL encoded)
    """
    service = EventService(session)
    event = service.apply_face_override(
        album_id=album_id,
        run_id=run_id,
        face_key=face_key,
        override_type="assign",
        to_person_id=request.to_person_id
    )
    if event is None:
        raise HTTPException(status_code=400, detail="Could not assign face")
    return event


@router.post("/{album_id}/runs/{run_id}/faces/{face_key}/split", response_model=EventResponse)
def split_face(
    album_id: str,
    run_id: str,
    face_key: str,
    from_person_id: str = Query(..., description="Person ID to split from"),
    session: Session = Depends(get_session)
):
    """Remove a face from a person (mark as unassigned)."""
    service = EventService(session)
    event = service.apply_face_override(
        album_id=album_id,
        run_id=run_id,
        face_key=face_key,
        override_type="split",
        from_person_id=from_person_id
    )
    if event is None:
        raise HTTPException(status_code=400, detail="Could not split face")
    return event


@router.post("/{album_id}/runs/{run_id}/faces/{face_key}/reassign", response_model=EventResponse)
def reassign_face(
    album_id: str,
    run_id: str,
    face_key: str,
    request: FaceReassignRequest,
    session: Session = Depends(get_session)
):
    """Move a face from one person to another."""
    service = EventService(session)
    event = service.apply_face_override(
        album_id=album_id,
        run_id=run_id,
        face_key=face_key,
        override_type="reassign",
        from_person_id=request.from_person_id,
        to_person_id=request.to_person_id
    )
    if event is None:
        raise HTTPException(status_code=400, detail="Could not reassign face")
    return event
