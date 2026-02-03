"""People router - API endpoints for managing detected people."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from sim_bench.api.database.session import get_session
from sim_bench.api.schemas.people import (
    PersonResponse,
    PersonListResponse,
    PersonRenameRequest,
    PersonMergeRequest,
    PersonSplitRequest,
    PersonImageResponse,
)
from sim_bench.api.services.people_service import PeopleService

router = APIRouter(prefix="/api/v1/people", tags=["people"])


@router.get("/{album_id}", response_model=list[PersonListResponse])
def list_people(
    album_id: str,
    run_id: Optional[str] = Query(None, description="Specific run ID (defaults to latest)"),
    session: Session = Depends(get_session)
):
    """List all detected people in an album."""
    service = PeopleService(session)
    return service.list_people(album_id, run_id)


@router.get("/{album_id}/{person_id}", response_model=PersonResponse)
def get_person(
    album_id: str,
    person_id: str,
    session: Session = Depends(get_session)
):
    """Get details of a specific person."""
    service = PeopleService(session)
    person = service.get_person(album_id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.get("/{album_id}/{person_id}/images", response_model=list[PersonImageResponse])
def get_person_images(
    album_id: str,
    person_id: str,
    session: Session = Depends(get_session)
):
    """Get all images containing a specific person."""
    service = PeopleService(session)
    person = service.get_person(album_id, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return service.get_person_images(album_id, person_id)


@router.patch("/{album_id}/{person_id}", response_model=PersonResponse)
def rename_person(
    album_id: str,
    person_id: str,
    request: PersonRenameRequest,
    session: Session = Depends(get_session)
):
    """Rename a person."""
    service = PeopleService(session)
    person = service.rename_person(album_id, person_id, request.name)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.post("/{album_id}/merge", response_model=PersonResponse)
def merge_people(
    album_id: str,
    request: PersonMergeRequest,
    session: Session = Depends(get_session)
):
    """Merge multiple people into one.

    The first person in the list becomes the merged person.
    All other people are deleted.
    """
    if len(request.person_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Merge requires at least 2 people"
        )

    service = PeopleService(session)
    merged = service.merge_people(album_id, request.person_ids)
    if merged is None:
        raise HTTPException(
            status_code=400,
            detail="Could not merge people (invalid IDs or insufficient valid people)"
        )
    return merged


@router.post("/{album_id}/{person_id}/split", response_model=list[PersonResponse])
def split_person(
    album_id: str,
    person_id: str,
    request: PersonSplitRequest,
    session: Session = Depends(get_session)
):
    """Split faces from a person into a new person.

    Returns the updated original person and the new person.
    """
    if not request.face_indices:
        raise HTTPException(
            status_code=400,
            detail="No face indices provided"
        )

    service = PeopleService(session)
    result = service.split_person(album_id, person_id, request.face_indices)
    if not result:
        raise HTTPException(status_code=404, detail="Person not found")
    return result
