"""Faces router - API endpoints for face management operations."""

from typing import Optional, List
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from sim_bench.api.database.session import get_session
from sim_bench.api.services.face_service import FaceService
from sim_bench.api.schemas.face import (
    FaceInfo,
    PersonDistance,
    BorderlineFace,
    PersonSummary,
    FaceAction,
    BatchChangeRequest,
    BatchChangeResponse,
)


router = APIRouter(
    prefix="/api/v1/albums/{album_id}/runs/{run_id}/faces",
    tags=["faces"]
)


@router.get("", response_model=List[FaceInfo])
def list_faces(
    album_id: str,
    run_id: str,
    status: Optional[str] = Query(
        None,
        description="Filter by status (comma-separated). Options: assigned, unassigned, untagged, not_a_face"
    ),
    session: Session = Depends(get_session)
):
    """
    List all faces for a pipeline run.

    Returns faces with their classification status and assignment info.
    """
    service = FaceService(session)

    status_filter = None
    if status:
        status_filter = [s.strip() for s in status.split(",")]

    return service.get_all_faces(album_id, run_id, status_filter)


@router.get("/needs-help", response_model=List[BorderlineFace])
def get_needs_help(
    album_id: str,
    run_id: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum faces to return"),
    session: Session = Depends(get_session)
):
    """
    Get borderline faces needing user decision.

    Returns faces where distance is between attach and reject thresholds,
    sorted by uncertainty (most uncertain first).
    """
    service = FaceService(session)
    return service.get_borderline_faces(album_id, run_id, limit)


@router.get("/people", response_model=List[PersonSummary])
def list_people(
    album_id: str,
    run_id: str,
    session: Session = Depends(get_session)
):
    """
    Get summary of all identified people.

    Returns list of people with face counts and thumbnails,
    sorted by face count descending.
    """
    service = FaceService(session)
    return service.get_people_summary(album_id, run_id)


@router.get("/{face_key:path}", response_model=FaceInfo)
def get_face(
    album_id: str,
    run_id: str,
    face_key: str,
    session: Session = Depends(get_session)
):
    """
    Get details for a single face.

    The face_key format is: image_path:face_N
    Note: face_key should be URL-encoded if it contains special characters.
    """
    service = FaceService(session)

    # Decode face_key
    decoded_key = unquote(face_key)

    face = service.get_face(album_id, run_id, decoded_key)
    if face is None:
        raise HTTPException(status_code=404, detail=f"Face not found: {decoded_key}")

    return face


@router.get("/{face_key:path}/distances", response_model=List[PersonDistance])
def get_face_distances(
    album_id: str,
    run_id: str,
    face_key: str,
    session: Session = Depends(get_session)
):
    """
    Get distances from a face to all people.

    Useful for showing "Assign to" menu sorted by likelihood.
    Returns list sorted by centroid_distance ascending.
    """
    service = FaceService(session)

    # Decode face_key
    decoded_key = unquote(face_key)

    return service.get_face_distances(album_id, run_id, decoded_key)


@router.post("/{face_key:path}/action", response_model=FaceInfo)
def apply_face_action(
    album_id: str,
    run_id: str,
    face_key: str,
    action: FaceAction,
    session: Session = Depends(get_session)
):
    """
    Apply a single action to a face (live mode).

    Actions:
    - assign: Assign to existing person (requires target_person_id) or create new (requires new_person_name)
    - unassign: Remove from current person, move to unassigned
    - untag: Mark as "don't care"
    - not_a_face: Mark as false positive
    """
    service = FaceService(session)

    # Decode face_key
    decoded_key = unquote(face_key)

    # Ensure action face_key matches path
    action.face_key = decoded_key

    success = service.apply_single_change(album_id, run_id, action)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to apply action")

    # Return updated face
    face = service.get_face(album_id, run_id, decoded_key)
    if face is None:
        raise HTTPException(status_code=404, detail="Face not found after action")

    return face


@router.post("/batch", response_model=BatchChangeResponse)
def apply_batch_changes(
    album_id: str,
    run_id: str,
    request: BatchChangeRequest,
    session: Session = Depends(get_session)
):
    """
    Apply multiple face changes at once (batch mode).

    If recluster=true (default), re-runs identity refinement after applying changes.
    This may auto-assign additional faces based on new exemplars from user assignments.
    """
    service = FaceService(session)

    return service.apply_batch_changes(
        album_id,
        run_id,
        request.changes,
        recluster=request.recluster
    )


@router.post("/person", response_model=PersonSummary)
def create_person(
    album_id: str,
    run_id: str,
    name: str = Query(..., description="Name for new person"),
    face_keys: List[str] = Query(..., description="Face keys to assign to the new person"),
    session: Session = Depends(get_session)
):
    """
    Create a new person from selected faces.

    The faces will be removed from their current assignments (if any)
    and assigned to the new person.
    """
    service = FaceService(session)

    person = service.create_person(album_id, run_id, name, face_keys)
    if person is None:
        raise HTTPException(status_code=400, detail="Failed to create person")

    return PersonSummary(
        person_id=person.id,
        name=person.name,
        thumbnail_base64=None,
        face_count=person.face_count,
        exemplar_count=min(5, person.face_count),
        cluster_tightness=None,
    )
