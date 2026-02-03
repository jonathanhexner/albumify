"""Albums router - API endpoints for album management."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from sim_bench.api.database.session import get_session
from sim_bench.api.schemas.album import AlbumCreate, AlbumResponse
from sim_bench.api.services.album_service import AlbumService

router = APIRouter(prefix="/api/v1/albums", tags=["albums"])


@router.post("/", response_model=AlbumResponse)
def create_album(
    request: AlbumCreate,
    session: Session = Depends(get_session)
):
    """Create a new album from a source directory."""
    service = AlbumService(session)
    album = service.create(request.name, request.source_path)
    return album


@router.get("/", response_model=list[AlbumResponse])
def list_albums(session: Session = Depends(get_session)):
    """List all albums."""
    service = AlbumService(session)
    return service.list_all()


@router.get("/{album_id}", response_model=AlbumResponse)
def get_album(album_id: str, session: Session = Depends(get_session)):
    """Get an album by ID."""
    service = AlbumService(session)
    album = service.get(album_id)
    if album is None:
        raise HTTPException(status_code=404, detail="Album not found")
    return album


@router.delete("/{album_id}")
def delete_album(album_id: str, session: Session = Depends(get_session)):
    """Delete an album by ID."""
    service = AlbumService(session)
    if not service.delete(album_id):
        raise HTTPException(status_code=404, detail="Album not found")
    return {"ok": True}
