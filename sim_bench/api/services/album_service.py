"""Album service - business logic for album management."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from sim_bench.api.database.models import Album


class AlbumService:
    """Service for managing albums."""

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None
    ):
        self._session = session
        self._logger = logger or logging.getLogger(__name__)

    def create(self, name: str, source_path: str) -> Album:
        """Create a new album."""
        self._logger.info(f"Creating album '{name}' from {source_path}")

        path = Path(source_path)
        if not path.exists():
            self._logger.error(f"Source path does not exist: {source_path}")
            raise ValueError(f"Source path does not exist: {source_path}")
        if not path.is_dir():
            self._logger.error(f"Source path is not a directory: {source_path}")
            raise ValueError(f"Source path is not a directory: {source_path}")

        image_extensions = {".jpg", ".jpeg", ".png", ".heic", ".raw"}
        image_count = sum(
            1 for f in path.rglob("*")
            if f.suffix.lower() in image_extensions
        )

        album = Album(
            id=str(uuid.uuid4()),
            name=name,
            source_path=str(path.resolve()),
            image_count=image_count
        )

        self._session.add(album)
        self._session.commit()
        self._session.refresh(album)

        self._logger.info(f"Created album {album.id} with {image_count} images")

        return album

    def get(self, album_id: str) -> Optional[Album]:
        """Get an album by ID."""
        return self._session.query(Album).filter(Album.id == album_id).first()

    def list_all(self) -> list[Album]:
        """List all albums."""
        return self._session.query(Album).order_by(Album.created_at.desc()).all()

    def delete(self, album_id: str) -> bool:
        """Delete an album by ID."""
        album = self.get(album_id)
        if album is None:
            self._logger.warning(f"Attempted to delete non-existent album: {album_id}")
            return False

        self._session.delete(album)
        self._session.commit()
        self._logger.info(f"Deleted album {album_id}")
        return True
