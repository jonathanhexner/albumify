"""BaseRepository — shared infrastructure for every entity Repository.

Subclasses take a SQLAlchemy Session in their constructor and translate
SQLAlchemy exceptions into the Repository error hierarchy. Concrete
query/mutation methods live on the subclass; this base only hosts the
patterns that every Repository needs.

Future Repositories (ClusterAnalysisRepository, …) inherit from this.
"""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Session

from face_cluster.repositories._errors import NotFoundError, ValidationError


class BaseRepository:
    """Hold the Session and translate ORM exceptions."""

    def __init__(self, session: Session):
        self._session = session

    @property
    def session(self) -> Session:
        return self._session

    @staticmethod
    def _translate_not_found(exc: NoResultFound, *, entity: str, key: object) -> NotFoundError:
        return NotFoundError(
            f"{entity} {key!r} not found",
            user_message=f"{entity} {key} doesn't exist.",
        )

    @staticmethod
    def _translate_integrity(exc: IntegrityError, *, entity: str) -> ValidationError:
        return ValidationError(
            f"{entity} integrity violation: {exc.orig}",
            user_message=f"Could not save {entity} — invalid input.",
        )


__all__ = ["BaseRepository"]
