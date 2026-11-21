"""Every read the public API makes, in one place (spec-120 R4, R11; the first slice of spec-119's repository layer).

Each method is filtered by the owner, so another user's album is simply not found (404 at the API), and takes at
most two queries: on Neon each round trip costs ~0.16 s. The job row (control tables) carries the owner; the run
rows (API tables) carry the results; both live in one database (spec-117 R2), so they are joined, not fetched
one by one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from sim_bench.api.database.models import Album, ImageMetricRow, Person, PipelineResult, PipelineRun
from sim_bench.control.models import Job

logger = logging.getLogger(__name__)

__all__ = ["AlbumView", "PersonView", "PhotoView", "PublicAlbumRepo", "BEST", "ALL", "MAX_NAME"]

BEST, ALL = "best", "all"
BEST_N = 30                 # the picks shown by default (productization plan: ship with the top 30)
MAX_NAME = 80


@dataclass(frozen=True, slots=True)
class AlbumView:
    job_id: str
    title: Optional[str]
    status: str                 # the job's status: queued, staging, running, ..., succeeded, failed
    created_at: datetime
    n_photos: int
    n_picked: Optional[int]
    error: Optional[str]


@dataclass(frozen=True, slots=True)
class PhotoView:
    image_path: str
    picked: bool
    score: Optional[float]
    reason: Optional[str]       # why it was dropped; None for a pick
    scene: Optional[int]


@dataclass(frozen=True, slots=True)
class PersonView:
    person_id: str
    name: Optional[str]
    face_count: int
    image_count: int
    thumbnail_image_path: Optional[str]
    thumbnail_bbox: Optional[list]


class PublicAlbumRepo:
    """``sessions`` opens a session that may write; ``read_sessions`` one for reads only (autocommit on Postgres:
    no BEGIN/COMMIT round trips). Both default to the same factory."""

    def __init__(self, sessions: Callable[[], Session], read_sessions: Optional[Callable[[], Session]] = None):
        self._sessions = sessions
        self._reads = read_sessions or sessions

    # --- albums -------------------------------------------------------------------------------------------------

    def _album_query(self, account_id: str):
        return (select(Job.id, Album.name, Job.status, Job.created_at, Job.n_inputs, PipelineResult.num_selected,
                       Job.error)
                .outerjoin(PipelineRun, PipelineRun.id == Job.id)
                .outerjoin(Album, Album.id == PipelineRun.album_id)
                .outerjoin(PipelineResult, PipelineResult.run_id == Job.id)
                .where(Job.owner_id == account_id))

    @staticmethod
    def _album(row) -> AlbumView:
        return AlbumView(job_id=row[0], title=row[1], status=row[2], created_at=row[3], n_photos=row[4] or 0,
                         n_picked=row[5], error=row[6])

    def list(self, account_id: str) -> List[AlbumView]:
        """The account's albums, newest first. One query."""
        with self._reads() as s:
            rows = s.execute(self._album_query(account_id).order_by(Job.created_at.desc())).all()
        return [self._album(r) for r in rows]

    def album(self, account_id: str, job_id: str) -> Optional[AlbumView]:
        """One album of the account, or None (also when it is someone else's). One query."""
        with self._reads() as s:
            row = s.execute(self._album_query(account_id).where(Job.id == job_id)).first()
        return self._album(row) if row else None

    def _owns(self, s: Session, account_id: str, job_id: str) -> bool:
        return s.scalar(select(Job.id).where(Job.id == job_id, Job.owner_id == account_id)) is not None

    # --- photos and people ---------------------------------------------------------------------------------------

    def photos(self, account_id: str, job_id: str, tier: str = BEST) -> Optional[List[PhotoView]]:
        """The best 30 picks (``tier="best"``) or every photo, picks first, best score first; None if the album is
        not the account's. Two queries."""
        if tier not in (BEST, ALL):
            raise ValueError(f"tier must be {BEST!r} or {ALL!r}")
        best_first = (ImageMetricRow.composite_score.is_(None), ImageMetricRow.composite_score.desc(),
                      ImageMetricRow.image_path)
        with self._reads() as s:
            if not self._owns(s, account_id, job_id):
                return None
            q = select(ImageMetricRow).where(ImageMetricRow.run_id == job_id)
            if tier == BEST:
                q = q.where(ImageMetricRow.is_selected.is_(True)).order_by(*best_first).limit(BEST_N)
            else:
                q = q.order_by(case((ImageMetricRow.is_selected.is_(True), 0), else_=1), *best_first)
            rows = s.scalars(q).all()
        return [PhotoView(image_path=r.image_path, picked=bool(r.is_selected), score=r.composite_score,
                          reason=None if r.is_selected else r.filter_reason, scene=r.cluster_id) for r in rows]

    def people(self, account_id: str, job_id: str) -> Optional[List[PersonView]]:
        """The album's people, most faces first; None if the album is not the account's. Two queries."""
        with self._reads() as s:
            if not self._owns(s, account_id, job_id):
                return None
            rows = s.execute(select(Person.id, Person.name, Person.face_count, Person.image_count,
                                    Person.thumbnail_image_path, Person.thumbnail_bbox)
                             .where(Person.run_id == job_id)
                             .order_by(Person.face_count.desc(), Person.person_index)).all()
        return [PersonView(person_id=r[0], name=r[1], face_count=r[2] or 0, image_count=r[3] or 0,
                           thumbnail_image_path=r[4], thumbnail_bbox=r[5]) for r in rows]

    def rename_person(self, account_id: str, job_id: str, person_id: str, name: str) -> Optional[PersonView]:
        """The only edit (R12). None if the album or person is not the account's. ``ValueError`` for an empty or
        over-long name. Two queries."""
        name = " ".join(name.split())
        if not name or len(name) > MAX_NAME:
            raise ValueError(f"a name has 1 to {MAX_NAME} characters")
        with self._sessions() as s:
            person = s.scalars(select(Person).join(Job, Job.id == Person.run_id)
                               .where(Person.id == person_id, Person.run_id == job_id,
                                      Job.owner_id == account_id)).first()
            if person is None:
                return None
            person.name = name
            s.commit()
            return PersonView(person_id=person.id, name=person.name, face_count=person.face_count or 0,
                              image_count=person.image_count or 0, thumbnail_image_path=person.thumbnail_image_path,
                              thumbnail_bbox=person.thumbnail_bbox)
