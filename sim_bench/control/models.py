"""Control-plane DB models (spec-112 A2): jobs + the rows behind each album.

Portable SQLAlchemy types only -- the same models run on a local SQLite file (Phase A) and on
Neon Postgres (Phase B). Every product row is keyed by ``job_id``. This metadata is separate from
``sim_bench.db`` (api/database/models.py, Alembic) and from the per-run ``run_db`` schema; the
three must not be cross-imported.

Rows are written only by the control plane (JobService / ResultsLoader / the API), never by the
worker -- the worker writes files (album.json + results bundle) and holds no DB credentials.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["ControlBase", "Job", "JobEvent", "Photo", "Face", "Person", "Scene", "UserEdit"]


class ControlBase(DeclarativeBase):
    pass


class Job(ControlBase):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    runner: Mapped[str] = mapped_column(String(16), nullable=False)
    pipeline: Mapped[str] = mapped_column(String(32), nullable=False, default="default")
    source: Mapped[Optional[str]] = mapped_column(Text)          # where the inputs came from
    n_inputs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text)
    failed_step: Mapped[Optional[str]] = mapped_column(String(64))
    theme: Mapped[Optional[dict]] = mapped_column(JSON)          # album_theme (spec-108)
    tag: Mapped[Optional[str]] = mapped_column(String(64))       # e.g. regression:slovakia (spec-114)
    worker: Mapped[Optional[str]] = mapped_column(String(128))   # host:pid:start of the process running it
    album_created_at: Mapped[Optional[str]] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class JobEvent(ControlBase):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)    # status | info | error
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")

    __table_args__ = (Index("ix_job_events_job", "job_id", "id"),)


class Photo(ControlBase):
    """One staged input photo. ``in_album`` False = removed before the album saw it (SIGHTING-128);
    its ``decisions`` still say why."""

    __tablename__ = "photos"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    photo_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    input_key: Mapped[Optional[str]] = mapped_column(String(512))     # e.g. input/a.jpg
    display_key: Mapped[Optional[str]] = mapped_column(String(512))   # e.g. display/a.jpg (picks)
    taken_at: Mapped[Optional[str]] = mapped_column(String(40))       # ISO 8601, EXIF local time
    lat: Mapped[Optional[float]] = mapped_column(Float)
    lon: Mapped[Optional[float]] = mapped_column(Float)
    in_album: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[Optional[str]] = mapped_column(String(16))         # pick | near_dup | ... | None
    reason_detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    beaten_by: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    beat: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    promotable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    pick_order: Mapped[Optional[int]] = mapped_column(Integer)
    scene_id: Mapped[Optional[int]] = mapped_column(Integer)          # album (canonical) scene id
    scene_label: Mapped[Optional[int]] = mapped_column(Integer)       # raw clusterer label
    caption: Mapped[str] = mapped_column(Text, nullable=False, default="")
    composite: Mapped[Optional[float]] = mapped_column(Float)
    iqa: Mapped[Optional[float]] = mapped_column(Float)
    ava: Mapped[Optional[float]] = mapped_column(Float)
    sharpness: Mapped[Optional[float]] = mapped_column(Float)
    occlusion: Mapped[Optional[float]] = mapped_column(Float)
    tilt: Mapped[Optional[float]] = mapped_column(Float)
    tilt_confidence: Mapped[Optional[float]] = mapped_column(Float)
    decisions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    __table_args__ = (Index("ix_photos_job_status", "job_id", "status"),)


class Face(ControlBase):
    __tablename__ = "faces"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    face_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    photo_id: Mapped[str] = mapped_column(String(255), nullable=False)
    person_id: Mapped[Optional[int]] = mapped_column(Integer)
    bbox: Mapped[list] = mapped_column(JSON, nullable=False)          # [x1, y1, x2, y2] px
    det_score: Mapped[Optional[float]] = mapped_column(Float)
    blur_score: Mapped[Optional[float]] = mapped_column(Float)
    pose: Mapped[Optional[list]] = mapped_column(JSON)                # [yaw, pitch, roll]
    is_core: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rejection_reason: Mapped[Optional[str]] = mapped_column(String(64))
    crop_key: Mapped[Optional[str]] = mapped_column(String(512))      # e.g. faces/12.jpg

    __table_args__ = (Index("ix_faces_job_photo", "job_id", "photo_id"),
                      Index("ix_faces_job_person", "job_id", "person_id"))


class Person(ControlBase):
    __tablename__ = "people"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    person_id: Mapped[str] = mapped_column(String(64), primary_key=True)   # album id, person_<n>
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer)
    face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sample_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    label: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # album order


class Scene(ControlBase):
    __tablename__ = "scenes"

    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    scene_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    n_photos: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_picks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_at: Mapped[Optional[str]] = mapped_column(String(40))
    end_at: Mapped[Optional[str]] = mapped_column(String(40))


class UserEdit(ControlBase):
    """An owner's change on top of the curation. Applied in order when the album is built."""

    __tablename__ = "user_edits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)    # promote | demote | rename
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (Index("ix_user_edits_job", "job_id", "id"),)
