"""Build the Album contract from control-DB rows + the owner's edits (spec-112 TA3.2, AC3/AC5).

The DB rows are the source of truth; ``album.json`` is an export. With no edits the built album
equals the worker's ``album.json`` (same picks, order, reasons, scores) -- that equality is what
proves nothing was lost in loading. Edits are applied on top, in the order they were made:

    promote  {photo_id}            a dropped photo joins the picks
    demote   {photo_id}            a pick leaves ("removed by you"), still shown, re-promotable
    rename   {person_id, label}    name a person

Edits live in their own table, so re-running the curation (re-load) never loses them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List

from sqlalchemy import select

from sim_bench.album.normalizer import chrono_key
from sim_bench.album.schema import (
    NOT_SELECTED, PICK, SET_ASIDE, Album, Counts, Dropped, Person as AlbumPerson, Pick,
    ScoreBreakdown,
)
from sim_bench.control.db import ControlDB
from sim_bench.control.models import Job, Person, Photo, UserEdit

__all__ = ["AlbumBuilder", "BuildInputs", "EditError", "EDIT_KINDS", "record_edit",
           "DEMOTED_DETAIL"]

EDIT_KINDS = ("promote", "demote", "rename")
DEMOTED_DETAIL = "removed by you"


class EditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BuildInputs:
    job_id: str
    apply_edits: bool = True


@dataclass
class _P:
    """Mutable working copy of a photo row while edits are applied."""
    row: Photo
    status: str
    detail: str
    beaten_by: List[str]
    beat: List[str]


class AlbumBuilder:
    def __init__(self, db: ControlDB):
        self._db = db

    def calc(self, inputs: BuildInputs) -> Album:
        job_id = inputs.job_id
        with self._db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            photos = list(s.scalars(select(Photo).where(Photo.job_id == job_id,
                                                        Photo.in_album.is_(True))))
            people = list(s.scalars(select(Person).where(Person.job_id == job_id)
                                    .order_by(Person.position)))
            edits = list(s.scalars(select(UserEdit).where(UserEdit.job_id == job_id)
                                   .order_by(UserEdit.id))) if inputs.apply_edits else []

        work = {p.photo_id: _P(p, p.status, p.reason_detail, list(p.beaten_by), list(p.beat))
                for p in photos}
        labels = {p.person_id: p.label for p in people}
        for e in edits:
            _apply(e, work, labels)

        chrono = chrono_key({sid: w.row.taken_at for sid, w in work.items()})
        pick_ids = sorted((sid for sid, w in work.items() if w.status == PICK), key=chrono)
        picks = [Pick(
            id=sid, order=i, scene_cluster=_scene(work[sid].row),
            score=ScoreBreakdown(composite=r.composite, sharpness=r.iqa, aesthetic=r.ava,
                                 occlusion=r.occlusion, tilt=r.tilt),
            beat=work[sid].beat, caption=r.caption,
        ) for i, sid in enumerate(pick_ids) for r in (work[sid].row,)]

        dropped = [Dropped(id=sid, reason=w.status, reason_detail=w.detail,
                           beaten_by=w.beaten_by, promotable=w.row.promotable)
                   for sid, w in sorted(work.items()) if w.status != PICK]

        album_people = [AlbumPerson(id=p.person_id, face_count=p.face_count,
                                    sample_ids=list(p.sample_ids), label=labels.get(p.person_id, ""))
                        for p in people]
        n_scenes = len({w.row.scene_id for w in work.values()
                        if w.row.scene_id is not None and w.row.scene_id >= 0})
        counts = Counts(uploaded=len(work), kept=len(picks), people=len(album_people),
                        scenes=n_scenes,
                        set_aside=sum(w.status == SET_ASIDE for w in work.values()))
        return Album(job_id=job_id, created_at=job.album_created_at or "", counts=counts,
                     picks=picks, dropped=dropped, people=album_people, theme=job.theme or {})


def _scene(row: Photo) -> int:
    return row.scene_id if row.scene_id is not None else -1


def _apply(edit: UserEdit, work: Dict[str, _P], labels: Dict[str, str]) -> None:
    p = edit.payload or {}
    if edit.kind == "rename":
        if p.get("person_id") in labels:
            labels[p["person_id"]] = str(p.get("label", ""))
        return
    w = work.get(p.get("photo_id"))
    if w is None:
        return   # the photo left the album in a later re-run; the edit no longer applies
    if edit.kind == "promote" and w.status != PICK:
        w.status, w.detail, w.beaten_by = PICK, "", []
        for other in work.values():
            if w.row.photo_id in other.beat:
                other.beat = [b for b in other.beat if b != w.row.photo_id]
    elif edit.kind == "demote" and w.status == PICK:
        w.status, w.detail, w.beat = NOT_SELECTED, DEMOTED_DETAIL, []


def record_edit(db: ControlDB, job_id: str, kind: str, payload: Dict) -> UserEdit:
    """Validate and store one edit. Raises EditError with a user-facing message."""
    if kind not in EDIT_KINDS:
        raise EditError(f"unknown edit kind {kind!r}; expected one of {EDIT_KINDS}")
    with db.session() as s:
        if s.get(Job, job_id) is None:
            raise KeyError(job_id)
        if kind == "rename":
            if s.get(Person, (job_id, payload.get("person_id", ""))) is None:
                raise EditError(f"no person {payload.get('person_id')!r} in job {job_id!r}")
            label = str(payload.get("label", "")).strip()
            if len(label) > 128:
                raise EditError("label is longer than 128 characters")
            payload = {"person_id": payload["person_id"], "label": label}
        else:
            photo = s.get(Photo, (job_id, payload.get("photo_id", "")))
            if photo is None or not photo.in_album:
                raise EditError(f"no album photo {payload.get('photo_id')!r} in job {job_id!r}")
            if kind == "promote" and not photo.promotable:
                raise EditError(f"photo {photo.photo_id!r} cannot be promoted")
            payload = {"photo_id": photo.photo_id}
        edit = UserEdit(job_id=job_id, ts=datetime.now(timezone.utc), kind=kind, payload=payload)
        s.add(edit)
        s.flush()
        return edit
