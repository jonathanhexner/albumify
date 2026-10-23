"""Jobs API (spec-112 TA5.1): submit a job, watch it, read its album, edit it.

Reads come from the control DB (rows) and the job store (image bytes); nothing here runs a model.
``POST /jobs`` takes a LOCAL folder path -- a Phase A, owner-only convenience. Browser upload with
presigned URLs replaces it before anything is public (spec-106 P2).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from sim_bench.api.schemas.jobs import (
    AlbumOut, CompareOut, EditIn, EditOut, FaceOut, JobAccepted, JobDetail, JobEventOut, JobLoad,
    JobOut, JobSubmit, PhotoOut,
)
from sim_bench.control.album_builder import AlbumBuilder, BuildInputs, EditError, record_edit
from sim_bench.control.compare import CompareJobsInputs, DifferentPhotos, JobComparer
from sim_bench.control.db import ControlDB
from sim_bench.control.jobs import SUCCEEDED, InvalidTransition
from sim_bench.control.models import Face, Photo
from sim_bench.serving.job_service import JobService, SubmitInputs
from sim_bench.serving.serving_config import RUNNERS, load_serving_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@lru_cache(maxsize=1)
def get_job_service() -> JobService:
    """One service per process (engine + config). Tests override this dependency."""
    return JobService(ControlDB(), load_serving_config())


def _job_or_404(svc: JobService, job_id: str):
    job = svc.repo.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return job


def _job_out(job) -> dict:
    return JobOut.model_validate(job, from_attributes=True).model_dump()


@router.post("", response_model=JobAccepted, status_code=202)
def submit_job(body: JobSubmit, background: BackgroundTasks,
               svc: JobService = Depends(get_job_service)):
    if body.runner and body.runner not in RUNNERS:
        raise HTTPException(422, f"runner must be one of {RUNNERS}")
    inputs = SubmitInputs(source=Path(body.source), job_id=body.job_id, runner=body.runner,
                          pipeline=body.pipeline, start=body.start, n=body.n)
    try:
        job_id = svc.create(inputs)
    except InvalidTransition as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    background.add_task(svc.run, job_id, inputs)
    return JobAccepted(job_id=job_id, status="queued")


@router.post("/{job_id}/load", response_model=JobAccepted)
def load_job_files(job_id: str, body: JobLoad, svc: JobService = Depends(get_job_service)):
    """Record + load a job whose files are already in the store (e.g. a Modal run pulled down)."""
    if not svc.store.paths(job_id).album_path.is_file():
        raise HTTPException(404, f"no album.json for {job_id!r} under {svc.store.root}")
    try:
        res = svc.load_existing(job_id, runner=body.runner, source=body.source)
    except InvalidTransition as e:
        raise HTTPException(409, str(e))
    return JobAccepted(job_id=job_id, status=res.status)


@router.get("", response_model=list[JobOut])
def list_jobs(limit: int = 100, svc: JobService = Depends(get_job_service)):
    svc.repo.reap_orphans()     # a killed worker cannot fail its own job (spec-114 AC6)
    return [_job_out(j) for j in svc.repo.list(limit)]


@router.get("/{job_id}", response_model=JobDetail)
def get_job(job_id: str, svc: JobService = Depends(get_job_service)):
    svc.repo.reap_orphans()
    job = _job_or_404(svc, job_id)
    events = [JobEventOut(ts=e.ts, kind=e.kind, message=e.message) for e in svc.repo.events(job_id)]
    return JobDetail(**_job_out(job), events=events)


@router.get("/{job_id}/album", response_model=AlbumOut)
def get_album(job_id: str, svc: JobService = Depends(get_job_service)):
    job = _job_or_404(svc, job_id)
    if job.status != SUCCEEDED:
        raise HTTPException(409, f"job {job_id!r} is {job.status}; the album exists once it succeeds")
    return AlbumBuilder(svc.db).calc(BuildInputs(job_id)).to_dict()


@router.get("/{job_id}/compare", response_model=CompareOut)
def compare_jobs(job_id: str, with_: str = Query(..., alias="with", description="reference job (A)"),
                 svc: JobService = Depends(get_job_service)):
    """spec-114: this job (B) against a reference job (A) on the same photos."""
    for jid in (job_id, with_):
        job = _job_or_404(svc, jid)
        if job.status != SUCCEEDED:
            raise HTTPException(409, f"job {jid!r} is {job.status}; compare finished jobs")
    try:
        return JobComparer(svc.db).calc(CompareJobsInputs(a=with_, b=job_id)).to_dict()
    except DifferentPhotos as e:
        raise HTTPException(422, str(e))


@router.get("/{job_id}/photos", response_model=list[PhotoOut])
def list_photos(job_id: str, svc: JobService = Depends(get_job_service)):
    """Every staged photo, including ones removed before the album saw them (with the reason)."""
    _job_or_404(svc, job_id)
    with svc.db.session() as s:
        n_faces = dict(s.execute(select(Face.photo_id, func.count()).where(Face.job_id == job_id)
                                 .group_by(Face.photo_id)).all())
        photos = list(s.scalars(select(Photo).where(Photo.job_id == job_id).order_by(Photo.photo_id)))
    return [PhotoOut(photo_id=p.photo_id, in_album=p.in_album, status=p.status,
                     reason_detail=p.reason_detail, taken_at=p.taken_at, scene_id=p.scene_id,
                     composite=p.composite, has_display=p.display_key is not None,
                     n_faces=n_faces.get(p.photo_id, 0), decisions=p.decisions) for p in photos]


@router.get("/{job_id}/faces", response_model=list[FaceOut])
def list_faces(job_id: str, person_id: int | None = None,
               svc: JobService = Depends(get_job_service)):
    _job_or_404(svc, job_id)
    with svc.db.session() as s:
        q = select(Face).where(Face.job_id == job_id)
        if person_id is not None:
            q = q.where(Face.person_id == person_id)
        faces = list(s.scalars(q.order_by(Face.face_id)))
    return [FaceOut(face_id=f.face_id, photo_id=f.photo_id, person_id=f.person_id,
                    det_score=f.det_score, is_core=f.is_core, rejection_reason=f.rejection_reason,
                    has_crop=f.crop_key is not None) for f in faces]


def _file(svc: JobService, job_id: str, key: str | None) -> FileResponse:
    path = svc.store.resolve(job_id, key or "")   # job folder or shared input store; never outside
    if path is None:
        raise HTTPException(404, "no file")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{job_id}/photos/{photo_id}/image")
def photo_image(job_id: str, photo_id: str, svc: JobService = Depends(get_job_service)):
    """The display copy for picks, else the staged 768 px input (dropped photos are shown too)."""
    with svc.db.session() as s:
        p = s.get(Photo, (job_id, photo_id))
    if p is None:
        raise HTTPException(404, f"no photo {photo_id!r} in job {job_id!r}")
    return _file(svc, job_id, p.display_key or p.input_key)


@router.get("/{job_id}/faces/{face_id}/crop")
def face_crop(job_id: str, face_id: int, svc: JobService = Depends(get_job_service)):
    with svc.db.session() as s:
        f = s.get(Face, (job_id, face_id))
    if f is None:
        raise HTTPException(404, f"no face {face_id} in job {job_id!r}")
    return _file(svc, job_id, f.crop_key)


@router.post("/{job_id}/edits", response_model=EditOut, status_code=201)
def add_edit(job_id: str, body: EditIn, svc: JobService = Depends(get_job_service)):
    job = _job_or_404(svc, job_id)
    if job.status != SUCCEEDED:
        raise HTTPException(409, f"job {job_id!r} is {job.status}; edit once it succeeds")
    try:
        e = record_edit(svc.db, job_id, body.kind, body.payload)
    except EditError as err:
        raise HTTPException(422, str(err))
    return EditOut(id=e.id, kind=e.kind, payload=e.payload)
