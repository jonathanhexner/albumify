"""Compare two jobs on the same photos (spec-114): what changed, and what each job did with every photo.

    A (reference, e.g. a regression baseline)  ┐
                                               ├─ AlbumBuilder (edits applied) -> AlbumComparator
    B (new run)                                ┘   + each photo's decision in A and in B
                                                   + people, largest first, with one face crop each

Refuses two jobs whose photo sets differ: a pick diff between different photos means nothing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from sqlalchemy import func, select

from sim_bench.album.regression import AlbumComparator, AlbumSummarizer, CompareInputs, CompareResult
from sim_bench.control.album_builder import AlbumBuilder, BuildInputs
from sim_bench.control.db import ControlDB
from sim_bench.control.models import Face, Photo

__all__ = ["CompareJobsInputs", "CompareJobsResult", "JobComparer", "DifferentPhotos"]

REASON_TEXT = {"near_dup": "near-duplicate", "not_selected": "not selected", "set_aside": "set aside"}


class DifferentPhotos(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CompareJobsInputs:
    a: str      # reference
    b: str      # new


@dataclass(frozen=True, slots=True)
class CompareJobsResult:
    a: str
    b: str
    result: CompareResult
    decisions: Dict[str, Dict[str, Dict]]     # job id -> photo id -> {state, detail, scene, score}
    people: Dict[str, List[Dict]]             # job id -> [{id, label, face_count, face_id}], largest first

    def to_dict(self) -> Dict:
        r = self.result
        return {"a": self.a, "b": self.b, "ok": r.ok, "failures": r.failures, "metrics": r.metrics,
                "checks": [asdict(c) for c in r.checks], "picks": asdict(r.picks),
                "decisions": self.decisions, "people": self.people}


class JobComparer:
    def __init__(self, db: ControlDB):
        self._db = db

    def calc(self, inputs: CompareJobsInputs) -> CompareJobsResult:
        rows = {j: self._photos(j) for j in (inputs.a, inputs.b)}
        a_ids, b_ids = set(rows[inputs.a]), set(rows[inputs.b])
        if a_ids != b_ids:
            raise DifferentPhotos(f"{inputs.a} has {len(a_ids)} photos, {inputs.b} has {len(b_ids)}, "
                                  f"{len(a_ids & b_ids)} shared: compare runs on the same photos")
        albums = {j: AlbumBuilder(self._db).calc(BuildInputs(j)).to_dict() for j in (inputs.a, inputs.b)}
        s = AlbumSummarizer()
        result = AlbumComparator().calc(CompareInputs(baseline=s.calc(albums[inputs.a]),
                                                      current=s.calc(albums[inputs.b])))
        return CompareJobsResult(
            a=inputs.a, b=inputs.b, result=result,
            decisions={j: _decisions(albums[j], rows[j]) for j in albums},
            people={j: self._people(j, albums[j]) for j in albums})

    def _photos(self, job_id: str) -> Dict[str, Photo]:
        with self._db.session() as s:
            return {p.photo_id: p for p in s.scalars(select(Photo).where(Photo.job_id == job_id))}

    def _people(self, job_id: str, album: Dict) -> List[Dict]:
        with self._db.session() as s:
            first_face = dict(s.execute(
                select(Face.person_id, func.min(Face.face_id))
                .where(Face.job_id == job_id, Face.crop_key.is_not(None), Face.person_id.is_not(None))
                .group_by(Face.person_id)).all())
        people = [{"id": p["id"], "label": p.get("label", ""), "face_count": p["face_count"],
                   "face_id": first_face.get(_cluster(p["id"]))} for p in album["people"]]
        return sorted(people, key=lambda p: (-p["face_count"], p["id"]))


def _cluster(person_id: str):
    tail = person_id.rsplit("_", 1)[-1]
    return int(tail) if tail.lstrip("-").isdigit() else None


def _decisions(album: Dict, photos: Dict[str, Photo]) -> Dict[str, Dict]:
    """Every photo -> what this job did with it, in words a reader can compare side by side."""
    out: Dict[str, Dict] = {}
    for pid, p in photos.items():
        why = "; ".join(f"{d.get('step')}: {d.get('reason')}" for d in p.decisions
                        if d.get("decision") == "rejected")
        out[pid] = {"state": "removed", "detail": f"removed by a filter: {why or 'no recorded reason'}",
                    "scene": p.scene_id, "score": p.composite}
    for d in album["dropped"]:
        detail = REASON_TEXT.get(d["reason"], d["reason"])
        if d["beaten_by"]:
            detail += f" of {', '.join(d['beaten_by'])}"
        if d["reason_detail"]:
            detail += f" ({d['reason_detail']})"
        out[d["id"]].update(state=d["reason"], detail=detail)
    for k in album["picks"]:
        out[k["id"]].update(state="pick", detail=f"kept, pick #{k['order'] + 1}"
                            + (f", beat {len(k['beat'])} near-duplicate(s)" if k["beat"] else ""))
    return out
