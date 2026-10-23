"""Load a finished job's files into the control DB (spec-112 TA3.1).

The worker writes files only (album.json + results bundle + displays + face crops). This is the one
place those files become rows:

    album.json  -> photo status / reason / beat / order / caption, people, theme   (authoritative
                   for everything the album shows)
    bundle.json -> every staged photo (incl. ones the album never saw), time, GPS, all scores,
                   pipeline decisions, every face with its person

Idempotent: a load replaces the job's rows in one transaction, so re-loading the same job leaves the
counts unchanged (AC2). ``user_edits`` are NOT touched -- they are the owner's, not the machine's,
and survive a re-run of the curation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sqlalchemy import delete

from sim_bench.album.normalizer import canonical_scene_ids, chrono_key
from sim_bench.album.schema import PICK
from sim_bench.control.db import ControlDB
from sim_bench.control.models import Face, Job, Person, Photo, Scene
from sim_bench.serving.job_store import JobStore

logger = logging.getLogger(__name__)

__all__ = ["LoadInputs", "LoadResult", "ResultsLoader", "MissingResults"]


class MissingResults(FileNotFoundError):
    pass


@dataclass(frozen=True, slots=True)
class LoadInputs:
    job_id: str


@dataclass(frozen=True, slots=True)
class LoadResult:
    job_id: str
    n_photos: int
    n_in_album: int
    n_picks: int
    n_faces: int
    n_people: int
    n_scenes: int
    warnings: List[str] = field(default_factory=list)


class ResultsLoader:
    """Files of one job (in a JobStore) -> rows (in the control DB)."""

    def __init__(self, db: ControlDB, store: JobStore):
        self._db = db
        self._store = store

    def calc(self, inputs: LoadInputs) -> LoadResult:
        job_id = inputs.job_id
        album = self._store.read_album(job_id)
        bundle = self._store.read_results(job_id)
        if album is None or bundle is None:
            missing = [n for n, v in (("album.json", album), ("results/bundle.json", bundle)) if v is None]
            raise MissingResults(f"job {job_id!r}: missing {', '.join(missing)} under "
                                 f"{self._store.paths(job_id).root}")

        warnings: List[str] = []
        photos = self._photo_rows(job_id, album, bundle, warnings)
        faces = self._face_rows(job_id, bundle)
        people = [Person(job_id=job_id, person_id=p["id"], cluster_id=_cluster_id(p["id"]),
                         face_count=int(p.get("face_count", 0)),
                         sample_ids=list(p.get("sample_ids", [])), label=p.get("label", ""),
                         position=i)
                  for i, p in enumerate(album.get("people", []))]
        scenes = _scene_rows(job_id, photos)

        with self._db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise KeyError(f"job {job_id!r} is not in the control DB; create it first")
            for model in (Photo, Face, Person, Scene):
                s.execute(delete(model).where(model.job_id == job_id))
            s.add_all(photos + faces + people + scenes)
            job.theme = album.get("theme") or {}
            job.album_created_at = album.get("created_at")
            job.n_inputs = len(photos)

        for w in warnings:
            logger.warning("job %s: %s", job_id, w)
        res = LoadResult(job_id=job_id, n_photos=len(photos),
                         n_in_album=sum(p.in_album for p in photos),
                         n_picks=sum(p.status == PICK for p in photos), n_faces=len(faces),
                         n_people=len(people), n_scenes=len(scenes), warnings=warnings)
        logger.info("job %s loaded: %d photos (%d in album, %d picks), %d faces, %d people, "
                    "%d scenes", job_id, res.n_photos, res.n_in_album, res.n_picks, res.n_faces,
                    res.n_people, res.n_scenes)
        return res

    # --- rows -----------------------------------------------------------------

    def _photo_rows(self, job_id: str, album: Dict, bundle: Dict, warnings: List[str]) -> List[Photo]:
        paths = self._store.paths(job_id)
        files = self._store.input_file_keys(job_id)
        displays = {p.stem: p.name for p in paths.display_dir.iterdir()} \
            if paths.display_dir.is_dir() else {}
        brows = {p["photo_id"]: p for p in bundle.get("photos", [])}
        picks = {p["id"]: p for p in album.get("picks", [])}
        dropped = {d["id"]: d for d in album.get("dropped", [])}

        unknown = (set(picks) | set(dropped)) - set(brows)
        if unknown:
            # the album names a photo the bundle does not -- keep it, but say so
            warnings.append(f"{len(unknown)} album photo(s) missing from the bundle: "
                            f"{sorted(unknown)[:5]}")
            for sid in unknown:
                brows[sid] = {"photo_id": sid, "scores": {}, "decisions": []}

        taken = {sid: b.get("taken_at") for sid, b in brows.items()}
        raw = {sid: b["scene_label"] for sid, b in brows.items() if b.get("scene_label") is not None}
        scene_of = canonical_scene_ids(raw, chrono_key(taken))
        for sid, pick in picks.items():
            if scene_of.get(sid, -1) != pick.get("scene_cluster", -1):
                warnings.append(f"scene id of pick {sid} differs: album "
                                f"{pick.get('scene_cluster')} vs rebuilt {scene_of.get(sid)}")

        rows = []
        for sid, b in sorted(brows.items()):
            sc = b.get("scores", {})
            pick, drop = picks.get(sid), dropped.get(sid)
            score = (pick or {}).get("score", {})
            rows.append(Photo(
                job_id=job_id, photo_id=sid,
                input_key=files.get(sid),
                display_key=f"display/{displays[sid]}" if sid in displays else None,
                taken_at=b.get("taken_at"), lat=b.get("lat"), lon=b.get("lon"),
                in_album=bool(pick or drop),
                status=PICK if pick else (drop["reason"] if drop else None),
                reason_detail=(drop or {}).get("reason_detail", ""),
                beaten_by=list((drop or {}).get("beaten_by", [])),
                beat=list((pick or {}).get("beat", [])),
                promotable=bool((drop or {}).get("promotable", True)),
                pick_order=(pick or {}).get("order"),
                scene_id=pick["scene_cluster"] if pick else scene_of.get(sid),
                scene_label=b.get("scene_label"),
                caption=(pick or {}).get("caption", ""),
                # the album's numbers win for picks (what the user was shown); bundle otherwise
                composite=score.get("composite", sc.get("composite")),
                iqa=score.get("sharpness", sc.get("iqa")),
                ava=score.get("aesthetic", sc.get("ava")),
                sharpness=sc.get("sharpness"),
                occlusion=score.get("occlusion", sc.get("occlusion")),
                tilt=score.get("tilt", sc.get("tilt")),
                tilt_confidence=sc.get("tilt_confidence"),
                decisions=list(b.get("decisions", [])),
            ))
        return rows

    def _face_rows(self, job_id: str, bundle: Dict) -> List[Face]:
        faces_dir = self._store.paths(job_id).faces_dir
        return [Face(job_id=job_id, face_id=int(f["face_id"]), photo_id=f["photo_id"],
                     person_id=f.get("person_id"), bbox=list(f["bbox"]),
                     det_score=f.get("det_score"), blur_score=f.get("blur_score"),
                     pose=f.get("pose"), is_core=bool(f.get("is_core")),
                     rejection_reason=f.get("rejection_reason"),
                     crop_key=f"faces/{int(f['face_id'])}.jpg"
                     if (faces_dir / f"{int(f['face_id'])}.jpg").is_file() else None)
                for f in bundle.get("faces", [])]


def _scene_rows(job_id: str, photos: List[Photo]) -> List[Scene]:
    by_scene: Dict[int, List[Photo]] = {}
    for p in photos:
        if p.in_album and p.scene_id is not None and p.scene_id >= 0:
            by_scene.setdefault(p.scene_id, []).append(p)
    rows = []
    for sid, members in sorted(by_scene.items()):
        times = sorted(p.taken_at for p in members if p.taken_at)
        rows.append(Scene(job_id=job_id, scene_id=sid, n_photos=len(members),
                          n_picks=sum(p.status == PICK for p in members),
                          start_at=times[0] if times else None,
                          end_at=times[-1] if times else None))
    return rows


def _cluster_id(person_id: str) -> Optional[int]:
    tail = person_id.rsplit("_", 1)[-1]
    return int(tail) if tail.lstrip("-").isdigit() else None
