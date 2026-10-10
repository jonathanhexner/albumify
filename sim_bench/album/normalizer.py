"""Normalize a PipelineContext into the `Album` contract (spec-106 P0).

The heart is `classify()` — a PURE function that derives each photo's status (pick / near-dup /
not-selected / set-aside) from state the pipeline already produces (selected_images, scene_clusters,
set_aside, scores). The user's point: that information already exists; this surfaces it formally.
`normalize()` is the thin translator that pulls the plain data off the context and assembles the Album.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sim_bench.album.schema import (
    Album, Counts, Dropped, Person, Pick, ScoreBreakdown,
    PICK, NEAR_DUP, NOT_SELECTED, SET_ASIDE,
)

__all__ = ["classify", "normalize", "canonical_scene_ids", "chrono_key"]


def classify(
    all_ids: set,
    selected: List[str],
    labels: Dict[str, int],
    set_aside: Dict[str, dict],
    scores: Dict[str, float],
) -> Dict[str, Tuple[str, str, List[str]]]:
    """Per-photo status derivation. Pure — plain data in, {id: (status, detail, beaten_by)} out.

    Priority: selected → pick; else triaged → set_aside; else same scene as a pick → near_dup;
    else a survivor that wasn't picked → not_selected.
    """
    selected_set = set(selected)
    clusters_with_pick = {labels[s] for s in selected_set if s in labels}
    out: Dict[str, Tuple[str, str, List[str]]] = {}
    for sid in all_ids:
        if sid in selected_set:
            out[sid] = (PICK, "", [])
        elif sid in set_aside:
            out[sid] = (SET_ASIDE, set_aside[sid].get("reason", "set aside in triage"), [])
        else:
            cid = labels.get(sid)
            if cid is not None and cid in clusters_with_pick:
                # sorted: iterating the set would give an arbitrary order, and this list is written
                # into album.json as `beaten_by` (spec-110 AC4 wants a reproducible album)
                winners = sorted(s for s in selected_set if labels.get(s) == cid)
                out[sid] = (NEAR_DUP, "a stronger shot of this moment was kept", winners)
            else:
                out[sid] = (NOT_SELECTED, "not among the top picks", [])
    return out


def _canonical_scene_ids(labels: Dict[str, int], chrono) -> Dict[str, int]:
    """Renumber scene clusters 0..n-1 by their earliest photo; noise (-1) is left alone.

    HDBSCAN's cluster *ids* depend on the row order it is handed, and the embedding dict the
    pipeline gives it is not order-stable (SIGHTING-126): the same photos can come back as cluster 0
    in one run and cluster 1 in the next, with the partition itself unchanged. The album therefore
    does its own numbering, which is what makes album.json reproducible (spec-110 AC4) and also
    makes scene 0 the first scene of the trip rather than an arbitrary one.
    """
    real = sorted({c for c in labels.values() if c >= 0})
    earliest = {c: min((sid for sid, cc in labels.items() if cc == c), key=chrono) for c in real}
    remap = {c: i for i, c in enumerate(sorted(real, key=lambda c: chrono(earliest[c])))}
    return {sid: remap.get(c, c) for sid, c in labels.items()}


def chrono_key(taken_at: Dict[str, Optional[str]]):
    """Album reading-order key over ISO capture times: timed photos first, by time, then by id.

    Same ordering as :func:`_chrono_sorter`, for callers holding plain ISO strings (spec-112: the
    control plane rebuilds the album from DB rows and must order it exactly as the worker did).
    """
    def key(sid: str):
        ts = taken_at.get(sid)
        return (0, ts, sid) if ts else (1, "", sid)

    return key


canonical_scene_ids = _canonical_scene_ids   # public name; the control plane reuses the numbering


def _chrono_sorter(context, stem):
    """Sort key for a photo id: EXIF capture time when available, else the id (filename stem).

    Both orderings are stable across runs -- that is the point. Ids sort chronologically for
    camera-named files (``20260822_181233``), so a trip with no EXIF time still reads in order.
    """
    times = {stem(p): getattr(m, "timestamp", None)
             for p, m in (getattr(context, "geo_metadata", {}) or {}).items()}

    def key(sid: str):
        ts = times.get(sid)
        return (0, ts.isoformat(), sid) if ts is not None else (1, "", sid)

    return key


def normalize(context, job_id: str, created_at: str) -> Album:
    """Assemble the Album from a finished PipelineContext. Thin translator over `classify()`."""
    sf = getattr(context, "straightened_from", {}) or {}

    def stem(p: str) -> str:
        return Path(sf.get(p, p)).stem

    def by_stem(d: dict) -> Dict[str, float]:
        return {stem(p): v for p, v in (d or {}).items()}

    scores = by_stem(getattr(context, "composite_scores", {}))
    iqa = by_stem(getattr(context, "iqa_scores", {}))
    ava = by_stem(getattr(context, "ava_scores", {}))
    occ = by_stem(getattr(context, "occlusion_scores", {}))
    tilt = by_stem(getattr(context, "tilt_angles", {}))
    captions = {stem(p): c for p, c in (getattr(context, "image_captions", {}) or {}).items()}
    labels = {stem(p): int(c) for p, c in (getattr(context, "scene_cluster_labels", {}) or {}).items()}
    selected = [stem(p) for p in (getattr(context, "selected_images", []) or [])]
    set_aside = {Path(e["image"]).stem: e for e in (getattr(context, "set_aside", []) or [])}

    # Canonical ordering + scene numbering: the pipeline's own orders are not stable across runs
    # (SIGHTING-126), and album.json has to be (spec-110 AC4).
    chrono = _chrono_sorter(context, stem)
    labels = _canonical_scene_ids(labels, chrono)

    all_ids = set(scores) | set(labels) | set(set_aside) | set(selected)
    status = classify(all_ids, selected, labels, set_aside, scores)

    # picks: chronological, which is also the album's reading order (Pick.order)
    picks = []
    for i, sid in enumerate(sorted(dict.fromkeys(selected), key=chrono)):
        cid = labels.get(sid, -1)
        near = sorted((d for d, (st, _, _) in status.items()
                       if st == NEAR_DUP and labels.get(d) == cid), key=chrono)
        picks.append(Pick(
            id=sid, order=i, scene_cluster=cid,
            score=ScoreBreakdown(composite=scores.get(sid), sharpness=iqa.get(sid),
                                 aesthetic=ava.get(sid), occlusion=occ.get(sid), tilt=tilt.get(sid)),
            beat=near, caption=captions.get(sid, ""),
        ))

    dropped = [Dropped(id=sid, reason=st, reason_detail=detail, beaten_by=beaten)
               for sid, (st, detail, beaten) in sorted(status.items())
               if st != PICK]

    people_clusters = getattr(context, "people_clusters", {}) or {}
    people = [Person(id=f"person_{pid}", face_count=len(members) if hasattr(members, "__len__") else 0)
              for pid, members in sorted(people_clusters.items(), key=lambda kv: str(kv[0]))]

    n_scenes = len({c for c in labels.values() if c >= 0})
    counts = Counts(uploaded=len(all_ids), kept=len(picks), people=len(people),
                    scenes=n_scenes, set_aside=len(set_aside))

    return Album(job_id=job_id, created_at=created_at, counts=counts, picks=picks,
                 dropped=dropped, people=people, theme=getattr(context, "album_theme", {}) or {})
