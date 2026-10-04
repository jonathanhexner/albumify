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

__all__ = ["classify", "normalize"]


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
                winners = [s for s in selected_set if labels.get(s) == cid]
                out[sid] = (NEAR_DUP, "a stronger shot of this moment was kept", winners)
            else:
                out[sid] = (NOT_SELECTED, "not among the top picks", [])
    return out


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

    all_ids = set(scores) | set(labels) | set(set_aside) | set(selected)
    status = classify(all_ids, selected, labels, set_aside, scores)

    # picks: ordered by selected_images order
    seen, picks = set(), []
    for i, sid in enumerate(selected):
        if sid in seen:
            continue
        seen.add(sid)
        cid = labels.get(sid, -1)
        near = [d for d, (st, _, _) in status.items() if st == NEAR_DUP and labels.get(d) == cid]
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
              for pid, members in people_clusters.items()]

    n_scenes = len({c for c in labels.values() if c >= 0})
    counts = Counts(uploaded=len(all_ids), kept=len(picks), people=len(people),
                    scenes=n_scenes, set_aside=len(set_aside))

    return Album(job_id=job_id, created_at=created_at, counts=counts, picks=picks,
                 dropped=dropped, people=people, theme=getattr(context, "album_theme", {}) or {})
