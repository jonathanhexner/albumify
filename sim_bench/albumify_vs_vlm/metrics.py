"""Spec-102 T4.2 — objective metrics (the defensible numbers, no human judge).

Three metrics, each computed for BOTH arms' K-sequences:
- duplicate_survival: using Albumify's scene clusters as the near-duplicate ground truth, how
  many picks are NOT the sole representative of their scene? (lower = more diverse album)
- coverage: precision/recall of persons + scenes against a hand-labelled roster (A6). Returns
  None until the roster is labelled (persons/scenes filled).
- defect_rate: fraction of picks flagged by our own detectors (occlusion/sharpness). Computed by
  a separate cheap post-hoc over just the ~40 picked images (kept out of this pure module).

Pure functions over stems + maps, so they unit-test without any pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DuplicateSurvival:
    k: int
    n_scenes_covered: int          # distinct Albumify scenes represented in the picks
    n_redundant: int               # picks beyond the first in an already-covered scene
    redundancy_rate: float         # n_redundant / k  (0 = every pick a distinct scene)


def _scene_of(stem: str, labels: dict[str, int]):
    """Albumify scene id; unknown/noise picks are each their own singleton (no redundancy)."""
    lab = labels.get(stem)
    return ("u", stem) if lab is None or lab == -1 else ("s", lab)


def duplicate_survival(order: list[str], scene_labels: dict[str, int]) -> DuplicateSurvival:
    """How diverse is this K-sequence w.r.t. Albumify's near-duplicate scene clusters?"""
    seen: set = set()
    redundant = 0
    for stem in order:
        key = _scene_of(stem, scene_labels)
        if key in seen:
            redundant += 1
        else:
            seen.add(key)
    k = len(order)
    return DuplicateSurvival(
        k=k, n_scenes_covered=len(seen), n_redundant=redundant,
        redundancy_rate=round(redundant / k, 4) if k else 0.0,
    )


@dataclass
class Coverage:
    kind: str                      # "persons" | "scenes"
    recall: float                  # fraction of roster entities that appear in the picks
    n_total: int
    n_covered: int
    missed: list[str]


def _coverage(order: set[str], entities: list[dict], id_key: str) -> Coverage:
    covered, missed = 0, []
    for ent in entities:
        imgs = set(ent.get("images", []))
        name = ent.get(id_key) or ent.get("id", "?")
        if imgs & order:
            covered += 1
        else:
            missed.append(name)
    n = len(entities)
    return Coverage(kind=id_key, recall=round(covered / n, 4) if n else 0.0,
                    n_total=n, n_covered=covered, missed=missed)


def coverage_from_roster(order: list[str], roster: dict) -> dict | None:
    """Person + scene recall against a labelled roster; None if roster not yet labelled."""
    persons = [p for p in roster.get("persons", []) if p.get("images")]
    scenes = [s for s in roster.get("scenes", []) if s.get("images")]
    if not persons and not scenes:
        return None
    oset = set(order)
    return {
        "persons": _coverage(oset, persons, "name").__dict__ if persons else None,
        "scenes": _coverage(oset, scenes, "label").__dict__ if scenes else None,
    }
