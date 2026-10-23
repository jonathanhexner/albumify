"""Albumify regression check: is a new album "close enough" to a saved baseline? (spec-113 R1)

Exact equality is the wrong test for a curation pipeline: a near-duplicate burst can legitimately flip
to a sibling frame, and a cluster can gain or lose a face. So an album is reduced to a small summary
(counts, people sizes, picks with the shots each pick beat) and compared with tolerances:

    counts   in-album, picks, scenes, people, set-aside  -> within a relative band (with an absolute floor)
    people   the largest N people's face counts          -> each within a relative band
    picks    a baseline pick is matched when the new album kept it, or kept a shot from the same
             near-duplicate group (the pick's `beat` list, either side). Recall and precision of
             that match must stay above a floor.

Pure functions over album.json dicts; the runner that produces albums lives in
``scripts/albumify_regression.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = ["Tolerances", "AlbumSummarizer", "CompareInputs", "Check", "PickDiff", "CompareResult",
           "AlbumComparator"]

SUMMARY_VERSION = 1


@dataclass(frozen=True, slots=True)
class Tolerances:
    count_rel: Dict[str, float] = field(default_factory=lambda: {
        "uploaded": 0.03, "kept": 0.10, "scenes": 0.10, "people": 0.15, "set_aside": 0.15})
    count_abs_floor: int = 2          # never fail on a difference this small
    top_people: int = 5
    people_size_rel: float = 0.20
    pick_recall_min: float = 0.85
    pick_precision_min: float = 0.85


class AlbumSummarizer:
    """album.json -> the summary a baseline stores."""

    def calc(self, album: Dict) -> Dict:
        return {
            "summary_version": SUMMARY_VERSION,
            "counts": {k: int(v) for k, v in album["counts"].items()},
            "people_sizes": sorted((int(p["face_count"]) for p in album["people"]), reverse=True),
            "picks": {p["id"]: sorted(p.get("beat") or []) for p in album["picks"]},
        }


@dataclass(frozen=True, slots=True)
class CompareInputs:
    baseline: Dict      # AlbumSummarizer output
    current: Dict


@dataclass(frozen=True, slots=True)
class Check:
    """One count or people-size comparison, kept for the report."""
    name: str
    baseline: int
    current: int
    allowed: int        # largest difference that still passes
    ok: bool


@dataclass(frozen=True, slots=True)
class PickDiff:
    """How the picks moved: same photo, a sibling from the same group, or no counterpart."""
    same: List[str]
    swapped: List[Tuple[str, str]]   # (baseline pick, current pick from the same group)
    missing: List[str]               # baseline picks whose group kept nothing now
    extra: List[str]                 # current picks from groups the baseline did not keep
    also_kept: List[str] = field(default_factory=list)   # further current picks from a baseline group


@dataclass(frozen=True, slots=True)
class CompareResult:
    failures: List[str]
    metrics: Dict[str, float]
    checks: List[Check] = field(default_factory=list)
    picks: Optional[PickDiff] = None

    @property
    def ok(self) -> bool:
        return not self.failures


class AlbumComparator:
    def __init__(self, tolerances: Tolerances = Tolerances()):
        self.tol = tolerances

    def calc(self, inputs: CompareInputs) -> CompareResult:
        base, cur, tol = inputs.baseline, inputs.current, self.tol
        failures: List[str] = []
        metrics: Dict[str, float] = {}
        checks: List[Check] = []

        for key, rel in tol.count_rel.items():
            b, c = base["counts"].get(key, 0), cur["counts"].get(key, 0)
            metrics[f"count.{key}"] = c
            checks.append(_check(key, b, c, rel, tol.count_abs_floor))
            if not checks[-1].ok:
                failures.append(f"{key}: {c} vs baseline {b} (allowed +/-{rel:.0%})")

        for i, (b, c) in enumerate(zip(_pad(base["people_sizes"], tol.top_people),
                                       _pad(cur["people_sizes"], tol.top_people)), start=1):
            checks.append(_check(f"person #{i} faces", b, c, tol.people_size_rel, tol.count_abs_floor))
            if not checks[-1].ok:
                failures.append(f"person #{i} size: {c} faces vs baseline {b} "
                                f"(allowed +/-{tol.people_size_rel:.0%})")

        recall = _matched_share(base["picks"], cur["picks"])
        precision = _matched_share(cur["picks"], base["picks"])
        metrics.update(pick_recall=round(recall, 3), pick_precision=round(precision, 3))
        if recall < tol.pick_recall_min:
            failures.append(f"pick recall {recall:.2f} < {tol.pick_recall_min} "
                            "(baseline picks with no kept shot from their group)")
        if precision < tol.pick_precision_min:
            failures.append(f"pick precision {precision:.2f} < {tol.pick_precision_min} "
                            "(new picks from groups the baseline did not keep)")
        return CompareResult(failures=failures, metrics=metrics, checks=checks,
                             picks=_pick_diff(base["picks"], cur["picks"]))


def _close(b: int, c: int, rel: float, floor: int) -> bool:
    return abs(c - b) <= max(floor, rel * max(b, c))


def _check(name: str, b: int, c: int, rel: float, floor: int) -> Check:
    return Check(name=name, baseline=b, current=c, allowed=int(max(floor, rel * max(b, c))),
                 ok=_close(b, c, rel, floor))


def _group(pid: str, picks: Dict[str, List[str]]) -> Set[str]:
    return {pid} | set(picks[pid])


def _pick_diff(base: Dict[str, List[str]], cur: Dict[str, List[str]]) -> PickDiff:
    """Same matching rule as recall/precision, but per pick, so a report can show each one."""
    swapped: List[Tuple[str, str]] = []
    missing: List[str] = []
    new = set(cur) - set(base)
    for b in sorted(set(base) - set(cur)):
        # prefer a newly kept photo; else the moment is covered by a pick both albums share
        partners = sorted(c for c in cur if _group(c, cur) & _group(b, base))
        partners = sorted(set(partners) & new) or partners
        if partners:
            swapped.append((b, partners[0]))
        else:
            missing.append(b)
    extra = sorted(c for c in new if not any(_group(c, cur) & _group(b, base) for b in base))
    also_kept = sorted(new - set(extra) - {c for _, c in swapped})
    return PickDiff(same=sorted(set(base) & set(cur)), swapped=swapped, missing=missing, extra=extra,
                    also_kept=also_kept)


def _pad(sizes: Sequence[int], n: int) -> List[int]:
    return (list(sizes) + [0] * n)[:n]


def _matched_share(picks: Dict[str, List[str]], other: Dict[str, List[str]]) -> float:
    """Share of ``picks`` whose near-duplicate group shares a photo with a group kept in ``other``."""
    if not picks:
        return 1.0 if not other else 0.0
    other_ids: Set[str] = set()
    for pid, beat in other.items():
        other_ids.add(pid)
        other_ids.update(beat)
    hit = sum(1 for pid, beat in picks.items() if other_ids & ({pid} | set(beat)))
    return hit / len(picks)
