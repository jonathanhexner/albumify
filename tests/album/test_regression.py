"""spec-113 R1 - AlbumComparator: "close enough" to a baseline, not identical."""
from __future__ import annotations

import copy

from sim_bench.album.regression import (AlbumComparator, AlbumSummarizer, CompareInputs,
                                        Tolerances)


def _album(picks=None, people=(20, 10, 6, 3, 2), counts=None):
    picks = picks if picks is not None else {f"p{i}": [f"p{i}_dup"] for i in range(20)}
    return {
        "counts": counts or {"uploaded": 100, "kept": len(picks), "scenes": 15,
                             "people": len(people), "set_aside": 8},
        "people": [{"id": f"person_{i}", "face_count": n} for i, n in enumerate(people)],
        "picks": [{"id": k, "beat": v} for k, v in picks.items()],
    }


def _cmp(base, cur):
    s = AlbumSummarizer()
    return AlbumComparator().calc(CompareInputs(baseline=s.calc(base), current=s.calc(cur)))


def test_identical_album_passes():
    assert _cmp(_album(), _album()).ok


def test_pick_swapped_for_its_near_duplicate_still_matches():
    picks = {f"p{i}": [f"p{i}_dup"] for i in range(20)}
    swapped = dict(picks)
    for i in range(5):                      # 5 of 20 bursts now keep the sibling frame
        del swapped[f"p{i}"]
        swapped[f"p{i}_dup"] = [f"p{i}"]
    res = _cmp(_album(picks), _album(swapped))
    assert res.ok and res.metrics["pick_recall"] == 1.0


def test_picks_from_other_moments_fail_recall_and_precision():
    picks = {f"p{i}": [] for i in range(20)}
    changed = {**{f"p{i}": [] for i in range(16)}, **{f"new{i}": [] for i in range(4)}}
    res = _cmp(_album(picks), _album(changed))
    assert not res.ok
    assert res.metrics["pick_recall"] == 0.8 and res.metrics["pick_precision"] == 0.8


def test_small_count_drift_is_tolerated_large_is_not():
    base = _album()
    near = copy.deepcopy(base)
    near["counts"]["scenes"] = 16           # +1: under the absolute floor
    far = copy.deepcopy(base)
    far["counts"]["scenes"] = 25
    assert _cmp(base, near).ok
    res = _cmp(base, far)
    assert not res.ok and any(f.startswith("scenes:") for f in res.failures)


def test_largest_person_shrinking_fails():
    res = _cmp(_album(people=(20, 10, 6, 3, 2)), _album(people=(12, 10, 6, 3, 2)))
    assert [f for f in res.failures if f.startswith("person #1")]


def test_people_count_jump_fails():
    """SIGHTING-132 shape: the old clusterer turned 80 people into 118."""
    base = _album(people=tuple([5] * 80))
    cur = _album(people=tuple([4] * 118))
    assert any(f.startswith("people:") for f in _cmp(base, cur).failures)


def test_tolerances_are_configurable():
    base, cur = _album(people=(20,)), _album(people=(17,))
    assert _cmp(base, cur).ok                       # default: 15% is inside the 20% band
    strict = AlbumComparator(Tolerances(people_size_rel=0.05, count_abs_floor=0))
    s = AlbumSummarizer()
    assert not strict.calc(CompareInputs(baseline=s.calc(base), current=s.calc(cur))).ok


def test_pick_diff_names_each_moved_pick():
    base = {"a": ["a2"], "b": ["b2"], "c": [], "d": []}
    cur = {"a": ["a2"], "a2": ["a"], "b2": ["b"], "c": [], "x": []}
    d = _cmp(_album(base), _album(cur)).picks
    assert d.same == ["a", "c"] and d.swapped == [("b", "b2")]
    assert d.missing == ["d"] and d.extra == ["x"] and d.also_kept == ["a2"]


def test_checks_carry_the_allowed_band():
    res = _cmp(_album(), _album(people=(30, 10, 6, 3, 2)))
    top = next(c for c in res.checks if c.name == "person #1 faces")
    assert (top.baseline, top.current, top.allowed, top.ok) == (20, 30, 6, False)
