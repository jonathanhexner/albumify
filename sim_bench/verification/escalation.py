"""Escalation trigger (spec-111 P0) — the cost-bounding core of the verification layer.

The whole layer is affordable only because it escalates to the VLM *only the borderline cases* — the
decisions the deterministic pipeline is least sure about. Confident decisions stand untouched. This is
the generic primitive every checkpoint uses to pick its borderline subset.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, TypeVar

__all__ = ["borderline"]

T = TypeVar("T")


def borderline(items: Sequence[T], margin: Callable[[T], float], band: float) -> List[T]:
    """Return the items within ``band`` of a decision boundary — the ones worth escalating.

    ``margin(item)`` is the item's signed distance from the fence (0 = exactly on the boundary;
    large magnitude = a confident decision). Items with ``abs(margin) <= band`` are ambiguous and
    get escalated to the VLM; the rest are left to the deterministic result.

    Examples of ``margin`` per checkpoint:
      - merge:     (cluster_distance - merge_threshold)      → near 0 = unsure whether to merge
      - best-of:   (score[top1] - score[top2])               → near 0 = unsure which is best
      - triage:    (confidence - 0.5)                        → near 0 = unsure junk vs keep
    """
    if band < 0:
        raise ValueError("band must be >= 0")
    return [it for it in items if abs(margin(it)) <= band]
