"""The competition: run every enabled axis, score it, keep the winner — or
decline to segment if even the best is below the floor (spec-022).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from geo_cluster.axes.base import AxisInputs, get_axes
from geo_cluster.scoring import QualityScorer
from geo_cluster.types import AxisScore, SegmentationOutcome

# Importing the axis modules registers them via @register_axis.
from geo_cluster.axes import geo_axis, time_axis, identity_axis, semantic_axis  # noqa: F401

DEFAULT_FLOOR = 0.45


class SegmentationSelector:
    def __init__(
        self,
        floor: float = DEFAULT_FLOOR,
        weights: Optional[dict] = None,
        enabled: Optional[list[str]] = None,
    ):
        self.floor = floor
        self.scorer = QualityScorer(weights)
        # None => all registered axes except semantic (off by default).
        self.enabled = enabled

    def calc(self, inputs: AxisInputs) -> SegmentationOutcome:
        axes = get_axes()
        enabled = self.enabled if self.enabled is not None else \
            [n for n in axes if n != "semantic"]

        scores: list[AxisScore] = []
        candidates = []
        for name in enabled:
            axis = axes.get(name)
            if axis is None:
                continue
            base = axis.propose(inputs)
            if base is None:
                scores.append(AxisScore(axis=name, overall=0.0, detail={"insufficient": True}))
                continue
            pert = axis.propose(replace(inputs, config=axis.perturbed_config(inputs.config)))
            score = self.scorer.calc(base, inputs, pert)
            scores.append(score)
            candidates.append((base, score))

        scores.sort(key=lambda s: s.overall, reverse=True)
        if not candidates:
            return SegmentationOutcome(winner=None, scores=scores, flat=True, floor=self.floor)

        best_seg, best_score = max(candidates, key=lambda c: c[1].overall)
        flat = best_score.overall < self.floor
        return SegmentationOutcome(
            winner=None if flat else best_seg, scores=scores, flat=flat, floor=self.floor,
        )
