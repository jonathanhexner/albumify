"""Axis-agnostic segmentation quality score (spec-022).

Answers "is this segmentation real, or imposed?" with four 0–1 signals:

  separation  segments well-separated in the axis's space (silhouette)
  coverage    fraction of the album the axis could place at all
  balance     sensible segment sizes (not one blob, not N singletons)
  stability   survives a small threshold nudge (Adjusted Rand Index)
  parsimony   few, large segments beat many tiny fragments (anti-fragmentation)

``overall`` is a weighted mean over whichever components are measurable
(separation is skipped for categorical axes / single-segment results).

Why parsimony exists: silhouette rewards many tiny tight clusters, so without
it a year of home photos fragments into hundreds of "segments" and still scores
well. Parsimony penalizes that directly. (Found by the spec-022 experiment.)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from geo_cluster.axes.base import AxisInputs
from geo_cluster.types import AxisScore, Segmentation

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "separation": 0.30,
    "coverage": 0.15,
    "balance": 0.15,
    "stability": 0.15,
    "parsimony": 0.25,
}


class QualityScorer:
    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def calc(
        self,
        seg: Segmentation,
        inputs: AxisInputs,
        perturbed: Optional[Segmentation] = None,
    ) -> AxisScore:
        total = len(inputs.metadata)
        covered = sum(s.size for s in seg.segments)
        coverage = covered / total if total else 0.0

        # A single bucket is the *flat* case, not a segmentation — score it 0
        # so it can never beat a real multi-segment proposal or clear the floor.
        if len(seg.segments) < 2:
            return AxisScore(
                axis=seg.axis, overall=0.0, separation=None, coverage=coverage,
                balance=0.0, stability=None, parsimony=0.0,
                detail={"n_segments": len(seg.segments), "single_bucket": True},
            )

        separation = self._separation(seg)
        balance = self._balance(seg)
        stability = self._stability(seg, perturbed)
        parsimony = self._parsimony(seg, covered)

        comps = {
            "separation": separation,
            "coverage": coverage,
            "balance": balance,
            "stability": stability,
            "parsimony": parsimony,
        }
        num = sum(self.weights[k] * v for k, v in comps.items() if v is not None)
        den = sum(self.weights[k] for k, v in comps.items() if v is not None)
        overall = num / den if den else 0.0

        return AxisScore(
            axis=seg.axis, overall=overall, separation=separation, coverage=coverage,
            balance=balance, stability=stability, parsimony=parsimony,
            detail={"n_segments": len(seg.segments), "covered": covered, "total": total},
        )

    # ---- components -------------------------------------------------------

    def _separation(self, seg: Segmentation) -> Optional[float]:
        """Silhouette of the segment labels in the axis's space, rescaled 0–1."""
        if seg.metric == "none" or not seg.score_space or len(seg.segments) < 2:
            return None
        paths = list(seg.score_space.keys())
        label_of = seg.path_to_label()
        labels = np.array([label_of[p] for p in paths])
        n_labels = len(set(labels))
        if n_labels < 2 or n_labels >= len(paths):
            return None
        X = np.array([np.atleast_1d(seg.score_space[p]) for p in paths])
        from sklearn.metrics import silhouette_score
        try:
            s = silhouette_score(X, labels, metric=seg.metric)
        except Exception as e:  # degenerate geometry -> treat as no separation
            logger.debug("silhouette failed for axis %s: %s", seg.axis, e)
            return None
        return float((s + 1.0) / 2.0)

    def _balance(self, seg: Segmentation) -> Optional[float]:
        """Normalized size entropy, penalized by the fraction of singletons."""
        sizes = np.array([s.size for s in seg.segments], dtype=float)
        k = len(sizes)
        if k < 2:
            return 0.0
        p = sizes / sizes.sum()
        entropy = -(p * np.log(p)).sum()
        norm_entropy = entropy / np.log(k)
        singleton_frac = float((sizes == 1).mean())
        return float(norm_entropy * (1.0 - singleton_frac))

    def _parsimony(self, seg: Segmentation, covered: int) -> float:
        """Few large segments score high; near-one-segment-per-photo scores low."""
        k = len(seg.segments)
        if covered <= 0:
            return 0.0
        frag_ratio = k / covered          # 1.0 == every photo is its own segment
        return float(max(0.0, 1.0 - frag_ratio))

    def _stability(self, seg: Segmentation, perturbed: Optional[Segmentation]) -> Optional[float]:
        """Agreement (ARI, clipped to 0–1) between base and threshold-nudged labels."""
        if perturbed is None:
            return None
        a, b = seg.path_to_label(), perturbed.path_to_label()
        common = [p for p in a if p in b]
        if len(common) < 2:
            return None
        la = [a[p] for p in common]
        lb = [b[p] for p in common]
        if len(set(la)) < 2 and len(set(lb)) < 2:
            return 1.0  # both say "one group" — perfectly stable agreement
        from sklearn.metrics import adjusted_rand_score
        return float(max(0.0, adjusted_rand_score(la, lb)))
