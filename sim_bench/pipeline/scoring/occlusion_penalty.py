"""Occlusion penalty for composite scoring (spec-097 Stage 1).

Third additive component of select_best's composite (spec-084 pattern):

    penalty = 0                                    if P(occluded) < gate
            = weight * P * area_factor(tiles)      otherwise, floored at max_penalty

- Driver is the LEARNED P(occluded) from score_occlusion (spec-096 CLIP probe),
  never raw blurriness — bokeh/night shots are safe by construction.
- gate defaults to 0.8, the ~86%-precision operating point measured on the
  adjudicated benchmark; below it the term is exactly 0.
- area_factor scales with how many tiles look occluded (small corner blob dents,
  large occlusion hurts): 0.4 + 0.6 * n_hot_tiles/9.
"""

from __future__ import annotations

import logging
from typing import Dict

from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class OcclusionPenaltyConfig:
    """Knobs for the occlusion penalty term (defaults per spec-097 Stage 1)."""

    def __init__(self, enabled: bool = True, gate: float = 0.8,
                 weight: float = -0.35, tile_threshold: float = 0.5,
                 max_penalty: float = -0.5):
        self.enabled = enabled
        self.gate = gate
        self.weight = weight              # negative: it is a penalty
        self.tile_threshold = tile_threshold
        self.max_penalty = max_penalty    # floor (most-negative value allowed)


class OcclusionPenaltyComputer:
    """penalty(path) from context.occlusion_scores / occlusion_tiles."""

    def __init__(self, config: OcclusionPenaltyConfig):
        self.config = config
        logger.info("OcclusionPenaltyComputer: enabled=%s gate=%.2f weight=%.2f",
                    config.enabled, config.gate, config.weight)

    def compute_penalty(self, image_path: str, context: PipelineContext) -> float:
        if not self.config.enabled:
            return 0.0
        p = self._lookup(context.occlusion_scores, image_path)
        if p is None or p < self.config.gate:
            return 0.0  # below the high-precision gate: no penalty at all
        tiles = self._lookup(context.occlusion_tiles, image_path) or []
        n_hot = sum(1 for t in tiles if t >= self.config.tile_threshold)
        area_factor = 0.4 + 0.6 * (n_hot / 9.0 if tiles else 0.0)
        penalty = self.config.weight * p * area_factor
        return max(penalty, self.config.max_penalty)

    @staticmethod
    def _lookup(d: dict, image_path: str):
        """Tolerate the two path-separator spellings (same rule as person_penalty)."""
        if image_path in d:
            return d[image_path]
        return d.get(image_path.replace("\\", "/"))


class OcclusionPenaltyFactory:
    @staticmethod
    def create(config: Dict) -> OcclusionPenaltyComputer:
        return OcclusionPenaltyComputer(OcclusionPenaltyConfig(
            enabled=config.get("enabled", True),
            gate=config.get("gate", 0.8),
            weight=config.get("weight", -0.35),
            tile_threshold=config.get("tile_threshold", 0.5),
            max_penalty=config.get("max_penalty", -0.5),
        ))
