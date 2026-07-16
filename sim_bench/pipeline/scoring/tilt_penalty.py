"""Tilt penalty for composite scoring (spec-099 / spec-101 option A).

Fourth additive component of select_best's composite. Penalise by the damage that
remains after the cheapest fix (rotate + inscribed crop) — so a crooked photo
loses points according to how badly straightening would cost it:

    0                                       if not confident, or |roll| < gate_deg
    −min(fov_weight·(1 − retained_area), cap)  if cleanly FIXABLE (crop keeps the
                                            subject + enough frame): small FOV cost
    −min(slope·(|roll| − gate_deg), cap)    if UNFIXABLE (crop would clip a
                                            prominent person, or drop below the
                                            area floor): full angle-based penalty

Selection therefore prefers the shot that straightens cleanly over one that can't,
and both over a level shot only by the (small) residual. The fixability call is the
SAME subject-aware gate the terminal straighten step uses (straighten_gate.decide),
so the score and the eventual pixel fix agree. Cheap: computed from roll + image
size + person box, no actual crop.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PIL import Image

from sim_bench.pipeline.context import PipelineContext
from sim_bench.quality_assessment.straighten_gate import GateConfig, decide

logger = logging.getLogger(__name__)


class TiltPenaltyConfig:
    """Knobs for the tilt penalty (defaults per spec-099 S3 / spec-101 option A).

    The gate knobs (conf_gate, gate_deg, min_retained_area, prominent_person_frac,
    preserve_aspect) MUST match the straighten_images step so score and fix agree.
    """

    def __init__(self, enabled: bool = True, conf_gate: float = 0.5,
                 gate_deg: float = 3.0, slope: float = 0.02, cap: float = 0.15,
                 fov_weight: float = 0.4, min_retained_area: float = 0.70,
                 prominent_person_frac: float = 0.15, preserve_aspect: bool = True):
        self.enabled = enabled
        self.conf_gate = conf_gate
        self.gate_deg = gate_deg
        self.slope = slope                  # per-degree penalty for UNFIXABLE tilts
        self.cap = cap                      # max magnitude
        self.fov_weight = fov_weight        # penalty per unit of lost area (FIXABLE)
        self.gate = GateConfig(conf_gate=conf_gate, gate_deg=gate_deg,
                               min_retained_area=min_retained_area,
                               prominent_person_frac=prominent_person_frac,
                               preserve_aspect=preserve_aspect)


class TiltPenaltyComputer:
    """penalty(path) from tilt_angles / tilt_confidences + fixability of the crop."""

    def __init__(self, config: TiltPenaltyConfig):
        self.config = config
        logger.info("TiltPenaltyComputer: enabled=%s conf_gate=%.2f gate_deg=%.1f fov_w=%.2f",
                    config.enabled, config.conf_gate, config.gate_deg, config.fov_weight)

    def compute_penalty(self, image_path: str, context: PipelineContext) -> float:
        if not self.config.enabled:
            return 0.0
        conf = self._lookup(context.tilt_confidences, image_path)
        angle = self._lookup(context.tilt_angles, image_path)
        if conf is None or angle is None:
            return 0.0
        if conf < self.config.conf_gate or abs(angle) < self.config.gate_deg:
            return 0.0  # not confident, or imperceptibly tilted -> a guess never moves a score

        dims = self._dims(image_path)
        if dims is None:
            # can't assess the crop -> treat as unfixable (conservative, full penalty)
            return -min(self.config.slope * (abs(angle) - self.config.gate_deg), self.config.cap)
        w, h = dims
        g = decide(angle, conf, w, h, self._person_bbox(context, image_path), self.config.gate)
        if g.straighten:  # cleanly fixable -> only the FOV cost of the crop
            return -min(self.config.fov_weight * (1.0 - g.retained_area), self.config.cap)
        # unfixable (would clip a prominent person, or crop below the area floor)
        return -min(self.config.slope * (abs(angle) - self.config.gate_deg), self.config.cap)

    @staticmethod
    def _dims(path: str) -> Optional[tuple]:
        try:
            return Image.open(path).size  # (w, h) — header only, no decode
        except Exception:
            return None

    @staticmethod
    def _person_bbox(context: PipelineContext, path: str) -> Optional[tuple]:
        persons = getattr(context, "persons", None) or {}
        p = persons.get(path) or persons.get(path.replace("\\", "/"))
        if not p or not p.get("person_detected") or not p.get("bbox"):
            return None
        b = p["bbox"]
        return (b["x"], b["y"], b["w"], b["h"])

    @staticmethod
    def _lookup(d: dict, image_path: str):
        if image_path in d:
            return d[image_path]
        return d.get(image_path.replace("\\", "/"))


class TiltPenaltyFactory:
    @staticmethod
    def create(config: Dict) -> TiltPenaltyComputer:
        return TiltPenaltyComputer(TiltPenaltyConfig(
            enabled=config.get("enabled", True),
            conf_gate=config.get("conf_gate", 0.5),
            gate_deg=config.get("gate_deg", 3.0),
            slope=config.get("slope", 0.02),
            cap=config.get("cap", 0.15),
            fov_weight=config.get("fov_weight", 0.4),
            min_retained_area=config.get("min_retained_area", 0.70),
            prominent_person_frac=config.get("prominent_person_frac", 0.15),
            preserve_aspect=config.get("preserve_aspect", True),
        ))
