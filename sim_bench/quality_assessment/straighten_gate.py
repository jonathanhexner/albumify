"""
Subject-aware decision for whether to auto-straighten a photo -- spec-101 S3.

Straighten only when the inscribed crop (a) keeps enough of the frame and (b)
does not clip a PROMINENT person out. "Prominent" = a YOLO person box (from the
existing detect_persons step -- no saliency, no new detector) covering at least
`prominent_person_frac` of the frame. Below-gate (near-level / low-confidence)
photos are left as-is; tilted-but-unfixable photos are declined so the tilt
penalty can demote them instead.

Pure geometry -- no image I/O -- so it is cheap enough to also drive the
selection-time fixability check.
"""

from dataclasses import dataclass
from typing import Optional

from sim_bench.quality_assessment.straighten import (
    aspect_preserving_rect,
    largest_inscribed_rect,
)


@dataclass(frozen=True)
class GateConfig:
    conf_gate: float = 0.5
    gate_deg: float = 3.0
    min_retained_area: float = 0.70
    prominent_person_frac: float = 0.15
    preserve_aspect: bool = True


@dataclass(frozen=True)
class GateResult:
    straighten: bool
    retained_area: float
    reason: str  # not_tilted | straighten | declined_area | declined_person


def _rect(w, h, roll, cfg):
    return (aspect_preserving_rect(w, h, roll) if cfg.preserve_aspect
            else largest_inscribed_rect(w, h, roll))


def decide(roll_deg: float, confidence: float, w: int, h: int,
           person_bbox_norm: Optional[tuple], cfg: GateConfig) -> GateResult:
    """person_bbox_norm = (x, y, bw, bh) in [0,1] frame coords, or None."""
    if confidence < cfg.conf_gate or abs(roll_deg) < cfg.gate_deg:
        return GateResult(False, 1.0, "not_tilted")  # ~level: nothing to do

    wr, hr = _rect(w, h, roll_deg, cfg)
    retained = (wr * hr) / (w * h) if w and h else 0.0
    if retained < cfg.min_retained_area:
        return GateResult(False, retained, "declined_area")

    if person_bbox_norm is not None:
        px, py, pw, ph = person_bbox_norm
        if pw * ph >= cfg.prominent_person_frac:
            half_cw, half_ch = (wr / w) / 2.0, (hr / h) / 2.0
            inside = (px >= 0.5 - half_cw and px + pw <= 0.5 + half_cw and
                      py >= 0.5 - half_ch and py + ph <= 0.5 + half_ch)
            if not inside:
                return GateResult(False, retained, "declined_person")

    return GateResult(True, retained, "straighten")
