"""Straighten images step (spec-101) — TERMINAL, straightens the selected winners.

Runs AFTER select_best (option A): selection already accounted for the crop cost
via the fixability-aware tilt_penalty, so this step just remediates the OUTPUT.
For each selected winner with a confident tilt, applies the subject-aware gate
(spec-101 S3, reusing detect_persons boxes); on straighten it writes a derived,
corner-clean JPEG to the image cache and repoints that entry in
context.selected_images to the derivative. Declined / near-level winners are left
unchanged; originals are never mutated on disk.

Runs terminal (depends_on select_best) so nothing downstream depends on it — which
keeps the heavy tilt machinery from being dragged into other pipelines by the
dependency resolver. Provenance: context.straightened_from[derived] = original.
Disable with config `enabled: false`.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.quality_assessment.straighten import straighten
from sim_bench.quality_assessment.straighten_gate import GateConfig, decide

logger = logging.getLogger(__name__)

STRAIGHTEN_VERSION = "v1"
_CACHE_DIR = Path.home() / ".sim_bench" / "image_cache" / "straightened"


@register_step
class StraightenImagesStep(BaseStep):
    """Level confident tilts (rotate + inscribed crop) before scoring."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="straighten_images",
            display_name="Straighten Images (auto)",
            description="Rotate + inscribed-crop the selected winners that are "
                        "confidently tilted, unless it would crop a prominent person (spec-101).",
            category="processing",
            requires={"selected_images", "tilt_angles", "tilt_confidences", "persons"},
            produces={"straightened_from"},
            depends_on=["select_best"],
            config_schema={"type": "object", "properties": {
                "enabled": {"type": "boolean", "default": True},
                "conf_gate": {"type": "number", "default": 0.5},
                "gate_deg": {"type": "number", "default": 3.0},
                "min_retained_area": {"type": "number", "default": 0.70},
                "prominent_person_frac": {"type": "number", "default": 0.15},
                "preserve_aspect": {"type": "boolean", "default": True},
            }},
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        if not config.get("enabled", True):
            logger.info("straighten_images disabled -> passthrough")
            return
        cfg = GateConfig(
            conf_gate=config.get("conf_gate", 0.5),
            gate_deg=config.get("gate_deg", 3.0),
            min_retained_area=config.get("min_retained_area", 0.70),
            prominent_person_frac=config.get("prominent_person_frac", 0.15),
            preserve_aspect=config.get("preserve_aspect", True),
        )
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        winners = list(context.selected_images or [])
        new_winners, mapping, reasons = [], {}, {}
        for i, raw in enumerate(winners):
            orig = str(raw)
            roll = context.tilt_angles.get(orig)
            conf = context.tilt_confidences.get(orig, 0.0)
            if roll is None:
                new_winners.append(orig)
                continue
            try:
                w, h = Image.open(orig).size
            except Exception as exc:  # unreadable -> leave as-is
                logger.warning("straighten skip %s: %s", orig, exc)
                new_winners.append(orig)
                continue

            res = decide(roll, conf, w, h, self._person_bbox(context, orig), cfg)
            reasons[res.reason] = reasons.get(res.reason, 0) + 1
            if not res.straighten:
                new_winners.append(orig)
                continue

            derived = self._straighten_to_cache(orig, roll, cfg)
            new_winners.append(derived)
            mapping[derived] = orig
            context.report_progress("straighten_images", (i + 1) / len(winners),
                                    f"Straighten {i + 1}/{len(winners)}")

        context.selected_images = new_winners
        context.straightened_from = mapping
        logger.info("straighten_images: %d/%d winners straightened, decisions=%s",
                    len(mapping), len(winners), reasons)

    @staticmethod
    def _person_bbox(context: PipelineContext, path: str) -> Optional[tuple]:
        p = (context.persons or {}).get(path) or {}
        if not p.get("person_detected") or not p.get("bbox"):
            return None
        b = p["bbox"]
        return (b["x"], b["y"], b["w"], b["h"])

    @staticmethod
    def _straighten_to_cache(orig: str, roll: float, cfg: GateConfig) -> str:
        key = hashlib.md5(
            f"{orig}|{roll:.2f}|{cfg.preserve_aspect}|{STRAIGHTEN_VERSION}".encode()
        ).hexdigest()[:16]
        out = _CACHE_DIR / f"{Path(orig).stem}_{key}.jpg"
        if not out.exists():
            with Image.open(orig) as pil:
                rgb = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
            fixed = straighten(rgb, roll, preserve_aspect=cfg.preserve_aspect)
            Image.fromarray(fixed).save(out, "JPEG", quality=95)
        return str(out)
