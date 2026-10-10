"""Build the ``PipelineSpec`` a serving worker runs: config load + per-step device placement.

spec-110 T2.1 (extracted from ``modal_app._curate``) so the Modal wrappers stay thin and the tricky
part -- which steps can actually be moved onto the GPU -- is unit-testable without Modal, a GPU, or
any image data.

Why placement is not a one-liner: a step only honors ``device`` if its typed config declares the
knob. ``triage_content`` / ``score_ava`` declare it but ``configs/pipeline.yaml`` omits it, and some
typed configs reject the field outright (SIGHTING-123). So the plan is built by *trying* the flip
through each step's own validator ("validator" = the Pydantic model registered for that step) and
recording what was refused, instead of assuming.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["SpecInputs", "SpecResult", "SpecBuilder", "default_config_path"]


def default_config_path() -> Path:
    """``<repo>/configs/pipeline.yaml`` -- the same path in the container (code mounts at /root)."""
    return Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"


@dataclass(frozen=True, slots=True)
class SpecInputs:
    """Per-call data for :meth:`SpecBuilder.calc`."""

    pipeline: str = "default"           # config key stem: default -> default_pipeline
    device: str = "cpu"                 # cpu | cuda
    config_path: Optional[Path] = None  # overrides the builder's path (tests, alt configs)


@dataclass(frozen=True, slots=True)
class SpecResult:
    """Resolved run plan: the steps, their configs, and where the device landed."""

    steps: List[str]
    step_configs: Dict[str, dict]
    flipped: List[str] = field(default_factory=list)  # now carry device=<device>
    refused: List[str] = field(default_factory=list)  # declare a knob, typed config rejects it

    def as_pipeline_spec(self):
        """Materialize the runnable ``PipelineSpec`` (imported lazily: keeps this module light)."""
        from sim_bench.pipeline.spec import PipelineSpec
        return PipelineSpec(steps=list(self.steps), step_configs=dict(self.step_configs))


class SpecBuilder:
    """Reads ``configs/pipeline.yaml`` and resolves per-step device placement for one run."""

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path

    def calc(self, inputs: SpecInputs) -> SpecResult:
        import yaml

        path = Path(inputs.config_path or self._config_path or default_config_path())
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        key = "default_pipeline" if inputs.pipeline == "default" else f"{inputs.pipeline}_pipeline"
        if key not in doc:
            raise KeyError(f"pipeline '{inputs.pipeline}' not found in {path} (looked for '{key}')")

        steps = list(doc[key])
        step_configs: Dict[str, dict] = {s: dict(doc.get(s) or {}) for s in steps}
        flipped, refused = self._place_device(step_configs, inputs.device)
        logger.info("device=%s flipped=%s refused(typed config forbids)=%s",
                    inputs.device, flipped, refused)
        return SpecResult(steps=steps, step_configs=step_configs, flipped=flipped, refused=refused)

    @staticmethod
    def _place_device(step_configs: Dict[str, dict], device: str) -> tuple[List[str], List[str]]:
        """Set ``device`` on every step that can accept it; report the two failure modes."""
        import sim_bench.pipeline.steps.all_steps  # noqa: F401  (registers every step)
        from sim_bench.pipeline.registry import get_registry
        from sim_bench.pipeline.steps.configs._validate import validate_step_config

        registry = get_registry()
        flipped: List[str] = []
        refused: List[str] = []
        for step, cfg in step_configs.items():
            schema = (registry.get_metadata(step).config_schema or {}).get("properties", {})
            if "device" not in cfg and "device" not in schema:
                continue  # no knob at all: the step's code decides (spec-109 P1 covers those)
            try:
                validate_step_config(step, {**cfg, "device": device})
            except Exception:  # noqa: BLE001 -- any validation error means "cannot take device"
                refused.append(step)
                continue
            cfg["device"] = device
            flipped.append(step)
        return flipped, refused
