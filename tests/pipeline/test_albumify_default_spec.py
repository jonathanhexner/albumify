"""spec-079 — Albumify's default pipeline must be a valid PipelineSpec.

Contract guard: Albumify now submits ``PipelineSpec(steps=default_pipeline,
step_configs=...)`` to the shared ``execute_spec`` primitive, which validates it
before running. If someone adds a step to ``configs/pipeline.yaml`` with a
mis-typed param or an unknown name, this fails at test time instead of at the
user's pipeline run.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from sim_bench.pipeline.spec import PipelineSpec, validate_spec

PIPELINE_YAML = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"


def _albumify_default_spec() -> PipelineSpec:
    doc = yaml.safe_load(PIPELINE_YAML.read_text(encoding="utf-8"))
    steps = doc["default_pipeline"]
    step_configs = {name: (doc.get(name) or {}) for name in steps}
    return PipelineSpec(steps=steps, step_configs=step_configs)


def test_albumify_default_pipeline_validates_clean():
    spec = _albumify_default_spec()
    problems = validate_spec(spec)
    assert problems == [], "Albumify default_pipeline is not a valid spec:\n" + "\n".join(problems)
