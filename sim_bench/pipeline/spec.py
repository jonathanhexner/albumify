"""The pipeline IS the config: an ordered step list + per-step params.

One ``PipelineSpec``, submitted by any frontend (FC v2 UI, Albumify API) or the
standalone CLI to the one runner (``sim_bench.pipeline.run.run_pipeline``). Given
an identical spec, the run is identical by construction — there is no per-app
hardcoded pipeline.

``validate_spec()`` is the preliminary check a pipeline must pass before running:
mandatory steps present (transitively), no unknown steps, dependencies satisfiable
(no cycles), and every step's params valid against its typed schema. It composes
the pieces that already exist (``PipelineBuilder.validate_pipeline`` for
steps/deps, ``validate_step_config`` for typed params).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import sim_bench.pipeline.steps.all_steps  # noqa: F401 — registers all steps
from sim_bench.pipeline.builder import PipelineBuilder
from sim_bench.pipeline.registry import get_registry
from sim_bench.pipeline.steps.configs._validate import validate_step_config

# Steps a runnable pipeline must include (after dependency resolution).
DEFAULT_MANDATORY_STEPS = ("discover_images",)


@dataclass
class PipelineSpec:
    """A pipeline, fully described by data: ordered steps + per-step params."""

    steps: List[str]
    step_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_fcparams(
        cls, params, *, producer_steps: Iterable[str], clustering_steps: Iterable[str]
    ) -> "PipelineSpec":
        """Build a spec from a typed ``FCParams``.

        Producer steps run with default config; clustering steps are fed the
        FCParams knobs via ``params.to_step_configs()`` — so the typed contract,
        not a hand-mapped dict, defines the clustering config.
        """
        steps = list(producer_steps) + list(clustering_steps)
        step_configs: Dict[str, Dict[str, Any]] = {s: {} for s in producer_steps}
        step_configs.update(params.to_step_configs())
        return cls(steps=steps, step_configs=step_configs)


class PipelineSpecError(ValueError):
    """Raised when a ``PipelineSpec`` fails validation before running."""


def validate_spec(
    spec: PipelineSpec, mandatory: Iterable[str] = DEFAULT_MANDATORY_STEPS
) -> List[str]:
    """Preliminary check. Returns a list of human-readable problems ([] == ok)."""
    registry = get_registry()
    builder = PipelineBuilder(registry)
    problems: List[str] = []

    # 1. unknown steps / unknown dependencies.
    problems.extend(builder.validate_pipeline(spec.steps))

    # 2. resolve order (also surfaces circular dependencies) + mandatory-step check.
    #    Resolve only known steps — get_execution_order raises on an unknown name,
    #    which (1) has already reported.
    known_steps = [s for s in spec.steps if registry.has_step(s)]
    try:
        resolved = set(builder.get_execution_order(known_steps))
    except ValueError as e:  # circular dependency
        problems.append(str(e))
        resolved = set(known_steps)
    for m in mandatory:
        if m not in resolved:
            problems.append(f"missing mandatory step: {m}")

    # 3. per-step typed param validity.
    for name, cfg in spec.step_configs.items():
        if not registry.has_step(name):
            continue  # unknown step already reported in (1) if it's in steps
        try:
            validate_step_config(name, cfg)
        except Exception as e:  # pydantic.ValidationError etc.
            problems.append(f"invalid config for '{name}': {e}")

    return problems


def validate_spec_or_raise(
    spec: PipelineSpec, mandatory: Iterable[str] = DEFAULT_MANDATORY_STEPS
) -> None:
    """Validate, raising ``PipelineSpecError`` listing all problems."""
    problems = validate_spec(spec, mandatory)
    if problems:
        raise PipelineSpecError(
            "Pipeline spec invalid:\n  - " + "\n  - ".join(problems)
        )


__all__ = [
    "PipelineSpec",
    "PipelineSpecError",
    "validate_spec",
    "validate_spec_or_raise",
    "DEFAULT_MANDATORY_STEPS",
]
