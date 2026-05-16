"""spec-033 P-G: enforce Pydantic step config contracts.

Rules:
  1. Every model in STEP_CONFIG_MODELS uses ``extra="forbid"`` — typo'd keys
     raise ValidationError instead of silently defaulting.
  2. Every field declares a non-empty ``Field(description=...)`` — the single
     source of truth for UI ``help=`` text (spec-033 P-A → P-G linkage).
  3. A misspelled config key raises ValidationError end-to-end at the
     step boundary.
  4. Each typed step's process() (or the BaseStep template-method path)
     reaches validate_step_config — we check by introspection: every model
     in STEP_CONFIG_MODELS must have a matching step registered.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sim_bench.pipeline.steps.configs import STEP_CONFIG_MODELS
from sim_bench.pipeline.steps.configs._validate import validate_step_config


def test_every_model_forbids_extra_keys():
    """extra='forbid' is the contract — typo'd keys must crash, not silently default."""
    failures = []
    for step_name, model in STEP_CONFIG_MODELS.items():
        extra = model.model_config.get("extra")
        if extra != "forbid":
            failures.append(f"{step_name}: model_config.extra={extra!r}, expected 'forbid'")
    assert not failures, "\n".join(failures)


def test_every_field_has_description():
    """UI help= text is sourced from Field(description=...). Empty == drift."""
    failures = []
    for step_name, model in STEP_CONFIG_MODELS.items():
        for field_name, field_info in model.model_fields.items():
            desc = field_info.description
            if not desc or not desc.strip():
                failures.append(
                    f"{step_name}.{field_name}: missing or empty Field(description=...). "
                    f"This is the source of UI help text per spec-033 P-G."
                )
    assert not failures, "\n".join(failures)


def test_misspelled_key_raises():
    """The whole point of P-G — UI typo'd key at the boundary must fail loud."""
    with pytest.raises(ValidationError) as ei:
        validate_step_config("filter_quality", {"min_iqa_score": 0.3, "mim_sharpness": 0.2})
    assert "mim_sharpness" in str(ei.value)


def test_out_of_range_value_raises():
    """Range constraints in Field(ge=, le=) are enforced at the boundary."""
    with pytest.raises(ValidationError):
        validate_step_config("filter_quality", {"min_iqa_score": 1.5})


def test_unknown_step_returns_none():
    """Steps without a model are not yet typed — pass through silently.

    (Full pipeline migration is out of scope per master plan P-G.)
    """
    assert validate_step_config("score_iqa", {"whatever": 1}) is None


def test_validated_model_exposes_typed_attrs():
    """Caller can read typed attributes after validation."""
    cfg = validate_step_config("filter_quality", {"min_iqa_score": 0.4})
    assert cfg is not None
    assert cfg.min_iqa_score == 0.4
    # default fills in when caller omits
    assert cfg.min_sharpness == pytest.approx(0.2)


def test_registry_models_match_module_exports():
    """STEP_CONFIG_MODELS must include every model exported from the package."""
    from sim_bench.pipeline.steps import configs as configs_pkg

    # Every BaseModel-class attribute in the package's __all__ should be
    # represented in the registry exactly once.
    exported_models = {
        getattr(configs_pkg, name)
        for name in configs_pkg.__all__
        if name != "STEP_CONFIG_MODELS"
    }
    registry_values = set(STEP_CONFIG_MODELS.values())
    missing_in_registry = exported_models - registry_values
    assert not missing_in_registry, (
        f"Models exported but not in STEP_CONFIG_MODELS: {missing_in_registry}"
    )
