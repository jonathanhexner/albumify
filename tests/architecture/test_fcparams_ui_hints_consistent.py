"""spec-041 drift guard — UI hints on FCParams must be consistent with
the validation bounds.

If a field declares both ``ge``/``le`` (validation contract) and
``ui_min``/``ui_max`` (display range), the display range must be a
subset of the validation range. Otherwise the UI can display a value
that the contract rejects, or vice versa.

This test would have caught the yaw_max crash from 2026-05-23:
``Field(30.0, ge=5.0, le=999.0)`` with a slider hardcoded to
``max_value=90.0`` — except in that case the UI hint did not exist
on FCParams at all (the slider literal was independent of FCParams).
Now that every UI-bound field carries its own ``ui_min``/``ui_max``,
this test enforces that the two ranges agree.
"""
from __future__ import annotations

import pytest
from annotated_types import Ge, Gt, Le, Lt

from face_cluster.fc_params import FCParams


def _bound(metadata, cls):
    """Return the constraint value for a given annotated-types class, or None."""
    for m in metadata:
        if isinstance(m, cls):
            # Ge/Le → .ge / .le ; Gt/Lt → .gt / .lt
            return getattr(m, cls.__name__.lower(), None)
    return None


def _ui_fields():
    """Yield (name, field_info, hints) for every UI-bound field."""
    for name, info in FCParams.model_fields.items():
        h = info.json_schema_extra
        if isinstance(h, dict) and "ui_widget" in h:
            yield name, info, h


def test_every_ui_field_has_widget_label_group_order():
    """Required keys on the json_schema_extra block."""
    required = {"ui_widget", "ui_label", "ui_group", "ui_order"}
    for name, _info, hints in _ui_fields():
        missing = required - set(hints)
        assert not missing, f"FCParams.{name} missing UI hints: {missing}"


def test_widget_type_is_known():
    valid = {"slider", "number_input", "checkbox"}
    for name, _info, hints in _ui_fields():
        assert hints["ui_widget"] in valid, (
            f"FCParams.{name} has unknown ui_widget={hints['ui_widget']!r}; "
            f"must be one of {sorted(valid)}"
        )


def test_numeric_widgets_have_ui_range():
    """slider / number_input must declare ui_min and ui_max."""
    for name, _info, hints in _ui_fields():
        if hints["ui_widget"] in ("slider", "number_input"):
            for key in ("ui_min", "ui_max"):
                assert key in hints, f"FCParams.{name} ({hints['ui_widget']}) missing {key!r}"


def test_ui_range_is_subset_of_validation_range():
    """ui_min ≥ ge and ui_max ≤ le. Display range can be tighter than the
    contract — never wider."""
    for name, info, hints in _ui_fields():
        if hints["ui_widget"] not in ("slider", "number_input"):
            continue
        # Find validation bounds.
        ge = _bound(info.metadata, Ge)
        le = _bound(info.metadata, Le)
        ui_min = hints.get("ui_min")
        ui_max = hints.get("ui_max")
        if ge is not None and ui_min is not None:
            assert ui_min >= ge, (
                f"FCParams.{name}: ui_min={ui_min} < ge={ge}. The UI would let "
                "the user enter a value below the validation floor."
            )
        if le is not None and ui_max is not None:
            assert ui_max <= le, (
                f"FCParams.{name}: ui_max={ui_max} > le={le}. The UI would let "
                "the user enter a value above the validation ceiling."
            )


def test_default_is_within_ui_range():
    """The Field default must be displayable in the widget's UI range."""
    for name, info, hints in _ui_fields():
        if hints["ui_widget"] not in ("slider", "number_input"):
            continue
        default = info.default
        if default is None:
            # Optional[int|float] with ui_zero_is_none: 0 is the display default.
            continue
        ui_min = hints.get("ui_min")
        ui_max = hints.get("ui_max")
        if ui_min is not None:
            assert default >= ui_min, (
                f"FCParams.{name}: default={default} < ui_min={ui_min}"
            )
        if ui_max is not None:
            assert default <= ui_max, (
                f"FCParams.{name}: default={default} > ui_max={ui_max}"
            )


def test_zero_is_none_only_on_optional_fields():
    """The 0→None sentinel only makes sense on Optional fields whose
    declared default IS None."""
    for name, info, hints in _ui_fields():
        if not hints.get("ui_zero_is_none"):
            continue
        assert info.default is None, (
            f"FCParams.{name}: ui_zero_is_none=True but default is "
            f"{info.default!r}, not None. The sentinel doesn't apply."
        )


def test_group_names_are_known():
    """Catch typos like ui_group='clusters' (plural)."""
    valid_groups = {"cluster", "quality", "exemplars", "optional", "merge", "cap"}
    for name, _info, hints in _ui_fields():
        g = hints["ui_group"]
        assert g in valid_groups, (
            f"FCParams.{name} declares unknown ui_group={g!r}; "
            f"must be one of {sorted(valid_groups)}"
        )


def test_ui_orders_are_unique_within_group():
    """Two fields in the same group must not share the same ui_order
    — otherwise the layout is non-deterministic."""
    seen: dict[tuple[str, int], str] = {}
    for name, _info, hints in _ui_fields():
        key = (hints["ui_group"], hints["ui_order"])
        if key in seen:
            pytest.fail(
                f"FCParams.{name} and FCParams.{seen[key]} both declare "
                f"ui_group={key[0]!r} ui_order={key[1]}"
            )
        seen[key] = name
