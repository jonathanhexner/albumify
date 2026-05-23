"""spec-041 follow-up — drift guard between ``UI_SPEC`` and ``FCParams``.

The split that landed after the yaw_max=999 crash:

* ``face_cluster/fc_params.py`` owns the validation contract.
* ``app/face_clustering_v2/ui_spec.py`` owns the UI rendering hints.

Drift between them is now structurally impossible for ranges (UI_SPEC
has no min/max field). What's left to check:

1. Every name in ``UI_SPEC`` references a real FCParams field.
2. Every ``zero_is_none`` flag is on a genuinely Optional field whose
   FCParams default is ``None``.
3. The widget choice is consistent with the field's Python type
   (``checkbox`` only for bool, etc.).
4. Order numbers are unique within a group (deterministic layout).
"""
from __future__ import annotations

import pytest

from app.face_clustering_v2.ui_spec import UI_SPEC
from face_cluster.fc_params import FCParams


def test_every_ui_spec_entry_references_a_real_fcparams_field():
    missing = set(UI_SPEC) - set(FCParams.model_fields)
    assert not missing, (
        f"UI_SPEC has entries with no matching FCParams field: {sorted(missing)}"
    )


def test_zero_is_none_only_on_optional_fields_with_none_default():
    for name, spec in UI_SPEC.items():
        if not spec.zero_is_none:
            continue
        info = FCParams.model_fields[name]
        assert info.default is None, (
            f"UI_SPEC[{name!r}].zero_is_none=True but FCParams default is "
            f"{info.default!r}, not None. The sentinel doesn't apply."
        )


def test_checkbox_widget_only_on_bool_fields():
    for name, spec in UI_SPEC.items():
        if spec.widget != "checkbox":
            continue
        ann = FCParams.model_fields[name].annotation
        assert ann is bool, (
            f"UI_SPEC[{name!r}].widget='checkbox' but FCParams annotation "
            f"is {ann!r}, not bool."
        )


def test_numeric_widget_not_on_bool_fields():
    for name, spec in UI_SPEC.items():
        if spec.widget in ("slider", "number_input"):
            ann = FCParams.model_fields[name].annotation
            assert ann is not bool, (
                f"UI_SPEC[{name!r}].widget={spec.widget!r} but the field is a bool."
            )


def test_orders_unique_within_group():
    seen: dict[tuple[str, int], str] = {}
    for name, spec in UI_SPEC.items():
        key = (spec.group, spec.order)
        if key in seen:
            pytest.fail(
                f"UI_SPEC[{name!r}] and UI_SPEC[{seen[key]!r}] both declare "
                f"group={key[0]!r} order={key[1]}"
            )
        seen[key] = name


def test_groups_are_known():
    valid = {"cluster", "quality", "exemplars", "optional", "merge", "cap"}
    for name, spec in UI_SPEC.items():
        assert spec.group in valid, (
            f"UI_SPEC[{name!r}].group={spec.group!r} not in {sorted(valid)}"
        )


def test_no_ui_spec_field_carries_a_min_or_max():
    """Structural guard: FieldUI has no min/max attribute. If someone adds
    one later, this test fails so they remember why it's absent."""
    from app.face_clustering_v2.ui_spec import FieldUI
    forbidden = {"min", "max", "min_value", "max_value", "ge", "le", "default"}
    declared = {f.name for f in __import__('dataclasses').fields(FieldUI)}
    overlap = declared & forbidden
    assert not overlap, (
        f"FieldUI declares {sorted(overlap)} — those belong on FCParams, "
        "not on UI_SPEC. Don't re-introduce the drift we just removed."
    )
