"""spec-041 — Streamlit widget factory.

Joins two sources of truth:

* ``face_cluster.fc_params.FCParams`` — *the contract*. Owns range
  bounds (``ge``/``le``), defaults, descriptions. Streamlit-unaware.
* ``app.face_clustering_v2.ui_spec`` — *how to render*. Owns widget
  type, label, group, order, step, sentinel flag. Carries NO range
  or default; would not type-check if it tried.

For each field, the factory looks up both, then makes the right
``st.<widget>`` call. There is no third place where a range could be
declared, so drift between "what's legal" and "what the widget shows"
is structurally impossible.

Deterministic widget key: ``f"v2_{field_name}"``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import streamlit as st
from annotated_types import Ge, Gt, Le, Lt
from pydantic.fields import FieldInfo

from app.face_clustering_v2.ui_spec import UI_SPEC, FieldUI, fields_by_group
from face_cluster.fc_params import FCParams

WIDGET_KEY_PREFIX = "v2_"


def widget_key(field_name: str, *, key_prefix: str = "") -> str:
    """Deterministic Streamlit widget key for an FCParams field.

    ``key_prefix`` namespaces the key per-tab so two tabs can each render
    the same FCParams field without colliding. Default ``""`` matches
    legacy ``v2_<field>`` keys — load-bearing because the History tab's
    "Load Run" populates session_state under these keys.
    """
    return WIDGET_KEY_PREFIX + key_prefix + field_name


def _bounds(info: FieldInfo) -> Tuple[Optional[float], Optional[float]]:
    """Extract (min, max) from Pydantic's annotated-types metadata.

    Returns (None, None) when the field has no Ge/Le (e.g., a bool or
    an int without explicit bounds). Gt/Lt (exclusive) are treated as
    inclusive — none of our FCParams fields use them today.
    """
    lo: Optional[float] = None
    hi: Optional[float] = None
    for m in info.metadata:
        if isinstance(m, Ge):
            lo = m.ge
        elif isinstance(m, Gt):
            lo = m.gt
        elif isinstance(m, Le):
            hi = m.le
        elif isinstance(m, Lt):
            hi = m.lt
    return lo, hi


def _zero_for(step: Any) -> Any:
    """Type-matching zero for Optional[int|float] widgets with the 'off' sentinel."""
    return 0 if isinstance(step, int) else 0.0


def _ui_default(field_name: str) -> Any:
    """The value the widget shows when no session_state entry exists.

    For ``zero_is_none`` fields, ``None`` displays as 0 / 0.0.
    """
    info = FCParams.model_fields[field_name]
    spec = UI_SPEC[field_name]
    default = info.default
    if spec.zero_is_none and default is None:
        return _zero_for(spec.step)
    return default


def render_field(
    field_name: str,
    *,
    readonly: bool = False,
    value_override: Any = None,
    key_prefix: str = "",
) -> Any:
    """Render the widget for an FCParams field and return its value.

    Args:
        field_name: name of a UI-bound field on ``FCParams``.
        readonly: when True, render as a non-editable display. Used by
            the History tab to show "what config this run used". Pure
            label/value pair (st.text); not an input. No session_state
            interaction. Returns ``value_override`` (or the field default).
        value_override: when ``readonly=True``, the value to display.
            Ignored in editable mode (session_state and default own that).

    Raises:
        KeyError if the field has no ``UI_SPEC`` entry (caller bug).
    """
    if field_name not in UI_SPEC:
        raise KeyError(
            f"FCParams field {field_name!r} has no UI_SPEC entry — "
            "add one in app/face_clustering_v2/ui_spec.py or stop calling "
            "render_field for it."
        )
    info = FCParams.model_fields[field_name]
    spec: FieldUI = UI_SPEC[field_name]

    # Readonly path: just a label + value display. No widget, no
    # session_state, no validation bounds. Used by view-side
    # "read past run's config" displays — see HISTORY_TAB_PRD §7.
    if readonly:
        display_value = value_override if value_override is not None else info.default
        if spec.zero_is_none and display_value is None:
            display_value = _zero_for(spec.step)
        # Render as a label : value line. st.text preserves alignment;
        # st.markdown could be used for richer formatting later.
        st.text(f"{spec.label}: {display_value}")
        return display_value

    lo, hi = _bounds(info)
    label = spec.label
    help_text = info.description or None
    key = widget_key(field_name, key_prefix=key_prefix)
    default = _ui_default(field_name)

    # When ``key`` is already in session_state (Load button populated it,
    # or the user interacted with the widget previously), session_state
    # owns the value and passing ``value=`` triggers a Streamlit warning.
    # Only seed ``value=`` on the very first render of this widget.
    seed_default = key not in st.session_state

    if spec.widget == "checkbox":
        if seed_default:
            return st.checkbox(label, value=bool(default), key=key, help=help_text)
        return st.checkbox(label, key=key, help=help_text)

    kwargs: dict[str, Any] = {
        "label": label,
        "key": key,
        "help": help_text,
    }
    if seed_default:
        kwargs["value"] = default
    if lo is not None:
        kwargs["min_value"] = lo
    if hi is not None:
        kwargs["max_value"] = hi
    if spec.step is not None:
        kwargs["step"] = spec.step

    if spec.widget == "number_input":
        return st.number_input(**kwargs)
    if spec.widget == "slider":
        return st.slider(**kwargs)
    raise ValueError(f"Unknown UI_SPEC widget {spec.widget!r} on {field_name!r}")


def render_group(
    group: str, *, columns: Optional[int] = None, key_prefix: str = "",
) -> None:
    """Render every field in ``UI_SPEC`` tagged with this group, in order."""
    fields = fields_by_group().get(group, [])
    if not fields:
        return
    if columns is None:
        for name in fields:
            render_field(name, key_prefix=key_prefix)
        return
    cols = st.columns(columns)
    for i, name in enumerate(fields):
        with cols[i % columns]:
            render_field(name, key_prefix=key_prefix)


def value_from_state(field_name: str, *, key_prefix: str = "") -> Any:
    """Read the widget value from session_state, applying zero→None for sentinels."""
    spec = UI_SPEC.get(field_name)
    key = widget_key(field_name, key_prefix=key_prefix)
    if key not in st.session_state:
        return None
    v = st.session_state[key]
    if spec is not None and spec.zero_is_none and (v == 0 or v == 0.0):
        return None
    return v


def build_params_from_state(*, key_prefix: str = "") -> Optional[FCParams]:
    """Reconstruct an FCParams from the current widget session_state.

    Renders ``st.error`` and returns None on ValidationError. Fields
    without a UI_SPEC entry are left at their FCParams default.
    """
    from pydantic import ValidationError
    payload: dict[str, Any] = {}
    for name in UI_SPEC:
        if widget_key(name, key_prefix=key_prefix) in st.session_state:
            payload[name] = value_from_state(name, key_prefix=key_prefix)
    try:
        return FCParams(**payload)
    except ValidationError as e:
        st.error(f"Invalid configuration:\n```\n{e}\n```")
        return None


def load_params_into_state(params: FCParams, *, key_prefix: str = "") -> None:
    """Push every UI-bound field of ``params`` into ``st.session_state``."""
    dumped = params.model_dump()
    for name, spec in UI_SPEC.items():
        if name not in dumped:
            continue
        value = dumped[name]
        if spec.zero_is_none and value is None:
            value = _zero_for(spec.step)
        st.session_state[widget_key(name, key_prefix=key_prefix)] = value
