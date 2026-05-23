"""spec-041 — generic Streamlit widget factory driven by ``FCParams``.

Every UI-bound field on ``face_cluster.fc_params.FCParams`` carries a
``json_schema_extra`` block with the display hints (widget type,
display range, step, label, help, group, order). This module reads
those hints and renders the right ``st.<widget>`` call. No widget
literal is hand-written anywhere; adding a knob means adding a Field
on ``FCParams`` — no UI change needed.

Deterministic widget key: ``f"v2_{field_name}"``. That lets
``_profile_bar.py`` set ``st.session_state["v2_<field>"] = value`` for
every field without a name-mapping table.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from face_cluster.fc_params import FCParams

WIDGET_KEY_PREFIX = "v2_"


def widget_key(field_name: str) -> str:
    """Deterministic Streamlit widget key for an FCParams field."""
    return WIDGET_KEY_PREFIX + field_name


def _hints(field_name: str) -> Dict[str, Any]:
    """Return the json_schema_extra hints dict (empty if field has none)."""
    info = FCParams.model_fields[field_name]
    extra = info.json_schema_extra
    if not isinstance(extra, dict):
        return {}
    return extra


def _ui_default(field_name: str) -> Any:
    """The value the widget shows when no session_state entry exists.

    For ``ui_zero_is_none`` fields, ``None`` displays as 0 / 0.0 — that's
    the UI sentinel for "disabled".
    """
    info = FCParams.model_fields[field_name]
    h = _hints(field_name)
    default = info.default if info.default is not None else None
    if h.get("ui_zero_is_none") and default is None:
        # type-aware zero
        if h.get("ui_widget") == "number_input":
            step = h.get("ui_step")
            return 0 if isinstance(step, int) else 0.0
    return default


def render_field(field_name: str) -> Any:
    """Render the widget for an FCParams field and return its value.

    Raises KeyError if the field has no UI hints — caller bug.
    """
    info = FCParams.model_fields[field_name]
    h = _hints(field_name)
    if not h:
        raise KeyError(f"FCParams field {field_name!r} has no UI hints")

    widget = h["ui_widget"]
    label = h.get("ui_label", field_name)
    help_text = h.get("ui_help", "") or None
    key = widget_key(field_name)
    # Default value used only on first render — once the widget owns a
    # session_state entry, Streamlit ignores `value=`.
    default = _ui_default(field_name)

    if widget == "checkbox":
        return st.checkbox(label, value=bool(default), key=key, help=help_text)

    if widget == "number_input":
        kwargs: Dict[str, Any] = {
            "label": label,
            "key": key,
            "help": help_text,
        }
        if "ui_min" in h:
            kwargs["min_value"] = h["ui_min"]
        if "ui_max" in h:
            kwargs["max_value"] = h["ui_max"]
        if "ui_step" in h:
            kwargs["step"] = h["ui_step"]
        kwargs["value"] = default
        return st.number_input(**kwargs)

    if widget == "slider":
        kwargs = {
            "label": label,
            "key": key,
            "help": help_text,
        }
        if "ui_min" in h:
            kwargs["min_value"] = h["ui_min"]
        if "ui_max" in h:
            kwargs["max_value"] = h["ui_max"]
        if "ui_step" in h:
            kwargs["step"] = h["ui_step"]
        kwargs["value"] = default
        return st.slider(**kwargs)

    raise ValueError(f"Unknown ui_widget {widget!r} on field {field_name!r}")


def render_group(group: str, *, columns: Optional[int] = None) -> None:
    """Render every field tagged with ``ui_group=group`` in order.

    When ``columns`` is given, lays out widgets across N columns; otherwise
    one widget per row.
    """
    fields = FCParams.ui_fields_by_group().get(group, [])
    if not fields:
        return
    if columns is None:
        for name in fields:
            render_field(name)
        return
    cols = st.columns(columns)
    for i, name in enumerate(fields):
        with cols[i % columns]:
            render_field(name)


def value_from_state(field_name: str) -> Any:
    """Read the widget value from session_state, applying zero→None for sentinels."""
    h = _hints(field_name)
    key = widget_key(field_name)
    if key not in st.session_state:
        return None
    v = st.session_state[key]
    if h.get("ui_zero_is_none"):
        if v == 0 or v == 0.0:
            return None
    return v


def build_params_from_state() -> Optional[FCParams]:
    """Reconstruct an FCParams from the current widget session_state.

    Renders ``st.error`` and returns None on ValidationError. Fields
    without UI hints are left at their FCParams default.
    """
    from pydantic import ValidationError
    payload: Dict[str, Any] = {}
    for name, info in FCParams.model_fields.items():
        h = info.json_schema_extra
        if not isinstance(h, dict) or "ui_widget" not in h:
            continue
        key = widget_key(name)
        if key not in st.session_state:
            continue
        payload[name] = value_from_state(name)
    try:
        return FCParams(**payload)
    except ValidationError as e:
        st.error(f"Invalid configuration:\n```\n{e}\n```")
        return None


def load_params_into_state(params: FCParams) -> None:
    """Push every UI-bound field of ``params`` into ``st.session_state``."""
    for name, value in params.model_dump().items():
        info = FCParams.model_fields.get(name)
        if info is None:
            continue
        h = info.json_schema_extra
        if not isinstance(h, dict) or "ui_widget" not in h:
            continue
        key = widget_key(name)
        if h.get("ui_zero_is_none") and value is None:
            value = 0 if isinstance(h.get("ui_step"), int) else 0.0
        st.session_state[key] = value
