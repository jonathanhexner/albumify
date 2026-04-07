"""Dynamic parameter slider component."""

from typing import Any, Dict, Optional

import streamlit as st


def render_param_sliders(
    param_definitions: Dict[str, Dict],
    current_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render sliders from parameter definitions.

    Args:
        param_definitions: Mapping of param_name → {type, min, max, default, optional}.
        current_values: Optional override for initial slider values.

    Returns:
        Dict of current parameter values.
    """
    values: Dict[str, Any] = {}
    for name, defn in param_definitions.items():
        is_optional = defn.get("optional", False)
        default = (current_values or {}).get(name, defn["default"])

        if is_optional:
            # Optional params: checkbox to enable, then slider if enabled
            col1, col2 = st.columns([1, 3])
            with col1:
                enabled = st.checkbox(f"Enable {name}", value=default is not None, key=f"{name}_enabled")
            if enabled:
                with col2:
                    if defn["type"] == "int":
                        values[name] = st.slider(name, int(defn["min"]), int(defn["max"]),
                                                 int(default) if default else int(defn["default"]),
                                                 key=f"{name}_slider", label_visibility="collapsed")
                    else:
                        values[name] = st.slider(name, float(defn["min"]), float(defn["max"]),
                                                 float(default) if default else float(defn["default"]),
                                                 step=0.05, key=f"{name}_slider", label_visibility="collapsed")
            else:
                values[name] = None
        else:
            # Required params: just render slider
            if defn["type"] == "int":
                values[name] = st.slider(name, int(defn["min"]), int(defn["max"]), int(default))
            else:
                values[name] = st.slider(name, float(defn["min"]), float(defn["max"]), float(default), step=0.05)
    return values
