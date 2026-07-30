"""spec-041 — profile save / load bar for the FC App v2 Run tab.

Profiles are ``FCParams`` JSON files in
``~/.sim_bench/profiles_v2/``. Round-trip:

* Load picks a file, reads it through ``FCParams.load(path)``, calls
  ``widget_factory.load_params_into_state(params)`` which writes every
  UI-bound field into ``st.session_state`` under its deterministic key,
  then triggers a rerun so the widgets pick up the new state.
* Save reads the current widget values via
  ``widget_factory.build_params_from_state()``, gets back a typed
  ``FCParams``, writes it via ``FCParams.save(path)``.

The v2 profile dir is distinct from the legacy FC App's
``~/.sim_bench/profiles/`` to avoid colliding with the old shape.

This module no longer holds a field→widget-key mapping. The factory
owns that mapping via ``widget_factory.widget_key(field_name)``, which
is just ``f"v2_{field_name}"``. Adding a new knob touches FCParams; this
bar requires no change.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from app.face_clustering_v2.widget_factory import (
    build_params_from_state,
    load_params_into_state,
)
from face_cluster.fc_params import FCParams

PROFILE_DIR = Path.home() / ".sim_bench" / "profiles_v2"


def render_profile_bar() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profiles = sorted(p.name for p in PROFILE_DIR.glob("*.json"))

    with st.expander("Profiles", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            options = ["(none)"] + profiles
            selected = st.selectbox(
                "Load profile", options=options, index=0,
                key="v2_profile_select",
                help=f"Profiles directory: {PROFILE_DIR}",
            )
        with c2:
            st.write("")
            st.write("")
            if st.button("Load", key="v2_profile_load_btn",
                         disabled=(selected == "(none)")):
                path = PROFILE_DIR / selected
                try:
                    params = FCParams.load(path)
                except (ValidationError, OSError) as e:
                    st.error(f"Could not load profile: {e}")
                else:
                    load_params_into_state(params)
                    st.session_state["v2_profile_last_loaded"] = selected
                    st.success(f"Loaded `{selected}` — widgets updated.")
                    st.rerun()

        c3, c4 = st.columns([2, 1])
        with c3:
            new_name = st.text_input(
                "Save current settings as",
                value="",
                placeholder="my_profile.json (or just my_profile)",
                key="v2_profile_save_name",
                help="Saved to the profiles directory above. .json extension auto-added.",
            )
        with c4:
            st.write("")
            st.write("")
            if st.button("Save", key="v2_profile_save_btn",
                         disabled=(not new_name.strip())):
                params = build_params_from_state()
                if params is not None:
                    name = new_name.strip()
                    if not name.endswith(".json"):
                        name += ".json"
                    path = PROFILE_DIR / name
                    try:
                        params.save(path)
                    except OSError as e:
                        st.error(f"Could not save profile: {e}")
                    else:
                        st.success(f"Saved `{name}` to {PROFILE_DIR}")
                        st.rerun()
