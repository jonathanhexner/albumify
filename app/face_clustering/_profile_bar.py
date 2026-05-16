"""Reusable profile load/save bars used by Run and Recluster tabs.

A profile is a JSON blob persisted by `face_cluster.profile_store.ProfileStore`
under `~/.sim_bench/profiles/<name>.json`.  The Run and Recluster tabs share
the same on-disk profile namespace but each owns a distinct subset of
session-state keys (`run_*` vs `rc_*`).  This helper takes the caller's
allow-list of session-state keys plus a widget-key prefix so the two tabs
can render identical-looking bars without colliding on Streamlit widget IDs.
"""
from __future__ import annotations

from typing import Iterable

import streamlit as st

from face_cluster.profile_store import ProfileStore


def render_profile_bar(param_keys: Iterable[str], widget_prefix: str) -> None:
    """Selectbox + Load button. Filters loaded profile to `param_keys`."""
    store = ProfileStore()
    names = store.list_names()
    if not names:
        return
    col_sel, col_load = st.columns([3, 1])
    sel_key  = f"{widget_prefix}prf_select"
    load_key = f"{widget_prefix}prf_load"
    with col_sel:
        selected = st.selectbox("Load profile", ["(none)"] + names, key=sel_key)
    with col_load:
        st.write("")
        if st.button("Load", key=load_key) and selected != "(none)":
            params = store.load(selected)
            allowed = set(param_keys)
            for k, v in params.items():
                if k in allowed:
                    st.session_state[k] = v
            st.toast(f"Loaded profile '{selected}'")
            st.rerun()


def render_profile_save_bar(param_keys: Iterable[str], widget_prefix: str) -> None:
    """Name input + Save / Save-as-default buttons.  Persists only `param_keys`."""
    col_name, col_save, col_default = st.columns([2, 1, 1])
    name_key    = f"{widget_prefix}prf_name"
    save_key    = f"{widget_prefix}prf_save"
    default_key = f"{widget_prefix}prf_default"
    allowed     = set(param_keys)
    snapshot    = lambda: {k: st.session_state[k] for k in allowed if k in st.session_state}
    with col_name:
        st.text_input("Profile name", key=name_key,
                      label_visibility="collapsed", placeholder="profile name")
    with col_save:
        if st.button("Save profile", key=save_key):
            name = st.session_state.get(name_key, "").strip()
            if name:
                ProfileStore().save(name, snapshot())
                st.toast(f"Saved profile '{name}'")
    with col_default:
        if st.button("Save as default", key=default_key):
            ProfileStore().save("default", snapshot())
            st.toast("Saved as default")
