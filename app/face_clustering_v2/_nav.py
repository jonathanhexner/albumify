"""spec-080 — session-state navigation for the v2 app.

Replaces ``st.tabs`` (which can't be switched programmatically) with a
horizontal radio bound to ``st.session_state['active_page']``. Two wins:

* ``navigate_to(name)`` lets any "Open" button switch the view (face grid →
  Face Analysis, Gallery → Cluster Analysis, …) — impossible with st.tabs.
* Only the ACTIVE page's render function runs each rerun, instead of all 11
  tab bodies — much lighter (the old eager render is what made the e2e slow).
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import streamlit as st

PAGE_KEY = "active_page"
_PENDING = "_pending_nav"


def navigate_to(page_name: str) -> None:
    """Switch to ``page_name`` on the next rerun (cross-view 'Open' actions).

    Sets a PENDING flag rather than the radio's key directly: the radio is
    already instantiated by ``render_nav`` earlier in the run, and Streamlit
    forbids mutating a live widget's key. ``render_nav`` applies the pending
    nav BEFORE it creates the radio next run.
    """
    st.session_state[_PENDING] = page_name
    st.rerun()


def render_nav(pages: List[Tuple[str, Callable[[], None]]]) -> None:
    """Render the top nav selector + the currently-active page.

    ``pages`` is an ordered list of ``(label, render_fn)``. Honors a pending
    ``navigate_to`` request and a one-time ``?page=<label>`` query param, both
    applied BEFORE the radio is created.
    """
    names = [name for name, _ in pages]
    pending = st.session_state.pop(_PENDING, None)
    if pending in names:
        st.session_state[PAGE_KEY] = pending
    qp = st.query_params.get("page")
    if qp in names and PAGE_KEY not in st.session_state:
        st.session_state[PAGE_KEY] = qp
    active = st.radio(
        "view", names, horizontal=True, key=PAGE_KEY, label_visibility="collapsed",
    )
    dict(pages)[active]()
