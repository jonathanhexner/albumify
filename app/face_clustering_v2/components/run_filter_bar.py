"""spec-042 H2 — filter-bar component for the History tab.

Three side-by-side widgets that gather the user's filter inputs and
return a typed ``HistoryQuery``. Streamlit-only. No DB access.
"""
from __future__ import annotations

from datetime import date
from typing import Sequence

import streamlit as st

from face_cluster.views.history import HistoryQuery


def render_filter_bar(albums: Sequence[str]) -> HistoryQuery:
    """Render the filter bar (album, date range, free-text) for the History tab.

    Args:
        albums: list of distinct album names — used to populate the dropdown.
            Passed in so the component doesn't touch the service itself
            (separation of concerns: caller queries; we render).

    Returns:
        ``HistoryQuery`` reflecting the user's current selections. Empty
        widgets map to None / empty fields, meaning "no filter on that axis".

    Side effects: writes to ``st.session_state`` under widget keys
    ``hist_album``, ``hist_date_range``, ``hist_text``. The 2:2:3 column
    widths keep the wider text-search field anchored on the right.
    """
    col_album, col_date, col_search = st.columns([2, 2, 3])
    with col_album:
        album_choice = st.selectbox(
            "Album",
            options=["(all)"] + list(albums),
            index=0,
            key="hist_album",
        )
    with col_date:
        date_range = st.date_input(
            "Date range",
            value=[],
            key="hist_date_range",
        )
    with col_search:
        text = st.text_input(
            "Search (album / run name / comment)",
            key="hist_text",
        )

    return HistoryQuery(
        album=None if album_choice == "(all)" else album_choice,
        date_from=_pick_date(date_range, 0),
        date_to=_pick_date(date_range, 1),
        text=text.strip() or None,
    )


def _pick_date(date_range, index: int) -> date | None:
    """Safely extract a ``date`` from the st.date_input return value.

    ``st.date_input(value=[])`` returns either a tuple of dates, a
    single ``date``, or ``()`` depending on user interaction. This
    helper hides that quirk.
    """
    if isinstance(date_range, (list, tuple)) and len(date_range) > index:
        candidate = date_range[index]
        if isinstance(candidate, date):
            return candidate
    return None
