"""spec-042 H2 — generic run table component.

Renders a list of row objects as a Streamlit dataframe with single-row
selection. Driven by a ``ColumnSpec`` list — works for any tabular
view (History, future tabs).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd
import streamlit as st

from face_cluster.views._specs import ColumnSpec, rows_to_records


def render_run_table(
    rows: Sequence[Any],
    columns: Sequence[ColumnSpec],
    *,
    key: str = "v2_run_table",
) -> Optional[int]:
    """Render a single-row-selectable dataframe over ``rows``.

    Args:
        rows: list of row objects (e.g., ``RunRow``). Each row must
            expose attributes named by the ``ColumnSpec.field`` /
            ``fallback_fields`` entries, plus an ``id`` attribute (used
            as the selection return value).
        columns: declarative column spec; controls what shows and how.
        key: Streamlit widget key. Override when rendering multiple
            tables on the same page (e.g., runs + actions).

    Returns:
        The selected row's ``id``, or None when no row is selected.

    Side effects: writes to ``st.session_state[key]`` via Streamlit's
    selection callback.
    """
    if not rows:
        st.info("No runs match the current filters.")
        return None

    records = rows_to_records(rows, columns)
    df = pd.DataFrame(records)
    event = st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    sel_indices = (
        event.selection.get("rows", []) if hasattr(event, "selection") else []
    )
    if not sel_indices:
        return None
    selected_row = rows[sel_indices[0]]
    return getattr(selected_row, "id", None)
