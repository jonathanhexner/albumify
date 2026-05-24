"""spec-042 H2 — recent-actions sub-table for the History tab.

Renders the list of non-pipeline actions (merge_apply, profile_save,
ml_train, model_load) plus a selectbox to inspect a payload.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import streamlit as st

from face_cluster.views._specs import rows_to_records
from face_cluster.views.history import (
    ACTION_COLUMNS,
    ActionRow,
    HistoryService,
)


def render_actions_table(
    actions: Sequence[ActionRow],
    service: HistoryService,
) -> None:
    """Render the recent-actions table + payload inspector.

    Args:
        actions: rows returned by ``HistoryService.list_other_actions()``.
        service: used to fetch the payload when a row is selected from
            the inspector dropdown.

    Side effects: writes to ``st.session_state["hist_action_selector"]``.
    """
    if not actions:
        st.info("No other actions recorded yet.")
        return

    records = rows_to_records(actions, ACTION_COLUMNS)
    df = pd.DataFrame(records)
    st.dataframe(df, hide_index=True, use_container_width=True)

    selected_id = _render_payload_selector(actions)
    if selected_id is not None:
        with st.expander("Full Payload (debug)", expanded=False):
            st.json(service.get_action_payload(selected_id))


def _render_payload_selector(actions: Sequence[ActionRow]) -> Optional[int]:
    """Selectbox to pick an action by (timestamp, type) and return its id."""
    if not actions:
        return None
    option_labels = {
        a.id: f"{(a.started_at or '')[:19]}  {a.action_type}"
        for a in actions
    }
    selected_id = st.selectbox(
        "Inspect action payload",
        options=list(option_labels.keys()),
        format_func=lambda aid: option_labels[aid],
        key="hist_action_selector",
    )
    return selected_id
