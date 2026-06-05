"""spec-071 — gate-badge strip for a merge decision.

Renders ✓/✗ per gate (cross / exemplar / support / margin / diameter) with the
measured value-vs-threshold detail beneath. Pure presentation over the typed
``GateBadge`` list the service produces.
"""
from __future__ import annotations

from typing import List

import streamlit as st

from face_cluster.views.merged_clusters import GateBadge


def render_merge_gate_badges(badges: List[GateBadge]) -> None:
    """One column per gate: a coloured ✓/✗/– header + the detail caption."""
    if not badges:
        return
    cols = st.columns(len(badges))
    for col, b in zip(cols, badges):
        if b.passed is True:
            icon, colour = "✓", "green"
        elif b.passed is False:
            icon, colour = "✗", "red"
        else:
            icon, colour = "–", "gray"
        col.markdown(f"**:{colour}[{icon} {b.name}]**")
        col.caption(b.detail)
