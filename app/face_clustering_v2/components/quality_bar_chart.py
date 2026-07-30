"""spec-065 — stacked bar chart of per-gate pass / fail counts.

Pure rendering. Input is a list of ``GateCounts`` from
:class:`face_cluster.views.quality.QualityService`. No domain logic; no
DB access.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from face_cluster.views.quality import GateCounts


def render_quality_bar_chart(
    gates: Sequence[GateCounts],
    *,
    key: str = "v2_quality_bar_chart",
) -> None:
    """Render a stacked bar (pass green / reject red) per gate.

    Empty gates list renders an info message instead of an empty chart.
    """
    if not gates:
        st.info(
            "No filter_decisions recorded for this run. The Quality tab "
            "is empty until a run writes per-gate verdicts (spec-032)."
        )
        return

    long: list[dict] = []
    for g in gates:
        long.append({"gate": g.gate_name, "verdict": "passed",   "count": g.n_passed})
        long.append({"gate": g.gate_name, "verdict": "rejected", "count": g.n_rejected})
    df = pd.DataFrame(long)
    fig = px.bar(
        df,
        x="gate",
        y="count",
        color="verdict",
        barmode="stack",
        color_discrete_map={"passed": "#2ca02c", "rejected": "#d62728"},
        title="Per-gate pass / reject counts",
    )
    fig.update_layout(xaxis_title="", yaxis_title="items", legend_title="")
    st.plotly_chart(fig, use_container_width=True, key=key)


__all__ = ["render_quality_bar_chart"]
