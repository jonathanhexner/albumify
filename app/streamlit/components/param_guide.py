"""spec-090 follow-up: surface the profile-parameter tutorial inside the app.

Renders `docs/guides/profile_parameters.html` (a one-time reference page) inline in
an expander, so Configure & Run and the Results report can link to it without a
static file server.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_GUIDE_PATH = Path(__file__).resolve().parents[3] / "docs" / "guides" / "profile_parameters.html"


def render_param_guide_link(*, key: str, height: int = 720) -> None:
    """Show a collapsible "Parameter guide" that renders the tutorial HTML inline."""
    with st.expander("📖 Parameter guide — what each setting means", expanded=False):
        if _GUIDE_PATH.exists():
            components.html(_GUIDE_PATH.read_text(encoding="utf-8"), height=height, scrolling=True)
        else:
            st.info(
                "Parameter guide not generated yet "
                f"(expected at `{_GUIDE_PATH}`)."
            )
