"""Status display component."""

import streamlit as st

from ...state import StateManager
from ...config import Settings


def render_status_panel() -> None:
    """Render application status panel."""
    st.subheader("📊 Status")

    _render_agent_status()
    _render_image_status()


def _render_agent_status() -> None:
    """Render agent initialization status."""
    if StateManager.has_agent():
        agent = StateManager.get_agent()
        agent_config = Settings.AGENTS.get(agent.agent_type)
        agent_name = agent_config.name if agent_config else agent.agent_type

        st.markdown(
            f'<p class="status-ready">✅ Agent: {agent_name}</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-warning">⚠️ Agent: Not initialized</p>',
            unsafe_allow_html=True
        )


def _render_image_status() -> None:
    """Render image library status."""
    if StateManager.has_images():
        count = StateManager.get_image_count()
        st.markdown(
            f'<p class="status-ready">✅ Images: {count} loaded</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-warning">⚠️ Images: None loaded</p>',
            unsafe_allow_html=True
        )
