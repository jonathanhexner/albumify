"""Sidebar UI component."""

import streamlit as st
import logging

from ...config import Settings, AgentType
from ...core.services import AgentService, ImageService
from ...core.exceptions import AgentError, ImageLoadError, ValidationError
from ...state import StateManager
from .status import render_status_panel

logger = logging.getLogger(__name__)


def render_sidebar() -> None:
    """Render complete sidebar with all controls."""
    with st.sidebar:
        st.title("⚙️ Setup")

        _render_agent_section()
        st.divider()

        _render_image_section()
        st.divider()

        render_status_panel()
        st.divider()

        _render_controls_section()


def _render_agent_section() -> None:
    """Render agent initialization section."""
    st.subheader("1️⃣ Initialize Agent")

    # Agent type selection
    available_agents = Settings.get_available_agents()

    agent_type = st.selectbox(
        "Agent Type",
        options=list(available_agents.keys()),
        format_func=lambda x: available_agents[x].name,
        help="Template agent is recommended - no API key required"
    )

    # Show description
    agent_config = available_agents[agent_type]
    st.caption(f"ℹ️ {agent_config.description}")

    if agent_config.requires_api:
        st.warning("⚠️ This agent requires an API key")

    # Initialize button
    if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
        _handle_agent_initialization(agent_type)


def _handle_agent_initialization(agent_type: AgentType) -> None:
    """Handle agent initialization."""
    try:
        with st.spinner(f"Initializing {agent_type.value} agent..."):
            agent_instance = AgentService.initialize_agent(agent_type)
            StateManager.set_agent(agent_instance)
            st.success(f"✅ Agent initialized: {agent_instance.agent_type}")
            st.rerun()

    except AgentError as e:
        st.error(f"❌ {str(e)}")
        logger.error(f"Agent initialization failed: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.exception("Unexpected error during agent initialization")


def _render_image_section() -> None:
    """Render image directory input section."""
    st.subheader("2️⃣ Load Images")

    # Get current directory from state
    current_library = StateManager.get_image_library()
    current_dir = str(current_library.directory) if current_library else ""

    directory = st.text_input(
        "Image Directory Path",
        value=current_dir,
        placeholder="D:/Photos/MyVacation",
        help="Full path to your photo folder"
    )

    # Load button
    if directory and directory != current_dir:
        if st.button("📁 Load Images", use_container_width=True):
            _handle_image_loading(directory)

    # Show current status
    if StateManager.has_images():
        count = StateManager.get_image_count()
        st.success(f"✅ {count} images loaded")


def _handle_image_loading(directory: str) -> None:
    """Handle image directory loading."""
    try:
        with st.spinner("Loading images..."):
            library = ImageService.load_directory(directory)
            StateManager.set_image_library(library)
            st.success(f"✅ Loaded {library.count} images")
            st.rerun()

    except (ImageLoadError, ValidationError) as e:
        st.error(f"❌ {str(e)}")
        logger.error(f"Image loading failed: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.exception("Unexpected error during image loading")


def _render_controls_section() -> None:
    """Render control buttons."""
    st.subheader("🎛️ Controls")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            StateManager.clear_chat_only()
            st.rerun()

    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            StateManager.reset_all()
            st.rerun()
