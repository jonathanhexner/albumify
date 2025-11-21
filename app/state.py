"""
State management for the Streamlit app.

Handles all session state initialization and updates.
"""

import streamlit as st
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def init_state():
    """Initialize all session state variables."""
    defaults = {
        'agent': None,
        'agent_type': None,
        'conversation': [],
        'image_dir': "",
        'image_paths': [],
        'current_workflow': None,
        'initialized': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_agent(agent, agent_type: str):
    """Set the agent instance."""
    st.session_state.agent = agent
    st.session_state.agent_type = agent_type
    st.session_state.initialized = True


def add_message(role: str, content: str, workflow=None):
    """Add a message to the conversation."""
    st.session_state.conversation.append({
        'role': role,
        'content': content,
        'workflow': workflow,
        'timestamp': datetime.now()
    })


def set_image_directory(path: str) -> tuple[bool, str, List[str]]:
    """
    Set image directory and load images.

    Returns:
        (success, message, image_paths)
    """
    path_obj = Path(path)

    if not path_obj.exists():
        return False, f"Directory not found: {path}", []

    if not path_obj.is_dir():
        return False, f"Path is not a directory: {path}", []

    # Load images
    image_paths = []
    for pattern in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_paths.extend(path_obj.glob(pattern))

    image_paths = [str(p) for p in sorted(image_paths)]

    if not image_paths:
        return False, f"No images found in {path}", []

    # Update state
    st.session_state.image_dir = path
    st.session_state.image_paths = image_paths

    return True, f"Loaded {len(image_paths)} images", image_paths


def clear_conversation():
    """Clear conversation history."""
    st.session_state.conversation = []
    st.session_state.current_workflow = None


def reset_app():
    """Reset entire app state."""
    st.session_state.agent = None
    st.session_state.agent_type = None
    st.session_state.conversation = []
    st.session_state.current_workflow = None
    st.session_state.initialized = False


def get_context() -> dict:
    """Get current context for agent queries."""
    context = {}

    if st.session_state.image_paths:
        context['image_paths'] = st.session_state.image_paths

    if st.session_state.image_dir:
        context['image_dir'] = st.session_state.image_dir

    return context


# Convenience getters
def has_agent() -> bool:
    """Check if agent is initialized."""
    return st.session_state.agent is not None


def has_images() -> bool:
    """Check if images are loaded."""
    return len(st.session_state.image_paths) > 0


def get_agent():
    """Get current agent."""
    return st.session_state.agent


def get_conversation():
    """Get conversation history."""
    return st.session_state.conversation


def get_image_count() -> int:
    """Get number of loaded images."""
    return len(st.session_state.image_paths)
