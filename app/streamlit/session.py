"""Session state management for Streamlit app."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import streamlit as st

from .models import Album, PipelineProgress, PipelineStatus, Person


@dataclass
class SessionState:
    """Centralized session state container."""

    # Current album
    current_album_id: Optional[str] = None
    current_album: Optional[Album] = None

    # Pipeline state
    pipeline_status: PipelineStatus = PipelineStatus.IDLE
    pipeline_progress: Optional[PipelineProgress] = None
    pipeline_error: Optional[str] = None

    # Cached data
    albums: List[Album] = field(default_factory=list)
    people: List[Person] = field(default_factory=list)

    # UI state
    selected_cluster_id: Optional[int] = None
    selected_person_id: Optional[str] = None
    filter_selected_only: bool = False
    view_mode: str = "grid"  # grid, list, clusters

    # Notifications
    notifications: List[Dict[str, Any]] = field(default_factory=list)

    # API connection
    api_connected: bool = False
    last_error: Optional[str] = None


def get_session() -> SessionState:
    """Get or create session state."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = SessionState()
    return st.session_state.app_state


def reset_session() -> None:
    """Reset session state to defaults."""
    st.session_state.app_state = SessionState()


def set_current_album(album: Album) -> None:
    """Set the current album."""
    state = get_session()
    state.current_album = album
    state.current_album_id = album.album_id
    # Reset related state
    state.selected_cluster_id = None
    state.selected_person_id = None


def clear_current_album() -> None:
    """Clear the current album."""
    state = get_session()
    state.current_album = None
    state.current_album_id = None
    state.selected_cluster_id = None
    state.selected_person_id = None
    state.pipeline_progress = None
    state.pipeline_status = PipelineStatus.IDLE


def update_pipeline_progress(progress: PipelineProgress) -> None:
    """Update pipeline progress in session."""
    state = get_session()
    state.pipeline_progress = progress
    state.pipeline_status = progress.status


def set_pipeline_error(error: str) -> None:
    """Set pipeline error in session."""
    state = get_session()
    state.pipeline_error = error
    state.pipeline_status = PipelineStatus.FAILED


def clear_pipeline_state() -> None:
    """Clear pipeline state."""
    state = get_session()
    state.pipeline_progress = None
    state.pipeline_error = None
    state.pipeline_status = PipelineStatus.IDLE


def add_notification(message: str, type: str = "info") -> None:
    """Add a notification to show the user.

    Args:
        message: The notification message
        type: One of 'info', 'success', 'warning', 'error'
    """
    state = get_session()
    state.notifications.append({"message": message, "type": type})


def pop_notifications() -> List[Dict[str, Any]]:
    """Get and clear all notifications."""
    state = get_session()
    notifications = state.notifications.copy()
    state.notifications = []
    return notifications


def set_api_connected(connected: bool) -> None:
    """Set API connection status."""
    state = get_session()
    state.api_connected = connected


def set_error(error: str) -> None:
    """Set last error message."""
    state = get_session()
    state.last_error = error


def clear_error() -> None:
    """Clear last error message."""
    state = get_session()
    state.last_error = None
