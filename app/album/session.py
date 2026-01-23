"""Session state management for album app.

Centralizes Streamlit session state handling.
Provides clean interface between UI and services.
"""

import streamlit as st
from typing import Optional, Dict, Any

from sim_bench.album import AlbumService, WorkflowResult
from sim_bench.config import get_global_config


class AlbumSession:
    """Manages session state for album organization app."""

    @staticmethod
    def initialize():
        """Initialize session state with defaults."""
        if 'album_service' not in st.session_state:
            st.session_state.album_service = None
        if 'workflow_result' not in st.session_state:
            st.session_state.workflow_result = None
        if 'config_overrides' not in st.session_state:
            st.session_state.config_overrides = {}

    @staticmethod
    def get_service(config_overrides: Optional[Dict[str, Any]] = None) -> AlbumService:
        """Get or create AlbumService with given config."""
        config = get_global_config().to_dict()

        if config_overrides:
            config = AlbumSession._deep_merge(config, config_overrides)

        # Always create fresh service with current config
        service = AlbumService(config)
        st.session_state.album_service = service
        return service

    @staticmethod
    def get_result() -> Optional[WorkflowResult]:
        """Get current workflow result."""
        return st.session_state.get('workflow_result')

    @staticmethod
    def set_result(result: WorkflowResult):
        """Store workflow result."""
        st.session_state.workflow_result = result

    @staticmethod
    def set_config_overrides(overrides: Dict[str, Any]):
        """Store configuration overrides."""
        st.session_state.config_overrides = overrides

    @staticmethod
    def get_config_overrides() -> Dict[str, Any]:
        """Get current configuration overrides."""
        return st.session_state.get('config_overrides', {})

    @staticmethod
    def clear():
        """Clear all session state."""
        st.session_state.album_service = None
        st.session_state.workflow_result = None
        st.session_state.config_overrides = {}

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = AlbumSession._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
