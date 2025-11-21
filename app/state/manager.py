"""State manager - bridges Streamlit session state with core models."""

import streamlit as st
from typing import Optional, List

from ..core.models import (
    AppState, AgentInstance, ImageLibrary,
    ChatMessage, MessageRole
)


class StateManager:
    """
    Manages application state using Streamlit session state.

    This class provides a clean interface between Streamlit's session state
    and our domain models, ensuring type safety and encapsulation.
    """

    STATE_KEY = '_app_state'

    @classmethod
    def initialize(cls) -> None:
        """Initialize state if not already initialized."""
        if cls.STATE_KEY not in st.session_state:
            st.session_state[cls.STATE_KEY] = AppState()

    @classmethod
    def get_state(cls) -> AppState:
        """Get current application state."""
        cls.initialize()
        return st.session_state[cls.STATE_KEY]

    # ========================================================================
    # Agent Operations
    # ========================================================================

    @classmethod
    def set_agent(cls, agent_instance: AgentInstance) -> None:
        """Set the current agent instance."""
        state = cls.get_state()
        state.agent = agent_instance

    @classmethod
    def get_agent(cls) -> Optional[AgentInstance]:
        """Get current agent instance."""
        return cls.get_state().agent

    @classmethod
    def has_agent(cls) -> bool:
        """Check if agent is initialized."""
        return cls.get_state().has_agent

    @classmethod
    def clear_agent(cls) -> None:
        """Clear agent instance."""
        state = cls.get_state()
        state.agent = None

    # ========================================================================
    # Image Library Operations
    # ========================================================================

    @classmethod
    def set_image_library(cls, library: ImageLibrary) -> None:
        """Set the current image library."""
        state = cls.get_state()
        state.image_library = library

    @classmethod
    def get_image_library(cls) -> Optional[ImageLibrary]:
        """Get current image library."""
        return cls.get_state().image_library

    @classmethod
    def has_images(cls) -> bool:
        """Check if images are loaded."""
        return cls.get_state().has_images

    @classmethod
    def get_image_count(cls) -> int:
        """Get number of loaded images."""
        library = cls.get_image_library()
        return library.count if library else 0

    @classmethod
    def clear_images(cls) -> None:
        """Clear image library."""
        state = cls.get_state()
        state.image_library = None

    # ========================================================================
    # Conversation Operations
    # ========================================================================

    @classmethod
    def add_message(cls, message: ChatMessage) -> None:
        """Add a message to the conversation."""
        state = cls.get_state()
        state.conversation.append(message)

    @classmethod
    def get_conversation(cls) -> List[ChatMessage]:
        """Get conversation history."""
        return cls.get_state().conversation

    @classmethod
    def clear_conversation(cls) -> None:
        """Clear conversation history."""
        state = cls.get_state()
        state.conversation = []

    @classmethod
    def has_conversation(cls) -> bool:
        """Check if there are any messages."""
        return len(cls.get_conversation()) > 0

    # ========================================================================
    # Context Operations
    # ========================================================================

    @classmethod
    def get_context(cls) -> dict:
        """Get context for agent queries."""
        return cls.get_state().get_context()

    # ========================================================================
    # Utility Operations
    # ========================================================================

    @classmethod
    def is_ready(cls) -> bool:
        """Check if app is ready to process queries."""
        return cls.get_state().is_ready

    @classmethod
    def reset_all(cls) -> None:
        """Reset entire application state."""
        st.session_state[cls.STATE_KEY] = AppState()

    @classmethod
    def clear_chat_only(cls) -> None:
        """Clear only the conversation, keep agent and images."""
        cls.clear_conversation()
