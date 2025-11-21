"""Chat interface components."""

import streamlit as st
import logging
from typing import Optional

from ...state import StateManager
from ...core.services import AgentService, ConversationService
from ...core.models import MessageRole
from ...core.exceptions import AgentError, ValidationError
from ...config.constants import EXAMPLE_QUERIES

logger = logging.getLogger(__name__)


def render_welcome_screen() -> None:
    """Render welcome screen when app is not ready."""
    st.info("👈 **Get Started:** Follow the steps in the sidebar to begin")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✨ What This App Does")
        st.markdown("""
- **Event Clustering** - Group photos by events (vacations, parties, etc.)
- **Face Recognition** - Organize by people in your photos
- **Landmark Detection** - Sort travel photos by location
- **Quality Assessment** - Find your best photos automatically
- **Smart Tagging** - Automatic scene and object detection
        """)

    with col2:
        st.markdown("### 🚀 How to Get Started")
        st.markdown("""
1. **Initialize Agent** - Select "Template Agent" and click initialize
2. **Load Images** - Enter your photo directory path
3. **Ask Questions** - Use natural language to organize your photos
4. **Get Results** - Agent will process and organize your photos
        """)

    # Show example queries
    st.markdown("### 💡 Example Questions You Can Ask")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUERIES):
        with cols[i % 2]:
            st.markdown(f"- {example}")


def render_chat_interface() -> None:
    """Render main chat interface."""
    st.markdown("---")
    st.subheader("💬 Chat with Your Photos")

    # Render conversation history
    _render_conversation_history()

    # Render example queries if no conversation yet
    if not StateManager.has_conversation():
        _render_example_queries()

    # Chat input
    user_input = st.chat_input(
        "Ask me to organize your photos...",
        disabled=not StateManager.is_ready()
    )

    if user_input:
        _handle_user_query(user_input)


def _render_conversation_history() -> None:
    """Render all messages in conversation."""
    conversation = StateManager.get_conversation()

    if not conversation:
        st.info("💭 No conversation yet. Start by asking a question below or clicking an example.")
        return

    for message in conversation:
        _render_message(message)


def _render_message(message) -> None:
    """Render a single chat message."""
    if message.role == MessageRole.USER:
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

            # Show workflow details if present
            workflow = message.metadata.get('workflow')
            if workflow:
                with st.expander("📋 Workflow Details", expanded=False):
                    _render_workflow_details(workflow)


def _render_workflow_details(workflow) -> None:
    """Render workflow execution details."""
    st.markdown(f"**Workflow:** `{workflow.get('name', 'Unknown')}`")

    if 'description' in workflow:
        st.caption(workflow['description'])

    # Show steps if available
    steps = workflow.get('steps', [])
    if steps:
        st.markdown("**Steps:**")
        for i, step in enumerate(steps, 1):
            status = step.get('status', 'unknown')
            status_icon = {
                'completed': '✅',
                'in_progress': '⏳',
                'pending': '⏸️',
                'failed': '❌',
                'skipped': '⏭️'
            }.get(status, '❓')

            st.markdown(f"{status_icon} **Step {i}:** {step.get('name', 'Unknown')}")


def _render_example_queries() -> None:
    """Render example query buttons."""
    st.markdown("### 💡 Try an Example")

    cols = st.columns(2)

    for i, example in enumerate(EXAMPLE_QUERIES):
        with cols[i % 2]:
            if st.button(
                example,
                key=f"example_{i}",
                use_container_width=True,
                disabled=not StateManager.is_ready()
            ):
                _handle_user_query(example)
                st.rerun()


def _handle_user_query(query: str) -> None:
    """Handle user query submission."""
    try:
        # Add user message
        user_message = ConversationService.create_message(
            MessageRole.USER,
            query
        )
        StateManager.add_message(user_message)

        # Process query
        with st.spinner("Processing your request..."):
            agent = StateManager.get_agent()
            context = StateManager.get_context()

            response = AgentService.process_query(agent, query, context)

            # Format response
            if response.success:
                content = ConversationService.format_success_message(response)
                metadata = {'workflow': response.data.get('workflow')}
            else:
                content = ConversationService.format_error_message(
                    Exception(response.message)
                )
                metadata = {}

            # Add assistant message
            assistant_message = ConversationService.create_message(
                MessageRole.ASSISTANT,
                content,
                metadata
            )
            StateManager.add_message(assistant_message)

        st.rerun()

    except (AgentError, ValidationError) as e:
        error_msg = ConversationService.format_error_message(e)
        error_message = ConversationService.create_message(
            MessageRole.ASSISTANT,
            error_msg
        )
        StateManager.add_message(error_message)
        st.rerun()

    except Exception as e:
        logger.exception("Unexpected error processing query")
        st.error(f"❌ Unexpected error: {str(e)}")
