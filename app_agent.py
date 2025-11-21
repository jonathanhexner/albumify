"""
AI Photo Organization App - Clean, modular design.

Simple Streamlit interface for organizing photos using AI agents.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.config import setup_logging
from sim_bench.agent.factory import create_agent

# Import modular components
from app.config import PAGE_CONFIG, CUSTOM_CSS, EXAMPLE_QUERIES
from app import state
from app import components

# Setup
setup_logging()
st.set_page_config(
    page_title=PAGE_CONFIG['title'],
    page_icon=PAGE_CONFIG['icon'],
    layout=PAGE_CONFIG['layout']
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_agent(agent_type: str):
    """Initialize the selected agent."""
    with st.spinner(f"Initializing {agent_type} agent..."):
        try:
            agent = create_agent(agent_type=agent_type)
            state.set_agent(agent, agent_type)
            st.success(f"✓ Agent initialized: {type(agent).__name__}")
        except NotImplementedError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")


def process_query(query: str):
    """Process user query through agent."""
    agent = state.get_agent()

    if not agent:
        st.error("No agent initialized")
        return

    if not state.has_images():
        st.warning("⚠️ No images loaded. Please specify an image directory first.")
        return

    # Add user message
    state.add_message("user", query)

    # Get context and process
    context = state.get_context()

    with st.spinner("Processing..."):
        try:
            response = agent.process_query(query, context)

            if response and response.success:
                # Add agent response
                state.add_message(
                    "agent",
                    response.message,
                    workflow=response.data.get('workflow')
                )
            else:
                error_msg = response.message if response else "Unknown error"
                state.add_message("agent", f"❌ Error: {error_msg}")

        except Exception as e:
            state.add_message("agent", f"❌ Exception: {str(e)}")


def render_sidebar():
    """Render the sidebar with all controls."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Agent selection
        agent_type = components.render_agent_selector()

        if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
            init_agent(agent_type)

        st.divider()

        # Image input
        components.render_image_input()

        st.divider()

        # Status
        components.render_status_panel()

        st.divider()

        # Tools (collapsed by default)
        with st.expander("🔧 View Available Tools"):
            components.render_tools_panel()

        st.divider()

        # Session controls
        components.render_session_controls()


def main():
    """Main app entry point."""
    # Initialize state
    state.init_state()

    # App header
    st.title("📸 AI Photo Organization")
    st.caption("Organize your photos using natural language")

    # Sidebar
    render_sidebar()

    # Main content
    if not state.has_agent():
        # Welcome screen
        components.render_welcome_screen()

    else:
        # Chat interface
        st.markdown("---")

        # Show conversation
        components.render_conversation()

        # Example queries (only if no conversation yet)
        if len(state.get_conversation()) == 0:
            example_query = components.render_example_queries(EXAMPLE_QUERIES)
            if example_query:
                process_query(example_query)
                st.rerun()

        # Chat input
        user_input = st.chat_input("Ask me to organize your photos...")

        if user_input:
            process_query(user_input)
            st.rerun()


if __name__ == "__main__":
    main()
