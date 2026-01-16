"""
AI Photo Organization App - Professional Architecture

Clear, simple, readable Streamlit app.
Every function does ONE thing and has a clear name.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, List
import logging

# Setup
from sim_bench.config import setup_logging
from sim_bench.agent.factory import create_agent
from sim_bench.agent.core.base import AgentResponse

setup_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Change these to customize the app
# ============================================================================

APP_TITLE = "📸 Photo Organization Assistant"
APP_SUBTITLE = "Organize your photos automatically"

AGENT_INFO = {
    "template": {
        "name": "Template Agent",
        "description": "Uses keyword matching to select pre-built workflows. No API key needed.",
        "available": True
    }
}

EXAMPLE_QUERIES = [
    "Organize my photos by event",
    "Find my best portraits",
    "Show me my vacation photos"
]

SUPPORTED_FORMATS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


# ============================================================================
# SESSION STATE - Simple state management
# ============================================================================

def init_session():
    """Initialize session state on first run."""
    if 'initialized' not in st.session_state:
        st.session_state.agent = None
        st.session_state.messages = []  # Chat history
        st.session_state.image_dir = ""
        st.session_state.image_paths = []
        st.session_state.initialized = True


def add_chat_message(role: str, content: str):
    """Add a message to chat history."""
    st.session_state.messages.append({"role": role, "content": content})


def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []


# ============================================================================
# CORE FUNCTIONS - Business logic
# ============================================================================

def initialize_agent() -> bool:
    """
    Initialize the agent.
    Returns True on success, False on failure.
    """
    try:
        st.session_state.agent = create_agent(agent_type='template')
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        st.error(f"Failed to initialize agent: {e}")
        return False


def load_images_from_directory(directory: str) -> tuple[bool, str, List[str]]:
    """
    Load images from directory.
    Returns: (success, message, image_paths)
    """
    path = Path(directory)

    if not path.exists():
        return False, f"Directory not found: {directory}", []

    if not path.is_dir():
        return False, f"Not a directory: {directory}", []

    # Find all images
    images = []
    for pattern in SUPPORTED_FORMATS:
        images.extend(path.glob(pattern))

    images = [str(p) for p in sorted(images)]

    if not images:
        return False, f"No images found in {directory}", []

    return True, f"Loaded {len(images)} images", images


def execute_query(query: str, image_paths: List[str]) -> AgentResponse:
    """
    Execute a query through the agent.

    Args:
        query: User's natural language query
        image_paths: List of image file paths

    Returns:
        AgentResponse with results or error
    """
    if not st.session_state.agent:
        return AgentResponse(
            success=False,
            message="Agent not initialized. Click 'Initialize Agent' first.",
            data={},
            metadata={}
        )

    if not image_paths:
        return AgentResponse(
            success=False,
            message="No images loaded. Please enter an image directory path first.",
            data={},
            metadata={}
        )

    # Execute through agent
    context = {'image_paths': image_paths}

    try:
        response = st.session_state.agent.process_query(query, context)
        return response
    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        return AgentResponse(
            success=False,
            message=f"Execution failed: {str(e)}",
            data={},
            metadata={}
        )


# ============================================================================
# UI COMPONENTS - Each function renders one UI element
# ============================================================================

def render_header():
    """Render app header."""
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def render_agent_init_section():
    """Render agent initialization section in sidebar."""
    st.subheader("1️⃣ Initialize Agent")

    agent_info = AGENT_INFO["template"]
    st.info(f"**{agent_info['name']}**\n\n{agent_info['description']}")

    if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
        with st.spinner("Initializing..."):
            if initialize_agent():
                st.success("✅ Agent ready!")
                st.rerun()


def render_image_directory_section():
    """Render image directory input section in sidebar."""
    st.subheader("2️⃣ Load Images")

    directory = st.text_input(
        "Image Directory Path",
        value=st.session_state.image_dir,
        placeholder="D:/Photos/MyVacation",
        help="Full path to your photo folder"
    )

    if directory and directory != st.session_state.image_dir:
        with st.spinner("Loading images..."):
            success, message, images = load_images_from_directory(directory)

            if success:
                st.session_state.image_dir = directory
                st.session_state.image_paths = images
                st.success(message)
            else:
                st.error(message)

    # Show status
    if st.session_state.image_paths:
        st.success(f"✅ {len(st.session_state.image_paths)} images loaded")


def render_status_section():
    """Render current status in sidebar."""
    st.subheader("📊 Status")

    agent_status = "✅ Ready" if st.session_state.agent else "❌ Not initialized"
    st.markdown(f"**Agent:** {agent_status}")

    image_status = f"✅ {len(st.session_state.image_paths)} loaded" if st.session_state.image_paths else "❌ No images"
    st.markdown(f"**Images:** {image_status}")


def render_controls_section():
    """Render control buttons in sidebar."""
    st.subheader("🎛️ Controls")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()

    with col2:
        if st.button("Reset All", use_container_width=True):
            # Clear everything
            st.session_state.agent = None
            st.session_state.messages = []
            st.session_state.image_dir = ""
            st.session_state.image_paths = []
            st.rerun()


def render_sidebar():
    """Render complete sidebar."""
    with st.sidebar:
        st.title("⚙️ Setup")

        render_agent_init_section()
        st.divider()

        render_image_directory_section()
        st.divider()

        render_status_section()
        st.divider()

        render_controls_section()


def render_welcome_screen():
    """Render welcome screen when agent not ready."""
    st.info("👈 **Get Started:** Follow the steps in the sidebar")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### What This App Does")
        st.markdown("""
- Organizes photos by event/similarity
- Finds your best quality photos
- Groups photos by people (faces)
- Sorts travel photos by location
        """)

    with col2:
        st.markdown("### How It Works")
        st.markdown("""
1. **Initialize the agent** (one-time setup)
2. **Load your photos** (enter directory path)
3. **Ask what you want** (natural language)
4. **Get organized results** (automatic)
        """)


def render_example_buttons():
    """Render example query buttons."""
    st.markdown("### 💡 Try an Example")

    cols = st.columns(len(EXAMPLE_QUERIES))

    for i, example in enumerate(EXAMPLE_QUERIES):
        with cols[i]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                return example

    return None


def render_chat_history():
    """Render all chat messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_chat_interface():
    """Render main chat interface."""
    st.markdown("---")
    st.subheader("💬 Chat")

    # Show chat history
    render_chat_history()

    # Example buttons if no chat yet
    if len(st.session_state.messages) == 0:
        clicked_example = render_example_buttons()
        if clicked_example:
            handle_user_input(clicked_example)
            st.rerun()

    # Chat input
    user_input = st.chat_input("Ask me to organize your photos...")
    if user_input:
        handle_user_input(user_input)
        st.rerun()


# ============================================================================
# EVENT HANDLERS - Handle user interactions
# ============================================================================

def handle_user_input(query: str):
    """
    Handle user query submission.
    This is the main orchestration function.
    """
    # Add user message to chat
    add_chat_message("user", query)

    # Execute query
    with st.spinner("Processing..."):
        response = execute_query(query, st.session_state.image_paths)

    # Add response to chat
    if response.success:
        add_chat_message("assistant", response.message)
    else:
        error_msg = f"❌ **Error:** {response.message}"
        add_chat_message("assistant", error_msg)


# ============================================================================
# MAIN - Entry point
# ============================================================================

def main():
    """Main app entry point."""
    # Setup
    st.set_page_config(page_title="Photo Organization", page_icon="📸", layout="wide")
    init_session()

    # Render
    render_header()
    render_sidebar()

    # Main content
    if not st.session_state.agent or not st.session_state.image_paths:
        render_welcome_screen()
    else:
        render_chat_interface()


if __name__ == "__main__":
    main()
