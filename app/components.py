"""
Reusable UI components for the Streamlit app.

Each function renders a specific UI element.
"""

import streamlit as st
from typing import Optional
from sim_bench.agent.workflows.base import Workflow, WorkflowStatus
from sim_bench.agent.tools.registry import get_registry
from app.config import AGENT_TYPES


def render_welcome_screen():
    """Render welcome screen when no agent is initialized."""
    st.info("👈 **Get Started:** Initialize an agent from the sidebar")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✨ Features")
        st.markdown("""
- **Event Clustering** - Group photos by event
- **Face Recognition** - Organize by people
- **Landmark Detection** - Sort by location
- **Quality Assessment** - Find your best photos
- **Smart Tagging** - Automatic scene detection
        """)

    with col2:
        st.markdown("### 💡 Example Tasks")
        st.markdown("""
- "Organize my photos by event"
- "Find my best 10 photos"
- "Group photos by person"
- "Show me portrait photos"
- "Organize travel photos by landmarks"
        """)


def render_agent_selector():
    """Render agent type selector with descriptions."""
    st.markdown("### Agent Type")

    # Only show available agents
    available_agents = {k: v for k, v in AGENT_TYPES.items() if v['available']}

    agent_type = st.radio(
        "Choose how the agent should work:",
        options=list(available_agents.keys()),
        format_func=lambda x: AGENT_TYPES[x]['label'],
        help="Template agent is recommended - no API key required"
    )

    # Show description
    agent_info = AGENT_TYPES[agent_type]
    st.caption(f"ℹ️ {agent_info['description']}")

    if agent_info['requires_api']:
        st.warning("⚠️ This agent requires an OpenAI API key")

    return agent_type


def render_image_input():
    """Render image directory input section."""
    st.markdown("### 📁 Photo Library")

    from app.state import set_image_directory, get_image_count

    input_dir = st.text_input(
        "Image Directory Path",
        value=st.session_state.image_dir,
        placeholder="D:/Photos/MyVacation",
        help="Full path to your photo folder"
    )

    if input_dir and input_dir != st.session_state.image_dir:
        success, message, _ = set_image_directory(input_dir)

        if success:
            st.success(f"✓ {message}")
        else:
            st.error(f"✗ {message}")

    # Show current status
    if get_image_count() > 0:
        st.info(f"📸 {get_image_count()} images ready")


def render_chat_message(role: str, content: str, workflow: Optional[Workflow] = None):
    """Render a single chat message."""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)

            # Show workflow if present
            if workflow:
                with st.expander("📋 Workflow Details", expanded=False):
                    render_workflow(workflow)


def render_workflow(workflow: Workflow):
    """Render workflow progress and steps."""
    st.markdown(f"**Workflow:** `{workflow.name}`")

    if workflow.description:
        st.caption(workflow.description)

    # Progress
    progress = workflow.get_progress()
    completed = progress['completed']
    total = progress['total_steps']

    st.progress(completed / total if total > 0 else 0)
    st.caption(f"Progress: {completed}/{total} steps completed")

    # Steps
    st.markdown("**Steps:**")
    for i, step in enumerate(workflow.steps, 1):
        status_icon = {
            WorkflowStatus.COMPLETED: "✅",
            WorkflowStatus.IN_PROGRESS: "⏳",
            WorkflowStatus.PENDING: "⏸️",
            WorkflowStatus.FAILED: "❌",
            WorkflowStatus.SKIPPED: "⏭️"
        }.get(step.status, "❓")

        st.markdown(f"{status_icon} **Step {i}:** {step.name} → `{step.tool_name}`")

        if step.error:
            st.error(f"Error: {step.error}")


def render_conversation():
    """Render full conversation history."""
    from app.state import get_conversation

    conversation = get_conversation()

    if not conversation:
        st.info("💬 Start a conversation by typing below or clicking an example")
        return

    for msg in conversation:
        render_chat_message(
            role=msg['role'],
            content=msg['content'],
            workflow=msg.get('workflow')
        )


def render_example_queries(examples: list):
    """Render clickable example query buttons."""
    st.markdown("### 💡 Try These Examples")

    cols = st.columns(2)

    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                return example

    return None


def render_tools_panel():
    """Render available tools panel in sidebar."""
    st.markdown("### 🔧 Available Tools")

    registry = get_registry()
    by_category = registry.get_tools_by_category()

    total_tools = sum(len(tools) for tools in by_category.values())
    st.caption(f"{total_tools} tools available")

    for category, tools in sorted(by_category.items(), key=lambda x: x[0].value):
        with st.expander(f"{category.value.title()} ({len(tools)})", expanded=False):
            for tool_name in sorted(tools):
                schema = registry.get_tool_schema(tool_name)
                st.markdown(f"**`{tool_name}`**")
                st.caption(schema['description'])
                st.divider()


def render_status_panel():
    """Render app status in sidebar."""
    from app.state import has_agent, has_images, get_image_count

    st.markdown("### 📊 Status")

    if has_agent():
        agent_type = st.session_state.agent_type
        st.success(f"✓ Agent: {AGENT_TYPES[agent_type]['label']}")
    else:
        st.warning("⚠ No agent initialized")

    if has_images():
        st.success(f"✓ Images: {get_image_count()}")
    else:
        st.warning("⚠ No images loaded")


def render_session_controls():
    """Render session management buttons."""
    from app.state import clear_conversation, reset_app

    st.markdown("### 🎛️ Controls")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_conversation()
            st.rerun()

    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            reset_app()
            st.rerun()
