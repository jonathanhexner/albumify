"""
Configuration for the AI Photo Organization App.

Contains all UI text, styling, examples, and settings.
"""

# Page configuration
PAGE_CONFIG = {
    "title": "AI Photo Organization",
    "icon": "📸",
    "layout": "wide",
}

# App text content
CONTENT = {
    "app_title": "📸 AI Photo Organization",
    "app_subtitle": "Organize your photos using natural language",

    "welcome_message": """
Welcome! This app helps you organize and analyze your photos using AI.

**Getting Started:**
1. Select an agent type (Template recommended - no API key needed)
2. Click "Initialize Agent"
3. Enter your photo directory path
4. Ask the agent what you want to do!
    """,

    "features": {
        "Event Clustering": "Group photos by events (vacations, parties, etc.)",
        "Face Recognition": "Organize by people in your photos",
        "Landmark Detection": "Sort travel photos by location",
        "Quality Assessment": "Find your best photos automatically",
        "Smart Tagging": "Automatic scene and object detection"
    }
}

# Example queries
EXAMPLE_QUERIES = [
    "Organize my photos by event",
    "Find my best 10 photos",
    "Group photos by person",
    "Show me all portrait photos",
    "Find photos from my vacation",
    "Organize travel photos by landmarks"
]

# Agent types
AGENT_TYPES = {
    "template": {
        "label": "Template Agent (Recommended)",
        "description": "Uses pre-defined workflows. No API key needed. ✓ Available now",
        "requires_api": False,
        "available": True
    },
    "workflow": {
        "label": "Workflow Agent (Coming Soon)",
        "description": "LLM plans custom workflows. Not yet implemented.",
        "requires_api": True,
        "available": False
    },
    "conversational": {
        "label": "Conversational Agent (Coming Soon)",
        "description": "Full LLM conversation. Not yet implemented.",
        "requires_api": True,
        "available": False
    }
}

# Supported image formats
IMAGE_FORMATS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

# Custom CSS styling
CUSTOM_CSS = """
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }

    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }

    .agent-message {
        background-color: #f1f8f4;
        border-left: 4px solid #4caf50;
    }

    /* Info boxes */
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }

    /* Workflow steps */
    .workflow-step {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-family: monospace;
    }

    .step-completed {
        background-color: #c8e6c9;
        color: #2e7d32;
    }

    .step-running {
        background-color: #fff9c4;
        color: #f57f17;
    }

    .step-pending {
        background-color: #eeeeee;
        color: #757575;
    }

    .step-failed {
        background-color: #ffcdd2;
        color: #c62828;
    }

    /* Tool cards */
    .tool-card {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #fafafa;
        border-radius: 0.25rem;
    }
</style>
"""
