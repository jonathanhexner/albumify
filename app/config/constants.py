"""Application constants and enumerations."""

from enum import Enum
from typing import List


class AgentType(Enum):
    """Available agent types."""
    TEMPLATE = "template"
    WORKFLOW = "workflow"
    CONVERSATIONAL = "conversational"


class ImageFormat(Enum):
    """Supported image formats."""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"


# Image file patterns
IMAGE_PATTERNS: List[str] = [
    "*.jpg", "*.jpeg", "*.png",
    "*.JPG", "*.JPEG", "*.PNG"
]

# Example queries for users
EXAMPLE_QUERIES: List[str] = [
    "Organize my photos by event",
    "Find my best 10 photos",
    "Group photos by person",
    "Show me all portrait photos",
    "Find photos from my vacation",
    "Organize travel photos by landmarks"
]
