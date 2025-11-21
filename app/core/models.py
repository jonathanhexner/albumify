"""Core data models for the application."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Any
from enum import Enum
from pathlib import Path


class AgentStatus(Enum):
    """Agent initialization status."""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


class MessageRole(Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A single chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Convert string role to enum if needed."""
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)


@dataclass
class ImageLibrary:
    """Represents a loaded image library."""
    directory: Path
    image_paths: List[Path]
    loaded_at: datetime = field(default_factory=datetime.now)

    @property
    def count(self) -> int:
        """Number of images in library."""
        return len(self.image_paths)

    @property
    def is_empty(self) -> bool:
        """Check if library has no images."""
        return self.count == 0


@dataclass
class AgentInstance:
    """Represents an initialized agent."""
    agent: Any  # The actual agent object
    agent_type: str
    initialized_at: datetime = field(default_factory=datetime.now)
    status: AgentStatus = AgentStatus.READY


@dataclass
class AppState:
    """Complete application state."""
    agent: Optional[AgentInstance] = None
    image_library: Optional[ImageLibrary] = None
    conversation: List[ChatMessage] = field(default_factory=list)

    @property
    def has_agent(self) -> bool:
        """Check if agent is initialized and ready."""
        return (
            self.agent is not None
            and self.agent.status == AgentStatus.READY
        )

    @property
    def has_images(self) -> bool:
        """Check if images are loaded."""
        return (
            self.image_library is not None
            and not self.image_library.is_empty
        )

    @property
    def is_ready(self) -> bool:
        """Check if app is ready to process queries."""
        return self.has_agent and self.has_images

    def get_context(self) -> dict:
        """Get context dictionary for agent queries."""
        context = {}

        if self.image_library:
            context['image_paths'] = [str(p) for p in self.image_library.image_paths]
            context['image_dir'] = str(self.image_library.directory)

        return context
