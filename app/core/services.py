"""Business logic services - framework-agnostic."""

import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from sim_bench.agent.factory import create_agent
from sim_bench.agent.core.base import AgentResponse

from .models import (
    AgentInstance, ImageLibrary, ChatMessage,
    MessageRole, AgentStatus
)
from .exceptions import AgentError, ImageLoadError, ValidationError
from ..config.constants import IMAGE_PATTERNS, AgentType

logger = logging.getLogger(__name__)


class AgentService:
    """Service for agent operations."""

    @staticmethod
    def initialize_agent(agent_type: AgentType) -> AgentInstance:
        """
        Initialize an agent of the specified type.

        Args:
            agent_type: Type of agent to initialize

        Returns:
            Initialized AgentInstance

        Raises:
            AgentError: If initialization fails
        """
        try:
            logger.info(f"Initializing agent: {agent_type.value}")
            agent = create_agent(agent_type=agent_type.value)

            return AgentInstance(
                agent=agent,
                agent_type=agent_type.value,
                status=AgentStatus.READY,
                initialized_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise AgentError(f"Agent initialization failed: {str(e)}") from e

    @staticmethod
    def process_query(
        agent_instance: AgentInstance,
        query: str,
        context: dict
    ) -> AgentResponse:
        """
        Process a user query through the agent.

        Args:
            agent_instance: The initialized agent
            query: User's natural language query
            context: Context dictionary with image_paths, etc.

        Returns:
            AgentResponse with results

        Raises:
            AgentError: If query processing fails
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

        if agent_instance.status != AgentStatus.READY:
            raise AgentError("Agent is not ready")

        if not context.get('image_paths'):
            raise ValidationError("No images in context")

        # Process query
        try:
            logger.info(f"Processing query: {query[:100]}")
            response = agent_instance.agent.process_query(query, context)
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise AgentError(f"Query execution failed: {str(e)}") from e


class ImageService:
    """Service for image operations."""

    @staticmethod
    def load_directory(directory_path: str) -> ImageLibrary:
        """
        Load images from a directory.

        Args:
            directory_path: Path to directory containing images

        Returns:
            ImageLibrary with loaded images

        Raises:
            ImageLoadError: If loading fails
            ValidationError: If path is invalid
        """
        # Validate path
        path = Path(directory_path)

        if not directory_path or not directory_path.strip():
            raise ValidationError("Directory path cannot be empty")

        if not path.exists():
            raise ImageLoadError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise ImageLoadError(f"Path is not a directory: {directory_path}")

        # Find images
        try:
            logger.info(f"Loading images from: {directory_path}")
            image_paths = []

            for pattern in IMAGE_PATTERNS:
                found = list(path.glob(pattern))
                image_paths.extend(found)
                logger.debug(f"Pattern {pattern}: found {len(found)} images")

            # Sort for consistent ordering
            image_paths = sorted(set(image_paths))

            if not image_paths:
                raise ImageLoadError(f"No images found in: {directory_path}")

            logger.info(f"Loaded {len(image_paths)} images")

            return ImageLibrary(
                directory=path,
                image_paths=image_paths,
                loaded_at=datetime.now()
            )

        except ImageLoadError:
            raise
        except Exception as e:
            logger.error(f"Failed to load images: {e}", exc_info=True)
            raise ImageLoadError(f"Image loading failed: {str(e)}") from e

    @staticmethod
    def validate_directory(directory_path: str) -> Tuple[bool, str]:
        """
        Validate a directory path without loading.

        Args:
            directory_path: Path to validate

        Returns:
            (is_valid, error_message)
        """
        if not directory_path or not directory_path.strip():
            return False, "Directory path cannot be empty"

        path = Path(directory_path)

        if not path.exists():
            return False, f"Directory not found: {directory_path}"

        if not path.is_dir():
            return False, f"Path is not a directory: {directory_path}"

        return True, ""


class ConversationService:
    """Service for conversation management."""

    @staticmethod
    def create_message(
        role: MessageRole,
        content: str,
        metadata: dict = None
    ) -> ChatMessage:
        """Create a new chat message."""
        return ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

    @staticmethod
    def format_error_message(error: Exception) -> str:
        """Format an error as a user-friendly message."""
        return f"❌ **Error:** {str(error)}"

    @staticmethod
    def format_success_message(response: AgentResponse) -> str:
        """Format a successful response message."""
        return response.message
