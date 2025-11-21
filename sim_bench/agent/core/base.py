"""
Abstract base classes for agent components.

Defines interfaces for:
- Planning strategies
- Execution strategies
- Memory management
- Agent coordination
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """
    Standardized agent response format.

    Attributes:
        success: Whether operation succeeded
        message: Human-readable response
        data: Structured result data
        metadata: Additional information (timing, tokens, etc.)
        next_steps: Suggested user actions
    """
    success: bool
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    next_steps: Optional[list] = None


class Agent(ABC):
    """
    Abstract base class for AI agents.

    Uses Template Method pattern - subclasses implement specific steps
    while base class orchestrates the overall flow.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup()

    def setup(self):
        """Override to perform agent-specific initialization."""
        pass

    @abstractmethod
    def process_query(self, query: str, context: Dict = None) -> AgentResponse:
        """
        Process user query and return response.

        Template method that orchestrates:
        1. Understanding query
        2. Planning workflow
        3. Executing workflow
        4. Formatting response

        Args:
            query: User's natural language request
            context: Optional context from previous interactions

        Returns:
            AgentResponse with results and suggested next steps
        """
        pass

    @abstractmethod
    def refine(self, feedback: str) -> AgentResponse:
        """
        Refine previous results based on user feedback.

        Args:
            feedback: User's feedback on previous results

        Returns:
            AgentResponse with refined results
        """
        pass

    def validate_query(self, query: str) -> bool:
        """Validate user query before processing."""
        if not query or not query.strip():
            self.logger.warning("Empty query received")
            return False
        return True
