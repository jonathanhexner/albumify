"""
Agent memory system using Memento pattern.

Manages conversation history, workflow results, and context for LLM.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn in agent interaction."""

    timestamp: datetime
    user_message: str
    agent_response: str
    workflow_name: Optional[str] = None
    workflow_results: Dict = field(default_factory=dict)
    tokens_used: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationTurn':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class AgentMemory:
    """
    Manages conversation history and workflow results.

    Uses Memento pattern to enable:
    - Save/restore state
    - Context retrieval for LLM
    - Result caching
    """

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.conversation_history: List[ConversationTurn] = []
        self.workflow_results: Dict[str, Any] = {}
        self.current_context: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_turn(
        self,
        user_message: str,
        agent_response: str,
        workflow_name: str = None,
        workflow_results: Dict = None,
        tokens_used: int = 0
    ):
        """
        Add conversation turn to memory.

        Args:
            user_message: User's input
            agent_response: Agent's response
            workflow_name: Name of workflow executed (if any)
            workflow_results: Results from workflow execution
            tokens_used: LLM tokens consumed
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=agent_response,
            workflow_name=workflow_name,
            workflow_results=workflow_results or {},
            tokens_used=tokens_used
        )

        self.conversation_history.append(turn)

        # Trim if exceeds max
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]

        logger.info(f"Added conversation turn. Total turns: {len(self.conversation_history)}")

    def add_step_result(self, step_name: str, result: Dict):
        """
        Store workflow step execution result.

        Args:
            step_name: Name of the workflow step
            result: Execution result data
        """
        self.workflow_results[step_name] = result
        self.current_context[step_name] = result

        logger.debug(f"Added step result: {step_name}")

    def get_context(self, max_recent_turns: int = 3) -> Dict:
        """
        Get current context for LLM prompting.

        Args:
            max_recent_turns: Number of recent turns to include

        Returns:
            Context dictionary with:
            - recent_conversation: Recent turns
            - workflow_results: Available data from previous workflows
            - statistics: Memory statistics
        """
        recent_turns = self.conversation_history[-max_recent_turns:] if self.conversation_history else []

        return {
            'recent_conversation': [
                {
                    'user': turn.user_message,
                    'agent': turn.agent_response,
                    'workflow': turn.workflow_name
                }
                for turn in recent_turns
            ],
            'workflow_results': self.workflow_results,
            'available_data': list(self.workflow_results.keys()),
            'num_turns': len(self.conversation_history),
            'metadata': self.metadata
        }

    def get_summary(self, num_turns: int = 5) -> str:
        """
        Get human-readable conversation summary.

        Args:
            num_turns: Number of recent turns to summarize

        Returns:
            Formatted string summary
        """
        if not self.conversation_history:
            return "No conversation history"

        recent = self.conversation_history[-num_turns:]
        lines = []

        for i, turn in enumerate(recent, 1):
            lines.append(f"Turn {i} ({turn.timestamp.strftime('%H:%M:%S')}):")
            lines.append(f"  User: {turn.user_message[:80]}...")
            lines.append(f"  Agent: {turn.agent_response[:80]}...")
            if turn.workflow_name:
                lines.append(f"  Workflow: {turn.workflow_name}")

        return "\n".join(lines)

    def get_workflow_result(self, workflow_name: str) -> Optional[Dict]:
        """
        Get results from a specific workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Workflow results or None if not found
        """
        return self.workflow_results.get(workflow_name)

    def clear_workflow_results(self):
        """Clear all workflow results (keep conversation)."""
        self.workflow_results.clear()
        self.current_context.clear()
        logger.info("Cleared workflow results")

    def save_state(self) -> Dict:
        """
        Save complete memory state (Memento pattern).

        Returns:
            Dictionary containing all memory state
        """
        return {
            'conversation_history': [turn.to_dict() for turn in self.conversation_history],
            'workflow_results': self.workflow_results,
            'current_context': self.current_context,
            'metadata': self.metadata,
            'max_turns': self.max_turns
        }

    def restore_state(self, state: Dict):
        """
        Restore memory from saved state.

        Args:
            state: Previously saved state dictionary
        """
        self.conversation_history = [
            ConversationTurn.from_dict(turn_data)
            for turn_data in state.get('conversation_history', [])
        ]
        self.workflow_results = state.get('workflow_results', {})
        self.current_context = state.get('current_context', {})
        self.metadata = state.get('metadata', {})
        self.max_turns = state.get('max_turns', 50)

        logger.info(f"Restored state with {len(self.conversation_history)} turns")

    def export_to_json(self, file_path: str):
        """
        Export memory to JSON file.

        Args:
            file_path: Path to save JSON file
        """
        state = self.save_state()
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Exported memory to: {file_path}")

    @classmethod
    def load_from_json(cls, file_path: str) -> 'AgentMemory':
        """
        Load memory from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            AgentMemory instance with loaded state
        """
        with open(file_path, 'r') as f:
            state = json.load(f)

        memory = cls()
        memory.restore_state(state)

        logger.info(f"Loaded memory from: {file_path}")
        return memory

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Statistics dictionary
        """
        total_tokens = sum(turn.tokens_used for turn in self.conversation_history)

        return {
            'total_turns': len(self.conversation_history),
            'total_tokens_used': total_tokens,
            'num_workflows_executed': len(set(
                turn.workflow_name for turn in self.conversation_history
                if turn.workflow_name
            )),
            'available_results': len(self.workflow_results),
            'memory_size_mb': len(json.dumps(self.save_state())) / (1024 * 1024)
        }
