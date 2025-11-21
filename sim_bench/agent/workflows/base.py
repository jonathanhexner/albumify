"""
Workflow base classes using Composite pattern.

A workflow is a directed acyclic graph (DAG) of steps, where each step
executes a tool and can depend on previous steps.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow or step execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """
    Single step in a workflow.

    Each step executes one tool with specific parameters and can depend
    on results from previous steps.
    """

    name: str
    tool_name: str
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

    def is_ready(self, completed_steps: Set[str]) -> bool:
        """
        Check if all dependencies are completed.

        Args:
            completed_steps: Set of completed step names

        Returns:
            True if all dependencies are in completed set
        """
        return all(dep in completed_steps for dep in self.dependencies)

    def can_retry(self) -> bool:
        """Check if step can be retried after failure."""
        return self.retries < self.max_retries

    def mark_completed(self, result: Dict):
        """Mark step as successfully completed."""
        self.status = WorkflowStatus.COMPLETED
        self.result = result
        logger.info(f"Step '{self.name}' completed successfully")

    def mark_failed(self, error: str):
        """Mark step as failed."""
        self.status = WorkflowStatus.FAILED
        self.error = error
        logger.error(f"Step '{self.name}' failed: {error}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'tool_name': self.tool_name,
            'params': self.params,
            'dependencies': self.dependencies,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'retries': self.retries
        }


@dataclass
class Workflow:
    """
    Complete workflow composed of multiple steps.

    Uses Composite pattern - workflows can contain sub-workflows.
    Implements DAG execution with dependency resolution.
    """

    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_next_step(self) -> Optional[WorkflowStep]:
        """
        Get next executable step based on dependencies and status.

        Returns:
            Next step to execute or None if workflow is complete or blocked
        """
        completed = self._get_completed_step_names()

        for step in self.steps:
            if step.status == WorkflowStatus.PENDING and step.is_ready(completed):
                return step

        return None

    def get_failed_steps(self) -> List[WorkflowStep]:
        """Get list of failed steps."""
        return [step for step in self.steps if step.status == WorkflowStatus.FAILED]

    def get_retriable_steps(self) -> List[WorkflowStep]:
        """Get failed steps that can be retried."""
        return [step for step in self.get_failed_steps() if step.can_retry()]

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(
            step.status in (WorkflowStatus.COMPLETED, WorkflowStatus.SKIPPED)
            for step in self.steps
        )

    def is_blocked(self) -> bool:
        """
        Check if workflow is blocked (has failures with no retriable steps).

        Returns:
            True if workflow cannot proceed
        """
        if self.is_complete():
            return False

        failed_steps = self.get_failed_steps()
        if not failed_steps:
            return False

        # Blocked if we have failures and no retriable steps
        return not self.get_retriable_steps()

    def get_progress(self) -> Dict[str, Any]:
        """
        Get workflow progress statistics.

        Returns:
            Dictionary with progress metrics
        """
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == WorkflowStatus.FAILED)
        pending = sum(1 for s in self.steps if s.status == WorkflowStatus.PENDING)

        return {
            'total_steps': total,
            'completed': completed,
            'failed': failed,
            'pending': pending,
            'progress_pct': (completed / total * 100) if total > 0 else 0,
            'is_complete': self.is_complete(),
            'is_blocked': self.is_blocked()
        }

    def validate_dependencies(self) -> List[str]:
        """
        Validate workflow DAG structure.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        step_names = {step.name for step in self.steps}

        for step in self.steps:
            # Check all dependencies exist
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")

        # Check for cycles (simple DFS)
        if self._has_cycle():
            errors.append("Workflow contains circular dependencies")

        return errors

    def _has_cycle(self) -> bool:
        """Check if workflow contains circular dependencies."""
        # Build adjacency list
        graph = {step.name: step.dependencies for step in self.steps}

        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step in self.steps:
            if step.name not in visited:
                if dfs(step.name):
                    return True

        return False

    def _get_completed_step_names(self) -> Set[str]:
        """Get set of completed step names."""
        return {
            step.name for step in self.steps
            if step.status == WorkflowStatus.COMPLETED
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps],
            'status': self.status.value,
            'results': self.results,
            'metadata': self.metadata,
            'progress': self.get_progress()
        }

    def get_summary(self) -> str:
        """Get human-readable workflow summary."""
        progress = self.get_progress()
        lines = [
            f"Workflow: {self.name}",
            f"Description: {self.description}",
            f"Progress: {progress['completed']}/{progress['total_steps']} steps completed",
            f"Status: {self.status.value}",
            ""
        ]

        if self.steps:
            lines.append("Steps:")
            for i, step in enumerate(self.steps, 1):
                status_symbol = {
                    WorkflowStatus.COMPLETED: "✓",
                    WorkflowStatus.FAILED: "✗",
                    WorkflowStatus.IN_PROGRESS: "→",
                    WorkflowStatus.PENDING: "○",
                    WorkflowStatus.SKIPPED: "⊘"
                }.get(step.status, "?")

                lines.append(f"  {i}. [{status_symbol}] {step.name} ({step.tool_name})")

                if step.dependencies:
                    lines.append(f"      Depends on: {', '.join(step.dependencies)}")

        return "\n".join(lines)
