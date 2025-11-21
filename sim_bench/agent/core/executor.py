"""
Workflow execution using Command pattern.

Executes workflows step-by-step, managing dependencies and error handling.
"""

from typing import Dict, Any, Set
import logging

from sim_bench.agent.workflows.base import Workflow, WorkflowStep, WorkflowStatus
from sim_bench.agent.tools.registry import ToolRegistry
from sim_bench.agent.core.memory import AgentMemory

logger = logging.getLogger(__name__)


class SimpleWorkflowExecutor:
    """
    Simple workflow executor for template-based workflows.

    Executes workflow steps in dependency order, handling errors and retries.
    """

    def __init__(self, tool_registry: ToolRegistry, memory: AgentMemory):
        """
        Initialize executor.

        Args:
            tool_registry: Tool registry for loading tools
            memory: Agent memory for storing results
        """
        self.tool_registry = tool_registry
        self.memory = memory

    def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute complete workflow.

        Args:
            workflow: Workflow to execute

        Returns:
            Dictionary with workflow results

        Raises:
            ValueError: If workflow validation fails
            RuntimeError: If workflow execution fails
        """
        logger.info(f"Executing workflow: {workflow.name}")

        # Validate workflow
        errors = workflow.validate_dependencies()
        if errors:
            raise ValueError(f"Workflow validation failed: {errors}")

        workflow.status = WorkflowStatus.IN_PROGRESS
        completed_steps: Set[str] = set()

        # Execute steps in dependency order
        while not workflow.is_complete():
            # Get next executable step
            next_step = workflow.get_next_step()

            if next_step is None:
                # Check if blocked
                if workflow.is_blocked():
                    workflow.status = WorkflowStatus.FAILED
                    raise RuntimeError(
                        f"Workflow blocked. Failed steps: "
                        f"{[s.name for s in workflow.get_failed_steps()]}"
                    )
                # All steps complete
                break

            # Execute step
            try:
                result = self._execute_step(next_step, workflow.results)
                next_step.mark_completed(result)
                completed_steps.add(next_step.name)

                # Store in workflow results
                workflow.results[next_step.name] = result

                # Update memory
                self.memory.add_step_result(next_step.name, result)

                logger.info(f"Step '{next_step.name}' completed successfully")

            except Exception as e:
                logger.error(f"Step '{next_step.name}' failed: {e}", exc_info=True)
                next_step.mark_failed(str(e))

                # Check if can retry
                if next_step.can_retry():
                    next_step.retries += 1
                    next_step.status = WorkflowStatus.PENDING
                    logger.info(f"Retrying step '{next_step.name}' ({next_step.retries}/{next_step.max_retries})")
                else:
                    logger.error(f"Step '{next_step.name}' failed permanently")

        # Finalize workflow status
        if workflow.is_complete():
            workflow.status = WorkflowStatus.COMPLETED
            logger.info(f"Workflow '{workflow.name}' completed successfully")
        else:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Workflow '{workflow.name}' failed")

        return workflow.results

    def _execute_step(self, step: WorkflowStep, previous_results: Dict) -> Dict[str, Any]:
        """
        Execute a single workflow step.

        Args:
            step: WorkflowStep to execute
            previous_results: Results from previous steps

        Returns:
            Step execution result

        Raises:
            Exception: If step execution fails
        """
        logger.debug(f"Executing step: {step.name}")

        step.status = WorkflowStatus.IN_PROGRESS

        # Get tool
        tool = self.tool_registry.get_tool(step.tool_name)

        # Resolve parameters from previous results
        params = self._resolve_params(step.params, previous_results)

        # Execute tool
        result = tool.run(**params)

        if not result.get('success', False):
            raise RuntimeError(f"Tool execution failed: {result.get('message', 'Unknown error')}")

        return result

    def _resolve_params(self, params: Dict, previous_results: Dict) -> Dict:
        """
        Resolve parameters, substituting references to previous step results.

        Supports syntax like:
        - "$step_name.data.key" - Reference to previous step result
        - "value" - Literal value

        Args:
            params: Parameter dictionary
            previous_results: Results from previous steps

        Returns:
            Resolved parameters
        """
        resolved = {}

        for key, value in params.items():
            if isinstance(value, str) and value.startswith('$'):
                # Reference to previous result
                resolved[key] = self._resolve_reference(value, previous_results)
            else:
                resolved[key] = value

        return resolved

    def _resolve_reference(self, reference: str, previous_results: Dict) -> Any:
        """
        Resolve a reference to a previous step result.

        Args:
            reference: Reference string (e.g., "$step_name.data.key")
            previous_results: Results from previous steps

        Returns:
            Resolved value

        Raises:
            KeyError: If reference cannot be resolved
        """
        # Remove leading $
        ref_path = reference[1:]

        # Split by dots
        parts = ref_path.split('.')

        # Navigate through results
        value = previous_results
        for part in parts:
            if isinstance(value, dict):
                value = value[part]
            else:
                raise KeyError(f"Cannot resolve reference: {reference}")

        return value
