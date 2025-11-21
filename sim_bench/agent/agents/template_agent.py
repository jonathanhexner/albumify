"""
Template-based agent implementation.

Uses pre-defined workflow templates with keyword matching.
No LLM required - fast and simple.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

from sim_bench.agent.core.base import Agent, AgentResponse
from sim_bench.agent.workflows.templates import WorkflowTemplates
from sim_bench.agent.core.executor import SimpleWorkflowExecutor
from sim_bench.agent.tools.registry import ToolRegistry
from sim_bench.agent.core.memory import AgentMemory

logger = logging.getLogger(__name__)


class TemplateAgent(Agent):
    """
    Agent that uses pre-defined workflow templates.

    Simpler than WorkflowAgent - no LLM needed, just template matching.
    Uses keyword-based matching to select appropriate workflow templates.
    """

    def __init__(self, tool_registry: ToolRegistry, memory: AgentMemory, config: Dict = None):
        """
        Initialize template agent.

        Args:
            tool_registry: Tool registry for executing workflow steps
            memory: Agent memory for storing conversation and results
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.tool_registry = tool_registry
        self.memory = memory
        self.executor = SimpleWorkflowExecutor(tool_registry, memory)

    def process_query(self, query: str, context: Dict = None) -> AgentResponse:
        """
        Process query by matching to template.

        Args:
            query: User query
            context: Optional context (must contain 'image_paths')

        Returns:
            AgentResponse with workflow results
        """
        context = context or {}

        # Check for image_paths in context
        if 'image_paths' not in context or not context['image_paths']:
            return AgentResponse(
                success=False,
                message="No images provided. Please specify image_paths in context.",
                data={'available_templates': WorkflowTemplates.list_templates()},
                metadata={}
            )

        # Simple keyword matching to templates
        query_lower = query.lower()

        workflow = None
        template_name = None

        # Person/people-based organization
        if ('person' in query_lower or 'people' in query_lower) and 'group' not in query_lower:
            if 'organize' in query_lower or 'who' in query_lower:
                template_name = 'organize_by_people'
                workflow = WorkflowTemplates.organize_by_people()
            elif 'best' in query_lower or 'portrait' in query_lower:
                template_name = 'find_best_portraits'
                workflow = WorkflowTemplates.find_best_portraits()
        # Group photos
        elif 'group' in query_lower and ('photo' in query_lower or 'picture' in query_lower):
            template_name = 'find_group_photos'
            workflow = WorkflowTemplates.find_group_photos()
        # Travel/landmark-based
        elif 'travel' in query_lower or 'landmark' in query_lower or 'place' in query_lower:
            template_name = 'organize_travel_photos'
            workflow = WorkflowTemplates.organize_travel_photos()
        # Event-based clustering
        elif 'event' in query_lower and ('organize' in query_lower or 'cluster' in query_lower):
            template_name = 'organize_by_event'
            workflow = WorkflowTemplates.organize_by_event()
        # Vacation (comprehensive)
        elif 'vacation' in query_lower:
            template_name = 'organize_vacation_photos'
            workflow = WorkflowTemplates.organize_vacation_photos()

        if workflow is None:
            available = WorkflowTemplates.list_templates()

            return AgentResponse(
                success=False,
                message=(
                    f"Could not match query to a template. Try:\n"
                    f"  - 'Organize my photos by event'\n"
                    f"  - 'Group my photos by person'\n"
                    f"  - 'Organize my travel photos by landmarks'\n"
                    f"  - 'Find my best group photos'\n"
                    f"  - 'Find my best portraits'"
                ),
                data={'available_templates': available},
                metadata={}
            )

        # Inject image_paths into workflow steps
        image_paths = context['image_paths']
        for step in workflow.steps:
            if 'image_paths' not in step.params:
                step.params['image_paths'] = image_paths

        # Execute workflow
        try:
            results = self.executor.execute_workflow(workflow)

            self.memory.add_turn(
                user_message=query,
                agent_response=f"Executed {template_name}",
                workflow_name=template_name,
                workflow_results=results
            )

            return AgentResponse(
                success=True,
                message=f"Successfully executed {template_name}. Processed {len(image_paths)} images.",
                data={'workflow': workflow, 'results': results},
                metadata={'workflow_name': template_name, 'num_images': len(image_paths)}
            )

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Workflow execution failed: {str(e)}",
                data={'workflow': workflow},
                metadata={'error': str(e), 'workflow_name': template_name}
            )

    def refine(self, feedback: str) -> AgentResponse:
        """Not supported for template agent."""
        return AgentResponse(
            success=False,
            message="Refinement not supported for template agent",
            data={},
            metadata={}
        )

    def execute(self, task: str, working_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a task (CLI interface compatibility).

        This method provides a simpler interface for CLI usage that:
        1. Extracts image paths from working directory
        2. Calls process_query with proper context
        3. Returns a dictionary compatible with CLI expectations

        Args:
            task: Task description (natural language query)
            working_directory: Optional working directory to search for images

        Returns:
            Dictionary with 'success', 'message', 'data' keys
        """
        # Build context from working directory if provided
        context = {}
        
        # Get working directory from config if not provided
        if not working_directory:
            working_directory = self.config.get('working_directory')
        
        if working_directory:
            work_dir = Path(working_directory)
            if work_dir.exists() and work_dir.is_dir():
                # Find images in working directory
                image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                image_paths = []
                for pattern in image_patterns:
                    image_paths.extend(work_dir.glob(pattern))
                    image_paths.extend(work_dir.glob(f'**/{pattern}'))
                
                if image_paths:
                    context['image_paths'] = [str(p) for p in sorted(set(image_paths))]
                    logger.info(f"Found {len(context['image_paths'])} images in {working_directory}")
                else:
                    logger.warning(f"No images found in {working_directory}")
            else:
                logger.warning(f"Working directory does not exist: {working_directory}")

        # If no images found, try to extract directory from task
        if 'image_paths' not in context or not context['image_paths']:
            # Try to extract directory path from task
            import re
            # Look for paths like "D:/Photos", "/path/to/photos", "C:\\Photos"
            path_pattern = r'([A-Z]:[\\/][^\s]+|[\\/][^\s]+)'
            matches = re.findall(path_pattern, task)
            if matches:
                potential_dir = Path(matches[0])
                if potential_dir.exists() and potential_dir.is_dir():
                    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                    image_paths = []
                    for pattern in image_patterns:
                        image_paths.extend(potential_dir.glob(pattern))
                        image_paths.extend(potential_dir.glob(f'**/{pattern}'))
                    
                    if image_paths:
                        context['image_paths'] = [str(p) for p in sorted(set(image_paths))]
                        logger.info(f"Extracted {len(context['image_paths'])} images from task path: {matches[0]}")

        # If still no images, return helpful error
        if 'image_paths' not in context or not context['image_paths']:
            return {
                'success': False,
                'message': (
                    "No images found. Please specify a directory with images.\n"
                    "Example: 'Analyze photos in D:/Photos/Vacation'"
                ),
                'data': {}
            }

        # Process query
        response = self.process_query(task, context)

        # Convert AgentResponse to dictionary format expected by CLI
        return {
            'success': response.success,
            'message': response.message,
            'data': response.data,
            'metadata': response.metadata
        }

