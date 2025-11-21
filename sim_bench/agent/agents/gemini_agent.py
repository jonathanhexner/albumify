"""
Gemini-based agent implementation.

Uses Google's Gemini LLM for workflow planning and execution.
"""

from typing import Dict, Any, Optional, List
import logging
import json
import time

from sim_bench.agent.core.base import Agent, AgentResponse
from sim_bench.agent.tools.registry import ToolRegistry
from sim_bench.agent.core.memory import AgentMemory
from sim_bench.agent.workflows.base import Workflow, WorkflowStep, WorkflowStatus
from sim_bench.agent.core.executor import SimpleWorkflowExecutor

logger = logging.getLogger(__name__)


class GeminiAgent(Agent):
    """
    Workflow agent that uses Google Gemini LLM for workflow planning.

    This is a workflow-type agent (uses LLM to plan workflows dynamically)
    with Gemini as the LLM provider. It should be created via factory:
    
    >>> agent = create_agent(
    ...     agent_type='workflow',
    ...     llm_provider='gemini',
    ...     model='gemini-pro'
    ... )

    Features:
    - Natural language understanding via Gemini
    - Dynamic workflow generation
    - Tool selection and parameter extraction
    - Context-aware responses
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        memory: AgentMemory,
        config: Dict = None
    ):
        """
        Initialize Gemini agent.

        Args:
            tool_registry: Tool registry for executing workflow steps
            memory: Agent memory for storing conversation and results
            config: Configuration dictionary with:
                - model: Gemini model name (default: 'gemini-pro')
                - api_key: Google API key (or from environment)
                - temperature: Sampling temperature (default: 0.7)
        """
        super().__init__(config)
        self.tool_registry = tool_registry
        self.memory = memory
        self.model_name = config.get('model', 'gemini-pro') if config else 'gemini-pro'
        self.temperature = config.get('temperature', 0.7) if config else 0.7
        self.executor = SimpleWorkflowExecutor(tool_registry, memory)
        
        # Initialize Gemini client
        self._gemini_client = None
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            
            api_key = self.config.get('api_key') or self._get_api_key_from_env()
            if not api_key:
                logger.warning("No Gemini API key found. Set GEMINI_API_KEY environment variable.")
                return
            
            genai.configure(api_key=api_key)
            self._gemini_client = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini client with model: {self.model_name}")
            
        except ImportError:
            logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}", exc_info=True)

    def _get_api_key_from_env(self) -> Optional[str]:
        """
        Get API key from environment variable or .env file.
        
        Checks:
        1. Environment variable GEMINI_API_KEY
        2. .env file in project root (if python-dotenv installed)
        
        Returns:
            API key string or None if not found
        """
        import os
        
        # First check environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key
        
        # Try loading from .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            
            # Look for .env file in project root (parent of sim_bench)
            project_root = Path(__file__).parent.parent.parent.parent
            env_file = project_root / '.env'
            
            if env_file.exists():
                load_dotenv(env_file)
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    logger.info("Loaded GEMINI_API_KEY from .env file")
                    return api_key
        except ImportError:
            # python-dotenv not installed, that's okay
            pass
        except Exception as e:
            logger.debug(f"Could not load .env file: {e}")
        
        return None

    def process_query(self, query: str, context: Dict = None) -> AgentResponse:
        """
        Process query using Gemini LLM to plan and execute workflows.

        This is the main method that:
        1. Takes a natural language query from the user
        2. Uses Gemini LLM to understand the intent
        3. Generates a workflow plan using available tools
        4. Executes the workflow step by step
        5. Returns formatted results

        Args:
            query: User's natural language query (e.g., "Find my best photos")
            context: Optional context dictionary containing:
                - image_paths: List of image file paths to process
                - previous_results: Results from previous queries
                - user_preferences: User-specific settings

        Returns:
            AgentResponse with:
                - success: Whether the operation succeeded
                - message: Human-readable response
                - data: Structured results (workflow results, selected images, etc.)
                - metadata: Additional info (tokens used, execution time, etc.)

        Implementation Plan:
        ----------
        This method is currently a placeholder. When implemented, it will:

        1. Get available tools from registry:
           - tool_schemas = self.tool_registry.get_all_schemas()
           
        2. Build system prompt with:
           - Available tools and their descriptions
           - Current context (image paths, previous results)
           - Conversation history from memory
           
        3. Call Gemini with function calling:
           - Use Gemini's function calling to select tools
           - Gemini returns tool name and parameters
           
        4. Build workflow from Gemini's tool selections:
           - Create WorkflowStep objects
           - Resolve dependencies
           - Create Workflow object
           
        5. Execute workflow:
           - Use SimpleWorkflowExecutor
           - Handle errors and retries
           
        6. Format response:
           - Summarize what was done
           - Include key results
           - Suggest next steps
        """
        if not self._gemini_client:
            return AgentResponse(
                success=False,
                message=(
                    "Gemini client not initialized. "
                    "Please set GEMINI_API_KEY environment variable or add it to .env file. "
                    "Get your key from: https://makersuite.google.com/app/apikey"
                ),
                data={},
                metadata={'agent_type': 'gemini', 'status': 'not_initialized'}
            )
        
        context = context or {}
        start_time = time.time()
        
        try:
            # Step 1: Get available tools
            tool_schemas = self.tool_registry.get_all_schemas()
            
            # Step 2: Build system prompt
            system_prompt = self._build_system_prompt(tool_schemas, context)
            
            # Step 3: Get conversation context
            memory_context = self.memory.get_context(max_recent_turns=3)
            
            # Step 4: Call Gemini to plan workflow
            workflow_plan = self._plan_workflow_with_gemini(
                query, system_prompt, memory_context, context
            )
            
            if not workflow_plan:
                return AgentResponse(
                    success=False,
                    message="Failed to generate workflow plan from query.",
                    data={'query': query, 'available_tools': self.tool_registry.list_tools()},
                    metadata={'agent_type': 'gemini', 'status': 'planning_failed'}
                )
            
            # Step 5: Convert plan to Workflow object
            workflow = self._build_workflow_from_plan(workflow_plan, context)
            
            # Step 6: Validate workflow
            errors = workflow.validate_dependencies()
            if errors:
                return AgentResponse(
                    success=False,
                    message=f"Generated workflow has errors: {', '.join(errors)}",
                    data={'workflow_plan': workflow_plan, 'errors': errors},
                    metadata={'agent_type': 'gemini', 'status': 'validation_failed'}
                )
            
            # Step 7: Execute workflow
            logger.info(f"Executing workflow: {workflow.name}")
            workflow_results = self.executor.execute_workflow(workflow)
            
            # Step 8: Format response
            execution_time = time.time() - start_time
            response_message = self._format_response_message(workflow, workflow_results)
            
            # Step 9: Update memory
            self.memory.add_turn(
                user_message=query,
                agent_response=response_message,
                workflow_name=workflow.name,
                workflow_results=workflow_results
            )
            
            return AgentResponse(
                success=True,
                message=response_message,
                data={
                    'workflow': workflow.to_dict(),
                    'results': workflow_results,
                    'selected_images': self._extract_selected_images(workflow_results)
                },
                metadata={
                    'agent_type': 'gemini',
                    'workflow_name': workflow.name,
                    'execution_time_seconds': execution_time,
                    'num_steps': len(workflow.steps)
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Error processing query: {str(e)}",
                data={'query': query},
                metadata={'agent_type': 'gemini', 'error': str(e), 'error_type': type(e).__name__}
            )

    def refine(self, feedback: str) -> AgentResponse:
        """
        Refine previous results based on user feedback.

        Args:
            feedback: User's feedback on previous results

        Returns:
            AgentResponse with refined results
        """
        if not self._gemini_client:
            return AgentResponse(
                success=False,
                message="Gemini client not initialized",
                data={},
                metadata={'agent_type': 'gemini', 'status': 'not_initialized'}
            )
        
        # Get last conversation turn
        if not self.memory.conversation_history:
            return AgentResponse(
                success=False,
                message="No previous query to refine. Please make an initial query first.",
                data={},
                metadata={}
            )
        
        last_turn = self.memory.conversation_history[-1]
        original_query = last_turn.user_message
        previous_results = last_turn.workflow_results
        
        # Build refinement prompt
        refinement_prompt = f"""
Previous query: {original_query}
Previous results: {json.dumps(previous_results, indent=2)[:500]}...

User feedback: {feedback}

Please refine the workflow based on this feedback. Generate a new workflow plan.
"""
        
        # Get tools and build system prompt
        tool_schemas = self.tool_registry.get_all_schemas()
        system_prompt = self._build_system_prompt(tool_schemas, {})
        
        # Plan refined workflow
        workflow_plan = self._plan_workflow_with_gemini(
            refinement_prompt,
            system_prompt,
            self.memory.get_context(),
            {'previous_results': previous_results}
        )
        
        if not workflow_plan:
            return AgentResponse(
                success=False,
                message="Failed to generate refined workflow plan.",
                data={'feedback': feedback},
                metadata={'agent_type': 'gemini', 'status': 'refinement_failed'}
            )
        
        # Build and execute refined workflow
        workflow = self._build_workflow_from_plan(workflow_plan, {'previous_results': previous_results})
        
        errors = workflow.validate_dependencies()
        if errors:
            return AgentResponse(
                success=False,
                message=f"Refined workflow has errors: {', '.join(errors)}",
                data={'workflow_plan': workflow_plan, 'errors': errors},
                metadata={'agent_type': 'gemini', 'status': 'validation_failed'}
            )
        
        # Execute refined workflow
        logger.info(f"Executing refined workflow: {workflow.name}")
        workflow_results = self.executor.execute_workflow(workflow)
        
        # Format response
        response_message = f"Refined workflow based on your feedback:\n\n{self._format_response_message(workflow, workflow_results)}"
        
        # Update memory
        self.memory.add_turn(
            user_message=f"[Refinement] {feedback}",
            agent_response=response_message,
            workflow_name=workflow.name,
            workflow_results=workflow_results
        )
        
        return AgentResponse(
            success=True,
            message=response_message,
            data={
                'workflow': workflow.to_dict(),
                'results': workflow_results,
                'selected_images': self._extract_selected_images(workflow_results)
            },
            metadata={
                'agent_type': 'gemini',
                'workflow_name': workflow.name,
                'is_refinement': True
            }
        )

    def execute(self, task: str, working_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a task (CLI interface compatibility).

        Args:
            task: Task description
            working_directory: Optional working directory

        Returns:
            Dictionary with 'success', 'message', 'data' keys
        """
        # Build context similar to TemplateAgent
        context = {}
        if working_directory:
            from pathlib import Path
            work_dir = Path(working_directory)
            if work_dir.exists() and work_dir.is_dir():
                image_patterns = ['*.jpg', '*.jpeg', '*.png']
                image_paths = []
                for pattern in image_patterns:
                    image_paths.extend(work_dir.glob(pattern))
                    image_paths.extend(work_dir.glob(f'**/{pattern}'))
                
                if image_paths:
                    context['image_paths'] = [str(p) for p in sorted(set(image_paths))]

        response = self.process_query(task, context)
        
        return {
            'success': response.success,
            'message': response.message,
            'data': response.data,
            'metadata': response.metadata
        }
    
    def _build_system_prompt(self, tool_schemas: List[Dict], context: Dict) -> str:
        """Build system prompt with available tools and context."""
        lines = [
            "You are a photo organization assistant. Your job is to plan workflows",
            "using available tools to accomplish user tasks.",
            "",
            "Available Tools:",
            "=" * 80
        ]
        
        for schema in tool_schemas:
            lines.append(f"\nTool: {schema['name']}")
            lines.append(f"Description: {schema['description']}")
            lines.append(f"Category: {schema['category'].value}")
            
            params = schema['parameters']
            if 'properties' in params:
                lines.append("Parameters:")
                for param_name, param_info in params['properties'].items():
                    param_desc = param_info.get('description', '')
                    param_type = param_info.get('type', '')
                    required = param_name in params.get('required', [])
                    req_marker = " (required)" if required else " (optional)"
                    lines.append(f"  - {param_name} ({param_type}): {param_desc}{req_marker}")
        
        lines.append("\n" + "=" * 80)
        lines.append("\nContext:")
        if context.get('image_paths'):
            lines.append(f"  - {len(context['image_paths'])} images provided")
        if context.get('previous_results'):
            lines.append(f"  - Previous results available: {list(context['previous_results'].keys())}")
        
        lines.append("\n" + "=" * 80)
        lines.append("\nInstructions:")
        lines.append("1. Analyze the user query")
        lines.append("2. Select appropriate tools to accomplish the task")
        lines.append("3. Plan a workflow with steps and dependencies")
        lines.append("4. Return a JSON workflow plan with this structure:")
        lines.append("""
{
  "name": "workflow_name",
  "description": "What this workflow does",
  "steps": [
    {
      "name": "step1_name",
      "tool_name": "tool_name_from_above",
      "params": {"param1": "value1", ...},
      "dependencies": [],
      "reason": "Why this step is needed"
    },
    {
      "name": "step2_name",
      "tool_name": "another_tool",
      "params": {"param1": "$step1_name.data.key"},
      "dependencies": ["step1_name"],
      "reason": "Uses results from step1"
    }
  ]
}
""")
        lines.append("\nImportant:")
        lines.append("- Use image_paths from context for tools that need images")
        lines.append("- Reference previous step results using $step_name.data.key syntax")
        lines.append("- Ensure dependencies are correct (step must exist before referencing)")
        lines.append("- Keep workflows simple and focused")
        
        return "\n".join(lines)
    
    def _plan_workflow_with_gemini(
        self,
        query: str,
        system_prompt: str,
        memory_context: Dict,
        context: Dict
    ) -> Optional[Dict]:
        """Use Gemini to generate workflow plan."""
        try:
            # Build user prompt
            user_prompt_parts = [
                f"User Query: {query}",
                ""
            ]
            
            # Add memory context if available
            if memory_context.get('recent_conversation'):
                user_prompt_parts.append("Recent Conversation:")
                for turn in memory_context['recent_conversation'][-2:]:
                    user_prompt_parts.append(f"  User: {turn['user']}")
                    user_prompt_parts.append(f"  Agent: {turn['agent']}")
                user_prompt_parts.append("")
            
            # Add image paths if available
            if context.get('image_paths'):
                image_paths = context['image_paths']
                user_prompt_parts.append(f"Images to process ({len(image_paths)} total):")
                if len(image_paths) <= 5:
                    for path in image_paths:
                        user_prompt_parts.append(f"  - {path}")
                else:
                    for path in image_paths[:3]:
                        user_prompt_parts.append(f"  - {path}")
                    user_prompt_parts.append(f"  ... and {len(image_paths) - 3} more")
                user_prompt_parts.append("")
            
            user_prompt_parts.append("Generate a workflow plan as JSON:")
            user_prompt = "\n".join(user_prompt_parts)
            
            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Call Gemini
            logger.info("Calling Gemini to plan workflow...")
            try:
                import google.generativeai as genai
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    response_mime_type='application/json'
                )
                response = self._gemini_client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            except Exception as e:
                # Fallback if response_mime_type not supported
                logger.warning(f"JSON mode not supported, using text mode: {e}")
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature
                )
                response = self._gemini_client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            
            # Parse JSON response
            try:
                plan_text = response.text.strip()
                # Remove markdown code blocks if present
                if plan_text.startswith('```'):
                    lines = plan_text.split('\n')
                    plan_text = '\n'.join(lines[1:-1]) if lines[-1].startswith('```') else '\n'.join(lines[1:])
                
                workflow_plan = json.loads(plan_text)
                logger.info(f"Generated workflow plan: {workflow_plan.get('name', 'unknown')}")
                return workflow_plan
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                logger.debug(f"Response text: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}", exc_info=True)
            return None
    
    def _build_workflow_from_plan(self, plan: Dict, context: Dict) -> Workflow:
        """Convert workflow plan to Workflow object."""
        steps = []
        
        for step_plan in plan.get('steps', []):
            # Resolve image_paths if not in params
            params = step_plan.get('params', {}).copy()
            if 'image_paths' not in params and context.get('image_paths'):
                params['image_paths'] = context['image_paths']
            
            step = WorkflowStep(
                name=step_plan['name'],
                tool_name=step_plan['tool_name'],
                params=params,
                dependencies=step_plan.get('dependencies', [])
            )
            steps.append(step)
        
        workflow = Workflow(
            name=plan.get('name', 'generated_workflow'),
            description=plan.get('description', ''),
            steps=steps
        )
        
        return workflow
    
    def _format_response_message(self, workflow: Workflow, results: Dict) -> str:
        """Format human-readable response message."""
        lines = [
            f"Completed workflow: {workflow.name}",
            f"Description: {workflow.description}",
            ""
        ]
        
        # Summarize steps
        lines.append("Steps executed:")
        for i, step in enumerate(workflow.steps, 1):
            status = "✓" if step.status == WorkflowStatus.COMPLETED else "✗"
            lines.append(f"  {i}. {status} {step.name} ({step.tool_name})")
        
        # Summarize key results
        if results:
            lines.append("\nResults:")
            for step_name, result in results.items():
                if isinstance(result, dict) and result.get('success'):
                    data = result.get('data', {})
                    if 'ranked_images' in data:
                        top_images = data['ranked_images'][:3]
                        lines.append(f"  {step_name}: Selected {len(data.get('ranked_images', []))} images")
                    elif 'selected_images' in data:
                        lines.append(f"  {step_name}: {len(data['selected_images'])} images selected")
                    elif 'message' in result:
                        lines.append(f"  {step_name}: {result['message']}")
        
        return "\n".join(lines)
    
    def _extract_selected_images(self, results: Dict) -> List[str]:
        """Extract selected image paths from workflow results."""
        selected = []
        
        for step_name, result in results.items():
            if isinstance(result, dict) and result.get('success'):
                data = result.get('data', {})
                
                # Check for ranked_images
                if 'ranked_images' in data:
                    selected.extend([img for img, _ in data['ranked_images']])
                
                # Check for selected_images
                if 'selected_images' in data:
                    selected.extend(data['selected_images'])
        
        return list(set(selected))  # Remove duplicates

