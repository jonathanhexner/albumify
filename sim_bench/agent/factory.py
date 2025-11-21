"""
Factory for creating configured agent instances.

Handles initialization of all agent components with proper dependency injection.
"""

from typing import Dict, Optional
import logging

from sim_bench.agent.core.base import Agent
from sim_bench.agent.core.memory import AgentMemory
from sim_bench.agent.tools.registry import ToolRegistry, get_registry
from sim_bench.config import GlobalConfig

logger = logging.getLogger(__name__)


def create_agent(
    agent_type: str = 'workflow',
    llm_provider: str = 'openai',
    model: str = 'gpt-4-turbo-preview',
    config: Optional[Dict] = None,
    tool_registry: Optional[ToolRegistry] = None,
    memory: Optional[AgentMemory] = None
) -> Agent:
    """
    Factory function to create configured agent instances.

    Args:
        agent_type: Type of agent:
            - 'template': Pre-defined workflows (no LLM needed)
            - 'workflow': LLM-planned workflows (requires LLM provider)
            - 'conversational': Free-form LLM conversation (requires LLM provider)
        llm_provider: LLM provider for workflow/conversational agents:
            - 'openai': OpenAI models (GPT-4, etc.)
            - 'anthropic': Anthropic models (Claude, etc.)
            - 'gemini': Google Gemini models
        model: Model name/identifier (e.g., 'gpt-4', 'claude-3-opus', 'gemini-pro')
        config: Additional configuration
        tool_registry: Optional custom tool registry (uses global if None)
        memory: Optional custom memory (creates new if None)

    Returns:
        Configured Agent instance

    Raises:
        ValueError: If agent_type or llm_provider unknown
        ImportError: If required dependencies missing

    Example:
        >>> # Template agent (no LLM)
        >>> agent = create_agent(agent_type='template')
        
        >>> # Workflow agent with OpenAI
        >>> agent = create_agent(
        ...     agent_type='workflow',
        ...     llm_provider='openai',
        ...     model='gpt-4'
        ... )
        
        >>> # Workflow agent with Gemini
        >>> agent = create_agent(
        ...     agent_type='workflow',
        ...     llm_provider='gemini',
        ...     model='gemini-pro'
        ... )
    """
    config = config or {}

    # Load global config if not provided
    if 'global_config' not in config:
        try:
            config['global_config'] = GlobalConfig()
        except Exception as e:
            logger.warning(f"Could not load global config: {e}")
            config['global_config'] = None

    # Get tool registry (use global if not provided)
    if tool_registry is None:
        tool_registry = get_registry()
        logger.info(f"Using global tool registry with {len(tool_registry.list_tools())} tools")

    # Create memory if not provided
    if memory is None:
        memory = AgentMemory()
        logger.info("Created new agent memory")

    # Create agent based on type
    if agent_type == 'template':
        # Template agent doesn't need LLM
        return _create_template_agent(
            config=config,
            tool_registry=tool_registry,
            memory=memory
        )
    elif agent_type == 'workflow':
        # Workflow agent uses LLM to plan workflows
        return _create_workflow_agent(
            llm_provider=llm_provider,
            model=model,
            config=config,
            tool_registry=tool_registry,
            memory=memory
        )
    elif agent_type == 'conversational':
        # Conversational agent for free-form chat
        return _create_conversational_agent(
            llm_provider=llm_provider,
            model=model,
            config=config,
            tool_registry=tool_registry,
            memory=memory
        )
    else:
        raise ValueError(
            f"Unknown agent_type: {agent_type}. "
            f"Supported: 'template', 'workflow', 'conversational'"
        )


def _create_workflow_agent(
    llm_provider: str,
    model: str,
    config: Dict,
    tool_registry: ToolRegistry,
    memory: AgentMemory
) -> Agent:
    """
    Create workflow-based agent (uses LLM to plan workflows).

    Args:
        llm_provider: LLM provider name ('openai', 'anthropic', 'gemini')
        model: Model name (e.g., 'gpt-4', 'claude-3-opus', 'gemini-pro')
        config: Configuration dictionary
        tool_registry: Tool registry instance
        memory: Agent memory instance

    Returns:
        WorkflowAgent instance (or GeminiAgent if provider is 'gemini')
    """
    # If using Gemini, use the GeminiAgent implementation
    if llm_provider == 'gemini':
        logger.info("Creating workflow agent with Gemini LLM provider")
        from sim_bench.agent.agents.gemini_agent import GeminiAgent
        # Update config with model name
        config = config or {}
        config['model'] = model
        return GeminiAgent(tool_registry, memory, config)
    
    # For other providers, use generic workflow agent (not yet implemented)
    logger.warning(f"WorkflowAgent with {llm_provider} not yet implemented, using placeholder")

    from sim_bench.agent.core.base import Agent, AgentResponse

    class PlaceholderWorkflowAgent(Agent):
        """Placeholder until full implementation."""

        def __init__(self, tool_registry, memory, config):
            super().__init__(config)
            self.tool_registry = tool_registry
            self.memory = memory
            self.llm_provider = llm_provider
            self.model = model

        def process_query(self, query: str, context: Dict = None) -> AgentResponse:
            """Placeholder implementation."""
            return AgentResponse(
                success=False,
                message=f"WorkflowAgent with {self.llm_provider} not yet implemented. Use TemplateAgent for now, or use Gemini provider.",
                data={},
                metadata={'agent_type': 'placeholder', 'llm_provider': self.llm_provider}
            )

        def refine(self, feedback: str) -> AgentResponse:
            """Placeholder implementation."""
            return self.process_query(feedback)

    return PlaceholderWorkflowAgent(tool_registry, memory, config)


def _create_template_agent(
    config: Dict,
    tool_registry: ToolRegistry,
    memory: AgentMemory
) -> Agent:
    """
    Create template-based agent (uses pre-defined workflows).

    Args:
        config: Configuration dictionary
        tool_registry: Tool registry instance
        memory: Agent memory instance

    Returns:
        TemplateAgent instance
    """
    from sim_bench.agent.agents.template_agent import TemplateAgent
    
    return TemplateAgent(tool_registry, memory, config)


def _create_conversational_agent(
    llm_provider: str,
    model: str,
    config: Dict,
    tool_registry: ToolRegistry,
    memory: AgentMemory
) -> Agent:
    """
    Create conversational agent (free-form LLM conversation).

    Args:
        llm_provider: LLM provider name
        model: Model name
        config: Configuration dictionary
        tool_registry: Tool registry instance
        memory: Agent memory instance

    Returns:
        ConversationalAgent instance
    """
    logger.warning("ConversationalAgent not yet implemented")
    # Placeholder for now
    return _create_workflow_agent(llm_provider, model, config, tool_registry, memory)




def list_agent_types() -> Dict[str, str]:
    """
    Get list of available agent types.

    Returns:
        Dictionary mapping agent type to description
    """
    return {
        'template': 'Pre-defined workflows (no LLM needed, faster)',
        'workflow': 'LLM-planned workflows (most flexible, requires LLM provider)',
        'conversational': 'Free-form conversation (requires LLM provider)'
    }


def list_llm_providers() -> Dict[str, str]:
    """
    Get list of available LLM providers.

    Returns:
        Dictionary mapping provider name to description
    """
    return {
        'openai': 'OpenAI models (GPT-4, GPT-3.5, etc.) - requires OPENAI_API_KEY',
        'anthropic': 'Anthropic models (Claude, etc.) - requires ANTHROPIC_API_KEY',
        'gemini': 'Google Gemini models - requires GEMINI_API_KEY'
    }
