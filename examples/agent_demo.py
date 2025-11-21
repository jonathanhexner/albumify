"""
AI Agent Demo - Photo Organization with Template-Based Workflows

Demonstrates the agent system without requiring LLM API keys.
Uses pre-defined workflow templates.
"""

import sys
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.config import setup_logging
from sim_bench.agent.factory import create_agent, list_agent_types
from sim_bench.agent.workflows.templates import WorkflowTemplates

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def demo_template_agent():
    """
    Demo template-based agent (no LLM required).

    Uses pre-defined workflows that are matched based on keywords.
    """
    logger.info("="*80)
    logger.info("AI Agent Demo - Template-Based Workflows")
    logger.info("="*80)

    # Create template agent
    logger.info("\n1. Creating Template Agent")
    agent = create_agent(agent_type='template')
    logger.info(f"   Agent created: {type(agent).__name__}")

    # List available templates
    logger.info("\n2. Available Workflow Templates:")
    templates = WorkflowTemplates.list_templates()
    for name, description in templates.items():
        logger.info(f"   - {name}: {description}")

    # Example queries (would need actual images to execute)
    example_queries = [
        "Organize my photos by event",
        "Find my best portraits",
        "Organize my vacation photos"
    ]

    logger.info("\n3. Example Queries (not executed - need images):")
    for query in example_queries:
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Response: Agent would execute matching workflow\n")

    logger.info("="*80)
    logger.info("Demo Complete!")
    logger.info("="*80)

    logger.info("\nTo actually execute workflows:")
    logger.info("  1. Provide image paths")
    logger.info("  2. Call: response = agent.process_query(query)")
    logger.info("  3. Check: response.success, response.data, response.message")


def demo_workflow_structure():
    """
    Demo workflow structure without execution.

    Shows how workflows are composed of steps with dependencies.
    """
    logger.info("\n" + "="*80)
    logger.info("Workflow Structure Demo")
    logger.info("="*80)

    # Get example workflow
    workflow = WorkflowTemplates.organize_by_event(
        top_n_per_event=3,
        min_cluster_size=5
    )

    logger.info(f"\nWorkflow: {workflow.name}")
    logger.info(f"Description: {workflow.description}")
    logger.info(f"\nSteps ({len(workflow.steps)}):")

    for i, step in enumerate(workflow.steps, 1):
        logger.info(f"\n  Step {i}: {step.name}")
        logger.info(f"    Tool: {step.tool_name}")
        logger.info(f"    Params: {step.params}")
        logger.info(f"    Dependencies: {step.dependencies if step.dependencies else 'None'}")

    # Validate workflow
    errors = workflow.validate_dependencies()
    if errors:
        logger.error(f"\nWorkflow validation errors: {errors}")
    else:
        logger.info("\n[OK] Workflow is valid (no circular dependencies)")

    logger.info("\n" + "="*80)


def demo_tool_registry():
    """
    Demo tool registry - shows all available tools.
    """
    logger.info("\n" + "="*80)
    logger.info("Tool Registry Demo")
    logger.info("="*80)

    from sim_bench.agent.tools.registry import get_registry

    registry = get_registry()

    logger.info(f"\nTotal tools registered: {len(registry.list_tools())}")

    # Group by category
    by_category = registry.get_tools_by_category()

    logger.info("\nTools by category:")
    for category, tools in sorted(by_category.items(), key=lambda x: x[0].value):
        logger.info(f"\n  {category.value.upper()} ({len(tools)} tools):")
        for tool_name in tools:
            schema = registry.get_tool_schema(tool_name)
            logger.info(f"    - {tool_name}: {schema['description']}")

    # Show example for one tool
    logger.info("\n" + "-"*80)
    logger.info("Example Tool: clip_tag_images")
    logger.info("-"*80)

    examples = registry.get_tool_examples('clip_tag_images')
    for i, example in enumerate(examples, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Query: {example['query']}")
        logger.info(f"  Description: {example['description']}")
        logger.info(f"  Params: {example['params']}")

    logger.info("\n" + "="*80)


def demo_agent_types():
    """Show available agent types."""
    logger.info("\n" + "="*80)
    logger.info("Available Agent Types")
    logger.info("="*80)

    agent_types = list_agent_types()

    for agent_type, description in agent_types.items():
        logger.info(f"\n  {agent_type}:")
        logger.info(f"    {description}")

    logger.info("\n" + "="*80)


def main():
    """Run all demos."""
    logger.info("\n" + "="*80)
    logger.info("AI AGENT SYSTEM DEMONSTRATION")
    logger.info("="*80)

    demo_agent_types()
    demo_template_agent()
    demo_workflow_structure()
    demo_tool_registry()

    logger.info("\n" + "="*80)
    logger.info("ALL DEMOS COMPLETE")
    logger.info("="*80)

    logger.info("\nNext Steps:")
    logger.info("  1. Provide actual image paths to execute workflows")
    logger.info("  2. Integrate with Streamlit for UI")
    logger.info("  3. Add LLM integration for WorkflowAgent")
    logger.info("  4. Create custom workflows for specific use cases")


if __name__ == "__main__":
    main()
