#!/usr/bin/env python
"""
Standalone CLI for running the photo organization agent.

This allows testing and using the agent without the Streamlit app.

Usage:
    # Interactive mode
    python run_agent_cli.py

    # Single task mode
    python run_agent_cli.py --task "Find all blurry photos in D:/Photos"

    # With specific tools enabled
    python run_agent_cli.py --tools quality,clustering --task "..."
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from sim_bench.agent.factory import create_agent


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Photo Organization Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (ask for tasks)
  python run_agent_cli.py

  # Execute single task
  python run_agent_cli.py --task "Analyze quality of images in D:/Photos/Vacation"

  # Enable specific tools only
  python run_agent_cli.py --tools quality,clustering --task "..."

  # Change model
  python run_agent_cli.py --model claude-sonnet-3-5 --task "..."
        """
    )

    parser.add_argument(
        '--task',
        type=str,
        help='Task description for the agent to execute'
    )

    parser.add_argument(
        '--tools',
        type=str,
        help='Comma-separated list of tool categories to enable (quality,clustering,analysis,face,landmark)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='claude-3-5-sonnet-20241022',
        help='Model to use (default: claude-3-5-sonnet-20241022)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum agent iterations (default: 10)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--working-dir',
        type=str,
        help='Working directory for agent operations (default: current directory)'
    )

    return parser.parse_args()


def interactive_mode(agent, args):
    """Run agent in interactive mode."""
    print("\n" + "="*80)
    print("Photo Organization Agent - Interactive Mode")
    print("="*80)
    print("\nType 'exit' or 'quit' to stop")
    print("Type 'help' for example tasks\n")

    while True:
        try:
            task = input("\n> What would you like me to do? ")

            if not task.strip():
                continue

            if task.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            if task.lower() == 'help':
                print_help_examples()
                continue

            # Execute task
            print(f"\n{'='*80}")
            print(f"Executing: {task}")
            print(f"{'='*80}\n")

            result = agent.execute(task, working_directory=args.working_dir)

            print(f"\n{'='*80}")
            print("Result:")
            print(f"{'='*80}")
            print(result.get('message', 'No message'))

            if result.get('data'):
                print("\nData:")
                print(result['data'])

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logging.exception("Error executing task")


def print_help_examples():
    """Print example tasks."""
    examples = [
        "Analyze quality of all photos in D:/Photos/Vacation",
        "Find all blurry images in D:/Photos",
        "Cluster similar photos in D:/Photos/Events",
        "Find photos with faces in D:/Photos/Family",
        "Select the best photo from each burst sequence in D:/Photos",
        "Analyze landmarks in photos in D:/Photos/Travel",
    ]

    print("\nExample tasks:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")


def single_task_mode(agent, task: str):
    """Execute a single task and exit."""
    print(f"\n{'='*80}")
    print(f"Task: {task}")
    print(f"{'='*80}\n")

    result = agent.execute(task, working_directory=args.working_dir)

    print(f"\n{'='*80}")
    print("Result:")
    print(f"{'='*80}")
    print(result.get('message', 'No message'))

    if result.get('data'):
        print("\nData:")
        print(result['data'])

    return 0 if result.get('success') else 1


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Parse tool categories
    tool_categories = None
    if args.tools:
        tool_categories = [t.strip() for t in args.tools.split(',')]
        logger.info(f"Enabled tool categories: {tool_categories}")

    # Create agent config
    config = {
        'model': args.model,
        'max_iterations': args.max_iterations,
        'working_directory': args.working_dir or str(Path.cwd()),
        'enabled_tool_categories': tool_categories
    }

    # Create agent
    logger.info(f"Creating agent with model: {args.model}")
    agent = create_agent(
        agent_type='template',  # Default to template agent (no LLM needed)
        config=config
    )

    # Run agent
    try:
        if args.task:
            # Single task mode
            return single_task_mode(agent, args.task)
        else:
            # Interactive mode
            interactive_mode(agent, args)
            return 0

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
