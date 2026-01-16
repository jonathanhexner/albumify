"""
Test agent with actual image processing (if images are available).

Quick test to verify agent can process queries with real images.
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.config import setup_logging
from sim_bench.agent.factory import create_agent

setup_logging()
logger = logging.getLogger(__name__)


def test_template_agent_with_images():
    """Test template agent with image context."""
    logger.info("=" * 80)
    logger.info("Testing Template Agent with Images")
    logger.info("=" * 80)

    # Create agent
    logger.info("\n1. Creating template agent...")
    agent = create_agent(agent_type='template')
    logger.info(f"   Agent created: {type(agent).__name__}")

    # Test without images
    logger.info("\n2. Testing without images (should fail gracefully)...")
    response = agent.process_query("Organize my photos by event")
    logger.info(f"   Success: {response.success}")
    logger.info(f"   Message: {response.message}")

    # Test with mock images
    logger.info("\n3. Testing with mock image paths...")
    mock_images = [f"photo_{i}.jpg" for i in range(10)]
    context = {'image_paths': mock_images}

    response = agent.process_query("Organize my photos by event", context)
    logger.info(f"   Success: {response.success}")
    logger.info(f"   Message: {response.message}")

    if response.success:
        logger.info(f"   Workflow: {response.metadata.get('workflow_name')}")
        logger.info(f"   Images processed: {response.metadata.get('num_images')}")

        # Check workflow in response
        workflow = response.data.get('workflow')
        if workflow:
            logger.info(f"\n   Workflow status: {workflow.status.value}")
            logger.info(f"   Steps completed: {sum(1 for s in workflow.steps if s.status.value == 'completed')}/{len(workflow.steps)}")
    else:
        logger.error(f"   Error: {response.metadata.get('error')}")

    logger.info("\n" + "=" * 80)


def test_with_real_images():
    """Test with real images if available."""
    logger.info("=" * 80)
    logger.info("Testing with Real Images (if available)")
    logger.info("=" * 80)

    # Check for sample images
    sample_dirs = [
        Path("D:/Similar Images/DataSets/InriaHolidaysFull"),
        Path("samples"),
        Path("test_images")
    ]

    image_paths = []
    for dir_path in sample_dirs:
        if dir_path.exists():
            images = list(dir_path.glob("*.jpg"))[:5]  # Take only 5 images for quick test
            if images:
                image_paths = [str(p) for p in images]
                logger.info(f"\n Found {len(image_paths)} images in {dir_path}")
                break

    if not image_paths:
        logger.warning("\n No real images found. Skipping real image test.")
        logger.info("   Place some .jpg files in 'samples/' directory to test with real images")
        return

    # Create agent
    logger.info("\n1. Creating template agent...")
    agent = create_agent(agent_type='template')

    # Test query
    logger.info("\n2. Processing query with real images...")
    context = {'image_paths': image_paths}

    try:
        response = agent.process_query("Organize my photos by event", context)
        logger.info(f"   Success: {response.success}")
        logger.info(f"   Message: {response.message}")

        if response.success:
            workflow = response.data.get('workflow')
            if workflow:
                logger.info(f"\n   Workflow: {workflow.name}")
                logger.info(f"   Status: {workflow.status.value}")

                # Show step results
                for step in workflow.steps:
                    logger.info(f"\n   Step: {step.name}")
                    logger.info(f"     Status: {step.status.value}")
                    if step.error:
                        logger.error(f"     Error: {step.error}")

        logger.info("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"\n   Error during execution: {e}", exc_info=True)


if __name__ == "__main__":
    test_template_agent_with_images()
    print("\n")
    test_with_real_images()
