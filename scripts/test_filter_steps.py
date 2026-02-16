"""Test the new filter_faces and score_face_frontal steps."""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Import steps to register them
from sim_bench.pipeline.steps.all_steps import *
from sim_bench.pipeline.registry import get_registry
from sim_bench.pipeline.context import PipelineContext

def main():
    # Check steps are registered
    registry = get_registry()
    print("\n=== Registered Steps ===")
    all_steps = registry.list_step_names()
    print(f"Total steps: {len(all_steps)}")

    # Check new steps
    new_steps = ["filter_faces", "score_face_frontal"]
    for step_name in new_steps:
        if registry.has_step(step_name):
            print(f"  [OK] {step_name} is registered")
        else:
            print(f"  [MISSING] {step_name} is NOT registered")

    # Try to get and instantiate the steps
    print("\n=== Step Instantiation ===")
    for step_name in new_steps:
        try:
            step = registry.get_step(step_name)
            print(f"  [OK] {step_name}: {step.metadata.display_name}")
            print(f"       requires: {step.metadata.requires}")
            print(f"       produces: {step.metadata.produces}")
            print(f"       depends_on: {step.metadata.depends_on}")
        except Exception as e:
            print(f"  [ERROR] {step_name}: {e}")

    # Check pipeline config
    print("\n=== Pipeline Config ===")
    import yaml
    config_path = Path("D:/sim-bench/configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    default_pipeline = config.get("default_pipeline", [])
    print(f"default_pipeline has {len(default_pipeline)} steps:")
    for i, step in enumerate(default_pipeline):
        marker = " <-- NEW" if step in new_steps else ""
        print(f"  {i+1}. {step}{marker}")

if __name__ == "__main__":
    main()
