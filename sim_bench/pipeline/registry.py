"""Step registry for discovering and managing pipeline steps."""

from typing import Type
from sim_bench.pipeline.base import PipelineStep, StepMetadata


class StepRegistry:
    """Registry for discovering and managing pipeline steps."""

    def __init__(self):
        self._steps: dict[str, PipelineStep] = {}
        self._step_classes: dict[str, Type[PipelineStep]] = {}

    def register(self, step_class: Type[PipelineStep]) -> None:
        """Register a step class."""
        instance = step_class()
        name = instance.metadata.name
        self._steps[name] = instance
        self._step_classes[name] = step_class

    def register_instance(self, step: PipelineStep) -> None:
        """Register a step instance directly."""
        name = step.metadata.name
        self._steps[name] = step

    def get(self, name: str) -> PipelineStep:
        """Get a step by name."""
        if name not in self._steps:
            raise KeyError(f"Step not found: {name}. Available: {list(self._steps.keys())}")
        return self._steps[name]

    def get_metadata(self, name: str) -> StepMetadata:
        """Get step metadata by name."""
        return self.get(name).metadata

    def list_steps(self, category: str = None) -> list[StepMetadata]:
        """List all registered steps, optionally filtered by category."""
        steps = [step.metadata for step in self._steps.values()]
        if category:
            steps = [s for s in steps if s.category == category]
        return steps

    def list_step_names(self) -> list[str]:
        """List all registered step names."""
        return list(self._steps.keys())

    def has_step(self, name: str) -> bool:
        """Check if a step is registered."""
        return name in self._steps

    def find_by_produces(self, context_key: str) -> list[str]:
        """Find steps that produce a given context key."""
        return [
            name for name, step in self._steps.items()
            if context_key in step.metadata.produces
        ]

    def find_by_requires(self, context_key: str) -> list[str]:
        """Find steps that require a given context key."""
        return [
            name for name, step in self._steps.items()
            if context_key in step.metadata.requires
        ]


# Global registry instance
_global_registry: StepRegistry = None


def get_registry() -> StepRegistry:
    """Get the global step registry, creating it if needed."""
    global _global_registry
    if _global_registry is None:
        _global_registry = StepRegistry()
    return _global_registry


def register_step(step_class: Type[PipelineStep]) -> Type[PipelineStep]:
    """Decorator to register a step class with the global registry."""
    get_registry().register(step_class)
    return step_class
