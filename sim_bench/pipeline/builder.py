"""Pipeline builder with automatic dependency resolution."""

from sim_bench.pipeline.base import PipelineStep
from sim_bench.pipeline.registry import StepRegistry


class PipelineBuilder:
    """Build ordered pipeline from step names with automatic dependency resolution."""

    def __init__(self, registry: StepRegistry):
        self._registry = registry

    def build(self, step_names: list[str], auto_resolve: bool = True) -> list[PipelineStep]:
        """
        Build ordered pipeline from step names.

        Args:
            step_names: List of step names to include
            auto_resolve: If True, automatically add missing dependencies

        Returns:
            Ordered list of PipelineStep instances
        """
        if auto_resolve:
            resolved_names = self._resolve_dependencies(step_names)
        else:
            resolved_names = step_names

        ordered_names = self._topological_sort(resolved_names)
        return [self._registry.get(name) for name in ordered_names]

    def _resolve_dependencies(self, step_names: list[str]) -> list[str]:
        """Add missing dependencies to the step list."""
        resolved = set()
        to_process = list(step_names)

        while to_process:
            name = to_process.pop(0)
            if name in resolved:
                continue

            resolved.add(name)
            step = self._registry.get(name)

            for dep in step.metadata.depends_on:
                if dep not in resolved and self._registry.has_step(dep):
                    to_process.append(dep)

        return list(resolved)

    def _topological_sort(self, step_names: list[str]) -> list[str]:
        """Sort steps so dependencies come before dependents."""
        in_degree: dict[str, int] = {name: 0 for name in step_names}
        graph: dict[str, list[str]] = {name: [] for name in step_names}

        for name in step_names:
            step = self._registry.get(name)
            for dep in step.metadata.depends_on:
                if dep in step_names:
                    graph[dep].append(name)
                    in_degree[name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            queue.sort()
            name = queue.pop(0)
            result.append(name)

            for dependent in graph[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(step_names):
            missing = set(step_names) - set(result)
            raise ValueError(f"Circular dependency detected involving: {missing}")

        return result

    def validate_pipeline(self, step_names: list[str]) -> list[str]:
        """
        Validate that a pipeline is well-formed.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        for name in step_names:
            if not self._registry.has_step(name):
                errors.append(f"Unknown step: {name}")
                continue

            step = self._registry.get(name)
            for dep in step.metadata.depends_on:
                if dep not in step_names and not self._registry.has_step(dep):
                    errors.append(f"Step '{name}' depends on unknown step '{dep}'")

        return errors

    def get_execution_order(self, step_names: list[str]) -> list[str]:
        """Get the execution order for given steps (with dependencies resolved)."""
        steps = self.build(step_names, auto_resolve=True)
        return [s.metadata.name for s in steps]
