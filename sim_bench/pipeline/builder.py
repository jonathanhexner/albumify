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
        """Add missing dependencies, preserving the author's declared order.

        Order matters: the topological sort tie-breaks by position in this list, so the
        pipeline.yaml order is honored among independent steps (e.g. triage before scoring).
        Listed steps keep their given order; auto-added dependencies are appended after.
        """
        resolved: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name not in seen:
                seen.add(name)
                resolved.append(name)

        for name in step_names:          # keep the author's order first
            add(name)
        i = 0
        while i < len(resolved):         # then pull in any missing (transitive) deps
            step = self._registry.get(resolved[i])
            for dep in step.metadata.depends_on:
                if dep not in seen and self._registry.has_step(dep):
                    add(dep)
            i += 1
        return resolved

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

        # Tie-break by the author's declared order (index in step_names), NOT alphabetically.
        # Among steps whose dependencies are all satisfied, the one listed earliest in the
        # pipeline runs first — so pipeline.yaml order is meaningful (e.g. triage before scoring).
        rank = {name: i for i, name in enumerate(step_names)}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            queue.sort(key=lambda n: rank[n])
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
