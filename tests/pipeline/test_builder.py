"""Tests for PipelineBuilder - dependency resolution and ordering."""

import pytest
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import StepRegistry
from sim_bench.pipeline.builder import PipelineBuilder


class StepA(BaseStep):
    """Step with no dependencies."""
    def __init__(self):
        self._metadata = StepMetadata(
            name="step_a",
            display_name="Step A",
            description="First step",
            category="test",
            requires=set(),
            produces={"output_a"},
            depends_on=[]
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        pass


class StepB(BaseStep):
    """Step that depends on A."""
    def __init__(self):
        self._metadata = StepMetadata(
            name="step_b",
            display_name="Step B",
            description="Depends on A",
            category="test",
            requires={"output_a"},
            produces={"output_b"},
            depends_on=["step_a"]
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        pass


class StepC(BaseStep):
    """Step that depends on B."""
    def __init__(self):
        self._metadata = StepMetadata(
            name="step_c",
            display_name="Step C",
            description="Depends on B",
            category="test",
            requires={"output_b"},
            produces={"output_c"},
            depends_on=["step_b"]
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        pass


class StepD(BaseStep):
    """Step that depends on both A and B."""
    def __init__(self):
        self._metadata = StepMetadata(
            name="step_d",
            display_name="Step D",
            description="Depends on A and B",
            category="test",
            requires={"output_a", "output_b"},
            produces={"output_d"},
            depends_on=["step_a", "step_b"]
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        pass


class StepM(BaseStep):
    """Sibling of StepZ — both depend only on step_a (same dependency level)."""
    def __init__(self):
        self._metadata = StepMetadata(
            name="step_m", display_name="Step M", description="dep A", category="test",
            requires={"output_a"}, produces={"output_m"}, depends_on=["step_a"])

    def process(self, context: PipelineContext, config: dict) -> None:
        pass


class StepZ(BaseStep):
    """Sibling of StepM — alphabetically after it, but may be declared first."""
    def __init__(self):
        self._metadata = StepMetadata(
            name="step_z", display_name="Step Z", description="dep A", category="test",
            requires={"output_a"}, produces={"output_z"}, depends_on=["step_a"])

    def process(self, context: PipelineContext, config: dict) -> None:
        pass


@pytest.fixture
def registry():
    """Create a registry with test steps."""
    reg = StepRegistry()
    reg.register(StepA)
    reg.register(StepB)
    reg.register(StepC)
    reg.register(StepD)
    reg.register(StepM)
    reg.register(StepZ)
    return reg


@pytest.fixture
def builder(registry):
    """Create a builder with the test registry."""
    return PipelineBuilder(registry)


class TestDependencyResolution:
    """Tests for automatic dependency resolution."""

    def test_single_step_no_deps(self, builder):
        """Step with no dependencies should just return itself."""
        steps = builder.build(["step_a"], auto_resolve=True)
        assert len(steps) == 1
        assert steps[0].metadata.name == "step_a"

    def test_auto_adds_missing_dependency(self, builder):
        """Requesting step_b should auto-add step_a."""
        steps = builder.build(["step_b"], auto_resolve=True)
        names = [s.metadata.name for s in steps]

        assert "step_a" in names
        assert "step_b" in names
        assert names.index("step_a") < names.index("step_b")

    def test_chain_dependencies(self, builder):
        """Requesting step_c should add step_a and step_b in order."""
        steps = builder.build(["step_c"], auto_resolve=True)
        names = [s.metadata.name for s in steps]

        assert names == ["step_a", "step_b", "step_c"]

    def test_multiple_dependencies(self, builder):
        """Step with multiple dependencies should have all resolved."""
        steps = builder.build(["step_d"], auto_resolve=True)
        names = [s.metadata.name for s in steps]

        assert "step_a" in names
        assert "step_b" in names
        assert "step_d" in names
        # A must come before B (B depends on A)
        assert names.index("step_a") < names.index("step_b")
        # Both A and B must come before D
        assert names.index("step_a") < names.index("step_d")
        assert names.index("step_b") < names.index("step_d")

    def test_no_duplicates(self, builder):
        """Requesting steps with shared dependencies shouldn't duplicate."""
        steps = builder.build(["step_b", "step_c"], auto_resolve=True)
        names = [s.metadata.name for s in steps]

        # step_a should only appear once
        assert names.count("step_a") == 1
        assert names.count("step_b") == 1


class TestTopologicalSort:
    """Tests for correct execution ordering."""

    def test_preserves_dependency_order(self, builder):
        """Dependencies must always come before dependents."""
        # Request in reverse order
        steps = builder.build(["step_c", "step_b", "step_a"], auto_resolve=False)
        names = [s.metadata.name for s in steps]

        assert names.index("step_a") < names.index("step_b")
        assert names.index("step_b") < names.index("step_c")

    def test_independent_steps_keep_declared_order(self, builder):
        """spec-107: among steps at the same dependency level, the author's declared
        order wins over alphabetical — so pipeline.yaml order is meaningful (e.g. triage
        before scoring). step_z is declared before step_m despite sorting after it."""
        steps = builder.build(["step_a", "step_z", "step_m"], auto_resolve=True)
        names = [s.metadata.name for s in steps]
        assert names.index("step_z") < names.index("step_m")

    def test_resolve_preserves_order_of_listed_steps(self, builder):
        """Auto-resolve keeps listed steps in declared order (not set-arbitrary)."""
        steps = builder.build(["step_z", "step_m"], auto_resolve=True)  # both auto-add step_a
        names = [s.metadata.name for s in steps]
        assert names[0] == "step_a"                      # dep first
        assert names.index("step_z") < names.index("step_m")

    def test_complex_dag(self, builder):
        """Test with a more complex dependency graph."""
        steps = builder.build(["step_c", "step_d"], auto_resolve=True)
        names = [s.metadata.name for s in steps]

        # A must be first
        assert names[0] == "step_a"
        # B must come after A but before C and D
        assert names.index("step_a") < names.index("step_b")
        assert names.index("step_b") < names.index("step_c")
        assert names.index("step_b") < names.index("step_d")


class TestValidation:
    """Tests for pipeline validation."""

    def test_unknown_step_error(self, builder):
        """Requesting unknown step should raise error."""
        with pytest.raises(KeyError):
            builder.build(["nonexistent_step"])

    def test_validate_returns_errors_for_unknown(self, builder):
        """validate_pipeline should return errors for unknown steps."""
        errors = builder.validate_pipeline(["step_a", "unknown_step"])
        assert len(errors) > 0
        assert "unknown_step" in errors[0].lower() or "Unknown" in errors[0]


class TestExecutionPlan:
    """Tests for get_execution_plan."""

    def test_execution_plan_format(self, registry):
        """Execution plan should have correct format."""
        from sim_bench.pipeline.executor import PipelineExecutor

        executor = PipelineExecutor(registry)
        plan = executor.get_execution_plan(["step_c"])

        assert len(plan) == 3
        assert plan[0]["order"] == 1
        assert plan[0]["name"] == "step_a"
        assert plan[1]["order"] == 2
        assert plan[1]["name"] == "step_b"
        assert plan[2]["order"] == 3
        assert plan[2]["name"] == "step_c"


class TestRealRegistry:
    """Checks on the real step registry and the default pipeline (SIGHTING-132, spec-113)."""

    @pytest.fixture
    def registry(self):
        import sim_bench.pipeline.steps.all_steps  # noqa: F401  registers every step
        from sim_bench.pipeline.registry import get_registry
        return get_registry()

    def test_every_dependency_names_a_registered_step(self, registry):
        """The builder silently skips an unknown dependency, so a stale name (a removed step)
        would order nothing. SIGHTING-132 was the opposite case: the stale step still existed
        and the builder pulled it into every run."""
        stale = {n: [d for d in registry.get_metadata(n).depends_on if not registry.has_step(d)]
                 for n in registry.list_step_names()}
        assert {n: d for n, d in stale.items() if d} == {}

    def test_people_steps_run_after_clustering_and_before_selection(self, registry):
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(Path("configs/pipeline.yaml").read_text(encoding="utf-8"))
        order = PipelineBuilder(registry).get_execution_order(cfg["default_pipeline"])
        pos = {name: i for i, name in enumerate(order)}
        assert pos["assign_people_clusters"] < pos["identity_refinement"] < pos["cluster_by_identity"]
        assert pos["cluster_by_identity"] < pos["select_best"]
