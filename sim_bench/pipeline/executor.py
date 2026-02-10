"""Pipeline executor - runs pipeline steps in order."""

import logging
import time
from dataclasses import dataclass, field
from typing import Generator

from sim_bench.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.registry import StepRegistry
from sim_bench.pipeline.builder import PipelineBuilder


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_name: str
    success: bool
    duration_ms: int
    error_message: str = None


@dataclass
class PipelineResult:
    """Result of executing a full pipeline."""
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    total_duration_ms: int = 0
    error_message: str = None

    @property
    def failed_step(self) -> str:
        """Get the name of the first failed step, if any."""
        for result in self.step_results:
            if not result.success:
                return result.step_name
        return None


class PipelineExecutor:
    """Executes pipeline steps in order with progress reporting."""

    def __init__(self, registry: StepRegistry):
        self._registry = registry
        self._builder = PipelineBuilder(registry)

    def execute(
        self,
        context: PipelineContext,
        step_names: list[str],
        config: PipelineConfig = None
    ) -> PipelineResult:
        """
        Execute a pipeline.

        Args:
            context: Pipeline context with input data
            step_names: List of step names to execute
            config: Pipeline configuration

        Returns:
            PipelineResult with success status and timing info
        """
        if config is None:
            config = PipelineConfig()

        context.on_progress = config.progress_callback

        steps = self._builder.build(step_names, auto_resolve=True)
        resolved_names = [s.metadata.name for s in steps]
        logger.info(f"Pipeline steps (after dependency resolution): {resolved_names}")

        result = PipelineResult(success=True)
        start_time = time.time()

        for i, step in enumerate(steps):
            step_result = self._execute_step(step, context, config)
            result.step_results.append(step_result)

            if not step_result.success:
                result.success = False
                result.error_message = f"Step '{step.metadata.name}' failed: {step_result.error_message}"
                if config.fail_fast:
                    break

        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result

    def execute_streaming(
        self,
        context: PipelineContext,
        step_names: list[str],
        config: PipelineConfig = None
    ) -> Generator[StepResult, None, PipelineResult]:
        """
        Execute pipeline with streaming results.

        Yields StepResult after each step completes.
        Returns final PipelineResult.
        """
        if config is None:
            config = PipelineConfig()

        context.on_progress = config.progress_callback

        steps = self._builder.build(step_names, auto_resolve=True)
        step_results = []
        start_time = time.time()
        success = True
        error_message = None

        for step in steps:
            step_result = self._execute_step(step, context, config)
            step_results.append(step_result)
            yield step_result

            if not step_result.success:
                success = False
                error_message = f"Step '{step.metadata.name}' failed: {step_result.error_message}"
                if config.fail_fast:
                    break

        return PipelineResult(
            success=success,
            step_results=step_results,
            total_duration_ms=int((time.time() - start_time) * 1000),
            error_message=error_message
        )

    def _execute_step(
        self,
        step: PipelineStep,
        context: PipelineContext,
        config: PipelineConfig
    ) -> StepResult:
        """Execute a single step."""
        step_name = step.metadata.name
        step_config = config.get_step_config(step_name)
        start_time = time.time()

        validation_errors = step.validate(context)
        if validation_errors:
            return StepResult(
                step_name=step_name,
                success=False,
                duration_ms=0,
                error_message=f"Validation failed: {'; '.join(validation_errors)}"
            )

        step.process(context, step_config)
        duration_ms = int((time.time() - start_time) * 1000)

        return StepResult(
            step_name=step_name,
            success=True,
            duration_ms=duration_ms
        )

    def get_execution_plan(self, step_names: list[str]) -> list[dict]:
        """
        Get execution plan showing step order and dependencies.

        Returns list of dicts with step info.
        """
        steps = self._builder.build(step_names, auto_resolve=True)
        return [
            {
                "order": i + 1,
                "name": step.metadata.name,
                "display_name": step.metadata.display_name,
                "depends_on": step.metadata.depends_on,
            }
            for i, step in enumerate(steps)
        ]
