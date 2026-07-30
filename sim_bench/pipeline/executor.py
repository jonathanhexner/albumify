"""Pipeline executor - runs pipeline steps in order."""

import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Iterable

from sim_bench.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.config import PipelineConfig
from sim_bench.pipeline.registry import StepRegistry
from sim_bench.pipeline.builder import PipelineBuilder


def _summarize_one(context: PipelineContext, key: str) -> str:
    """Render a single context key as ``key=<shape>`` for log lines.

    Best-effort: lists / dicts / numpy arrays show their length, scalars
    show ``=<scalar>``, missing keys show ``=missing``, None shows
    ``=None``. Used by the per-step in/out telemetry in ``_execute_step``.
    """
    if not hasattr(context, key):
        return f"{key}=missing"
    val = getattr(context, key)
    if val is None:
        return f"{key}=None"
    try:
        return f"{key}={len(val)}"
    except TypeError:
        return f"{key}=<scalar>"


def _summarize_keys(context: PipelineContext, keys: Iterable[str]) -> str:
    """Comma-join ``_summarize_one`` over a set of context keys."""
    ordered = sorted(keys) if keys else []
    if not ordered:
        return "—"
    return ", ".join(_summarize_one(context, k) for k in ordered)


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
        config: PipelineConfig = None,
        on_step_complete=None,
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

            # Notify caller of per-step completion (for DB persistence)
            if on_step_complete:
                try:
                    on_step_complete(step_result)
                except Exception as e:
                    logger.warning(f"on_step_complete callback failed: {e}")

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
            # spec-041 follow-up #2: log the validation failure with the
            # input shape so the diagnostic doesn't depend on someone
            # scrolling back through prior step logs to figure out what
            # context state caused the validator to fire.
            in_summary = _summarize_keys(context, step.metadata.requires)
            logger.error(
                "%s: validation failed (in[%s]): %s",
                step_name, in_summary, "; ".join(validation_errors),
            )
            return StepResult(
                step_name=step_name,
                success=False,
                duration_ms=0,
                error_message=f"Validation failed: {'; '.join(validation_errors)}"
            )

        try:
            step.process(context, step_config)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Step '{step_name}' failed after {duration_ms}ms: {e}", exc_info=True)
            return StepResult(
                step_name=step_name,
                success=False,
                duration_ms=duration_ms,
                error_message=f"{type(e).__name__}: {e}"
            )
        finally:
            # SIGHTING-117: free this step's model before the next step loads its
            # own, so peak RSS is one model instead of the sum of all. Runs on
            # success AND failure. Best-effort: a release() bug must never mask
            # the step's real result.
            try:
                step.release()
            except Exception:
                logger.warning("%s: release() failed (non-fatal)", step_name, exc_info=True)

        duration_ms = int((time.time() - start_time) * 1000)

        # spec-041 follow-up #2: one INFO line per successful step with
        # input / output counts pulled from metadata. Standardized so any
        # post-mortem reads "what was the shape going in / coming out"
        # without each step having to log its own.
        in_summary = _summarize_keys(context, step.metadata.requires)
        out_summary = _summarize_keys(context, step.metadata.produces)
        logger.info(
            "%s: in[%s] -> out[%s] (%dms)",
            step_name, in_summary, out_summary, duration_ms,
        )

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
