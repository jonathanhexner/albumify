"""Pipeline configuration."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    fail_fast: bool = True
    step_configs: dict[str, dict] = field(default_factory=dict)
    progress_callback: Callable[[str, float, str], None] = None
    # spec-104 v1: run each step in a subprocess that exits after it finishes, so the
    # OS reclaims 100% of whatever model(s) the step loaded (the only reliable free()
    # for torch CPU memory on Windows — see SIGHTING-117). Default OFF: the in-process
    # path is byte-identical when this is False.
    isolate_steps: bool = False
    # Wall-clock ceiling for a single isolated step. On expiry the child is
    # terminated and the step returns a failed StepResult — so a hung step (deadlock,
    # infinite loop, stalled download) can't hang the parent forever. Generous by
    # default (heavy model load + inference); lower it for bounded/interactive runs.
    isolate_step_timeout_s: float = 600.0

    def get_step_config(self, step_name: str) -> dict:
        """Get configuration for a specific step."""
        return self.step_configs.get(step_name, {})
