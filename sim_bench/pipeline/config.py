"""Pipeline configuration."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    fail_fast: bool = True
    step_configs: dict[str, dict] = field(default_factory=dict)
    progress_callback: Callable[[str, float, str], None] = None

    def get_step_config(self, step_name: str) -> dict:
        """Get configuration for a specific step."""
        return self.step_configs.get(step_name, {})
