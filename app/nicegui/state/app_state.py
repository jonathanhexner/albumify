"""Application state management."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppState:
    """Global application state."""

    current_album: Optional[dict] = None
    current_job_id: Optional[str] = None
    pipeline_status: Optional[dict] = None
    pipeline_result: Optional[dict] = None

    selected_steps: list[str] = field(default_factory=list)
    step_configs: dict[str, dict] = field(default_factory=dict)

    progress: float = 0.0
    current_step: str = ""
    progress_message: str = ""

    error_message: Optional[str] = None

    def reset_pipeline(self) -> None:
        """Reset pipeline-related state."""
        self.current_job_id = None
        self.pipeline_status = None
        self.pipeline_result = None
        self.progress = 0.0
        self.current_step = ""
        self.progress_message = ""
        self.error_message = None


app_state = AppState()
