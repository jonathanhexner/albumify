"""Type definitions and enums for album module."""

from enum import Enum, auto
from typing import Callable, Optional


class WorkflowStage(Enum):
    """Pipeline stages for progress tracking."""
    DISCOVER = auto()
    PREPROCESS = auto()
    ANALYZE = auto()
    FILTER_QUALITY = auto()
    FILTER_PORTRAIT = auto()
    EXTRACT_FEATURES = auto()
    CLUSTER = auto()
    SELECT = auto()
    EXPORT = auto()
    COMPLETE = auto()


# Type alias for progress callbacks
ProgressCallback = Callable[[WorkflowStage, float, Optional[str]], None]
