"""
Telemetry and performance tracking for album workflow.

Tracks timing for each operation to identify bottlenecks and monitor performance.
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OperationTiming:
    """Performance metrics for a single operation."""
    name: str
    duration_sec: float
    count: int = 1
    avg_per_item: float = field(init=False)
    
    def __post_init__(self):
        """Calculate average time per item."""
        self.avg_per_item = self.duration_sec / self.count if self.count > 0 else 0.0


@dataclass
class WorkflowTelemetry:
    """Complete telemetry data for a workflow run."""
    run_id: str
    total_duration_sec: float = 0.0
    timings: List[OperationTiming] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def export_json(self, path: Path):
        """
        Export telemetry data as JSON file.
        
        Args:
            path: Output file path
        """
        data = {
            'run_id': self.run_id,
            'total_duration_sec': self.total_duration_sec,
            'timings': [asdict(t) for t in self.timings],
            'metadata': self.metadata
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        
        logger.info(f"Telemetry exported to {path}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'run_id': self.run_id,
            'total_duration': f"{self.total_duration_sec:.2f}s",
            'num_operations': len(self.timings),
            'slowest_operation': max(self.timings, key=lambda t: t.duration_sec).name if self.timings else None
        }


class TimingTracker:
    """
    Context manager for timing operations.
    
    Usage:
        telemetry = WorkflowTelemetry(run_id="abc123")
        
        with TimingTracker(telemetry, "analyze_images", count=100):
            # ... analysis code ...
            pass
        
        # Timing automatically recorded in telemetry
    """
    
    def __init__(self, telemetry: WorkflowTelemetry, name: str, count: int = 1):
        """
        Initialize timing tracker.
        
        Args:
            telemetry: WorkflowTelemetry instance to record to
            name: Operation name
            count: Number of items processed (for per-item average)
        """
        self.telemetry = telemetry
        self.name = name
        self.count = count
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        logger.debug(f"[TIMING] Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record result."""
        duration = time.time() - self.start_time
        timing = OperationTiming(self.name, duration, self.count)
        self.telemetry.timings.append(timing)
        
        logger.debug(
            f"[TIMING] Completed: {self.name} - "
            f"{duration:.2f}s total, {timing.avg_per_item:.3f}s per item"
        )
        
        return False  # Don't suppress exceptions
