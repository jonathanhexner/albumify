"""Shared exception types for the run-DB layer (spec-057)."""
from __future__ import annotations


class RunExporterError(RuntimeError):
    """Raised when input data violates the writer's contract."""
