"""Typed config for score_iqa step (spec-040 Phase 2)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ScoreIQAConfig(BaseModel):
    """Empty config — score_iqa has no tunable parameters today."""
    model_config = ConfigDict(extra="forbid")
