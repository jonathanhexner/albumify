"""Typed config for score_ava step (spec-040 Phase 2)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScoreAVAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = Field(
        default="models/album_app/ava_resnet50.pt",
        description="Path to the AVA ResNet50 checkpoint.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device for the AVA model (cpu, cuda, mps). The step always read this "
                    "knob; the typed config forbade it until 2026-10-09 (SIGHTING-123), so AVA "
                    "could never leave the CPU.",
    )
