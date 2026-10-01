"""VLM provider seam (spec-108).

The semantic layer's VLM-backed tiers (richer theme, keeper captions P2, theme-relevance P3) depend
only on this small protocol — never on a concrete model — so the backend can be swapped (local
MiniCPM-V, an API, a worker subprocess) without touching pipeline code.

**Chosen backend: MiniCPM-V 4.5 local** (2026-10-01) — $0/token + strangers' photos never leave the
box. Operational note: MiniCPM-V 4.5 pins `transformers==4.51` (this repo's env is 4.57) and an 8B
VLM is impractical on CPU, so the live backend runs on the spec-109 GPU worker in its own pinned
environment (or a local `.venv-vlm` / ollama GGUF for dev). See `MiniCPMProvider` for the
implementation contract; it is intentionally not loaded in the shared env.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

__all__ = ["VLMProvider"]


@runtime_checkable
class VLMProvider(Protocol):
    """Minimal vision-language interface the semantic layer calls.

    Implementations must be lazy (load the model on first use, like ``RamTagger``) and degrade to a
    safe empty string on per-image failure rather than raising, so one bad image can't fail a run.
    """

    def caption(self, image_path: str, prompt: str) -> str:
        """Return the model's free-text response to ``prompt`` about the image."""
        ...

    def caption_batch(self, image_paths: List[str], prompt: str) -> List[str]:
        """Caption several images with the same prompt. Default impl may loop ``caption``."""
        ...
