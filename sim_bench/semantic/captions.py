"""Caption helpers (spec-108 P2) — prompts + response cleanup. Pure, framework-agnostic.

Two products (both via the VLM seam):
* per-keeper caption — one vivid sentence per selected image (``caption(image, prompt)``).
* album caption — one headline for the whole trip, synthesized from the keeper captions + the
  tag-based theme via a text-only call (``generate(prompt)``).

The orchestration lives in the steps; this module only builds prompts and cleans model output, so it
is fully unit-testable without a model.
"""
from __future__ import annotations

import re
from typing import List

__all__ = ["KEEPER_PROMPT", "clean_caption", "build_album_prompt"]

KEEPER_PROMPT = (
    "Describe this travel photo in one vivid sentence of at most 20 words. "
    "Name the main subject, the setting, and the mood. "
    "Do not begin with 'This photo' or 'The image'."
)

_WS = re.compile(r"\s+")
_LEAD = re.compile(
    r"^(this (photo|image)|the (photo|image)|here (is|we see)|a picture of)"
    r"(\s+(shows|depicts|captures|features|contains|is of))?"   # eat the follow-on verb too
    r"[:,]?\s*",
    re.IGNORECASE)


def clean_caption(text: str, max_chars: int = 240) -> str:
    """Normalize a raw VLM caption: collapse whitespace, strip boilerplate leads, trim, cap length."""
    if not text:
        return ""
    s = _WS.sub(" ", text).strip().strip('"').strip()
    s = _LEAD.sub("", s)
    if s:
        s = s[0].upper() + s[1:]
    if len(s) > max_chars:                      # cut at the last sentence/space within the cap
        cut = s[:max_chars]
        s = cut[:cut.rfind(" ")].rstrip(",; ") + "…" if " " in cut else cut
    return s


def build_album_prompt(album_type: str, motifs: List[str], keeper_captions: List[str],
                       max_captions: int = 12) -> str:
    """Prompt for the album headline — synthesized (text-only) from keeper captions + theme."""
    sample = [c for c in keeper_captions if c][:max_captions]
    bullets = "\n".join(f"- {c}" for c in sample) or "- (no captions available)"
    motif_str = ", ".join(motifs[:6]) if motifs else "various"
    return (
        f"These are captions of the best photos from one trip.\n"
        f"Album type: {album_type}. Recurring subjects: {motif_str}.\n"
        f"Photo captions:\n{bullets}\n\n"
        f"Write ONE warm, specific sentence (max 18 words) that could title this album. "
        f"No quotes, no 'This album'."
    )
