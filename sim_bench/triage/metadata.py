"""Metadata-tier triage (spec-107 Tier 1): flag screenshots/downloads from file metadata alone.

Pure + fast — no model. Signals: filename, camera-EXIF presence (Make/Model), format, dimensions.
Conservative by design: only high-confidence junk -> ``set_aside``; ambiguous -> ``review``;
real camera photos -> ``keep``. Nothing is ever deleted (spec-107: notify + review, never silent).

Domain helper per spec-053: ``calc(inputs) -> result`` is the entry; ``classify_file(path)`` is a
convenience that reads the metadata off disk and calls ``calc``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Representative screen resolutions (phone + desktop), either orientation. A *weak* corroborating
# signal only — never decisive on its own. Not meant to be exhaustive.
_SCREEN_DIMS = {
    (1080, 1920), (1080, 2160), (1080, 2340), (1080, 2400), (1170, 2532), (1179, 2556),
    (1284, 2778), (1290, 2796), (1440, 2560), (1440, 3040), (1440, 3200), (828, 1792),
    (750, 1334), (1536, 2048), (1668, 2388), (1920, 1080), (2560, 1440), (3840, 2160),
    (1366, 768), (1280, 720),
}
_DOWNLOAD_HINTS = ("whatsapp", "download", "img-", "fb_img", "received_")


@dataclass
class TriageInputs:
    """Metadata needed to classify one image — decoupled from disk so ``calc`` is unit-testable."""
    filename: str
    has_camera_exif: bool                 # EXIF Make or Model present
    fmt: Optional[str] = None             # PIL format: "JPEG", "PNG", ...
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class TriageResult:
    decision: str                         # "keep" | "review" | "set_aside"
    reason: str
    confidence: float                     # 0.0 (keep) .. 1.0 (certain junk)
    signals: List[str] = field(default_factory=list)


def _is_screen_dim(w: Optional[int], h: Optional[int]) -> bool:
    return w is not None and h is not None and ((w, h) in _SCREEN_DIMS or (h, w) in _SCREEN_DIMS)


def calc(inp: TriageInputs) -> TriageResult:
    """Classify one image from its metadata. Pure."""
    name = (inp.filename or "").lower()
    signals: List[str] = []

    # Highest-confidence signal: the OS names screenshots explicitly.
    if name.startswith("screenshot") or name.startswith("screen_shot") or name.startswith("screen-"):
        return TriageResult("set_aside", "screenshot (filename)", 0.98, ["filename:screenshot"])

    download_like = any(h in name for h in _DOWNLOAD_HINTS)
    if download_like:
        signals.append("filename:download-like")

    if not inp.has_camera_exif:
        signals.append("no-camera-exif")
        png = (inp.fmt or "").upper() == "PNG"
        screen = _is_screen_dim(inp.width, inp.height)
        if png:
            signals.append("png")
        if screen:
            signals.append("screen-dimensions")
        # No camera metadata + a corroborating signal -> confident it never came from a camera.
        if png and screen:
            return TriageResult("set_aside", "no camera metadata, PNG at screen resolution", 0.9, signals)
        if png or screen or download_like:
            why = "PNG" if png else ("screen resolution" if screen else "download-like name")
            return TriageResult("review", f"no camera metadata ({why})", 0.6, signals)
        # No camera metadata but nothing else -> could be an edited/stripped real photo. Be gentle.
        return TriageResult("review", "no camera metadata (may be edited/exported)", 0.4, signals)

    # Has camera EXIF and not a screenshot name -> a real photo.
    return TriageResult("keep", "camera photo (EXIF present)", 0.0, ["camera-exif"])


def classify_file(path) -> TriageResult:
    """Read an image's metadata off disk and classify it. Unreadable files -> ``review`` (never drop)."""
    from PIL import Image  # lazy: keep import cost out of callers that only use calc()

    p = Path(path)
    try:
        with Image.open(p) as im:
            exif = im.getexif()
            has_cam = bool(exif.get(271) or exif.get(272))  # Make / Model
            return calc(TriageInputs(p.name, has_cam, im.format, im.width, im.height))
    except Exception:
        return TriageResult("review", "unreadable image", 0.5, ["unreadable"])
