"""Turn a folder of photos into a job's analysis-tier inputs (spec-112 TA4.2).

768 px longest edge, JPEG q85, EXIF kept (capture time / GPS feed scene clustering). One function
for every runner, so a local job and a Modal job of the same folder see byte-identical inputs.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from sim_bench.serving.job_store import INPUT_SUFFIXES

logger = logging.getLogger(__name__)

__all__ = ["downsample", "select_images", "collect_images", "ANALYSIS_EDGE"]

ANALYSIS_EDGE = 768


def downsample(path: Path, max_edge: int = ANALYSIS_EDGE) -> bytes:
    img = Image.open(path)
    exif = img.info.get("exif")
    img = img.convert("RGB")
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    kwargs = {"format": "JPEG", "quality": 85}
    if exif:
        kwargs["exif"] = exif
    img.save(buf, **kwargs)
    return buf.getvalue()


def select_images(src: Path, start: int = 0, n: Optional[int] = None) -> List[Path]:
    """Sorted images of ``src`` (non-recursive), sliced ``[start:start+n]``."""
    files = sorted(p for p in Path(src).iterdir()
                   if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES)
    return files[start:start + n] if n is not None else files[start:]


def collect_images(src: Path, start: int = 0, n: Optional[int] = None) -> List[Tuple[str, bytes]]:
    """:func:`select_images`, downsampled into memory.

    Output names keep the stem and become ``.jpg`` (the bytes are JPEG whatever the source was).
    """
    out = []
    for p in select_images(src, start, n):
        try:
            out.append((f"{p.stem}.jpg", downsample(p)))
        except Exception as e:   # one unreadable file must not sink the job; it is logged
            logger.warning("skipping unreadable image %s: %s", p, e)
    logger.info("collected %d images from %s (%.1f MB)", len(out), src,
                sum(len(b) for _, b in out) / 1e6)
    return out
