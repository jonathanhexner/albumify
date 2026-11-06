"""Staged 768 px copy -> the original photo (spec-116 R5).

Workers see staged copies: ``<input root>/<sha16>/<stem>.jpg`` here, ``/inputs/<sha16>/<stem>.jpg`` on the L4.
The regular pages must point at the originals, so a staged path is recognised by its input-store key (the last
two path parts), which is the same on both machines. ``scale`` is original pixels per staged pixel.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

from sim_bench.serving.job_store import JobStore

logger = logging.getLogger(__name__)

__all__ = ["PathMap"]

# <anything><sha16>/<name>.jpg, optionally followed by a face suffix (the pipeline's "<path>:face_<n>" keys)
_STAGED = re.compile(r"^(?:.*[\\/])?([0-9a-f]{16})[\\/]([^\\/]+\.jpg)(:face_\d+)?$", re.DOTALL)


def _split(path: str) -> Optional[Tuple[str, str]]:
    m = _STAGED.match(str(path))
    return (f"{m.group(1)}/{m.group(2)}", m.group(3) or "") if m else None


def _pixel_scale(original: Path, staged: Path) -> float:
    with Image.open(original) as a, Image.open(staged) as b:
        return a.size[0] / b.size[0]


@dataclass(frozen=True)
class PathMap:
    originals: Dict[str, str] = field(default_factory=dict)   # input-store key -> original path
    scales: Dict[str, float] = field(default_factory=dict)    # input-store key -> original px / staged px

    @classmethod
    def from_job(cls, store: JobStore, job_id: str) -> "PathMap":
        originals = store.input_originals(job_id)
        scales: Dict[str, float] = {}
        for key, original in originals.items():
            try:
                scales[key] = _pixel_scale(Path(original), store.inputs.path(key))
            except OSError as e:
                # the photo moved since staging: the path is still right, pixel boxes stay in staged pixels
                logger.warning("job %s: no pixel scale for %s: %s", job_id, original, e)
                scales[key] = 1.0
        return cls(originals=originals, scales=scales)

    def original(self, staged: str) -> str:
        """The original photo's path (keeping a ``:face_<n>`` suffix); anything else comes back unchanged."""
        parts = _split(staged)
        if parts is None or parts[0] not in self.originals:
            return staged
        return self.originals[parts[0]] + parts[1]

    def scale(self, staged: str) -> float:
        parts = _split(staged)
        return self.scales.get(parts[0], 1.0) if parts else 1.0
