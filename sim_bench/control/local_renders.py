"""Make a job's display images and face crops from the server's own copies (spec-115 W2, R10).

The worker returns results only (``album.json``, ``run.json``, ``results/``); pictures never come back. The server
already holds every photo at 768 px (the input store), and the results bundle carries each face's box, so it
cuts what the app shows itself:

    display/<photo>.jpg   the picks, a copy of the staged input (browsers apply its EXIF orientation)
    faces/<face_id>.jpg   the face box plus a margin, cut from the photo read upright

Boxes come from the detector, which reads photos EXIF-transposed (``image_cache``), so crops are cut from the
transposed image too (T2.1 gate, 2026-10-23: right face on 20 of 20, rotated photos included). Existing files
are kept, so an older job's worker-made displays and crops are left as they are.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageOps

from sim_bench.serving.job_store import JobStore

logger = logging.getLogger(__name__)

__all__ = ["LocalRenders", "LocalRendersInputs", "LocalRendersResult"]


@dataclass(frozen=True, slots=True)
class LocalRendersInputs:
    job_id: str
    album: Dict
    bundle: Dict


@dataclass(slots=True)
class LocalRendersResult:
    n_displays: int = 0
    n_crops: int = 0
    n_kept: int = 0                       # files already there (older jobs), left untouched
    missing_photos: List[str] = field(default_factory=list)
    unreadable_photos: List[str] = field(default_factory=list)


class LocalRenders:
    """Config in ``__init__``; one job per :meth:`calc`."""

    def __init__(self, store: JobStore, *, margin: float = 0.25, crop_size: int = 224, quality: int = 85):
        self._store, self.margin, self.crop_size, self.quality = store, margin, crop_size, quality

    def calc(self, inputs: LocalRendersInputs) -> LocalRendersResult:
        paths = self._store.paths(inputs.job_id)
        sources = {p.stem: p for p in self._store.input_images(inputs.job_id)}
        res = LocalRendersResult()

        paths.display_dir.mkdir(parents=True, exist_ok=True)
        for pick in inputs.album.get("picks", []):
            src = sources.get(pick["id"])
            if src is None:
                res.missing_photos.append(pick["id"])
                continue
            dest = paths.display_dir / src.name
            if dest.exists():
                res.n_kept += 1
                continue
            self.render_display(src, dest)
            res.n_displays += 1

        paths.faces_dir.mkdir(parents=True, exist_ok=True)
        by_photo: Dict[str, List[Dict]] = {}
        for face in inputs.bundle.get("faces", []):
            if face.get("bbox"):
                by_photo.setdefault(face["photo_id"], []).append(face)
        for photo_id, faces in by_photo.items():
            src = sources.get(photo_id)
            if src is None:
                res.missing_photos.append(photo_id)
                continue
            todo = [f for f in faces if not (paths.faces_dir / f"{int(f['face_id'])}.jpg").exists()]
            res.n_kept += len(faces) - len(todo)
            if not todo:
                continue
            try:
                with Image.open(src) as raw:
                    upright = ImageOps.exif_transpose(raw).convert("RGB")
            except Exception as e:   # one unreadable photo must not sink the job's load; it is reported
                res.unreadable_photos.append(photo_id)
                logger.warning("job %s: cannot read %s for face crops: %s", inputs.job_id, src.name, e)
                continue
            for f in todo:
                self.render_crop(upright, f["bbox"], paths.faces_dir / f"{int(f['face_id'])}.jpg")
                res.n_crops += 1

        if res.missing_photos:
            logger.warning("job %s: %d photo(s) named in the results are not staged: %s", inputs.job_id,
                           len(res.missing_photos), sorted(set(res.missing_photos))[:5])
        logger.info("job %s: rendered %d displays, %d face crops on this server (%d kept)", inputs.job_id,
                    res.n_displays, res.n_crops, res.n_kept)
        return res

    def render_display(self, src: Path, dest: Path) -> None:
        shutil.copyfile(src, dest)

    def render_crop(self, upright: Image.Image, box: Sequence[float], dest: Path) -> None:
        """``box`` is ``[x1, y1, x2, y2]`` in the upright image's pixels."""
        x1, y1, x2, y2 = (float(v) for v in box)
        mx, my = (x2 - x1) * self.margin, (y2 - y1) * self.margin
        crop = upright.crop((max(0, round(x1 - mx)), max(0, round(y1 - my)),
                             min(upright.width, round(x2 + mx)), min(upright.height, round(y2 + my))))
        crop.thumbnail((self.crop_size, self.crop_size))
        crop.save(dest, "JPEG", quality=self.quality)
