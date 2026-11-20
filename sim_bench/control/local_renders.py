"""Make a job's display images and face crops from the server's own copies (spec-115 W2, R10).

The worker returns results only (``album.json``, ``run.json``, ``results/``); pictures never come back. The server
already holds every photo at 768 px (the input store), and the results bundle carries each face's box, so it
cuts what the app shows itself:

    display/<photo>.jpg   the picks, a copy of the staged input (browsers apply its EXIF orientation)
    faces/<face_id>.jpg   the face box plus a margin, cut from the photo read upright

Boxes come from the detector, which reads photos EXIF-transposed (``image_cache``), so crops are cut from the
transposed image too (T2.1 gate, 2026-10-23: right face on 20 of 20, rotated photos included). Existing files
are kept, so an older job's worker-made displays and crops are left as they are.

spec-117 R7: the files go to the job store's object store under ``<job_id>/display/`` and ``<job_id>/faces/``:
the job folder on this PC (unchanged), or the R2 bucket. On the L4 the worker makes them itself (W4).
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageOps

from sim_bench.serving.job_store import JobStore
from sim_bench.storage.object_store import UPLOAD_WORKERS, job_key

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
    bad_boxes: List[int] = field(default_factory=list)          # face ids whose box could not be cut


class LocalRenders:
    """Config in ``__init__``; one job per :meth:`calc`."""

    def __init__(self, store: JobStore, *, margin: float = 0.25, crop_size: int = 224, quality: int = 85):
        self._store, self.margin, self.crop_size, self.quality = store, margin, crop_size, quality

    def calc(self, inputs: LocalRendersInputs) -> LocalRendersResult:
        job_id = self._store.paths(inputs.job_id).job_id          # validates the id
        objects = self._store.objects
        sources = {p.stem: p for p in self._store.input_images(job_id)}
        res = LocalRendersResult()
        present = set(objects.list(f"{job_id}/display/")) | set(objects.list(f"{job_id}/faces/"))   # one listing each
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS, thread_name_prefix="upload") as uploads:
            pending = []
            self._render_all(inputs, job_id, sources, present, res, uploads, pending)
            for done in pending:
                done.result()            # the first failed upload fails the load, as before
        if res.missing_photos:
            logger.warning("job %s: %d photo(s) named in the results are not staged: %s", inputs.job_id,
                           len(res.missing_photos), sorted(set(res.missing_photos))[:5])
        logger.info("job %s: rendered %d displays, %d face crops on this server (%d kept)", inputs.job_id,
                    res.n_displays, res.n_crops, res.n_kept)
        return res

    def _render_all(self, inputs: LocalRendersInputs, job_id: str, sources: Dict[str, Path], present: set,
                    res: LocalRendersResult, uploads: ThreadPoolExecutor, pending: list) -> None:
        """Cut every missing display and crop here, in order; ``uploads`` sends them to the object store."""
        for pick in inputs.album.get("picks", []):
            src = sources.get(pick["id"])
            if src is None:
                res.missing_photos.append(pick["id"])
                continue
            key = job_key(job_id, f"display/{src.name}")
            if key in present:
                res.n_kept += 1
                continue
            pending.append(uploads.submit(self._store.objects.put_file, key, src))
            res.n_displays += 1

        by_photo: Dict[str, List[Dict]] = {}
        for face in inputs.bundle.get("faces", []):
            if face.get("bbox"):
                by_photo.setdefault(face["photo_id"], []).append(face)
        for photo_id, faces in by_photo.items():
            src = sources.get(photo_id)
            if src is None:
                res.missing_photos.append(photo_id)
                continue
            todo = [f for f in faces if job_key(job_id, f"faces/{int(f['face_id'])}.jpg") not in present]
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
                try:
                    data = self.crop_bytes(upright, f["bbox"])
                    pending.append(uploads.submit(self._store.objects.put,
                                                  job_key(job_id, f"faces/{int(f['face_id'])}.jpg"), data))
                    res.n_crops += 1
                except (ValueError, OSError) as e:      # one bad box must not sink the load (review M6)
                    res.bad_boxes.append(int(f["face_id"]))
                    logger.warning("job %s: face %s box %s not rendered: %s", inputs.job_id, f["face_id"],
                                   f["bbox"], e)

    def render_display(self, src: Path, key: str) -> None:
        self._store.objects.put_file(key, src)   # atomic in both stores: never a half-written file (review M6)

    def render_crop(self, upright: Image.Image, box: Sequence[float], key: str) -> None:
        """``box`` is ``[x1, y1, x2, y2]`` in the upright image's pixels."""
        self._store.objects.put(key, self.crop_bytes(upright, box))

    def crop_bytes(self, upright: Image.Image, box: Sequence[float]) -> bytes:
        """The face crop as JPEG bytes; ``box`` is ``[x1, y1, x2, y2]`` in the upright image's pixels."""
        x1, y1, x2, y2 = (float(v) for v in box)
        mx, my = (x2 - x1) * self.margin, (y2 - y1) * self.margin
        left, top = max(0, round(x1 - mx)), max(0, round(y1 - my))
        right, bottom = min(upright.width, round(x2 + mx)), min(upright.height, round(y2 + my))
        if right <= left or bottom <= top:
            raise ValueError("box is empty or outside the photo")
        crop = upright.crop((left, top, right, bottom))
        crop.thumbnail((self.crop_size, self.crop_size))
        buf = io.BytesIO()
        crop.save(buf, "JPEG", quality=self.quality)
        return buf.getvalue()
