"""Spec-102 T1.1 — downsample a trip to a fixed working set both arms share.

Both the Albumify arm and the VLM arm MUST consume the *same* pixels (spec-102 D5/A1), so
this is the single front door: read each source image, apply EXIF orientation, convert to RGB,
resize so the longest edge is `max_edge`, and write a JPEG. The content hash is computed over
the *decoded pixel array* (not the JPEG bytes) so it is stable across libjpeg versions and is
what actually feeds the models. The per-image hashes fold into one `input_set_hash` that A1
checks to prove both arms saw an identical set.

Framework-agnostic (no PipelineContext); usable from a notebook or the CLI harness.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Filenames on all three trips are `YYYYMMDD_HHMMSS.*` (Google Photos export). Used for the
# free day-segmentation + chronological order; EXIF is the fallback (see roster.py).
_TS_RE = re.compile(r"(\d{8})_(\d{6})")

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")


@dataclass(frozen=True)
class DownsampleConfig:
    max_edge: int = 768       # longest-edge target (spec-102: 768px)
    quality: int = 80         # JPEG q
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS


@dataclass
class ImageRecord:
    src: str                  # absolute source path
    dst: str                  # absolute downsampled path
    stem: str                 # filename stem (stable ID used in contact sheets / picks)
    orig_wh: tuple[int, int]
    out_wh: tuple[int, int]
    content_sha256: str       # hash of the decoded RGB pixel array actually fed to the models


@dataclass
class DownsampleResult:
    trip: str
    out_dir: str
    config: DownsampleConfig
    records: list[ImageRecord] = field(default_factory=list)
    input_set_hash: str = ""  # A1: one hash proving the shared input set

    @property
    def n_images(self) -> int:
        return len(self.records)


def _register_heif_once() -> None:
    """HEIC support for the Germany set; no-op if pillow_heif is unavailable (JPG trips)."""
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:  # pragma: no cover - only bites on .heic trips without the dep
        logger.debug("pillow_heif not available; .heic files will be skipped")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(rgb)


def _fit_max_edge(wh: tuple[int, int], max_edge: int) -> tuple[int, int]:
    w, h = wh
    longest = max(w, h)
    if longest <= max_edge:
        return w, h
    scale = max_edge / longest
    return max(1, round(w * scale)), max(1, round(h * scale))


def _content_hash(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def discover(src_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Case-insensitive extension match, deduped and sorted (chronological via filename)."""
    exts = {e.lower() for e in extensions}
    found = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(set(found))


def downsample_trip(
    src_dir: Path,
    out_dir: Path,
    trip: str,
    config: DownsampleConfig | None = None,
) -> DownsampleResult:
    """Downsample every image under `src_dir` into `out_dir` and return the manifest."""
    cfg = config or DownsampleConfig()
    _register_heif_once()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = DownsampleResult(trip=trip, out_dir=str(out_dir), config=cfg)
    for src in discover(src_dir, cfg.extensions):
        try:
            arr = _load_rgb(src)
        except Exception as exc:  # keep going; log the skip so counts stay honest
            logger.warning("skip %s: %s", src.name, exc)
            continue
        orig_wh = (arr.shape[1], arr.shape[0])
        out_wh = _fit_max_edge(orig_wh, cfg.max_edge)
        img = Image.fromarray(arr)
        if out_wh != orig_wh:
            img = img.resize(out_wh, Image.LANCZOS)
        out_arr = np.asarray(img)
        dst = out_dir / f"{src.stem}.jpg"
        img.save(dst, "JPEG", quality=cfg.quality)
        result.records.append(
            ImageRecord(
                src=str(src),
                dst=str(dst),
                stem=src.stem,
                orig_wh=orig_wh,
                out_wh=out_wh,
                content_sha256=_content_hash(out_arr),
            )
        )

    result.input_set_hash = input_set_hash(result.records)
    logger.info("downsampled %d images for %s -> %s", result.n_images, trip, out_dir)
    return result


def input_set_hash(records: list[ImageRecord]) -> str:
    """A1: order-independent hash over (stem, content hash) pairs for the whole set."""
    h = hashlib.sha256()
    for stem, content in sorted((r.stem, r.content_sha256) for r in records):
        h.update(stem.encode())
        h.update(content.encode())
    return h.hexdigest()
