"""Track D — synthetic occlusion compositor (spec-096).

Manufactures labeled positives by compositing a defocused occluder blob onto a
clean photo. Parameters mirror what we LEARNED about real occluders:

  - color: WARM skin-toned (sunlit finger) or DARK (backlit finger / strap) —
    the dark variety is the one Haiku surfaced and the warm-only cue missed;
  - heavily defocused: strong Gaussian blur on blob AND its alpha (soft edge);
  - anchored to a frame edge/corner (fingers enter from the border);
  - coverage sets the severity label: <8% L1, 8-20% L2, >20% L3.

Deterministic per (image, seed) — reproducible datasets, group-split-safe
(synthetic variants inherit the base image's group).
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

WARM_COLORS = [(225, 160, 130), (235, 175, 140), (200, 130, 100), (240, 190, 160)]
DARK_COLORS = [(25, 22, 20), (45, 40, 38), (60, 50, 45), (35, 30, 40)]
CORNERS = ["tl", "tr", "bl", "br", "left", "right", "top", "bottom"]


def _rng(image_path: str, seed: int) -> random.Random:
    h = hashlib.sha1(f"{image_path}:{seed}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def level_for(coverage: float) -> int:
    return 1 if coverage < 0.08 else (2 if coverage < 0.20 else 3)


def composite(img: Image.Image, coverage: float, corner: str, dark: bool,
              rng: random.Random) -> Image.Image:
    """Blend a defocused blob covering ~``coverage`` of the frame at ``corner``."""
    img = img.convert("RGB")
    w, h = img.size
    area = coverage * w * h
    # ellipse ~ pi/4 * bw * bh; make it wider than tall-ish with jitter
    aspect = rng.uniform(0.6, 1.6)
    bw = int((area * 4 / 3.14159 * aspect) ** 0.5)
    bh = int(bw / aspect)

    cx = {"tl": 0, "bl": 0, "left": 0, "tr": w, "br": w, "right": w,
          "top": rng.randint(0, w), "bottom": rng.randint(0, w)}[corner]
    cy = {"tl": 0, "tr": 0, "top": 0, "bl": h, "br": h, "bottom": h,
          "left": rng.randint(0, h), "right": rng.randint(0, h)}[corner]
    cx += rng.randint(-bw // 4, bw // 4)
    cy += rng.randint(-bh // 4, bh // 4)

    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2],
              fill=int(255 * rng.uniform(0.82, 1.0)))  # slight translucency sometimes
    mask = mask.filter(ImageFilter.GaussianBlur(max(8, min(bw, bh) // 8)))  # soft defocus edge
    # blurring flattens peak opacity -> rescale so the blob core stays solid
    import numpy as _np
    m = _np.asarray(mask, dtype=_np.float32)
    if m.max() > 0:
        m = _np.clip(m * (235.0 / m.max()), 0, 255)
    mask = Image.fromarray(m.astype("uint8"))

    color = rng.choice(DARK_COLORS if dark else WARM_COLORS)
    jitter = lambda c: max(0, min(255, c + rng.randint(-15, 15)))
    blob = Image.new("RGB", (w, h), tuple(jitter(c) for c in color))
    blob = blob.filter(ImageFilter.GaussianBlur(6))  # subsurface-ish softness
    return Image.composite(blob, img, mask)


def synth_occlusion(image_path: str, out_path: str, seed: int = 0,
                    coverage: Optional[float] = None) -> Tuple[int, dict]:
    """Create one synthetic occluded variant; returns (level, params)."""
    rng = _rng(image_path, seed)
    cov = coverage if coverage is not None else rng.choice(
        [rng.uniform(0.03, 0.08), rng.uniform(0.08, 0.20), rng.uniform(0.20, 0.40)])
    corner = rng.choice(CORNERS)
    dark = rng.random() < 0.4  # 40% dark occluders (per the Haiku finding)
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    with Image.open(image_path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    img.thumbnail((1600, 1600))  # keep synth files reasonable
    out, in_frame_cov = _composite_measured(img, cov, corner, dark, rng)
    out.save(out_path, "JPEG", quality=88)
    params = {"coverage": round(in_frame_cov, 3), "corner": corner, "dark": dark, "seed": seed}
    return level_for(in_frame_cov), params


def _composite_measured(img, cov, corner, dark, rng):
    """Composite, retrying nominal coverage until IN-FRAME coverage hits target
    (edge-anchored blobs land ~half off-frame, so nominal ~2x under-covers)."""
    import numpy as _np
    target = cov
    nominal = cov * 1.6
    for _ in range(3):
        out = composite(img, nominal, corner, dark, rng)
        # measure actual altered area from pixel differences
        a = _np.asarray(img, dtype=_np.int16); b = _np.asarray(out, dtype=_np.int16)
        diff = (_np.abs(a - b).sum(axis=2) > 24)
        got = float(diff.mean())
        if got >= 0.6 * target:
            return out, got
        nominal *= 1.7
    return out, got
