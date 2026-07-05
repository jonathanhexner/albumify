"""Track E — classical 4-cue occlusion detector (spec-096 baseline).

Ported from the 2026-07-05 prototype (LEARNINGS entry: approach #3+#5). Score =
fraction of the frame covered by a candidate blob that is simultaneously:

  blur outlier   cell Laplacian variance < BLUR_FRAC * frame median
  low texture    cell std-dev < TEX_MAX
  warm-toned     YCrCb skin-range fraction > WARM_MIN
  border blob    connected component touching the frame edge, >= MIN_CELLS

plus the RING cue: a blob whose 1-cell surround is sharply in focus
(rel. sharpness > RING_SHARP_MAX) is rejected (in-focus wall frames, not a
lens obstruction). Known limitation: warm smooth walls still outscore real
fingers sometimes — that is WHY it is the baseline, not the product.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GRID = 16
WORK = 512
BLUR_FRAC = 0.20
TEX_MAX = 18.0
WARM_MIN = 0.35
MIN_CELLS = 0.05
RING_SHARP_MAX = 1.5


def _read(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:  # HEIC etc.
        try:
            from PIL import Image
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("classical: cannot read %s (%s)", path, e)
            return None
    h, w = img.shape[:2]
    s = WORK / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def _grids(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ch, cw = h // GRID, w // GRID
    sharp = np.zeros((GRID, GRID))
    tex = np.zeros((GRID, GRID))
    warm = np.zeros((GRID, GRID))
    for r in range(GRID):
        for c in range(GRID):
            y0, y1, x0, x1 = r * ch, (r + 1) * ch, c * cw, (c + 1) * cw
            g = gray[y0:y1, x0:x1]
            sharp[r, c] = cv2.Laplacian(g, cv2.CV_64F).var()
            tex[r, c] = g.std()
            cr = ycc[y0:y1, x0:x1, 1]
            cb = ycc[y0:y1, x0:x1, 2]
            warm[r, c] = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)).mean()
    return sharp, tex, warm


def occlusion_score(path: str) -> float:
    """0 (clean) .. ~1 (frame covered by an edge-connected blurry warm blob)."""
    img = _read(path)
    if img is None:
        return 0.0
    sharp, tex, warm = _grids(img)
    med = np.median(sharp[sharp > 0]) if (sharp > 0).any() else 1.0
    cand = ((sharp < BLUR_FRAC * med) & (tex < TEX_MAX) & (warm > WARM_MIN)).astype(np.uint8)

    n, labels = cv2.connectedComponents(cand)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    if not border:
        return 0.0
    blob = np.isin(labels, list(border))
    cov = blob.sum() / (GRID * GRID)
    if cov < MIN_CELLS:
        return 0.0

    # ring cue: sharp surroundings -> in-focus structure (wall/frame), not occlusion
    dil = cv2.dilate(blob.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ring = dil & ~blob
    if ring.any() and (sharp[ring] / med).mean() > RING_SHARP_MAX:
        return 0.0
    return float(cov)
