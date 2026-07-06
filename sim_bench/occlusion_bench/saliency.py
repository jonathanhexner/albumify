"""Saliency + blur-region localization research helpers (spec-097 Stage 2 research).

Three zero-install saliency variants (opencv-contrib is already the pinned cv2):
  SR       — spectral residual (fast, contrast-based)
  FG       — fine-grained static saliency
  SR*center— spectral residual weighted by a center prior (stabilizes scatter;
             consumer photos center their subjects)

Plus ``blur_bbox``: the occluded/blurry-region box from the LoG patch grid
(patches much flatter than the image's own norm, largest border-touching
component). Color-blind — catches warm AND dark occluders.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x0, y0, x1, y1


def _read(path: str, work: int = 640) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        try:
            from PIL import Image
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("saliency: cannot read %s (%s)", path, e)
            return None
    h, w = img.shape[:2]
    s = work / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def saliency_map(img: np.ndarray, variant: str = "sr") -> np.ndarray:
    """Normalized [0,1] saliency map for variant in {sr, fg, sr_center}."""
    if variant == "fg":
        eng = cv2.saliency.StaticSaliencyFineGrained_create()
    else:
        eng = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, m = eng.computeSaliency(img)
    m = m.astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 8)
    if variant == "sr_center":
        h, w = m.shape
        yy, xx = np.mgrid[0:h, 0:w]
        prior = np.exp(-(((yy - h / 2) / (0.6 * h)) ** 2 + ((xx - w / 2) / (0.6 * w)) ** 2))
        m = m * prior.astype(np.float32)
    rng = m.max() - m.min()
    return (m - m.min()) / (rng if rng > 1e-9 else 1.0)


def salient_bbox(m: np.ndarray, pct: float = 85.0) -> Optional[Box]:
    """Bounding box of the largest connected component above the pct-percentile."""
    thr = (m > np.percentile(m, pct)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thr)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    return int(x), int(y), int(x + w), int(y + h)


def blur_bbox(path: str, grid: int = 8) -> Optional[Box]:
    """Box around the blurry/occluded region: LoG patches with IQR far below the
    image's own IQR, largest border-touching component (fraction coords 0..1)."""
    from sim_bench.occlusion_bench.features_log import log_response
    r = log_response(path)
    if r is None:
        return None
    g_iqr = float(np.percentile(r, 75) - np.percentile(r, 25)) + 1e-9
    g_med = float(np.median(r)) + 1e-9
    h, w = r.shape
    ph, pw = h // grid, w // grid
    flat = np.zeros((grid, grid), np.uint8)
    for row in range(grid):
        for col in range(grid):
            patch = r[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw]
            iqr = np.percentile(patch, 75) - np.percentile(patch, 25)
            med = float(np.median(patch))
            # FLAT (low spread) but NOISY (sits on a sensor-noise floor).
            # Sky/walls are flat AND near-zero -> excluded. Defocused occluders
            # are flat with response ABOVE the floor (the LR's sign-flip insight).
            flat[row, col] = 1 if (iqr < 0.15 * g_iqr and med > 0.08 * g_med) else 0
    n, labels = cv2.connectedComponents(flat)
    best, area = None, 0
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        touches = ys.min() == 0 or ys.max() == grid - 1 or xs.min() == 0 or xs.max() == grid - 1
        if touches and len(ys) > area and len(ys) >= 3:
            area = len(ys)
            best = (xs.min() / grid, ys.min() / grid, (xs.max() + 1) / grid, (ys.max() + 1) / grid)
    return best


def dual_box_panel(path: str, out_path: str) -> dict:
    """One panel: [SR | FG | SR*center], green = salient box, red = blur box."""
    img = _read(path)
    if img is None:
        return {}
    h, w = img.shape[:2]
    bb = blur_bbox(path)
    red = None if bb is None else (int(bb[0] * w), int(bb[1] * h), int(bb[2] * w), int(bb[3] * h))
    tiles, info = [], {"blur_box": red is not None}
    for variant in ("sr", "fg", "sr_center"):
        m = saliency_map(img, variant)
        gb = salient_bbox(m)
        t = img.copy()
        if gb:
            cv2.rectangle(t, gb[:2], gb[2:], (0, 200, 0), 3)
        if red:
            cv2.rectangle(t, red[:2], red[2:], (0, 0, 255), 3)
        cv2.putText(t, variant, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(t)
        if gb and red:
            ix0, iy0 = max(gb[0], red[0]), max(gb[1], red[1])
            ix1, iy1 = min(gb[2], red[2]), min(gb[3], red[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            info[f"overlap_{variant}"] = round(inter / max((red[2]-red[0])*(red[3]-red[1]), 1), 2)
    cv2.imwrite(out_path, np.hstack(tiles))
    return info
