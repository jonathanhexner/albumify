"""LoG-statistics features for occlusion detection (spec-096 Track F, user-proposed).

Physics: a finger ~1 cm from the lens is defocused past the optics' cutoff — the
Laplacian-of-Gaussian response ``r = |LoG(patch)|`` is near-zero EVERYWHERE inside
the occluded region, at every scale. In-focus content has heavy-tailed r (edges).
Per-patch statistics of r capture this; a small classifier learns the boundary
that hand-tuned thresholds (the 4-cue detector) could not.

Per-patch stats (user's spec, verbatim): mean, std, entropy, near_zero_fraction,
p50, p90, p99, tail_ratio. ``near_zero`` eps is RELATIVE to the image's global
median response so the features are exposure/contrast invariant.

Image-level vector (weak supervision — we only have image labels): for each stat,
aggregates over ALL patches (min, p10, p50) + over BORDER patches only (min, p10)
— fingers enter from the frame edge — plus 2 global stats. 8*5+2 = 42 features.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GRID = 8          # 8x8 patches
WORK = 768        # max side for analysis
SIGMA = 2.0       # Gaussian scale before Laplacian
EPS_FRAC = 0.05   # near-zero threshold = EPS_FRAC * global median(r)

STAT_NAMES = ["mean", "std", "entropy", "near_zero_fraction",
              "p50", "p90", "p99", "tail_ratio"]


def _entropy(r: np.ndarray, bins: int = 32) -> float:
    hist, _ = np.histogram(r, bins=bins)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def patch_stats(r: np.ndarray, eps: float) -> Dict[str, float]:
    """The user's feature set over one patch's |LoG| response."""
    p50 = float(np.percentile(r, 50))
    return {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "entropy": _entropy(r),
        "near_zero_fraction": float(np.mean(r < eps)),
        "p50": p50,
        "p90": float(np.percentile(r, 90)),
        "p99": float(np.percentile(r, 99)),
        "tail_ratio": float(np.percentile(r, 95) / (p50 + 1e-6)),
    }


def log_response(path: str) -> Optional[np.ndarray]:
    """|LoG| response map of the (downscaled, grayscale) image."""
    img = cv2.imread(path)
    if img is None:  # HEIC etc. -> PIL fallback
        try:
            from PIL import Image
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("log_response: cannot read %s (%s)", path, e)
            return None
    h, w = img.shape[:2]
    s = WORK / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    smoothed = cv2.GaussianBlur(gray, (0, 0), SIGMA)
    return np.abs(cv2.Laplacian(smoothed, cv2.CV_64F))


def image_features(path: str) -> Optional[Dict[str, float]]:
    """42-dim image-level feature dict (aggregated per-patch LoG stats)."""
    r = log_response(path)
    if r is None:
        return None
    eps = EPS_FRAC * (np.median(r) + 1e-9)
    h, w = r.shape
    ph, pw = h // GRID, w // GRID

    per_patch: List[Dict[str, float]] = []
    border_idx: List[int] = []
    for row in range(GRID):
        for col in range(GRID):
            patch = r[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw]
            per_patch.append(patch_stats(patch, eps))
            if row in (0, GRID - 1) or col in (0, GRID - 1):
                border_idx.append(len(per_patch) - 1)

    feats: Dict[str, float] = {}
    for name in STAT_NAMES:
        vals = np.array([p[name] for p in per_patch])
        bvals = vals[border_idx]
        # BOTH tails: blur makes some stats LOW (tail_ratio, p90, mean) and
        # others HIGH (near_zero_fraction) — the classifier picks per stat.
        feats[f"{name}_min"] = float(vals.min())
        feats[f"{name}_p10"] = float(np.percentile(vals, 10))
        feats[f"{name}_p50"] = float(np.percentile(vals, 50))
        feats[f"{name}_p90"] = float(np.percentile(vals, 90))
        feats[f"{name}_max"] = float(vals.max())
        feats[f"{name}_border_min"] = float(bvals.min())
        feats[f"{name}_border_max"] = float(bvals.max())
    feats["global_near_zero_fraction"] = float(np.mean(r < eps))
    feats["global_tail_ratio"] = float(np.percentile(r, 95) / (np.percentile(r, 50) + 1e-6))
    return feats


FEATURE_NAMES: List[str] = (
    [f"{n}_{agg}" for n in STAT_NAMES
     for agg in ("min", "p10", "p50", "p90", "max", "border_min", "border_max")]
    + ["global_near_zero_fraction", "global_tail_ratio"]
)


def feature_vector(path: str) -> Optional[np.ndarray]:
    f = image_features(path)
    if f is None:
        return None
    return np.array([f[n] for n in FEATURE_NAMES])
