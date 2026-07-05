"""LoG explainability helpers (spec-096 review app) — pure, no Streamlit.

- ``overlay(path, stat)`` renders the image with the 8x8 patch grid colored by a
  chosen per-patch statistic, and outlines the winning cells (max overall in
  yellow, max border cell in cyan) — "the frame we are looking at".
- ``contributions(path)`` explains the LoG logistic regression for one image:
  per-feature contribution to the decision = coefficient x standardized value.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from sim_bench.occlusion_bench.features_log import (
    GRID, WORK, STAT_NAMES, FEATURE_NAMES, EPS_FRAC,
    log_response, patch_stats, feature_vector,
)


def patch_grid(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Per-patch stat grids: {stat_name: GRIDxGRID array}."""
    r = log_response(path)
    if r is None:
        return None
    eps = EPS_FRAC * (np.median(r) + 1e-9)
    h, w = r.shape
    ph, pw = h // GRID, w // GRID
    grids = {n: np.zeros((GRID, GRID)) for n in STAT_NAMES}
    for row in range(GRID):
        for col in range(GRID):
            st = patch_stats(r[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw], eps)
            for n in STAT_NAMES:
                grids[n][row, col] = st[n]
    return grids


def _is_border(row: int, col: int) -> bool:
    return row in (0, GRID - 1) or col in (0, GRID - 1)


def overlay(path: str, stat: str = "near_zero_fraction") -> Optional[Image.Image]:
    """Image + heatmap of ``stat`` per patch + outlines on the winning cells."""
    grids = patch_grid(path)
    if grids is None:
        return None
    g = grids[stat]
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    img = Image.open(path).convert("RGB")
    img.thumbnail((WORK, WORK))
    img = img.convert("RGBA")
    w, h = img.size
    cw, ch = w / GRID, h / GRID

    lo, hi = float(g.min()), float(g.max())
    rng = (hi - lo) or 1.0
    heat = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(heat)
    for row in range(GRID):
        for col in range(GRID):
            a = int(140 * (g[row, col] - lo) / rng)  # stronger stat -> redder
            d.rectangle([col * cw, row * ch, (col + 1) * cw, (row + 1) * ch],
                        fill=(255, 40, 40, a), outline=(255, 255, 255, 40))
    out = Image.alpha_composite(img, heat)
    d = ImageDraw.Draw(out)

    # winning cells: overall max (yellow), border max (cyan)
    r0, c0 = np.unravel_index(int(g.argmax()), g.shape)
    d.rectangle([c0 * cw, r0 * ch, (c0 + 1) * cw, (r0 + 1) * ch],
                outline=(255, 220, 0, 255), width=4)
    border_mask = np.array([[_is_border(r, c) for c in range(GRID)] for r in range(GRID)])
    gb = np.where(border_mask, g, -np.inf)
    r1, c1 = np.unravel_index(int(gb.argmax()), gb.shape)
    d.rectangle([c1 * cw, r1 * ch, (c1 + 1) * cw, (r1 + 1) * ch],
                outline=(0, 230, 230, 255), width=4)
    return out.convert("RGB")


class LogExplainer:
    """Fits the explanation LR once (on all data) and explains single images."""

    def __init__(self, root: str):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        d = np.load(os.path.join(root, "features_log.npz"), allow_pickle=True)
        self.names = [str(n) for n in d["names"]]
        self.scaler = StandardScaler().fit(d["X"])
        self.clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced")
        self.clf.fit(self.scaler.transform(d["X"]), d["y"])

    def explain(self, path: str, top_k: int = 10) -> Optional[Tuple[float, List[dict]]]:
        """(P(occluded), top contributions). contribution = coef x z-scored value."""
        v = feature_vector(path)
        if v is None:
            return None
        z = self.scaler.transform(v.reshape(1, -1))[0]
        contrib = self.clf.coef_[0] * z
        prob = float(self.clf.predict_proba(z.reshape(1, -1))[0, 1])
        order = np.argsort(-np.abs(contrib))[:top_k]
        rows = [{"feature": self.names[i], "value": float(v[i]),
                 "contribution": float(contrib[i]),
                 "pushes": "OCCLUDED" if contrib[i] > 0 else "clean"} for i in order]
        return prob, rows
