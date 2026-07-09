"""Production occlusion scorer (spec-097 Stage 1).

Wraps the spec-096 winner — CLIP ViT-B/32 embeddings, global+tile-max
logistic head (0.86 scene PR-AUC, adjudicated labels) — behind the standard
domain-helper interface: config in ``__init__``, per-call data in a typed
``Inputs`` dataclass, ``calc(inputs) -> Result`` as the primary entry.

The artifact (scaler + coefficients + version) lives in ``models/occlusion/``;
training/validation is owned by ``scripts/train_occlusion_artifact.py``.
Framework-agnostic: no PipelineContext here (spec-053).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TILE_GRID = 3  # 3x3 tiles, matching the benchmark's clip_embed layout
DEFAULT_ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "occlusion", "clip_b32_gmax_v1.npz")


@dataclass
class OcclusionInputs:
    image_paths: List[str]


@dataclass
class OcclusionResult:
    scores: Dict[str, float] = field(default_factory=dict)        # path -> P(occluded)
    tiles: Dict[str, List[float]] = field(default_factory=dict)   # path -> 9 tile scores
    skipped: List[str] = field(default_factory=list)              # unreadable paths


class OcclusionScorer:
    """CLIP global+tile-max logistic head; lazy CLIP load; CPU."""

    def __init__(self, artifact_path: Optional[str] = None):
        self._artifact_path = artifact_path or DEFAULT_ARTIFACT
        a = np.load(self._artifact_path, allow_pickle=True)
        self._mean = a["mean"].astype(np.float64)
        self._scale = a["scale"].astype(np.float64)
        self._coef = a["coef"].astype(np.float64).ravel()
        self._intercept = float(a["intercept"])
        self.version = str(a["version"])
        self._clip = None
        self._preprocess = None

    # -- embedding ----------------------------------------------------------
    def _ensure_clip(self):
        if self._clip is None:
            import clip
            self._clip, self._preprocess = clip.load("ViT-B/32", device="cpu")
            self._clip.eval()

    def embed(self, path: str) -> Optional[tuple]:
        """(global_emb, tile_embs[9]) — unit-normalized; None if unreadable."""
        import torch
        from PIL import Image, ImageOps
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            pass
        self._ensure_clip()
        try:
            with Image.open(path) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:
            logger.warning("occlusion: cannot read %s (%s)", path, e)
            return None
        w, h = img.size
        crops = [img]
        tw, th = w // TILE_GRID, h // TILE_GRID
        for ty in range(TILE_GRID):
            for tx in range(TILE_GRID):
                crops.append(img.crop((tx * tw, ty * th, (tx + 1) * tw, (ty + 1) * th)))
        with torch.no_grad():
            e = self._clip.encode_image(
                torch.stack([self._preprocess(c) for c in crops])).float()
        e = (e / e.norm(dim=-1, keepdim=True)).numpy()
        return e[0], e[1:]

    # -- head ---------------------------------------------------------------
    def _head(self, feat: np.ndarray) -> float:
        z = (feat - self._mean) / self._scale
        return float(1.0 / (1.0 + np.exp(-(z @ self._coef + self._intercept))))

    def score_embeddings(self, emb_global: np.ndarray, emb_tiles: np.ndarray):
        """(P(occluded), [9 per-tile scores]) from precomputed embeddings."""
        p = self._head(np.concatenate([emb_global, emb_tiles.max(axis=0)]))
        # per-tile: the head's opinion if THIS tile were the max — localization signal
        tile_scores = [self._head(np.concatenate([emb_global, t])) for t in emb_tiles]
        return p, tile_scores

    # -- primary entry (spec-053) --------------------------------------------
    def calc(self, inputs: OcclusionInputs) -> OcclusionResult:
        res = OcclusionResult()
        for path in inputs.image_paths:
            pair = self.embed(path)
            if pair is None:
                res.skipped.append(path)
                continue
            p, tiles = self.score_embeddings(*pair)
            res.scores[path] = p
            res.tiles[path] = tiles
        if res.skipped:
            logger.warning("occlusion: skipped %d unreadable images", len(res.skipped))
        return res
