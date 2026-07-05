"""Track B — CLIP embeddings for the occlusion benchmark (spec-096).

Extracts, per dataset image:
  - the GLOBAL image embedding (the classic linear-probe input; our control), and
  - 3x3 TILE embeddings (patch-level signal for a localized defect: an occlusion
    lives in one corner, so tile aggregates / max-pooled tile scores can see it
    where the global average washes it out).

Saved to ``clip_embeddings.npz`` (ids, y, groups, emb_global, emb_tiles).
Label-independent: extract once now, refit probes in seconds after label
corrections land.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TILE_GRID = 3  # 3x3 tiles


def _load_rgb(path: str):
    from PIL import Image, ImageOps
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        logger.warning("clip_embed: cannot read %s (%s)", path, e)
        return None


def extract(dataset_root: str, out_name: str = "clip_embeddings.npz",
            model_name: str = "ViT-B/32") -> dict:
    import clip
    import torch
    from sim_bench.occlusion_bench.dataset import load_manifest

    device = "cpu"
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    rows = load_manifest(dataset_root)
    ids: List[str] = []
    ys: List[int] = []
    gs: List[str] = []
    glob: List[np.ndarray] = []
    tiles: List[np.ndarray] = []

    with torch.no_grad():
        for i, r in enumerate(rows):
            sub = "positives" if r["label"] == "1" else "negatives"
            img = _load_rgb(os.path.join(dataset_root, sub, r["id"]))
            if img is None:
                continue
            w, h = img.size
            crops = [img]  # index 0 = global
            tw, th = w // TILE_GRID, h // TILE_GRID
            for ty in range(TILE_GRID):
                for tx in range(TILE_GRID):
                    crops.append(img.crop((tx * tw, ty * th, (tx + 1) * tw, (ty + 1) * th)))
            batch = torch.stack([preprocess(c) for c in crops]).to(device)
            emb = model.encode_image(batch).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            glob.append(emb[0].numpy())
            tiles.append(emb[1:].numpy())
            ids.append(r["id"]); ys.append(int(r["label"])); gs.append(r["group_id"])
            if (i + 1) % 50 == 0:
                logger.info("clip_embed %d/%d", i + 1, len(rows))

    out = os.path.join(dataset_root, out_name)
    np.savez_compressed(out, ids=np.array(ids), y=np.array(ys), groups=np.array(gs),
                        emb_global=np.array(glob, dtype=np.float32),
                        emb_tiles=np.array(tiles, dtype=np.float32))
    logger.info("saved %s (%d images)", out, len(ids))
    return {"n": len(ids), "path": out}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    extract(os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset"))
