"""Track B — CLIP probe variants over cached embeddings (spec-096).

Variants (all through the identical grouped-CV harness):
  global     — classic linear probe on the whole-image embedding (control)
  tile_max   — elementwise max over the 3x3 tile embeddings (localized signal,
               position-invariant; the 'CLIP with patches' idea)
  tile_mean  — elementwise mean over tiles
  global+max — concatenation of global and tile_max
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


def run(dataset_root: str) -> dict:
    from sim_bench.occlusion_bench.eval import cv_evaluate
    npz = os.path.join(dataset_root, "clip_embeddings.npz")
    while not os.path.isfile(npz):  # allow launch before extraction finishes
        logger.info("waiting for %s ...", npz)
        time.sleep(30)
    d = np.load(npz, allow_pickle=True)
    y, g = d["y"], d["groups"]
    eg, et = d["emb_global"], d["emb_tiles"]

    variants = {
        "global": eg,
        "tile_max": et.max(axis=1),
        "tile_mean": et.mean(axis=1),
        "global+max": np.concatenate([eg, et.max(axis=1)], axis=1),
    }
    results = {}
    for name, X in variants.items():
        r = cv_evaluate(X, y, g, C=0.01)  # stronger reg: 512-1024 dims vs 19 scenes
        results[name] = {k: v for k, v in r.items() if k != "folds"}
        logger.info("B/%s: scene=%.3f [%.3f,%.3f] image=%.3f", name,
                    r["scene_pr_auc_mean"], r["scene_pr_auc_min"],
                    r["scene_pr_auc_max"], r["image_pr_auc_mean"])
    with open(os.path.join(dataset_root, "results_track_b.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset"))
