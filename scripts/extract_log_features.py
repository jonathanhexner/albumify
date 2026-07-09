"""Extract Track-F LoG patch-statistic features for the occlusion dataset.

Writes features_log.npz (X, y, groups, names) aligned to manifest row order —
the same layout the refit script and the review app's explainer expect.
(Recommitted 2026-07-09: the original extraction ran inline and was never saved.)
"""

from __future__ import annotations

import logging
import os

import numpy as np

from sim_bench.occlusion_bench.dataset import load_manifest
from sim_bench.occlusion_bench.features_log import FEATURE_NAMES, feature_vector

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("extract_log")
ROOT = os.environ.get("OCCLUSION_DATASET", r"D:\occlusion_dataset")


def main() -> None:
    rows = load_manifest(ROOT)
    X, y, g, skipped = [], [], [], 0
    for i, r in enumerate(rows):
        sub = "positives" if r["label"] == "1" else "negatives"
        v = feature_vector(os.path.join(ROOT, sub, r["id"]))
        if v is None:
            # keep alignment with the manifest: zero-vector placeholder
            v = np.zeros(len(FEATURE_NAMES))
            skipped += 1
        X.append(v)
        y.append(int(r["label"]))
        g.append(r["group_id"])
        if (i + 1) % 100 == 0:
            logger.info("features %d/%d", i + 1, len(rows))
    out = os.path.join(ROOT, "features_log.npz")
    np.savez_compressed(out, X=np.array(X), y=np.array(y), groups=np.array(g),
                        names=np.array(FEATURE_NAMES))
    logger.info("saved %s (%d rows, %d unreadable->zeros)", out, len(X), skipped)


if __name__ == "__main__":
    main()
