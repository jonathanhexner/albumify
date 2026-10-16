"""Where InsightFace keeps its model files (spec-110 T1.3).

InsightFace defaults `FaceAnalysis(root=...)` to `~/.insightface` and downloads buffalo_l
(~330 MB) there on first use. On a serverless worker `$HOME` is container scratch that dies with the
container, so the weights were re-downloaded on every cold start while the mounted models Volume sat
unused. `INSIGHTFACE_HOME` redirects the root; unset (local dev) keeps the upstream default.

Two call sites construct `FaceAnalysis` -- detection (`insightface_pipeline.face_analyzer`) and
embeddings (`face_embedding.insightface_native`) -- so the lookup lives here rather than twice.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

__all__ = ["ENV_VAR", "model_root_kwargs"]

ENV_VAR = "INSIGHTFACE_HOME"


def model_root_kwargs() -> Dict[str, str]:
    """`{"root": <dir>}` when `INSIGHTFACE_HOME` is set, else `{}` (InsightFace applies its default)."""
    home = os.environ.get(ENV_VAR)
    if not home:
        return {}
    logger.info("InsightFace model root: %s (%s)", home, ENV_VAR)
    return {"root": home}
