"""A run's own face embeddings (spec-116 R14): ``results/embeddings.npy`` of its job, in the object store (spec-117).

The feature cache is keyed by the path the pipeline saw, which for a staged run is a copy (and for an L4 run lives
on the Volume), so pages that hold an original path read the run's embeddings first. Runs made before spec-116
have no job folder: ``get`` returns None and the caller falls back to the cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from sim_bench.serving.job_store import JobStore
from sim_bench.serving.serving_config import load_serving_config
from sim_bench.storage.object_store import make_object_store

__all__ = ["RunEmbeddings"]


_MAX_LOADED = 8
_loaded: Dict[Tuple[str, str], Dict] = {}


def _load(root: str, job_id: str) -> Dict[Tuple[str, int], np.ndarray]:
    """(photo id, face index) -> embedding. A finished run's files never change, so found ones are kept; a run
    still in flight has none yet and is looked up again."""
    found = _loaded.get((root, job_id))
    if found is None:
        found = _read(root, job_id)
        if found:
            while len(_loaded) >= _MAX_LOADED:
                _loaded.pop(next(iter(_loaded)))
            _loaded[(root, job_id)] = found
    return found


def _read(root: str, job_id: str) -> Dict[Tuple[str, int], np.ndarray]:
    """spec-117 R7: from the object store (the job folder, or R2 for a run the worker wrote there)."""
    import io
    import json

    store = JobStore(root, objects=make_object_store(root))
    try:
        store.paths(job_id)
    except ValueError:                # an old run id that is not a valid job id
        return {}
    raw = {name: store.read_object(job_id, f"results/{name}")
           for name in ("bundle.json", "embeddings.npy", "embedding_face_ids.npy")}
    if any(v is None for v in raw.values()):
        return {}
    bundle = json.loads(raw["bundle.json"].decode("utf-8"))
    row_of = {int(fid): i for i, fid in enumerate(np.load(io.BytesIO(raw["embedding_face_ids.npy"])))}
    matrix = np.load(io.BytesIO(raw["embeddings.npy"]))
    return {(f["photo_id"], int(f["face_index"])): matrix[row_of[int(f["face_id"])]]
            for f in bundle.get("faces", [])
            if f.get("face_index") is not None and int(f["face_id"]) in row_of}


class RunEmbeddings:
    def __init__(self, jobs_root: Optional[Path] = None):
        self._root = str(jobs_root or load_serving_config().jobs_root)

    def has(self, run_id: str) -> bool:
        return bool(_load(self._root, run_id))

    def get(self, run_id: str, image_path: str, face_index: int) -> Optional[np.ndarray]:
        stem = Path(str(image_path).replace("\\", "/")).stem
        return _load(self._root, run_id).get((stem, int(face_index)))
