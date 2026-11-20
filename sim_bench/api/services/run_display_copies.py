"""A run's 768 px copies of its photos (SIGHTING-152 C5): original path -> the staged copy in the input store.

Pages hold the original path; a thumbnail made from it decodes a full-size photo (~0.15 s each, ~90 s for a
600-photo album on first open). Every staged run (spec-116) already keeps a 768 px copy of each photo, listed in its
job's manifest, so thumbnails are made from that. Runs made before spec-116 have no manifest: the map is empty and
pages fall back to the original.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from sim_bench.serving.job_store import JobStore
from sim_bench.serving.serving_config import load_serving_config
from sim_bench.storage.object_store import make_object_store

logger = logging.getLogger(__name__)

__all__ = ["RunDisplayCopies", "normalise"]


def normalise(path: str) -> str:
    """One spelling per file, whichever slashes the pipeline or the manifest used."""
    return str(path).replace("\\", "/")


class RunDisplayCopies:
    def __init__(self, jobs_root: Optional[Path] = None):
        self._root = str(jobs_root or load_serving_config().jobs_root)

    def paths(self, run_id: str) -> Dict[str, str]:
        """Normalised original path -> the 768 px copy on this machine, for copies that exist here; {} if none."""
        store = JobStore(self._root, objects=make_object_store(self._root))
        try:
            originals = store.input_originals(run_id)
        except ValueError:                # an old run id that is not a valid job id
            return {}
        out = {}
        for key, original in originals.items():
            try:
                copy = store.inputs.path(key)
            except ValueError:            # a manifest key that tries to leave the input store
                continue
            if copy.is_file():
                out[normalise(original)] = str(copy)
        if originals and not out:
            logger.warning("run %s: none of its %d staged copies is on this machine", run_id, len(originals))
        return out
