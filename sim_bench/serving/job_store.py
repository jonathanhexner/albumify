"""Filesystem layout of a curation job -- the worker's storage contract (spec-110 T2.1).

A job *is* a directory. Nothing here knows about Modal: on Modal the root is a mounted Volume
(``/jobs``), locally it is any directory -- which is exactly what makes the storage-reference path
testable off Modal.

    <root>/<job_id>/input/       staged analysis-tier images (768 px long edge) -- the INPUT, never written by a run
    <root>/<job_id>/album.json   normalized Album (spec-106 contract) -- the RESULT the front end reads
    <root>/<job_id>/display/     display copies of the picks, for the front end
    <root>/<job_id>/run.json     run record: device, timing, step placement (debug trail, not a contract)
    <root>/<job_id>/results/     per-photo / per-face rows + embeddings (spec-112 results bundle)
    <root>/<job_id>/faces/       one JPEG crop per detected face, named <face_id>.jpg

The input dir staying read-only during a run is what makes a re-run idempotent: the pipeline writes
straighten derivatives next to its inputs, so a run works on a scratch copy (see
:mod:`sim_bench.serving.curate_job`) and never mutates the job's inputs.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["JobPaths", "JobStore", "INPUT_SUFFIXES"]

INPUT_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")

# A job id becomes a directory name, so it is validated rather than trusted: no separators, no
# traversal, no leading dot. The control plane (spec-106) generates these; the worker is remote.
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class JobPaths:
    """Resolved paths for one job. Pure data -- nothing is created by building this."""

    job_id: str
    root: Path
    input_dir: Path
    display_dir: Path
    album_path: Path
    run_path: Path
    results_dir: Path
    faces_dir: Path

    @property
    def bundle_path(self) -> Path:
        return self.results_dir / "bundle.json"


class JobStore:
    """Job directories under one root (a Modal Volume mount in production)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # --- layout ---------------------------------------------------------------

    def paths(self, job_id: str) -> JobPaths:
        if not _JOB_ID.match(job_id or ""):
            raise ValueError(
                f"invalid job_id {job_id!r}: expected 1-64 chars of [A-Za-z0-9_-] starting alphanumeric"
            )
        root = self.root / job_id
        return JobPaths(
            job_id=job_id,
            root=root,
            input_dir=root / "input",
            display_dir=root / "display",
            album_path=root / "album.json",
            run_path=root / "run.json",
            results_dir=root / "results",
            faces_dir=root / "faces",
        )

    # --- input ----------------------------------------------------------------

    def stage_inputs(self, job_id: str, images: Sequence[Tuple[str, bytes]],
                     *, replace: bool = True) -> List[Path]:
        """Write the job's images into ``input/``. ``replace`` clears the dir first (idempotent re-stage)."""
        paths = self.paths(job_id)
        if replace and paths.input_dir.exists():
            shutil.rmtree(paths.input_dir)
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for name, data in images:
            safe = Path(name).name  # a client-supplied name must not escape the job dir
            if not safe or safe.startswith("."):
                raise ValueError(f"invalid image name {name!r}")
            dest = paths.input_dir / safe
            dest.write_bytes(data)
            written.append(dest)
        logger.info("job %s: staged %d images (%.1f MB)", job_id, len(written),
                    sum(len(d) for _, d in images) / 1e6)
        return written

    def input_images(self, job_id: str) -> List[Path]:
        """The job's staged images, sorted by name (stable order -> stable album order)."""
        input_dir = self.paths(job_id).input_dir
        if not input_dir.is_dir():
            return []
        return sorted(p for p in input_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES)

    # --- result ---------------------------------------------------------------

    def write_album(self, job_id: str, album: Dict) -> Path:
        paths = self.paths(job_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.album_path.write_text(json.dumps(album, indent=2, default=str), encoding="utf-8")
        return paths.album_path

    def read_album(self, job_id: str) -> Optional[Dict]:
        path = self.paths(job_id).album_path
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def write_run_record(self, job_id: str, record: Dict) -> Path:
        paths = self.paths(job_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.run_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return paths.run_path

    def write_displays(self, job_id: str, sources: Sequence[Path]) -> List[Path]:
        """Copy the picks' images into ``display/``. Inputs are already analysis-tier, so this is a
        copy, not a re-encode -- a display tier wider than 768 px needs the original (spec-106)."""
        paths = self.paths(job_id)
        if paths.display_dir.exists():
            shutil.rmtree(paths.display_dir)
        paths.display_dir.mkdir(parents=True, exist_ok=True)
        out: List[Path] = []
        for src in sources:
            if not Path(src).is_file():
                logger.warning("job %s: display source missing: %s", job_id, src)
                continue
            dest = paths.display_dir / Path(src).name
            shutil.copyfile(src, dest)
            out.append(dest)
        return out

    # --- results bundle (spec-112) --------------------------------------------

    def write_results(self, job_id: str, manifest: Dict, embeddings=None,
                      embedding_face_ids: Sequence[int] = (),
                      crops: Optional[Dict[int, bytes]] = None) -> Path:
        """Write the results bundle: ``results/bundle.json`` (+ embeddings) and ``faces/<id>.jpg``.

        Overwrites a previous bundle (a re-run replaces, never accumulates). ``crops`` maps
        face_id -> encoded JPEG bytes.
        """
        import numpy as np

        paths = self.paths(job_id)
        for d in (paths.results_dir, paths.faces_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        paths.bundle_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        if embeddings is not None and len(embedding_face_ids):
            np.save(paths.results_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
            np.save(paths.results_dir / "embedding_face_ids.npy",
                    np.asarray(list(embedding_face_ids), dtype=np.int64))
        for fid, data in (crops or {}).items():
            (paths.faces_dir / f"{int(fid)}.jpg").write_bytes(data)
        return paths.bundle_path

    def read_results(self, job_id: str) -> Optional[Dict]:
        path = self.paths(job_id).bundle_path
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
