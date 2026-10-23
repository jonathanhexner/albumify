"""Filesystem layout of a curation job -- the worker's storage contract (spec-110 T2.1).

A job *is* a directory. Nothing here knows about Modal: on Modal the root is a mounted Volume
(``/jobs``), locally it is any directory -- which is exactly what makes the storage-reference path
testable off Modal.

    <root>/<job_id>/input/       the INPUT, never written by a run: either input/manifest.json naming
                                 photos in the shared input store (spec-112 A2), or (older jobs) the
                                 staged 768 px images themselves
    <root>/_inputs/<sha16>/<stem>.jpg   shared, content-addressed input store (:mod:`input_store`)
    <root>/<job_id>/album.json   normalized Album (spec-106 contract) -- the RESULT the front end reads
    <root>/<job_id>/display/     display copies of the picks, for the front end
    <root>/<job_id>/run.json     run record: device, timing, step placement (debug trail, not a contract)
    <root>/<job_id>/results/     per-photo / per-face rows + embeddings (spec-112 results bundle)
    <root>/<job_id>/faces/       one JPEG crop per detected face, named <face_id>.jpg

A run never writes beside its inputs (straightened copies go to their own cache), and the pipeline is
handed the input paths explicitly, so the same photo has the same path in every job and the
path-keyed feature cache is shared across jobs (spec-112 A2).
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

__all__ = ["JobPaths", "JobStore", "INPUT_SUFFIXES", "INPUTS_PREFIX"]

INPUT_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")

# A job id becomes a directory name, so it is validated rather than trusted: no separators, no
# traversal, no leading dot. The control plane (spec-106) generates these; the worker is remote.
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# A file key that starts with this names the shared input store, not the job's own folder.
INPUTS_PREFIX = "inputs/"
_MANIFEST = "manifest.json"


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

    def __init__(self, root: Path | str, inputs_root: Path | str | None = None):
        from sim_bench.serving.input_store import InputStore   # lazy: input_store -> staging -> here

        self.root = Path(root)
        # "_inputs" cannot collide with a job: job ids start alphanumeric
        self.inputs = InputStore(inputs_root if inputs_root is not None else self.root / "_inputs")

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

    def write_input_manifest(self, job_id: str, keys: Sequence[str]) -> Path:
        """Point the job at photos in the shared input store (replaces any previous inputs)."""
        paths = self.paths(job_id)
        if paths.input_dir.exists():
            shutil.rmtree(paths.input_dir)
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        for key in keys:
            self.inputs.path(key)   # validates every key before the manifest is written
        names = [Path(k).name for k in keys]
        if len(set(names)) != len(names):
            raise ValueError(f"job {job_id}: two inputs share a file name")
        manifest = paths.input_dir / _MANIFEST
        manifest.write_text(json.dumps({"inputs": list(keys)}, indent=2), encoding="utf-8")
        logger.info("job %s: %d inputs in the shared store", job_id, len(keys))
        return manifest

    def input_keys(self, job_id: str) -> Optional[List[str]]:
        """The job's input-store keys, or None for an older job whose images sit in ``input/``."""
        manifest = self.paths(job_id).input_dir / _MANIFEST
        if not manifest.is_file():
            return None
        return list(json.loads(manifest.read_text(encoding="utf-8"))["inputs"])

    def input_images(self, job_id: str) -> List[Path]:
        """The job's staged images, sorted by name (stable order -> stable album order)."""
        keys = self.input_keys(job_id)
        if keys is not None:
            return sorted((self.inputs.path(k) for k in keys), key=lambda p: p.name)
        input_dir = self.paths(job_id).input_dir
        if not input_dir.is_dir():
            return []
        return sorted(p for p in input_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES)

    def input_file_keys(self, job_id: str) -> Dict[str, str]:
        """photo id (file stem) -> file key for :meth:`resolve`, for every input of the job."""
        keys = self.input_keys(job_id)
        if keys is not None:
            return {Path(k).stem: INPUTS_PREFIX + k for k in keys}
        return {p.stem: f"input/{p.name}" for p in self.input_images(job_id)}

    def results_files(self, job_id: str) -> List[str]:
        """What a worker returns (spec-115 R8): ``album.json``, ``run.json`` and ``results/*``, relative to the job
        folder. Pictures are never among them; the server renders its own (``LocalRenders``)."""
        paths = self.paths(job_id)
        files = [p for p in (paths.album_path, paths.run_path) if p.is_file()]
        if paths.results_dir.is_dir():
            files += sorted(p for p in paths.results_dir.rglob("*") if p.is_file())
        return [p.relative_to(paths.root).as_posix() for p in files]

    def resolve(self, job_id: str, key: str) -> Optional[Path]:
        """A stored file key -> an existing file of THIS job: inside its folder, or a photo its manifest lists
        (spec-115 R7: another job's photo is not served through this job, even with the right key). Else None."""
        if not key:
            return None
        if key.startswith(INPUTS_PREFIX):
            if key[len(INPUTS_PREFIX):] not in (self.input_keys(job_id) or []):
                return None
            base, rel = self.inputs.root.resolve(), key[len(INPUTS_PREFIX):]
        else:
            base, rel = self.paths(job_id).root.resolve(), key
        path = (base / rel).resolve()
        if base not in path.parents or not path.is_file():
            return None
        return path

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
