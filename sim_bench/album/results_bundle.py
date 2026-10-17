"""Per-photo and per-face rows of a finished run -- the worker's results bundle (spec-112 TA1.2).

``album.json`` is the curated *result*; this is the *data* behind it, shaped as plain rows so the
control plane can load it into its DB (spec-112 A3) without the worker ever holding DB credentials.

    photos  one row per staged input photo: capture time, GPS, every score, scene label, in_album,
            and the pipeline decisions that touched it (so a photo filtered before scoring still
            carries why -- SIGHTING-128)
    faces   one row per detected face: photo, bbox, detector/quality signals, person id
    embeddings / embedding_face_ids   (n_faces, D) float32 + the face_id of each row

Photo ids are filename stems with ``straightened_from`` folded back (same rule as the album
normalizer), so a photo has one id across album.json, this bundle and storage keys.

Why not ``sim_bench.run_db.exporter.RunExporter``: it is face-centric (the images table is derived
from faces, with no capture time, GPS or album status), needs the face-clustering app's in-memory
ClusterResult objects, and keys rows by absolute path. Recorded in spec-112 updates.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["BundleInputs", "BundleResult", "ResultsBundler", "BUNDLE_VERSION"]

BUNDLE_VERSION = 1

# context dict attribute -> photo row score column
_SCORE_FIELDS = {
    "composite_scores": "composite",
    "iqa_scores": "iqa",
    "ava_scores": "ava",
    "sharpness_scores": "sharpness",
    "occlusion_scores": "occlusion",
    "tilt_angles": "tilt",
    "tilt_confidences": "tilt_confidence",
}


@dataclass(frozen=True, slots=True)
class BundleInputs:
    """Per-call data for :meth:`ResultsBundler.calc`."""

    context: object
    input_ids: Sequence[str]           # stems of every staged input, the universe of photos
    album_ids: Sequence[str] = ()      # ids present in album.json (picks + dropped)


@dataclass(frozen=True, slots=True)
class BundleResult:
    photos: List[Dict]
    faces: List[Dict]
    embeddings: Optional[np.ndarray] = None       # (n, D) float32, rows aligned with face_ids
    embedding_face_ids: List[int] = field(default_factory=list)
    crops: Dict[int, np.ndarray] = field(default_factory=dict)   # face_id -> aligned crop (HxWx3)

    def manifest(self) -> Dict:
        """The JSON part of the bundle (photos + faces + counts)."""
        return {
            "bundle_version": BUNDLE_VERSION,
            "n_photos": len(self.photos),
            "n_faces": len(self.faces),
            "photos": self.photos,
            "faces": self.faces,
        }


class ResultsBundler:
    """PipelineContext -> plain rows. Pure: reads the context, writes nothing."""

    def calc(self, inputs: BundleInputs) -> BundleResult:
        ctx = inputs.context
        sf = getattr(ctx, "straightened_from", {}) or {}

        def stem(p) -> str:
            p = str(p)
            return Path(sf.get(p, p)).stem

        def by_stem(d) -> Dict:
            return {stem(p): v for p, v in (d or {}).items()}

        scores = {col: by_stem(getattr(ctx, attr, {})) for attr, col in _SCORE_FIELDS.items()}
        labels = by_stem(getattr(ctx, "scene_cluster_labels", {}))
        geo = by_stem(getattr(ctx, "geo_metadata", {}))
        decisions = self._decisions_by_photo(ctx, stem)
        album_ids = set(inputs.album_ids)

        photos = []
        for pid in sorted(dict.fromkeys(inputs.input_ids)):
            meta = geo.get(pid)
            ts = getattr(meta, "timestamp", None)
            label = labels.get(pid)
            photos.append({
                "photo_id": pid,
                "taken_at": ts.isoformat() if ts is not None else None,
                "lat": _num(getattr(meta, "lat", None)),
                "lon": _num(getattr(meta, "lon", None)),
                "scene_label": int(label) if label is not None else None,
                "in_album": pid in album_ids,
                "scores": {col: _num(vals.get(pid)) for col, vals in scores.items()},
                "decisions": decisions.get(pid, []),
            })

        faces, embeddings, emb_ids, crops = self._faces(ctx, stem)
        logger.info("results bundle: %d photos, %d faces, %d embeddings",
                    len(photos), len(faces), len(emb_ids))
        return BundleResult(photos=photos, faces=faces, embeddings=embeddings,
                            embedding_face_ids=emb_ids, crops=crops)

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _decisions_by_photo(ctx, stem) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        for d in getattr(ctx, "step_decisions", []) or []:
            if getattr(d, "item_type", None) != "image":
                continue
            out.setdefault(stem(d.item_id), []).append(
                {"step": d.step, "decision": d.decision, "reason": d.reason})
        return out

    @staticmethod
    def _faces(ctx, stem):
        records = list(getattr(ctx, "face_records", []) or [])
        # people_clusters members are FaceRecords (assign_people_clusters); the removed
        # cluster_people step (SIGHTING-132, spec-113) produced face objects with original_path +
        # face_index and no face_id. Matching by (photo id, face index) works for both shapes.
        person_of: Dict[tuple, int] = {}
        for pid, members in (getattr(ctx, "people_clusters", {}) or {}).items():
            for m in members or []:
                path = getattr(m, "image_path", None) or getattr(m, "original_path", None)
                idx = getattr(m, "face_index", None)
                if path is not None and idx is not None:
                    person_of[(stem(path), int(idx))] = int(pid)

        faces, vecs, emb_ids, crops = [], [], [], {}
        for rec in records:
            fid = int(rec.face_id)
            path = getattr(rec, "image_path", None) or getattr(rec, "image_id", "")
            pose = getattr(rec, "pose", None)
            idx = getattr(rec, "face_index", None)
            faces.append({
                "face_id": fid,
                "face_index": int(idx) if idx is not None else None,
                "photo_id": stem(path),
                "bbox": [float(v) for v in rec.bbox],
                "det_score": _num(getattr(rec, "det_score", None)),
                "blur_score": _num(getattr(rec, "blur_score", None)),
                "pose": [float(v) for v in pose] if pose is not None else None,
                "is_core": bool(getattr(rec, "is_core", False)),
                "rejection_reason": getattr(rec, "rejection_reason", None),
                "person_id": person_of.get((stem(path), int(idx))) if idx is not None else None,
            })
            emb = getattr(rec, "embedding_normalized", None)
            if emb is None:
                emb = getattr(rec, "embedding", None)
            if emb is not None:
                vecs.append(np.asarray(emb, dtype=np.float32).ravel())
                emb_ids.append(fid)
            crop = getattr(rec, "aligned_face", None)
            if crop is not None:
                crops[fid] = np.asarray(crop)

        faces.sort(key=lambda f: f["face_id"])
        embeddings = np.stack(vecs) if vecs and len({v.shape for v in vecs}) == 1 else None
        if vecs and embeddings is None:
            logger.warning("results bundle: face embeddings have mixed shapes; not written")
            emb_ids = []
        return faces, embeddings, emb_ids, crops


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None
