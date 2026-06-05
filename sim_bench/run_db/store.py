"""Single read interface for face clustering run directories (spec-030 / spec-059).

Strict, fail-loud API: every method returns the requested data or raises
``RunStoreError``; no silent fallbacks. spec-059: every read goes through
SQLAlchemy ORM models. Only ``_load_and_validate()`` keeps raw sqlite3
because ``PRAGMA user_version`` must run before ORM machinery (locked
decision #3).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from sqlalchemy import func, select

from face_cluster.image_detail import FaceDetail, FaceFilterDecision, ImageDetail
from face_cluster.types import ClusterResult, FaceRecord, MergeDecisionRow
from sim_bench.run_db._schema import EXPECTED_ARTIFACTS, SCHEMA_VERSION
from sim_bench.run_db._session import make_run_db_sessionmaker
from sim_bench.run_db.models import (
    Cluster,
    ClusterAssignment,
    Face,
    FilterDecision,
    MergeDecision,
    RunMetadataRow,
)

logger = logging.getLogger(__name__)


def _safe_json(s: Optional[str]) -> Dict:
    """Parse JSON, return {} on None or malformed (image_detail metadata only)."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return {}


class RunStoreError(RuntimeError):
    """Raised when a run directory is missing, malformed, or schema-incompatible."""


class EmbeddingMatrix(NamedTuple):
    """Bulk numeric embeddings + their face_id alignment."""
    matrix: np.ndarray
    face_ids: np.ndarray


@dataclass
class FilterDecisionRow:
    """spec-032: one row of the filter_decisions table."""
    item_id: str
    item_type: str
    parent_id: Optional[str]
    filter_name: str
    rejected: bool
    reason: str
    measured: Dict


@dataclass(frozen=True, slots=True)
class ImageRow:
    """spec-077: one row of the ``images`` table (per-image scores + gate)."""
    image_path: str
    n_faces: int
    width_px: Optional[int]
    height_px: Optional[int]
    iqa_score: Optional[float]
    ava_score: Optional[float]
    sharpness_score: Optional[float]
    composite_score: Optional[float]
    filter_passed: bool


@dataclass
class RunMetadata:
    """Run-level facts mirroring the ``run_metadata`` table."""
    run_id: str
    source_album: str
    producer: str
    parent_run_id: Optional[str]
    config: Dict
    merge_thresholds: Optional[Dict]
    merge_iter_summary: Optional[Dict]
    n_images: int
    n_faces: int
    n_core: int
    n_clusters_base: int
    n_clusters_final: int
    n_merges: int
    n_iterations: int
    started_at: str
    finished_at: str
    schema_version: int


# SQLite int → Python bool conversion for known boolean columns in MergeDecisionRow.
_BOOL_MERGE_FIELDS = frozenset({
    "passes_cross", "passes_exemplar", "passes_support",
    "passes_margin", "passes_diameter", "actually_merged",
})


def _face_record_from_orm(r: Face, embedding: Optional[np.ndarray]) -> FaceRecord:
    """Hydrate an ORM Face row + npy embedding into a FaceRecord."""
    pose = (r.yaw, r.pitch, r.roll) if r.yaw is not None else None
    return FaceRecord(
        face_id=r.face_id,
        image_id=r.image_id or "",
        bbox=(r.bbox_x or 0.0, r.bbox_y or 0.0, r.bbox_w or 0.0, r.bbox_h or 0.0),
        embedding=embedding,
        embedding_normalized=embedding,
        pose=pose,
        blur_score=r.blur_score or 0.0,
        area=r.area or 0.0,
        is_core=bool(r.is_core),
        image_path=r.image_path,
        face_index=r.face_index,
        det_score=r.det_score,
        rejection_reason=r.rejection_reason,
        crop_path=r.crop_path if r.crop_path else None,
        # spec-073/072 fix: the faces table stores these v5 ratio columns but
        # the reader dropped them, so FaceRecord.area_ratio was always None —
        # Area % rendered empty and the area-% gate had no data on reload.
        area_ratio=r.area_ratio,
        bbox_x_ratio=r.bbox_x_ratio,
        bbox_y_ratio=r.bbox_y_ratio,
        bbox_w_ratio=r.bbox_w_ratio,
        bbox_h_ratio=r.bbox_h_ratio,
    )


def _face_detail_from_orm(
    r: Face,
    cluster_id: Optional[int],
    is_exemplar: bool,
    decisions: List[FaceFilterDecision],
) -> FaceDetail:
    """Hydrate an ORM Face row + cluster info + filter decisions into a FaceDetail."""
    pose = None
    if r.yaw is not None:
        pose = (float(r.yaw), float(r.pitch or 0.0), float(r.roll or 0.0))
    return FaceDetail(
        face_id=r.face_id,
        face_index=r.face_index,
        bbox=(r.bbox_x or 0.0, r.bbox_y or 0.0, r.bbox_w or 0.0, r.bbox_h or 0.0),
        crop_path=r.crop_path or None,
        det_score=r.det_score,
        blur_score=float(r.blur_score or 0.0),
        area=float(r.area or 0.0),
        pose=pose,
        is_core=bool(r.is_core),
        rejection_reason=r.rejection_reason,
        cluster_id=cluster_id,
        is_exemplar=is_exemplar,
        filter_decisions=decisions,
    )


class RunStore:
    """Read-only view of a face clustering run directory.

    Construction validates the layout on raw sqlite3 (``PRAGMA user_version``
    must precede ORM machinery), then builds a per-run sessionmaker. Every
    read opens a fresh session on entry and closes it on exit.
    """

    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir)
        self._pipeline_run: Dict = self._load_and_validate()
        self._db_path = self.run_dir / "face_clustering.db"
        self._embeddings_path = self.run_dir / "embeddings.npy"
        self._embedding_face_ids_path = self.run_dir / "embedding_face_ids.npy"
        self._crops_dir = self.run_dir / "crops"
        self._sessionmaker = make_run_db_sessionmaker(self.run_dir)

    # ------------------------------------------------------------------
    # Construction-time validation (locked decision #3 — raw sqlite3 only)
    # ------------------------------------------------------------------

    def _load_and_validate(self) -> Dict:
        if not self.run_dir.is_dir():
            raise RunStoreError(f"run directory not found: {self.run_dir}")

        pr_path = self.run_dir / "pipeline_run.json"
        if not pr_path.is_file():
            raise RunStoreError(
                f"required artifact 'pipeline_run.json' not found in {self.run_dir}"
            )
        try:
            payload = json.loads(pr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RunStoreError(f"pipeline_run.json is not valid JSON: {e}") from e

        sv = payload.get("schema_version")
        if sv != SCHEMA_VERSION:
            raise RunStoreError(
                f"schema_version mismatch in {self.run_dir}: run has {sv!r}, "
                f"RunStore expects {SCHEMA_VERSION}. "
                f"Run scripts/migrate_run_to_v4.py to upgrade."
            )

        for name in EXPECTED_ARTIFACTS:
            p = self.run_dir / name
            if name == "crops":
                if not p.is_dir():
                    raise RunStoreError(
                        f"required directory {name!r} not found in {self.run_dir}"
                    )
            elif not p.is_file():
                raise RunStoreError(
                    f"required artifact {name!r} not found in {self.run_dir}"
                )

        conn = None
        try:
            conn = sqlite3.connect(str(self.run_dir / "face_clustering.db"))
            db_sv = conn.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.Error as e:
            raise RunStoreError(f"face_clustering.db is unreadable: {e}") from e
        finally:
            if conn is not None:
                conn.close()
        if db_sv != SCHEMA_VERSION:
            raise RunStoreError(
                f"DB user_version {db_sv} != schema_version {SCHEMA_VERSION} in {self.run_dir}"
            )
        return payload

    # ------------------------------------------------------------------
    # Public reads
    # ------------------------------------------------------------------

    def metadata(self) -> RunMetadata:
        with self._sessionmaker() as session:
            row = session.execute(select(RunMetadataRow)).scalar_one_or_none()
        if row is None:
            raise RunStoreError(f"run_metadata table is empty in {self._db_path}")
        return RunMetadata(
            run_id=row.run_id,
            source_album=row.source_album,
            producer=row.producer,
            parent_run_id=row.parent_run_id,
            config=json.loads(row.config_json) if row.config_json else {},
            merge_thresholds=json.loads(row.merge_thresholds_json) if row.merge_thresholds_json else None,
            merge_iter_summary=json.loads(row.merge_iter_summary_json) if row.merge_iter_summary_json else None,
            n_images=row.n_images,
            n_faces=row.n_faces,
            n_core=row.n_core,
            n_clusters_base=row.n_clusters_base,
            n_clusters_final=row.n_clusters_final,
            n_merges=row.n_merges,
            n_iterations=row.n_iterations,
            started_at=row.started_at,
            finished_at=row.finished_at,
            schema_version=row.schema_version,
        )

    def faces(self) -> List[FaceRecord]:
        """Return all FaceRecord objects sorted by face_id."""
        with self._sessionmaker() as session:
            face_rows = session.execute(
                select(Face).order_by(Face.face_id)
            ).scalars().all()

        emb = self.embeddings()
        face_id_to_idx = {int(fid): i for i, fid in enumerate(emb.face_ids)}
        return [
            _face_record_from_orm(
                r, emb.matrix[face_id_to_idx[r.face_id]] if r.face_id in face_id_to_idx else None
            )
            for r in face_rows
        ]

    def merge_log(self) -> List[MergeDecisionRow]:
        """Return the full merge log; ints in _BOOL_MERGE_FIELDS converted to bool."""
        field_names = MergeDecisionRow.field_names()
        with self._sessionmaker() as session:
            rows = session.execute(
                select(MergeDecision).order_by(
                    MergeDecision.iteration, MergeDecision.cluster_a, MergeDecision.cluster_b
                )
            ).scalars().all()
        out: List[MergeDecisionRow] = []
        for r in rows:
            kwargs = {}
            for name in field_names:
                value = getattr(r, name)
                kwargs[name] = bool(value) if name in _BOOL_MERGE_FIELDS and value is not None else value
            out.append(MergeDecisionRow(**kwargs))
        return out

    def image_detail(self, image_path: str) -> ImageDetail:
        """spec-033 P-D — image-level scores + all faces + filter decisions for ``image_path``.

        Falls back to ``image_id`` match for FC App standalone runs (which
        store basenames). Raises if no Face row matches.
        """
        with self._sessionmaker() as session:
            img_row = session.execute(
                select(Face).where(Face.image_path == image_path).limit(1)
            ).scalar_one_or_none()
            if img_row is None:
                img_row = session.execute(
                    select(Face).where(Face.image_id == image_path).limit(1)
                ).scalar_one_or_none()
            if img_row is None:
                raise RunStoreError(
                    f"No face row found for image_path={image_path!r} in this run."
                )

            face_rows = session.execute(
                select(Face)
                .where((Face.image_path == image_path) | (Face.image_id == image_path))
                .order_by(Face.face_index, Face.face_id)
            ).scalars().all()

            face_ids = [r.face_id for r in face_rows]
            assignments: Dict[int, Tuple[int, bool]] = {}
            if face_ids:
                for fid, cid, is_exemplar, _it in session.execute(
                    select(
                        ClusterAssignment.face_id,
                        ClusterAssignment.cluster_id,
                        ClusterAssignment.is_exemplar,
                        func.max(ClusterAssignment.iteration),
                    )
                    .where(ClusterAssignment.face_id.in_(face_ids))
                    .group_by(ClusterAssignment.face_id)
                ).all():
                    assignments[fid] = (cid, bool(is_exemplar))

            face_decisions: Dict[str, List[FaceFilterDecision]] = {}
            for item_id, fname, rejected, reason, mjson in session.execute(
                select(
                    FilterDecision.item_id, FilterDecision.filter_name,
                    FilterDecision.rejected, FilterDecision.reason, FilterDecision.measured_json,
                ).where(FilterDecision.item_type == "face")
            ).all():
                face_decisions.setdefault(item_id, []).append(
                    FaceFilterDecision(
                        filter_name=fname, rejected=bool(rejected),
                        reason=reason, measured=_safe_json(mjson),
                    )
                )

            image_decisions = [
                FaceFilterDecision(
                    filter_name=fname, rejected=bool(rejected),
                    reason=reason, measured=_safe_json(mjson),
                )
                for fname, rejected, reason, mjson in session.execute(
                    select(
                        FilterDecision.filter_name, FilterDecision.rejected,
                        FilterDecision.reason, FilterDecision.measured_json,
                    )
                    .where(FilterDecision.item_type == "image")
                    .where(FilterDecision.item_id == image_path)
                ).all()
            ]

        faces = []
        for r in face_rows:
            cluster_id, is_exemplar = assignments.get(r.face_id, (None, False))
            decisions = (
                face_decisions.get(str(r.face_id))
                or face_decisions.get(f"{r.image_path}:{r.face_index}")
                or []
            )
            faces.append(_face_detail_from_orm(r, cluster_id, is_exemplar, decisions))

        return ImageDetail(
            image_path=img_row.image_path or image_path,
            image_id=img_row.image_id,
            iqa_score=img_row.iqa_score,
            ava_score=img_row.ava_score,
            sharpness_score=img_row.sharpness_score,
            scene_cluster_id=img_row.scene_cluster_id,
            faces=faces,
            image_filter_decisions=image_decisions,
        )

    def list_images(self) -> List[ImageRow]:
        """spec-077: one :class:`ImageRow` per row of the ``images`` table."""
        from sim_bench.run_db.models import Image
        with self._sessionmaker() as session:
            rows = session.execute(select(Image).order_by(Image.image_path)).scalars().all()
        return [
            ImageRow(
                image_path=r.image_path,
                n_faces=int(r.n_faces or 0),
                width_px=r.width_px,
                height_px=r.height_px,
                iqa_score=r.iqa_score,
                ava_score=r.ava_score,
                sharpness_score=r.sharpness_score,
                composite_score=r.composite_score,
                filter_passed=bool(r.filter_passed),
            )
            for r in rows
        ]

    def filter_decisions(self) -> List[FilterDecisionRow]:
        """spec-032 filter_decisions table rows. Empty list for pre-spec-032 runs."""
        with self._sessionmaker() as session:
            rows = session.execute(
                select(FilterDecision).order_by(
                    FilterDecision.item_id, FilterDecision.filter_name
                )
            ).scalars().all()
        out: List[FilterDecisionRow] = []
        for r in rows:
            try:
                measured = json.loads(r.measured_json) if r.measured_json else {}
            except json.JSONDecodeError as e:
                raise RunStoreError(
                    f"filter_decisions: malformed measured_json for "
                    f"{r.item_id} / {r.filter_name}: {e}"
                ) from e
            out.append(FilterDecisionRow(
                item_id=r.item_id, item_type=r.item_type, parent_id=r.parent_id,
                filter_name=r.filter_name, rejected=bool(r.rejected),
                reason=r.reason, measured=measured,
            ))
        return out

    def embeddings(self) -> EmbeddingMatrix:
        """Return (matrix, face_ids) loaded atomically from the two .npy files."""
        try:
            matrix = np.load(self._embeddings_path)
            face_ids = np.load(self._embedding_face_ids_path)
        except (OSError, ValueError) as e:
            raise RunStoreError(f"failed to load embeddings: {e}") from e
        if matrix.shape[0] != face_ids.shape[0]:
            raise RunStoreError(
                f"embeddings.npy / embedding_face_ids.npy length mismatch: "
                f"{matrix.shape[0]} vs {face_ids.shape[0]}"
            )
        return EmbeddingMatrix(matrix=matrix, face_ids=face_ids)

    def crop_path(self, face_id: int) -> Path:
        """Resolve face_id → on-disk crop path. Raises on unknown face / missing file."""
        with self._sessionmaker() as session:
            rel = session.execute(
                select(Face.crop_path).where(Face.face_id == face_id)
            ).scalar_one_or_none()
            if rel is None:
                # Disambiguate "no row" vs "crop_path IS NULL".
                exists = session.execute(
                    select(func.count()).select_from(Face).where(Face.face_id == face_id)
                ).scalar_one()
                if not exists:
                    raise RunStoreError(f"unknown face_id {face_id} in {self.run_dir}")
                raise RunStoreError(
                    f"face_id {face_id} has no crop_path recorded in {self.run_dir}"
                )
        path = (self.run_dir / rel).resolve()
        if not path.is_file():
            raise RunStoreError(f"crop file missing for face_id {face_id}: {path}")
        return path

    def clusters(self, iteration: Union[int, str]) -> ClusterResult:
        """Return a ClusterResult at the given iteration ('base', 'final', or int)."""
        iter_num = self._resolve_iteration(iteration)
        with self._sessionmaker() as session:
            cluster_rows = session.execute(
                select(
                    Cluster.cluster_id, Cluster.size, Cluster.diameter,
                    Cluster.avg_intra_dist, Cluster.origin, Cluster.parent_ids,
                ).where(Cluster.iteration == iter_num).order_by(Cluster.cluster_id)
            ).all()
            assign_rows = session.execute(
                select(
                    ClusterAssignment.face_id, ClusterAssignment.cluster_id,
                    ClusterAssignment.is_exemplar,
                ).where(ClusterAssignment.iteration == iter_num)
            ).all()
            n_faces = session.execute(select(func.count()).select_from(Face)).scalar_one()
            all_face_ids = session.execute(
                select(Face.face_id).order_by(Face.face_id)
            ).scalars().all()
            face_id_to_idx = {fid: i for i, fid in enumerate(all_face_ids)}

        if not cluster_rows:
            raise RunStoreError(
                f"no clusters recorded for iteration {iter_num} in {self._db_path}"
            )

        clusters: Dict[int, List[int]] = {cid: [] for (cid, *_) in cluster_rows}
        exemplars: Dict[int, List[int]] = {cid: [] for (cid, *_) in cluster_rows}
        labels = np.full(n_faces, -1, dtype=np.int32)
        for face_id, cid, is_exemplar in assign_rows:
            idx = face_id_to_idx.get(face_id)
            if idx is None:
                continue
            clusters.setdefault(cid, []).append(idx)
            labels[idx] = cid
            if is_exemplar:
                exemplars.setdefault(cid, []).append(idx)

        cluster_stats = {}
        for cid, size, diameter, avg_intra_dist, origin, parent_ids in cluster_rows:
            try:
                parents = json.loads(parent_ids) if parent_ids else []
            except (TypeError, ValueError):
                parents = []
            cluster_stats[cid] = {
                "diameter": diameter, "mean_dist": avg_intra_dist,
                "size": size, "origin": origin, "parent_ids": parents,
            }
        return ClusterResult(
            labels=labels, clusters=clusters, cluster_stats=cluster_stats,
            exemplars=exemplars, n_clusters=len(clusters),
            n_noise=int((labels == -1).sum()),
        )

    def iteration_count(self) -> int:
        """Max iteration recorded in merge_decisions (0 if no merges)."""
        with self._sessionmaker() as session:
            value = session.execute(select(func.max(MergeDecision.iteration))).scalar()
        return int(value) if value is not None else 0

    def _resolve_iteration(self, iteration: Union[int, str]) -> int:
        if isinstance(iteration, str):
            if iteration == "base":
                return 0
            if iteration == "final":
                return self.iteration_count()
            raise RunStoreError(
                f"iteration label must be 'base' or 'final', got {iteration!r}"
            )
        if not isinstance(iteration, int):
            raise RunStoreError(
                f"iteration must be int or 'base'/'final', got {type(iteration).__name__}"
            )
        return iteration
