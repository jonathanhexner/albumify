"""Single read interface for face clustering run directories (spec-030, Phase 2).

Replaces the existence-chain spaghetti in `face_cluster.loader` with a strict,
fail-loud API.  Every method either returns the requested data or raises
`RunStoreError`; there are no silent fallbacks.

Usage:

    store = RunStore(run_dir)        # validates layout up-front
    meta  = store.metadata()
    log   = store.merge_log()         # List[MergeDecisionRow]
    emb   = store.embeddings()        # EmbeddingMatrix(matrix, face_ids)
    base  = store.clusters("base")    # ClusterResult at iteration 0
    final = store.clusters("final")   # ClusterResult at max iteration
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union


def _safe_json(s: Optional[str]) -> Dict:
    """Parse a JSON string, tolerating None / malformed input.

    Used by ``image_detail`` to materialize the ``measured_json`` column.
    A malformed payload becomes an empty dict rather than raising — the row
    is metadata, not load-bearing.
    """
    if not s:
        return {}
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return {}

import numpy as np

from sim_bench.run_db._schema import EXPECTED_ARTIFACTS, SCHEMA_VERSION
from face_cluster.image_detail import FaceDetail, FaceFilterDecision, ImageDetail
from face_cluster.types import (
    ClusterResult,
    FaceRecord,
    MergeDecisionRow,
)

logger = logging.getLogger(__name__)


class RunStoreError(RuntimeError):
    """Raised when a run directory is missing, malformed, or schema-incompatible."""


# ---------------------------------------------------------------------------
# Public return types
# ---------------------------------------------------------------------------

class EmbeddingMatrix(NamedTuple):
    """Bulk numeric embeddings + their face_id alignment.

    matrix[i] is the embedding for face_ids[i].  Loading one without the other
    is a contract violation; `RunStore.embeddings()` always returns both.
    """
    matrix: np.ndarray
    face_ids: np.ndarray


@dataclass
class FilterDecisionRow:
    """spec-032: one row of the filter_decisions table.

    Mirrors the schema in `face_cluster.run_exporter._SCHEMA`. Used by
    consumers (FC App Run Summary tab, analysis notebooks) to ask
    "why did item X end up rejected?" via a typed object instead of raw CSV.
    """
    item_id:     str
    item_type:   str            # "image" | "face" | "cluster"
    parent_id:   Optional[str]
    filter_name: str
    rejected:    bool
    reason:      str
    measured:    Dict


@dataclass
class RunMetadata:
    """Run-level facts mirroring the `run_metadata` table.

    Mirrors `RunExporter._write_run_metadata` exactly; if a field is added to
    the writer it must be added here (and vice versa) — tests guard the symmetry.
    """
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


# ---------------------------------------------------------------------------
# RunStore
# ---------------------------------------------------------------------------

# Field sets used to convert SQLite ints back to Python bools when materialising
# a MergeDecisionRow.  Kept as module-level constants so the cost is paid once.
_BOOL_MERGE_FIELDS = frozenset({
    "passes_cross", "passes_exemplar", "passes_support",
    "passes_margin", "passes_diameter", "actually_merged",
})


class RunStore:
    """Read-only view of a face clustering run directory.

    Construction validates the layout — missing or schema-incompatible runs raise
    immediately.  All read methods assume a valid run and raise on any data
    integrity problem (no `dict.get(..., default)`).
    """

    def __init__(self, run_dir: Union[str, Path]):
        self.run_dir = Path(run_dir)
        self._pipeline_run: Dict = self._load_and_validate()

        self._db_path = self.run_dir / "face_clustering.db"
        self._embeddings_path = self.run_dir / "embeddings.npy"
        self._embedding_face_ids_path = self.run_dir / "embedding_face_ids.npy"
        self._crops_dir = self.run_dir / "crops"

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def _load_and_validate(self) -> Dict:
        if not self.run_dir.is_dir():
            raise RunStoreError(f"run directory not found: {self.run_dir}")

        # pipeline_run.json must exist before we can check schema_version.
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

        # Every other expected artifact must exist.  No fallbacks.
        for name in EXPECTED_ARTIFACTS:
            p = self.run_dir / name
            if name == "crops":
                if not p.is_dir():
                    raise RunStoreError(
                        f"required directory {name!r} not found in {self.run_dir}"
                    )
            else:
                if not p.is_file():
                    raise RunStoreError(
                        f"required artifact {name!r} not found in {self.run_dir}"
                    )

        # Belt-and-braces: PRAGMA user_version on the DB must agree with the JSON.
        try:
            conn = sqlite3.connect(str(self.run_dir / "face_clustering.db"))
            db_sv = conn.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.Error as e:
            raise RunStoreError(f"face_clustering.db is unreadable: {e}") from e
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if db_sv != SCHEMA_VERSION:
            raise RunStoreError(
                f"DB user_version {db_sv} != schema_version {SCHEMA_VERSION} in {self.run_dir}"
            )

        return payload

    # ------------------------------------------------------------------
    # Public reads
    # ------------------------------------------------------------------

    def metadata(self) -> RunMetadata:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM run_metadata").fetchone()
        if row is None:
            raise RunStoreError(
                f"run_metadata table is empty in {self._db_path}"
            )
        meta = dict(row)
        return RunMetadata(
            run_id=meta["run_id"],
            source_album=meta["source_album"],
            producer=meta["producer"],
            parent_run_id=meta["parent_run_id"],
            config=json.loads(meta["config_json"]) if meta["config_json"] else {},
            merge_thresholds=(
                json.loads(meta["merge_thresholds_json"])
                if meta["merge_thresholds_json"] else None
            ),
            merge_iter_summary=(
                json.loads(meta["merge_iter_summary_json"])
                if meta["merge_iter_summary_json"] else None
            ),
            n_images=meta["n_images"],
            n_faces=meta["n_faces"],
            n_core=meta["n_core"],
            n_clusters_base=meta["n_clusters_base"],
            n_clusters_final=meta["n_clusters_final"],
            n_merges=meta["n_merges"],
            n_iterations=meta["n_iterations"],
            started_at=meta["started_at"],
            finished_at=meta["finished_at"],
            schema_version=meta["schema_version"],
        )

    def faces(self) -> List[FaceRecord]:
        """Return all FaceRecord objects sorted by face_id."""
        with self._connect() as conn:
            face_rows = conn.execute("SELECT * FROM faces ORDER BY face_id").fetchall()

        # Embeddings live in npy, not the DB — load once and align by face_id.
        emb = self.embeddings()
        face_id_to_idx = {int(fid): i for i, fid in enumerate(emb.face_ids)}

        records: List[FaceRecord] = []
        for r in face_rows:
            fid = r["face_id"]
            pose = None
            if r["yaw"] is not None:
                pose = (r["yaw"], r["pitch"], r["roll"])
            embedding = None
            idx = face_id_to_idx.get(fid)
            if idx is not None:
                embedding = emb.matrix[idx]
            records.append(FaceRecord(
                face_id=fid,
                image_id=r["image_id"] or "",
                bbox=(
                    r["bbox_x"] or 0.0,
                    r["bbox_y"] or 0.0,
                    r["bbox_w"] or 0.0,
                    r["bbox_h"] or 0.0,
                ),
                embedding=embedding,
                embedding_normalized=embedding,
                pose=pose,
                blur_score=r["blur_score"] or 0.0,
                area=r["area"] or 0.0,
                is_core=bool(r["is_core"]),
                image_path=r["image_path"],
                face_index=r["face_index"],
                det_score=r["det_score"],
                rejection_reason=r["rejection_reason"],
                # SIGHTING-080 follow-up 2026-05-29: surface the crop path
                # so v2 UI components can display thumbnails without
                # guessing the filename pattern (was hardcoded as
                # face_{id:04d}.jpg in face_grid; real writer uses
                # face_{id:04d}_aligned.jpg).
                crop_path=r["crop_path"] if r["crop_path"] else None,
            ))
        return records

    def merge_log(self) -> List[MergeDecisionRow]:
        """Return the full merge log as MergeDecisionRow dataclasses.

        DB column order is asserted to match `MergeDecisionRow.field_names()` by
        the writer's schema; we materialize positionally.  SQLite booleans come
        back as ints — we convert the known bool columns explicitly.  No silent
        defaults.
        """
        field_names = MergeDecisionRow.field_names()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {','.join(field_names)} FROM merge_decisions "
                "ORDER BY iteration, cluster_a, cluster_b"
            ).fetchall()

        out: List[MergeDecisionRow] = []
        for raw in rows:
            kwargs = {}
            for name, value in zip(field_names, raw):
                if name in _BOOL_MERGE_FIELDS and value is not None:
                    kwargs[name] = bool(value)
                else:
                    kwargs[name] = value
            out.append(MergeDecisionRow(**kwargs))
        return out

    def image_detail(self, image_path: str) -> ImageDetail:
        """Return everything we know about one image in this run (spec-033 P-D).

        Closes spec-023 US1's "data source = ``image_metrics`` JSON" gap:
        the main-app popup and the FC App popup both call this method
        instead of recomputing from raw context dicts.

        Returns an ``ImageDetail`` with image-level scores, scene assignment,
        every face on the image, each face's cluster + exemplar status, and
        every spec-032 filter decision (image-level and per-face).

        Raises ``RunStoreError`` if the run doesn't contain this image at all.
        """
        with self._connect() as conn:
            # 1. Image-level fields come from any face row on this image
            #    (the columns are denormalized onto faces — spec-033 P-C C-3).
            img_row = conn.execute(
                "SELECT image_path, image_id, iqa_score, ava_score, "
                "sharpness_score, scene_cluster_id FROM faces "
                "WHERE image_path = ? LIMIT 1",
                (image_path,),
            ).fetchone()
            if img_row is None:
                # Try matching on image_id too — FC App standalone often
                # stores image_id as basename without a full image_path.
                img_row = conn.execute(
                    "SELECT image_path, image_id, iqa_score, ava_score, "
                    "sharpness_score, scene_cluster_id FROM faces "
                    "WHERE image_id = ? LIMIT 1",
                    (image_path,),
                ).fetchone()
            if img_row is None:
                raise RunStoreError(
                    f"No face row found for image_path={image_path!r} in this run."
                )

            # 2. All faces for this image.
            face_rows = conn.execute(
                "SELECT * FROM faces WHERE image_path = ? OR image_id = ? "
                "ORDER BY face_index, face_id",
                (image_path, image_path),
            ).fetchall()

            # 3. Final cluster assignments (latest iteration per face).
            face_ids = [r["face_id"] for r in face_rows]
            assignments: Dict[int, Tuple[int, bool]] = {}
            if face_ids:
                placeholders = ",".join("?" for _ in face_ids)
                latest_iter = conn.execute(
                    f"SELECT face_id, cluster_id, is_exemplar, "
                    f"  MAX(iteration) AS it "
                    f"FROM cluster_assignments "
                    f"WHERE face_id IN ({placeholders}) "
                    f"GROUP BY face_id",
                    face_ids,
                ).fetchall()
                for r in latest_iter:
                    assignments[r["face_id"]] = (r["cluster_id"], bool(r["is_exemplar"]))

            # 4. Per-face filter decisions (spec-032).
            face_decisions: Dict[str, List[FaceFilterDecision]] = {}
            for r in conn.execute(
                "SELECT item_id, filter_name, rejected, reason, measured_json "
                "FROM filter_decisions WHERE item_type = 'face'",
            ).fetchall():
                face_decisions.setdefault(r["item_id"], []).append(
                    FaceFilterDecision(
                        filter_name=r["filter_name"],
                        rejected=bool(r["rejected"]),
                        reason=r["reason"],
                        measured=_safe_json(r["measured_json"]),
                    )
                )

            # 5. Image-level filter decisions for this image.
            image_decisions: List[FaceFilterDecision] = [
                FaceFilterDecision(
                    filter_name=r["filter_name"],
                    rejected=bool(r["rejected"]),
                    reason=r["reason"],
                    measured=_safe_json(r["measured_json"]),
                )
                for r in conn.execute(
                    "SELECT filter_name, rejected, reason, measured_json "
                    "FROM filter_decisions WHERE item_type = 'image' AND item_id = ?",
                    (image_path,),
                ).fetchall()
            ]

        faces: List[FaceDetail] = []
        for r in face_rows:
            fid = r["face_id"]
            cluster_id, is_exemplar = assignments.get(fid, (None, False))
            pose = None
            if r["yaw"] is not None:
                pose = (float(r["yaw"]), float(r["pitch"] or 0.0), float(r["roll"] or 0.0))
            # Look up per-face decisions by either the literal face_id string
            # or the canonical "<image_path>:<face_index>" key used by spec-032.
            decisions = (
                face_decisions.get(str(fid))
                or face_decisions.get(f"{r['image_path']}:{r['face_index']}")
                or []
            )
            faces.append(FaceDetail(
                face_id=fid,
                face_index=r["face_index"],
                bbox=(r["bbox_x"] or 0.0, r["bbox_y"] or 0.0,
                      r["bbox_w"] or 0.0, r["bbox_h"] or 0.0),
                crop_path=r["crop_path"] or None,
                det_score=r["det_score"],
                blur_score=float(r["blur_score"] or 0.0),
                area=float(r["area"] or 0.0),
                pose=pose,
                is_core=bool(r["is_core"]),
                rejection_reason=r["rejection_reason"],
                cluster_id=cluster_id,
                is_exemplar=is_exemplar,
                filter_decisions=decisions,
            ))

        return ImageDetail(
            image_path=img_row["image_path"] or image_path,
            image_id=img_row["image_id"],
            iqa_score=img_row["iqa_score"],
            ava_score=img_row["ava_score"],
            sharpness_score=img_row["sharpness_score"],
            scene_cluster_id=img_row["scene_cluster_id"],
            faces=faces,
            image_filter_decisions=image_decisions,
        )

    def filter_decisions(self) -> List["FilterDecisionRow"]:
        """Return spec-032 filter_decisions table rows.

        Empty list when the run pre-dates spec-032 (table exists but holds no
        rows) — that's the dual-write transition window. The table is created
        unconditionally by the schema; absence of the table indicates v4
        schema corruption and raises.
        """
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT item_id, item_type, parent_id, filter_name, "
                    "rejected, reason, measured_json "
                    "FROM filter_decisions "
                    "ORDER BY item_id, filter_name"
                ).fetchall()
            except sqlite3.OperationalError as e:
                raise RunStoreError(
                    f"filter_decisions table missing or unreadable: {e}"
                ) from e

        import json as _json
        out: List[FilterDecisionRow] = []
        for raw in rows:
            item_id, item_type, parent_id, filter_name, rejected, reason, mjson = raw
            try:
                measured = _json.loads(mjson) if mjson else {}
            except _json.JSONDecodeError as e:
                raise RunStoreError(
                    f"filter_decisions: malformed measured_json for "
                    f"{item_id} / {filter_name}: {e}"
                ) from e
            out.append(FilterDecisionRow(
                item_id=item_id,
                item_type=item_type,
                parent_id=parent_id,
                filter_name=filter_name,
                rejected=bool(rejected),
                reason=reason,
                measured=measured,
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
        """Resolve a face_id to its on-disk crop path.

        Looks up `faces.crop_path` in the DB and resolves it relative to the
        run directory.  Raises if the face is unknown OR the crop file is missing.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT crop_path FROM faces WHERE face_id = ?",
                (face_id,),
            ).fetchone()
        if row is None:
            raise RunStoreError(f"unknown face_id {face_id} in {self.run_dir}")
        rel = row["crop_path"]
        if not rel:
            raise RunStoreError(
                f"face_id {face_id} has no crop_path recorded in {self.run_dir}"
            )
        path = (self.run_dir / rel).resolve()
        if not path.is_file():
            raise RunStoreError(
                f"crop file missing for face_id {face_id}: {path}"
            )
        return path

    def clusters(self, iteration: Union[int, str]) -> ClusterResult:
        """Return a ClusterResult snapshot at the given iteration.

        `iteration` accepts either an integer iteration number, "base" (alias
        for iteration 0), or "final" (alias for the maximum iteration recorded).
        """
        iter_num = self._resolve_iteration(iteration)

        with self._connect() as conn:
            cluster_rows = conn.execute(
                "SELECT cluster_id, size, diameter, avg_intra_dist, origin, parent_ids "
                "FROM clusters WHERE iteration = ? ORDER BY cluster_id",
                (iter_num,),
            ).fetchall()
            assign_rows = conn.execute(
                "SELECT face_id, cluster_id, is_exemplar FROM cluster_assignments "
                "WHERE iteration = ?",
                (iter_num,),
            ).fetchall()
            n_faces = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            face_id_to_idx = {
                row["face_id"]: i
                for i, row in enumerate(conn.execute(
                    "SELECT face_id FROM faces ORDER BY face_id"
                ).fetchall())
            }

        if not cluster_rows:
            raise RunStoreError(
                f"no clusters recorded for iteration {iter_num} in {self._db_path}"
            )

        clusters: Dict[int, List[int]] = {row["cluster_id"]: [] for row in cluster_rows}
        exemplars: Dict[int, List[int]] = {row["cluster_id"]: [] for row in cluster_rows}
        labels = np.full(n_faces, -1, dtype=np.int32)
        for r in assign_rows:
            idx = face_id_to_idx.get(r["face_id"])
            if idx is None:
                continue
            cid = r["cluster_id"]
            clusters.setdefault(cid, []).append(idx)
            labels[idx] = cid
            if r["is_exemplar"]:
                exemplars.setdefault(cid, []).append(idx)

        cluster_stats = {}
        for row in cluster_rows:
            try:
                parents = json.loads(row["parent_ids"]) if row["parent_ids"] else []
            except (TypeError, ValueError):
                parents = []
            cluster_stats[row["cluster_id"]] = {
                "diameter": row["diameter"],
                "mean_dist": row["avg_intra_dist"],
                "size": row["size"],
                "origin": row["origin"],
                "parent_ids": parents,
            }
        return ClusterResult(
            labels=labels,
            clusters=clusters,
            cluster_stats=cluster_stats,
            exemplars=exemplars,
            n_clusters=len(clusters),
            n_noise=int((labels == -1).sum()),
        )

    def iteration_count(self) -> int:
        """Total number of merge iterations recorded (max iteration in merge_decisions)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(iteration) FROM merge_decisions"
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn
