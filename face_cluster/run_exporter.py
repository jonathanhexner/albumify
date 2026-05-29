"""Single writer for face clustering run artifacts (spec-030).

Replaces the duplicated write paths in `face_cluster.export`, `face_cluster.result_db`,
and `sim_bench.pipeline.steps.face_cluster_export`. Both apps (Albumify and the
standalone Face Clustering App) call into this module so a run directory has the
same byte-level layout regardless of which app produced it.

A run directory contains exactly five top-level artifacts (FR-003):

    face_clustering.db        — relational store: faces, scores, clusters,
                                cluster_assignments, merge_decisions, run_metadata
    embeddings.npy            — (n_faces, 512) float32 matrix
    embedding_face_ids.npy    — (n_faces,) int32, row i ↔ face_id at index i
    pipeline_run.json         — small human-readable run pointer (run_id, status,
                                schema_version, db_path)
    crops/face_NNNN_aligned.jpg — per-face JPEG crops (one file each)

Nothing else lives in a run directory. Consumers read everything through `RunStore`
(introduced in Phase 2 of spec-030); this module is the only writer.
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import pandas as pd

from face_cluster.db import (
    EXPECTED_ARTIFACTS,
    FACES_SCHEMA,
    FACE_SCORES_SCHEMA,
    IMAGES_SCHEMA,
    SCENE_CLUSTERS_SCHEMA,
    SCENE_CLUSTER_ASSIGNMENTS_SCHEMA,
    SCHEMA_DDL,
    SCHEMA_VERSION,
)
from face_cluster.types import (
    ClusterResult,
    FaceRecord,
    MergeDecisionRow,
)

logger = logging.getLogger(__name__)


# Schema (DDL + version + artifact allow-list) lives in `face_cluster/db/`.
# Re-export for callers that historically imported these names from this
# module (RunStore, tests, scripts).
__all__ = (
    "RunExporter", "RunExporterError",
    "RunExportInputs", "RunExportResult",
    "EXPECTED_ARTIFACTS", "SCHEMA_VERSION",
)


# spec-053: typed boundary for RunExporter.calc(). The Inputs dataclass
# is wide (19 fields) because the export contract is wide — but pinning
# it as one typed object is still better than the ad-hoc kwargs callers
# pass today. A future spec may split RunExporter.

@dataclass(frozen=True, slots=True)
class RunExportInputs:
    """Per-call data for RunExporter.calc().

    Wraps every kwarg of ``export()``. Optional fields default to None
    or empty so callers only set what they have.
    """
    faces: List[Any]
    base_cluster_result: Any
    merged_cluster_result: Optional[Any]
    core_indices: List[int]
    merge_log: Optional[List[Dict]]
    merge_metadata: Optional[Dict]
    config: Any
    source_album: str
    producer: str
    run_id: str
    started_at: str
    finished_at: str
    parent_run_id: Optional[str] = None
    crop_source_dir: Optional[Path] = None
    filters: Any = None
    image_scores: Optional[Dict[str, Dict[str, float]]] = None
    image_paths: Optional[List[str]] = None
    scene_clusters: Optional[List[Dict]] = None
    scene_cluster_assignments: Optional[List[Dict]] = None


@dataclass(frozen=True, slots=True)
class RunExportResult:
    """Output of RunExporter.calc()."""
    output_dir: Path
    db_path: Path


class RunExporterError(RuntimeError):
    """Raised when input data violates the writer's contract."""


_VALID_PRODUCERS = ("albumify", "fc_app", "remerge", "manual_merge")


class RunExporter:
    """Writes a complete face-clustering run directory.

    Construct with the target output directory, then call `export(...)` exactly once.
    The exporter is deliberately stateless across calls; reuse for multiple runs is
    not supported.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # spec-053 — single pipeline entry point
    # ------------------------------------------------------------------

    def calc(self, inputs: "RunExportInputs") -> "RunExportResult":
        """Single pipeline entry point (spec-053). Thin facade over
        ``export()`` — the kwarg surface is wide because the export
        contract is wide; pinning it in a typed dataclass keeps the
        boundary grep-able."""
        self.export(
            faces=inputs.faces,
            base_cluster_result=inputs.base_cluster_result,
            merged_cluster_result=inputs.merged_cluster_result,
            core_indices=inputs.core_indices,
            merge_log=inputs.merge_log,
            merge_metadata=inputs.merge_metadata,
            config=inputs.config,
            source_album=inputs.source_album,
            producer=inputs.producer,
            run_id=inputs.run_id,
            started_at=inputs.started_at,
            finished_at=inputs.finished_at,
            parent_run_id=inputs.parent_run_id,
            crop_source_dir=inputs.crop_source_dir,
            filters=inputs.filters,
            image_scores=inputs.image_scores,
            image_paths=inputs.image_paths,
            scene_clusters=inputs.scene_clusters,
            scene_cluster_assignments=inputs.scene_cluster_assignments,
        )
        return RunExportResult(
            output_dir=self.output_dir,
            db_path=self.output_dir / "face_clustering.db",
        )

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def export(
        self,
        *,
        faces: List[FaceRecord],
        base_cluster_result: ClusterResult,
        merged_cluster_result: Optional[ClusterResult],
        core_indices: List[int],
        merge_log: Optional[List[Dict]],
        merge_metadata: Optional[Dict],
        config,
        source_album: str,
        producer: str,
        run_id: str,
        started_at: str,
        finished_at: str,
        parent_run_id: Optional[str] = None,
        crop_source_dir: Optional[Path] = None,
        filters=None,  # face_cluster.filter_context.FilterContext | None
        image_scores: Optional[Dict[str, Dict[str, float]]] = None,
        image_paths: Optional[List[str]] = None,
        scene_clusters: Optional[List[Dict]] = None,
        scene_cluster_assignments: Optional[List[Dict]] = None,
    ) -> None:
        """Write the full 5-artifact layout into self.output_dir.

        Parameters
        ----------
        faces, base_cluster_result, merged_cluster_result, core_indices, merge_log,
        merge_metadata, config, source_album:
            In-memory pipeline state.  See `face_cluster.pipeline.PipelineResult`.
        producer:
            "albumify" | "fc_app" | "remerge" | "manual_merge".
        run_id, started_at, finished_at, parent_run_id:
            Run-level metadata that doesn't live in the algorithm state.
        crop_source_dir:
            If set, JPEG crops are copied from this directory into <output_dir>/crops/.
            Used during Phase 1 dual-write where crops were already produced by the
            legacy writer.  When None, crops are generated from `face.aligned_face`.

        Raises
        ------
        RunExporterError:
            If `producer` is not in the allow-list, or if `merge_log` rows contain
            unknown keys (strict-write contract, FR-005).
        """
        if producer not in _VALID_PRODUCERS:
            raise RunExporterError(
                f"producer must be one of {_VALID_PRODUCERS}, got {producer!r}"
            )

        merge_log = merge_log or []
        self._strict_validate_merge_log(merge_log)

        merged_cr = merged_cluster_result or base_cluster_result
        crop_manifest = self._write_crops(faces, crop_source_dir)
        self._write_embeddings_npy(faces)

        db_path = self.output_dir / "face_clustering.db"
        # Replace any prior DB for this run; idempotent re-export is a clean rewrite.
        if db_path.exists():
            db_path.unlink()

        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_DDL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

            self._write_faces_and_scores(conn, faces, crop_manifest, image_scores)
            self._write_clusters_and_assignments(
                conn, base_cluster_result, merged_cr, merge_log,
                core_indices, faces,
            )
            self._write_merges(conn, merge_log)
            self._write_filter_decisions(conn, filters)
            # spec-040 Phase 4 (schema v5) — scene-side + image-level persistence.
            # Closes REVIEW.md B3 + B6: write the 3 new tables that previously
            # had DDL + Pandera schemas but no writer. Image rows derived from
            # the face list (one row per distinct image_path) plus any
            # image_paths that produced zero faces. Scene tables remain empty
            # until the scene-clustering producer side ships in a follow-up.
            self._write_images(conn, faces, image_paths, image_scores)
            self._write_scene_clusters(conn, scene_clusters)
            self._write_scene_cluster_assignments(conn, scene_cluster_assignments)
            self._write_run_metadata(
                conn,
                faces=faces,
                base_cr=base_cluster_result,
                merged_cr=merged_cr,
                merge_log=merge_log,
                merge_metadata=merge_metadata,
                config=config,
                source_album=source_album,
                producer=producer,
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                parent_run_id=parent_run_id,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        self._write_pipeline_run_json(
            run_id=run_id,
            source_album=source_album,
            producer=producer,
            parent_run_id=parent_run_id,
            started_at=started_at,
            finished_at=finished_at,
        )

        logger.info(
            f"RunExporter wrote {len(EXPECTED_ARTIFACTS)} artifacts to {self.output_dir}"
        )

    # ------------------------------------------------------------------
    # Strict validation of merge_log rows (FR-005)
    # ------------------------------------------------------------------

    def _strict_validate_merge_log(self, merge_log: List[Dict]) -> None:
        """Reject any row whose key set != MergeDecisionRow.field_names().

        The merger's job is to produce a complete row; if a key is missing or extra,
        that's a contract violation we want to surface immediately, not paper over
        with default values.
        """
        expected = set(MergeDecisionRow.field_names())
        for i, entry in enumerate(merge_log):
            actual = set(entry.keys())
            if actual != expected:
                missing = expected - actual
                extra = actual - expected
                raise RunExporterError(
                    f"merge_log[{i}] has key set != MergeDecisionRow.field_names(); "
                    f"missing={sorted(missing)} extra={sorted(extra)}"
                )

    # ------------------------------------------------------------------
    # Faces + scores
    # ------------------------------------------------------------------

    @staticmethod
    def _write_faces_and_scores(
        conn: sqlite3.Connection,
        faces: List[FaceRecord],
        crop_manifest: Dict[int, str],
        image_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        image_scores = image_scores or {}
        face_rows = []
        score_rows = []
        for face in faces:
            bbox = face.bbox or (0.0, 0.0, 0.0, 0.0)
            yaw, pitch, roll = (face.pose or (None, None, None))
            # spec-033 P-C C-3: per-image scores looked up by image_path (the
            # canonical key used elsewhere).  Falls back to image_id for
            # FC-App-style runs.
            img_key = face.image_path or face.image_id or ""
            img_score = image_scores.get(img_key) or image_scores.get(face.image_id, {})
            face_rows.append((
                face.face_id,
                face.image_path or face.image_id,
                face.image_id,
                face.face_index,
                _maybe_float(bbox[0] if len(bbox) > 0 else None),
                _maybe_float(bbox[1] if len(bbox) > 1 else None),
                _maybe_float(bbox[2] if len(bbox) > 2 else None),
                _maybe_float(bbox[3] if len(bbox) > 3 else None),
                crop_manifest.get(face.face_id, ""),
                _maybe_float(face.det_score),
                float(face.blur_score),
                float(face.area),
                _maybe_float(yaw),
                _maybe_float(pitch),
                _maybe_float(roll),
                1 if face.is_core else 0,
                face.rejection_reason,
                # spec-033 P-C C-3: image-level context fields
                _maybe_float(img_score.get("iqa")),
                _maybe_float(img_score.get("ava")),
                _maybe_float(img_score.get("sharpness")),
                img_score.get("scene_cluster_id"),
                # spec-040 Phase 4 (schema v5): canonical unit-normalized geometry.
                # SIGHTING-064 fix. NULL on legacy code paths that don't compute
                # them; populated by the new unified producer steps once they land.
                _maybe_float(getattr(face, "area_ratio", None)),
                _maybe_float(getattr(face, "bbox_x_ratio", None)),
                _maybe_float(getattr(face, "bbox_y_ratio", None)),
                _maybe_float(getattr(face, "bbox_w_ratio", None)),
                _maybe_float(getattr(face, "bbox_h_ratio", None)),
            ))

            pose_score = None
            if face.pose and face.pose[0] is not None:
                pose_score = max(0.0, 1.0 - abs(face.pose[0]) / 90.0)
            score_rows.append((
                face.face_id, pose_score, None, None, None,
                1 if face.is_core else 0,
            ))

        # spec-033 P-H: Pandera-validate before commit.  A NULL in a
        # non-nullable column (blur_score, area, is_core, ...) is the
        # SIGHTING-059 failure mode this guards against.
        _faces_columns = [
            "face_id", "image_path", "image_id", "face_index",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "crop_path",
            "det_score", "blur_score", "area", "yaw", "pitch", "roll",
            "is_core", "rejection_reason",
            "iqa_score", "ava_score", "sharpness_score", "scene_cluster_id",
            # spec-040 Phase 4 (v5) — canonical ratios
            "area_ratio", "bbox_x_ratio", "bbox_y_ratio", "bbox_w_ratio", "bbox_h_ratio",
        ]
        FACES_SCHEMA.validate(pd.DataFrame(face_rows, columns=_faces_columns))

        _scores_columns = [
            "face_id", "pose_score", "eyes_score", "expression_score",
            "frontal_score", "is_clusterable",
        ]
        FACE_SCORES_SCHEMA.validate(pd.DataFrame(score_rows, columns=_scores_columns))

        conn.executemany(
            "INSERT INTO faces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            face_rows,
        )
        conn.executemany(
            "INSERT INTO face_scores VALUES (?,?,?,?,?,?)",
            score_rows,
        )

    # ------------------------------------------------------------------
    # Clusters + per-iteration assignments
    # ------------------------------------------------------------------

    @staticmethod
    def _write_clusters_and_assignments(
        conn: sqlite3.Connection,
        base_cr: ClusterResult,
        merged_cr: ClusterResult,
        merge_log: List[Dict],
        core_indices: List[int],
        faces: List[FaceRecord],
    ) -> None:
        # Iteration 0 — base clustering result.
        cluster_rows: List[Tuple] = []
        assign_rows: List[Tuple] = []
        for cid, members in base_cr.clusters.items():
            stats = base_cr.cluster_stats.get(cid, {})
            cluster_rows.append((
                cid, 0, len(members),
                _maybe_float(stats.get("diameter")),
                _maybe_float(stats.get("mean_dist")),
                "base", "[]",
            ))
            exemplars = set(base_cr.exemplars.get(cid, []))
            for node_idx in members:
                face_idx = (
                    core_indices[node_idx]
                    if core_indices and node_idx < len(core_indices)
                    else node_idx
                )
                if 0 <= face_idx < len(faces):
                    assign_rows.append((
                        faces[face_idx].face_id,
                        cid,
                        0,
                        1 if node_idx in exemplars else 0,
                        _maybe_float(getattr(faces[face_idx], "d10_score", None)),
                    ))

        # Final iteration — only when at least one merge actually executed.
        max_iter = max((int(e["iteration"]) for e in merge_log), default=0)
        any_merged = any(e["actually_merged"] for e in merge_log)
        if any_merged and merged_cr is not base_cr:
            parent_map = _build_parent_map(merge_log)
            for cid, members in merged_cr.clusters.items():
                stats = merged_cr.cluster_stats.get(cid, {})
                parents = parent_map.get(cid, [])
                cluster_rows.append((
                    cid, max_iter, len(members),
                    _maybe_float(stats.get("diameter")),
                    _maybe_float(stats.get("mean_dist")),
                    "auto_merge" if parents else "base",
                    json.dumps(parents),
                ))
                exemplars = set(merged_cr.exemplars.get(cid, []))
                for node_idx in members:
                    face_idx = (
                        core_indices[node_idx]
                        if core_indices and node_idx < len(core_indices)
                        else node_idx
                    )
                    if 0 <= face_idx < len(faces):
                        assign_rows.append((
                            faces[face_idx].face_id,
                            cid,
                            max_iter,
                            1 if node_idx in exemplars else 0,
                            None,
                        ))

        conn.executemany(
            "INSERT INTO clusters VALUES (?,?,?,?,?,?,?)",
            cluster_rows,
        )
        # face_id+iteration is the PK; tolerate duplicates that arise when a face
        # appears in both base and merged with the same (face_id, iteration).
        conn.executemany(
            "INSERT OR REPLACE INTO cluster_assignments VALUES (?,?,?,?,?)",
            assign_rows,
        )

    # ------------------------------------------------------------------
    # Merge decisions — full 28-field fidelity (FR-004)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_merges(conn: sqlite3.Connection, merge_log: List[Dict]) -> None:
        if not merge_log:
            return
        field_order = MergeDecisionRow.field_names()
        rows: List[Tuple] = []
        for entry in merge_log:
            rows.append(tuple(_to_sql(entry[name], name) for name in field_order))
        placeholders = ",".join(["?"] * len(field_order))
        conn.executemany(
            f"INSERT INTO merge_decisions VALUES ({placeholders})",
            rows,
        )

    # ------------------------------------------------------------------
    # spec-032: filter decisions
    # ------------------------------------------------------------------

    @staticmethod
    def _write_filter_decisions(conn: sqlite3.Connection, filters) -> None:
        """Persist FilterContext to the filter_decisions table.

        No-op when filters is None or empty — keeps the schema consistent
        for runs that don't use the new contract yet (P1 dual-write window).
        """
        if filters is None or len(filters) == 0:
            return
        rows: List[Tuple] = []
        for item, decision in filters.all_decisions():
            rows.append((
                item.item_id,
                item.item_type,
                item.parent_id,
                decision.filter_name,
                1 if decision.rejected else 0,
                decision.reason,
                json.dumps(decision.measured, default=str),
            ))
        conn.executemany(
            "INSERT INTO filter_decisions "
            "(item_id, item_type, parent_id, filter_name, rejected, reason, measured_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    # ------------------------------------------------------------------
    # spec-040 Phase 4 (schema v5): images + scene-side persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _write_images(
        conn: sqlite3.Connection,
        faces: List[FaceRecord],
        image_paths: Optional[List[str]] = None,
        image_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Persist one row per distinct image to the ``images`` table.

        Rows derived from two sources unioned:
          * every distinct ``face.image_path`` in ``faces`` (image had >=1 face)
          * every entry in ``image_paths`` (images discovered upstream, even if
            they produced zero faces)

        ``image_scores`` (keyed by image_path) joins iqa/ava/sharpness/scene
        scores onto the corresponding row. Width/height are populated from
        the first face on that image that carries ``image_width_px``.

        Pandera-validates the resulting DataFrame before INSERT (spec-033 P-H).
        Empty input writes nothing but still validates an empty DataFrame so
        the schema contract is exercised on every run.
        """
        image_scores = image_scores or {}
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        def _norm(p: str) -> str:
            """Normalize path separators so face_records (forward slashes) and
            caller-supplied paths (potentially backslash on Windows) align.
            """
            return str(p).replace("\\", "/")

        # Group faces by image_path → carries n_faces + width/height.
        by_path: Dict[str, Dict[str, object]] = {}
        for face in faces:
            p = _norm(face.image_path) if face.image_path else None
            if not p:
                continue
            entry = by_path.setdefault(p, {
                "n_faces": 0,
                "width_px": None,
                "height_px": None,
            })
            entry["n_faces"] = int(entry["n_faces"]) + 1
            if entry["width_px"] is None and face.image_width_px:
                entry["width_px"] = int(face.image_width_px)
            if entry["height_px"] is None and face.image_height_px:
                entry["height_px"] = int(face.image_height_px)

        # Union with upstream image_paths so zero-face images get a row.
        all_paths = sorted(set(by_path) | {_norm(p) for p in (image_paths or [])})

        rows: List[Tuple] = []
        for path in all_paths:
            entry = by_path.get(path, {"n_faces": 0, "width_px": None, "height_px": None})
            scores = image_scores.get(path, {}) or {}
            rows.append((
                path,
                Path(path).name,
                entry["width_px"],
                entry["height_px"],
                int(entry["n_faces"]),
                _maybe_float(scores.get("iqa")),
                _maybe_float(scores.get("ava")),
                _maybe_float(scores.get("sharpness")),
                _maybe_float(scores.get("composite")),
                scores.get("scene_cluster_id"),
                1,  # filter_passed default — refined when filter wiring lands
                now,
            ))

        # Pandera validation (spec-033 P-H). Empty DF still validates the contract.
        df = pd.DataFrame(rows, columns=[
            "image_path", "image_id", "width_px", "height_px", "n_faces",
            "iqa_score", "ava_score", "sharpness_score", "composite_score",
            "scene_cluster_id", "filter_passed", "created_at",
        ])
        IMAGES_SCHEMA.validate(df)

        if rows:
            conn.executemany(
                "INSERT INTO images "
                "(image_path, image_id, width_px, height_px, n_faces, "
                " iqa_score, ava_score, sharpness_score, composite_score, "
                " scene_cluster_id, filter_passed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    @staticmethod
    def _write_scene_clusters(
        conn: sqlite3.Connection,
        scene_clusters: Optional[List[Dict]] = None,
    ) -> None:
        """Persist scene clusters to the ``scene_clusters`` table.

        Empty until the scene-clustering producer side writes
        ``context.scene_clusters: List[SceneClusterRecord]`` (out of scope for
        spec-040 T2 — tracked as future scene-side parity work). Pandera-
        validates the empty DataFrame so the contract is exercised even when
        no rows land.

        Each input dict must carry: ``scene_cluster_id`` (int), ``iteration``
        (int), ``size`` (int), ``method`` (str). Optional: ``exemplar_image_path``
        (str), ``avg_intra_distance`` (float).
        """
        rows: List[Tuple] = []
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for sc in scene_clusters or []:
            rows.append((
                int(sc["scene_cluster_id"]),
                int(sc.get("iteration", 0)),
                int(sc["size"]),
                str(sc.get("method", "")),
                sc.get("exemplar_image_path"),
                _maybe_float(sc.get("avg_intra_distance")),
                sc.get("created_at", now),
            ))
        df = pd.DataFrame(rows, columns=[
            "scene_cluster_id", "iteration", "size", "method",
            "exemplar_image_path", "avg_intra_distance", "created_at",
        ])
        SCENE_CLUSTERS_SCHEMA.validate(df)
        if rows:
            conn.executemany(
                "INSERT INTO scene_clusters "
                "(scene_cluster_id, iteration, size, method, "
                " exemplar_image_path, avg_intra_distance, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    @staticmethod
    def _write_scene_cluster_assignments(
        conn: sqlite3.Connection,
        scene_cluster_assignments: Optional[List[Dict]] = None,
    ) -> None:
        """Persist (image_path, scene_cluster_id, iteration) tuples.

        Each input dict must carry ``image_path``, ``scene_cluster_id``,
        ``iteration``. Optional: ``distance_to_centroid``. Pandera-validates
        before INSERT.
        """
        rows: List[Tuple] = []
        for sca in scene_cluster_assignments or []:
            rows.append((
                str(sca["image_path"]),
                int(sca["scene_cluster_id"]),
                int(sca.get("iteration", 0)),
                _maybe_float(sca.get("distance_to_centroid")),
            ))
        df = pd.DataFrame(rows, columns=[
            "image_path", "scene_cluster_id", "iteration", "distance_to_centroid",
        ])
        SCENE_CLUSTER_ASSIGNMENTS_SCHEMA.validate(df)
        if rows:
            conn.executemany(
                "INSERT INTO scene_cluster_assignments "
                "(image_path, scene_cluster_id, iteration, distance_to_centroid) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    # ------------------------------------------------------------------
    # Run metadata (absorbs merge_metadata.json + export_summary.json fields)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_run_metadata(
        conn: sqlite3.Connection,
        *,
        faces: List[FaceRecord],
        base_cr: ClusterResult,
        merged_cr: ClusterResult,
        merge_log: List[Dict],
        merge_metadata: Optional[Dict],
        config,
        source_album: str,
        producer: str,
        run_id: str,
        started_at: str,
        finished_at: str,
        parent_run_id: Optional[str],
    ) -> None:
        n_merges = sum(1 for e in merge_log if e["actually_merged"])
        n_iterations = max((int(e["iteration"]) for e in merge_log), default=0)

        try:
            config_json = json.dumps(
                {k: v for k, v in vars(config).items() if not k.startswith("_")},
                default=str,
            )
        except TypeError:
            config_json = "{}"

        thresholds_json = None
        iter_summary_json = None
        if merge_metadata:
            # Split merge_metadata into the two columns it absorbs.  Thresholds
            # block lives under `cluster_thresholds` / `global_threshold` keys
            # historically; iteration summary lives elsewhere in metadata.
            thresholds = {
                k: merge_metadata[k]
                for k in (
                    "cluster_thresholds", "global_threshold",
                    "merge_exemplar_threshold", "merge_candidate_threshold",
                )
                if k in merge_metadata
            }
            if thresholds:
                thresholds_json = json.dumps(thresholds, default=str)

            iter_summary = {
                k: merge_metadata[k]
                for k in ("n_iterations", "n_candidates_proposed")
                if k in merge_metadata
            }
            if iter_summary:
                iter_summary_json = json.dumps(iter_summary, default=str)

        n_images = len({f.image_path or f.image_id for f in faces})
        n_core = sum(1 for f in faces if f.is_core)

        conn.execute(
            "INSERT INTO run_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                source_album,
                producer,
                parent_run_id,
                config_json,
                thresholds_json,
                iter_summary_json,
                n_images,
                len(faces),
                n_core,
                base_cr.n_clusters,
                merged_cr.n_clusters,
                n_merges,
                n_iterations,
                started_at,
                finished_at,
                SCHEMA_VERSION,
            ),
        )

    # ------------------------------------------------------------------
    # Embeddings — leave the relational store, live in npy (FR-006)
    # ------------------------------------------------------------------

    def _write_embeddings_npy(self, faces: List[FaceRecord]) -> None:
        EMB_DIM = 512
        matrix = np.zeros((len(faces), EMB_DIM), dtype=np.float32)
        face_ids = np.array([f.face_id for f in faces], dtype=np.int32)
        for i, face in enumerate(faces):
            emb = face.embedding_normalized if face.embedding_normalized is not None else face.embedding
            if emb is not None:
                matrix[i] = np.asarray(emb, dtype=np.float32)
        np.save(self.output_dir / "embeddings.npy", matrix)
        np.save(self.output_dir / "embedding_face_ids.npy", face_ids)

    # ------------------------------------------------------------------
    # Pipeline run header (FR-007)
    # ------------------------------------------------------------------

    def _write_pipeline_run_json(
        self,
        *,
        run_id: str,
        source_album: str,
        producer: str,
        parent_run_id: Optional[str],
        started_at: str,
        finished_at: str,
    ) -> None:
        payload = {
            "run_id": run_id,
            "source_album": source_album,
            "producer": producer,
            "parent_run_id": parent_run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "db_path": "face_clustering.db",
        }
        path = self.output_dir / "pipeline_run.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Crops — copy from a sibling source dir during Phase 1 dual-write,
    # otherwise generate from face.aligned_face arrays.
    # ------------------------------------------------------------------

    def _write_crops(
        self,
        faces: List[FaceRecord],
        crop_source_dir: Optional[Path],
    ) -> Dict[int, str]:
        crops_dir = self.output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[int, str] = {}

        if crop_source_dir is not None:
            src = Path(crop_source_dir)
            for face in faces:
                fname = f"face_{face.face_id:04d}_aligned.jpg"
                src_path = src / fname
                if not src_path.exists():
                    continue
                dst_path = crops_dir / fname
                if dst_path.resolve() != src_path.resolve():
                    shutil.copyfile(src_path, dst_path)
                manifest[face.face_id] = f"crops/{fname}"
            return manifest

        for face in faces:
            if face.aligned_face is None:
                continue
            fname = f"face_{face.face_id:04d}_aligned.jpg"
            dst_path = crops_dir / fname
            Image.fromarray(face.aligned_face).save(dst_path, "JPEG", quality=95)
            manifest[face.face_id] = f"crops/{fname}"
        return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_sql(value, field_name: str):
    """Convert a Python value into a SQLite-storable form.

    SQLite has no boolean or infinity type; bools become ints, and `inf` is stored
    via Python's REAL handling (SQLite preserves it as a float).  None passes through.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, float):
        # Preserve inf / nan as-is; SQLite REAL columns round-trip them via Python.
        return value
    if isinstance(value, (int, str)):
        return value
    # numpy scalars
    if hasattr(value, "item"):
        return value.item()
    raise RunExporterError(
        f"unsupported type {type(value).__name__} for field {field_name!r}"
    )


def _build_parent_map(merge_log: List[Dict]) -> Dict[int, List[int]]:
    """Union-find over actually_merged=True rows → {final_id: [original parent ids]}."""
    if not merge_log:
        return {}
    parent: Dict[int, int] = {}

    def _find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    originals: Dict[int, List[int]] = {}
    for entry in merge_log:
        if not entry["actually_merged"]:
            continue
        a, b = int(entry["cluster_a"]), int(entry["cluster_b"])
        originals.setdefault(a, [a])
        originals.setdefault(b, [b])
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra
            originals[ra] = originals.get(ra, [ra]) + originals.get(rb, [rb])

    out: Dict[int, List[int]] = {}
    for cid in originals:
        root = _find(cid)
        members = originals.get(root, [])
        if len(members) > 1:
            out[root] = sorted(set(members) - {root})
    return out
