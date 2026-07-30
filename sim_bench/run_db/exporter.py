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
    FACES_SCHEMA,
    FACE_SCORES_SCHEMA,
    IMAGES_SCHEMA,
    SCENE_CLUSTERS_SCHEMA,
    SCENE_CLUSTER_ASSIGNMENTS_SCHEMA,
)
from sim_bench.run_db._schema import (
    EXPECTED_ARTIFACTS,
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


from sim_bench.run_db._errors import RunExporterError  # re-exported


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
        filter_verdicts=None,  # List[QualityVerdict] from the v2 quality gate (SIGHTING-093 G1)
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
            # SIGHTING-093 G1: the v2 quality gate emits per-face verdicts (not a
            # FilterContext); persist them to the same table so the Quality /
            # Excluded-Faces tabs have data on fresh v2 runs.
            if filter_verdicts:
                from sim_bench.run_db.writers.filter_decisions_writer import (
                    write_filter_decisions_from_verdicts,
                )
                write_filter_decisions_from_verdicts(conn, filter_verdicts, faces)
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
        from sim_bench.run_db.writers.faces_writer import write_faces
        write_faces(conn, faces, crop_manifest, image_scores)

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
        from sim_bench.run_db.writers.clusters_writer import write_clusters
        write_clusters(conn, base_cr, merged_cr, merge_log, core_indices, faces)

    # ------------------------------------------------------------------
    # Merge decisions — full 28-field fidelity (FR-004)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_merges(conn: sqlite3.Connection, merge_log: List[Dict]) -> None:
        from sim_bench.run_db.writers.merges_writer import write_merges
        write_merges(conn, merge_log)

    # ------------------------------------------------------------------
    # spec-032: filter decisions
    # ------------------------------------------------------------------

    @staticmethod
    def _write_filter_decisions(conn: sqlite3.Connection, filters) -> None:
        from sim_bench.run_db.writers.filter_decisions_writer import write_filter_decisions
        write_filter_decisions(conn, filters)

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
        from sim_bench.run_db.writers.images_writer import write_images
        write_images(conn, faces, image_paths, image_scores)

    @staticmethod
    def _write_scene_clusters(
        conn: sqlite3.Connection,
        scene_clusters: Optional[List[Dict]] = None,
    ) -> None:
        from sim_bench.run_db.writers.scenes_writer import write_scene_clusters
        write_scene_clusters(conn, scene_clusters)

    @staticmethod
    def _write_scene_cluster_assignments(
        conn: sqlite3.Connection,
        scene_cluster_assignments: Optional[List[Dict]] = None,
    ) -> None:
        from sim_bench.run_db.writers.scenes_writer import write_scene_cluster_assignments
        write_scene_cluster_assignments(conn, scene_cluster_assignments)

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
        from sim_bench.run_db.writers.run_metadata_writer import write_run_metadata
        write_run_metadata(
            conn,
            faces=faces,
            base_cr=base_cr,
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

    # ------------------------------------------------------------------
    # Embeddings — leave the relational store, live in npy (FR-006)
    # ------------------------------------------------------------------

    def _write_embeddings_npy(self, faces: List[FaceRecord]) -> None:
        from sim_bench.run_db.artifact_writers.embeddings_writer import write_embeddings
        write_embeddings(self.output_dir, faces)

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
        from sim_bench.run_db.artifact_writers.pipeline_run_writer import write_pipeline_run
        write_pipeline_run(
            self.output_dir,
            run_id=run_id,
            source_album=source_album,
            producer=producer,
            parent_run_id=parent_run_id,
            started_at=started_at,
            finished_at=finished_at,
        )

    # ------------------------------------------------------------------
    # Crops — copy from a sibling source dir during Phase 1 dual-write,
    # otherwise generate from face.aligned_face arrays.
    # ------------------------------------------------------------------

    def _write_crops(
        self,
        faces: List[FaceRecord],
        crop_source_dir: Optional[Path],
    ) -> Dict[int, str]:
        from sim_bench.run_db.artifact_writers.crops_writer import write_crops
        return write_crops(self.output_dir, faces, crop_source_dir)


