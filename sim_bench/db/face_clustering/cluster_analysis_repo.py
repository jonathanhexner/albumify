"""spec-045 Phase 1 — ClusterAnalysisRepository (query-shape, B0b).

Reads the per-run ``face_clustering.db`` for the Cluster Analysis tab. Composes
:class:`face_cluster.run_store.RunStore` for schema validation + heavyweight
artifacts (``faces()``, ``embeddings()``, ``metadata()``, ``merge_log()``,
``clusters(iteration)``) and issues its own SQL only where RunStore doesn't
already expose the shape we need (currently: per-row reads from the ``clusters``
and ``cluster_assignments`` tables for typed :class:`ClusterRow` /
:class:`Assignment` returns).

Locked decisions (from spec-045 §"Locked decisions"):

* **Query-shape Repository** — does not own the per-run schema. RunStore's
  ``PRAGMA user_version`` + ``EXPECTED_ARTIFACTS`` checks remain the schema
  authority. We never CREATE/ALTER/DROP per-run tables.
* **Config-based constructor** — takes a :class:`ClusterAnalysisRepoConfig`
  (validated up front); no SQLAlchemy ``Session`` in the signature (per-run DBs
  aren't Alembic-managed). Inherits :class:`BaseRepository` to keep the
  error-translation idiom available, but passes ``None`` for the session.
* **No bare -1 for noise** — every cluster-id comparison routes through
  :data:`NOISE_LABEL` / :func:`is_noise` from
  ``sim_bench.pipeline.clustering_labels`` (commit ``9824d84``).

Phase 1 ships read methods + typed Config/Criteria. Phase 2 adds the
``save_manual_merge_snapshot`` mutation.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from face_cluster.repositories._base_repository import BaseRepository
from face_cluster.repositories._errors import ValidationError
from sim_bench.run_db.store import RunMetadata, RunStore
from face_cluster.types import ClusterResult, FaceRecord, MergeDecisionRow
from face_cluster.views._base import Assignment, ClusterRow
from face_cluster.views.cluster_analysis import ForceMergeResult
from sim_bench.pipeline.clustering_labels import NOISE_LABEL, is_noise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed configuration (spec §5.3 / §5.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClusterAnalysisRepoConfig:
    """Construction-time configuration for the Repository.

    Fields:
        run_dir: Path to the run directory containing face_clustering.db.
                 Required; ``__init__`` validates existence + DB presence.
        read_only: When True, every mutation method raises ValidationError.
                   Phase 2's force-merge writer honours this flag.
        log_queries: When True, the Repository emits one INFO log per query.
    """
    run_dir: Path
    read_only: bool = False
    log_queries: bool = False


@dataclass(frozen=True, slots=True)
class ClusterAnalysisCriteria:
    """Filter for ``find_assignments`` and related read methods.

    Defaults are conservative: ``include_noise=False`` so the noise cluster
    (``NOISE_LABEL``) is excluded from every read unless the caller asks for
    it explicitly. ``face_ids`` is a tuple (not list) for hashability — lets
    the Service memoize on a Criteria instance.
    """
    cluster_id: Optional[int] = None
    face_ids: Optional[Tuple[int, ...]] = None
    iteration: str = "final"
    exemplars_only: bool = False
    include_noise: bool = False


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class ClusterAnalysisRepository(BaseRepository):
    """Typed reads over a single run's face_clustering.db.

    Construction: pass a validated :class:`ClusterAnalysisRepoConfig`. Internally
    builds a :class:`RunStore` (which runs the strict schema/artifact checks).

    All reads return typed dataclasses — never raw dicts or sqlite3.Row.
    """

    def __init__(self, config: ClusterAnalysisRepoConfig) -> None:
        # query-shape Repository: no SQLAlchemy Session (per-run DB isn't
        # Alembic-managed). Pass None to satisfy BaseRepository's contract;
        # the static error-translation helpers remain available.
        super().__init__(session=None)  # type: ignore[arg-type]
        if not isinstance(config, ClusterAnalysisRepoConfig):
            raise ValueError(
                f"ClusterAnalysisRepository expects ClusterAnalysisRepoConfig, "
                f"got {type(config).__name__}"
            )
        if not config.run_dir.is_dir():
            raise ValueError(f"run_dir does not exist or is not a directory: {config.run_dir}")
        db_path = config.run_dir / "face_clustering.db"
        if not db_path.is_file():
            raise ValueError(f"face_clustering.db not found in run_dir: {config.run_dir}")
        self._config = config
        self._db_path = db_path
        # RunStore() runs schema-version + EXPECTED_ARTIFACTS checks. If any
        # check fails it raises RunStoreError — we let that propagate so the
        # caller sees the underlying schema/artifact mismatch.
        self._run_store = RunStore(config.run_dir)

    # ------------------------------------------------------------------
    # Read methods (spec §6.1 / tasks T005)
    # ------------------------------------------------------------------

    def get_cluster_rows(self, iteration: str = "final") -> List[ClusterRow]:
        """Return one :class:`ClusterRow` per real cluster, sorted by cluster_id.

        Excludes the noise cluster (``NOISE_LABEL``) — the noise bucket is not
        a cluster from the UI's perspective. Nearest-cluster fields are
        placeholders (the Service fills them in via async compute).
        """
        it = self._resolve_iteration(iteration)
        with self._connect() as conn:
            cluster_rows = conn.execute(
                "SELECT cluster_id, size, diameter, avg_intra_dist "
                "FROM clusters WHERE iteration = ? ORDER BY cluster_id",
                (it,),
            ).fetchall()
            exemplar_counts = dict(conn.execute(
                "SELECT cluster_id, COUNT(*) FROM cluster_assignments "
                "WHERE iteration = ? AND is_exemplar = 1 GROUP BY cluster_id",
                (it,),
            ).fetchall())
        out: List[ClusterRow] = []
        for r in cluster_rows:
            cid = int(r["cluster_id"])
            if is_noise(cid):
                continue
            out.append(ClusterRow(
                cluster_id=cid,
                size=int(r["size"]),
                diameter=float(r["diameter"] or 0.0),
                avg_intra_dist=float(r["avg_intra_dist"] or 0.0),
                n_exemplars=int(exemplar_counts.get(cid, 0)),
                # Compute fields filled by Service.compute_detail_async (Phase 4):
                nearest_cluster_id=NOISE_LABEL,
                nearest_cluster_dist=0.0,
                merge_candidate=False,
            ))
        self._log(f"get_cluster_rows(iteration={it}) -> {len(out)} clusters")
        return out

    def get_cluster_ids(self, iteration: str = "final") -> List[int]:
        """Cluster ids in display order (same order as :meth:`get_cluster_rows`)."""
        return [r.cluster_id for r in self.get_cluster_rows(iteration)]

    def find_assignments(self, criteria: ClusterAnalysisCriteria) -> List[Assignment]:
        """Return :class:`Assignment` rows matching the criteria.

        Honors ``include_noise`` — noise rows are dropped by default.
        """
        it = self._resolve_iteration(criteria.iteration)
        sql = ["SELECT face_id, cluster_id, is_exemplar FROM cluster_assignments WHERE iteration = ?"]
        params: List[object] = [it]
        if criteria.cluster_id is not None:
            sql.append("AND cluster_id = ?")
            params.append(int(criteria.cluster_id))
        if criteria.face_ids:
            placeholders = ",".join("?" for _ in criteria.face_ids)
            sql.append(f"AND face_id IN ({placeholders})")
            params.extend(int(f) for f in criteria.face_ids)
        if criteria.exemplars_only:
            sql.append("AND is_exemplar = 1")
        if not criteria.include_noise:
            sql.append(f"AND cluster_id != {NOISE_LABEL}")
        sql.append("ORDER BY cluster_id, face_id")
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), params).fetchall()
        return [
            Assignment(
                face_id=int(r["face_id"]),
                cluster_id=int(r["cluster_id"]),
                is_exemplar=bool(r["is_exemplar"]),
                iteration=criteria.iteration,
            )
            for r in rows
        ]

    def get_face_records(self, face_ids: List[int]) -> List[FaceRecord]:
        """Return the :class:`FaceRecord` for each known face_id.

        Unknown ids are silently dropped (the caller may pass ids from a
        criterion that's broader than the run). Returned order matches the
        input order for known ids.
        """
        if not face_ids:
            return []
        all_records = {r.face_id: r for r in self._run_store.faces()}
        return [all_records[fid] for fid in face_ids if fid in all_records]

    def get_run_metadata(self) -> RunMetadata:
        """Return :class:`RunMetadata` for this run (delegates to RunStore)."""
        return self._run_store.metadata()

    def get_merge_log(self) -> List[MergeDecisionRow]:
        """Return the full merge log (delegates to RunStore)."""
        return self._run_store.merge_log()

    def get_cluster_result(self, iteration: str = "final") -> ClusterResult:
        """Return :class:`ClusterResult` at the given iteration.

        spec-045 follow-up (2026-05-29): we resolve ``"final"`` locally
        against the ``clusters`` table and pass the integer iteration to
        RunStore, bypassing RunStore's own "final" resolver. RunStore
        computes "final" as ``MAX(iteration) FROM merge_decisions``,
        which is wrong when the merger ran an iteration but didn't
        actually merge anything (clusters stay at the previous
        iteration; merge_decisions has rows at N; clusters has none at
        N → RunStoreError). Our ``_resolve_iteration`` queries the
        ``clusters`` table directly and returns the right number.

        Filed as a follow-up sighting against RunStore — the fix
        belongs there long-term, but this Repository must not crash on
        the common no-merge case in the meantime.
        """
        it = self._resolve_iteration(iteration)
        return self._run_store.clusters(it)

    # ------------------------------------------------------------------
    # Mutations (spec §"Locked decisions" #3 / tasks T013)
    # ------------------------------------------------------------------

    def save_manual_merge_snapshot(
        self,
        *,
        cluster_a: int,
        cluster_b: int,
        merge_round: int,
        config,
    ) -> ForceMergeResult:
        """Write a fresh ``merge_snap_{round}/`` snapshot dir under the parent.

        Delegates the on-disk write to the legacy
        :func:`face_cluster.manual_merge_snapshot.save_manual_merge_snapshot`
        (same format the legacy "Force Merge" path produces, so a follow-up
        remerge run can load it). Parent run dir is never mutated — the
        snapshot is written as a sibling.

        Args:
            cluster_a / cluster_b: the two clusters to merge.
            merge_round:           1-based iteration counter. Caller (Service)
                                   decides the next round by reading the
                                   existing snapshot dirs.
            config:                ``face_cluster.config.PipelineConfig`` of
                                   the parent run; saved into the snapshot for
                                   audit/remerge.

        Returns: :class:`ForceMergeResult` with the snapshot path and merge metadata.

        Raises:
            ValidationError:  if ``read_only=True`` or either cluster id is unknown.
        """
        # late import: keeps the read-only path free of the legacy writer
        from face_cluster.manual_merge_snapshot import (
            save_manual_merge_snapshot as _legacy_save,
        )

        if self._config.read_only:
            raise ValidationError(
                "ClusterAnalysisRepository is read-only; save_manual_merge_snapshot refused",
                user_message="This run is open in read-only mode.",
            )
        cluster_result = self.get_cluster_result("final")
        if cluster_a not in cluster_result.clusters:
            raise ValidationError(
                f"cluster_a={cluster_a} not found in run {self._config.run_dir}",
                user_message=f"Cluster {cluster_a} doesn't exist in this run.",
            )
        if cluster_b not in cluster_result.clusters:
            raise ValidationError(
                f"cluster_b={cluster_b} not found in run {self._config.run_dir}",
                user_message=f"Cluster {cluster_b} doesn't exist in this run.",
            )

        faces = self._run_store.faces()
        meta = self._run_store.metadata()
        snapshot_dir = (
            self._config.run_dir.parent
            / f"{self._config.run_dir.name}_merge_snap_{merge_round}"
        )

        _legacy_save(
            faces=faces,
            merged_cluster_result=cluster_result,
            approved_pairs=[(cluster_a, cluster_b)],
            rejected_pairs=[],
            config=config,
            output_dir=snapshot_dir,
            parent_output_dir=self._config.run_dir,
            parent_run_id=meta.run_id,
            merge_round=merge_round,
        )
        return ForceMergeResult(
            snapshot_dir=snapshot_dir,
            merge_round=merge_round,
            parent_run_dir=self._config.run_dir,
            new_cluster_id=min(cluster_a, cluster_b),
            n_merged=1,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a per-call sqlite3 connection. Caller is responsible for closing
        (used via ``with`` blocks above). Read-only Repository methods don't
        need pooling — opening is fast and avoids cross-thread surprises in
        the Streamlit polling loop."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _resolve_iteration(self, iteration: str) -> int:
        """Map the ``"base"`` / ``"final"`` / int aliases used by the public API
        to the integer iteration stored in the DB. Mirrors RunStore's own
        resolution semantics so cluster_rows and faces line up."""
        if isinstance(iteration, int):
            return iteration
        if iteration == "base":
            return 0
        if iteration == "final":
            with self._connect() as conn:
                row = conn.execute("SELECT MAX(iteration) FROM clusters").fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        raise ValueError(f"unknown iteration alias: {iteration!r} (expected 'base', 'final', or int)")

    def _log(self, msg: str) -> None:
        if self._config.log_queries:
            logger.info("ClusterAnalysisRepository: %s", msg)


__all__ = [
    "ClusterAnalysisRepoConfig",
    "ClusterAnalysisCriteria",
    "ClusterAnalysisRepository",
]
