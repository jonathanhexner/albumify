"""spec-045 / spec-059 — ClusterAnalysisRepository (query-shape, ORM reads).

Reads the per-run ``face_clustering.db`` for the Cluster Analysis tab.
Composes :class:`sim_bench.run_db.store.RunStore` for schema validation +
heavyweight artifacts; issues its own ORM queries for typed ClusterRow /
Assignment returns. spec-059: every public method opens a fresh per-run
Session; raw SQL has been removed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from sqlalchemy import func, select

from face_cluster.repositories._base_repository import BaseRepository
from face_cluster.repositories._errors import ValidationError
from face_cluster.types import ClusterResult, FaceRecord, MergeDecisionRow
from face_cluster.views._base import Assignment, ClusterRow
from face_cluster.views.cluster_analysis import ForceMergeResult
from sim_bench.pipeline.clustering_labels import NOISE_LABEL, is_noise
from sim_bench.run_db._session import make_run_db_sessionmaker
from sim_bench.run_db.models import Cluster, ClusterAssignment
from sim_bench.run_db.store import RunMetadata, RunStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed configuration (spec §5.3 / §5.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClusterAnalysisRepoConfig:
    """Construction-time configuration for the Repository."""
    run_dir: Path
    read_only: bool = False
    log_queries: bool = False


@dataclass(frozen=True, slots=True)
class ClusterAnalysisCriteria:
    """Filter for ``find_assignments`` and related read methods."""
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
    builds a :class:`RunStore` (which runs the strict schema/artifact checks)
    plus a per-run-DB sessionmaker (spec-059).

    All reads return typed dataclasses — never raw dicts or sqlite3.Row.
    """

    def __init__(self, config: ClusterAnalysisRepoConfig) -> None:
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
        # RunStore runs schema-version + EXPECTED_ARTIFACTS checks up-front.
        self._run_store = RunStore(config.run_dir)
        # Per-run-DB sessionmaker (engine held internally).
        self._sessionmaker = make_run_db_sessionmaker(config.run_dir)

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def get_cluster_rows(self, iteration: str = "final") -> List[ClusterRow]:
        """Return one :class:`ClusterRow` per real cluster, sorted by cluster_id.

        Excludes the noise cluster — the noise bucket is not a cluster from
        the UI's perspective. Nearest-cluster fields are placeholders (the
        Service fills them in via async compute).
        """
        it = self._resolve_iteration(iteration)
        with self._sessionmaker() as session:
            cluster_rows = session.execute(
                select(Cluster.cluster_id, Cluster.size, Cluster.diameter, Cluster.avg_intra_dist)
                .where(Cluster.iteration == it)
                .order_by(Cluster.cluster_id)
            ).all()
            exemplar_counts = dict(session.execute(
                select(ClusterAssignment.cluster_id, func.count())
                .where(ClusterAssignment.iteration == it)
                .where(ClusterAssignment.is_exemplar == 1)
                .group_by(ClusterAssignment.cluster_id)
            ).all())
        out: List[ClusterRow] = []
        for cid, size, diameter, avg_intra_dist in cluster_rows:
            cid = int(cid)
            if is_noise(cid):
                continue
            out.append(ClusterRow(
                cluster_id=cid,
                size=int(size),
                diameter=float(diameter or 0.0),
                avg_intra_dist=float(avg_intra_dist or 0.0),
                n_exemplars=int(exemplar_counts.get(cid, 0)),
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
        stmt = select(
            ClusterAssignment.face_id,
            ClusterAssignment.cluster_id,
            ClusterAssignment.is_exemplar,
        ).where(ClusterAssignment.iteration == it)
        if criteria.cluster_id is not None:
            stmt = stmt.where(ClusterAssignment.cluster_id == int(criteria.cluster_id))
        if criteria.face_ids:
            stmt = stmt.where(ClusterAssignment.face_id.in_([int(f) for f in criteria.face_ids]))
        if criteria.exemplars_only:
            stmt = stmt.where(ClusterAssignment.is_exemplar == 1)
        if not criteria.include_noise:
            stmt = stmt.where(ClusterAssignment.cluster_id != NOISE_LABEL)
        stmt = stmt.order_by(ClusterAssignment.cluster_id, ClusterAssignment.face_id)
        with self._sessionmaker() as session:
            rows = session.execute(stmt).all()
        return [
            Assignment(
                face_id=int(face_id),
                cluster_id=int(cluster_id),
                is_exemplar=bool(is_exemplar),
                iteration=criteria.iteration,
            )
            for face_id, cluster_id, is_exemplar in rows
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

        Resolves ``"final"`` locally against the ``clusters`` table and
        passes the integer iteration to RunStore. See the rationale on the
        original sqlite version: RunStore's own "final" resolver uses
        ``MAX(iteration) FROM merge_decisions`` which is wrong when the
        merger ran but didn't merge anything.
        """
        it = self._resolve_iteration(iteration)
        return self._run_store.clusters(it)

    # ------------------------------------------------------------------
    # Mutations
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

        Raises:
            ValidationError: if ``read_only=True`` or either cluster id is unknown.
        """
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

    def _resolve_iteration(self, iteration: str) -> int:
        """Map the ``"base"`` / ``"final"`` / int aliases to the integer
        iteration stored in the DB. Mirrors RunStore's resolution semantics
        so cluster_rows and faces line up.
        """
        if isinstance(iteration, int):
            return iteration
        if iteration == "base":
            return 0
        if iteration == "final":
            with self._sessionmaker() as session:
                value = session.execute(select(func.max(Cluster.iteration))).scalar()
            return int(value) if value is not None else 0
        raise ValueError(f"unknown iteration alias: {iteration!r} (expected 'base', 'final', or int)")

    def _log(self, msg: str) -> None:
        if self._config.log_queries:
            logger.info("ClusterAnalysisRepository: %s", msg)


__all__ = [
    "ClusterAnalysisRepoConfig",
    "ClusterAnalysisCriteria",
    "ClusterAnalysisRepository",
]
