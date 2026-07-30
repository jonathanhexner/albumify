"""spec-063 — Service for the v2 Recluster tab.

Re-clusters a prior run's face_records with a fresh ``FCParams`` set. Skips
the expensive producer chain (detect / align / embed) and writes a new
snapshot run dir under ``~/.sim_bench/runs/<uuid4-hex>/`` whose
``pipeline_run.json`` records ``parent_run_id`` pointing back at the input.

Architecture rules (spec §"Architecture rules"):

* Streamlit-free — no ``streamlit`` imports here. The tab is the only
  layer that knows about Streamlit.
* Synchronous compute — SIGHTING-079 lesson; recluster on a typical
  album is ~5-15 s and Streamlit's request/response model doesn't poll
  background threads. The tab wraps the call in ``st.spinner``.
* Typed I/O — every public method returns a typed dataclass or list of
  typed entries (RunPickerEntry / ReclusterResult). No bare dicts.

Lineage:

* Input  prior run dir  ──>  ``RunStore.faces()``  ──>  ``context.face_records``
* Output snapshot dir   <──  ``RunExporter`` with ``parent_run_id`` set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from face_cluster._paths import sim_bench_data_dir
from face_cluster.fc_app_runner import FCAppRunner
from face_cluster.fc_params import FCParams
from face_cluster.repositories import (
    RunHistoryCriteria,
    RunHistoryRepoConfig,
    RunHistoryRepository,
)
from face_cluster.run_layout import allocate_run_dir
from sim_bench.pipeline.context import PipelineContext
from sim_bench.run_db.exporter import RunExporter
from sim_bench.run_db.store import RunStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReclusterResult:
    """Outcome of one recluster operation.

    The snapshot dir is a complete v5 run that the History tab + the
    Cluster Analysis tab open exactly like any other run.
    """
    snapshot_dir: Path
    n_clusters: int
    n_faces: int
    parent_run_id: str


# ---------------------------------------------------------------------------
# Picker-entry projection — shape matches app.face_clustering_v2.components.run_picker
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunPickerEntry:
    """One row in the recent-runs picker.

    Mirrors :class:`app.face_clustering_v2.components.run_picker.RunPickerEntry`
    structurally (the tab can pass either type to a render component) without
    importing the Streamlit-bound module — the Service must stay
    Streamlit-free.
    """
    run_id: str
    output_dir: Path
    album: str
    started_at: str
    n_faces: int
    n_clusters: int
    status: str
    is_orphan: bool = False


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

# Allow-list of producers RunExporter will accept for the snapshot.
# "fc_app" matches the legacy producer string the History tab already
# understands; the parent_run_id field in pipeline_run.json carries the
# lineage that distinguishes a recluster from a fresh run.
_SNAPSHOT_PRODUCER = "fc_app"


class ReclusterService:
    """Typed read + compute API for the v2 Recluster tab.

    Construction takes a :class:`RunHistoryRepository` because the picker
    lists the most recent v2 runs from the global action_log — same source
    of truth as the History tab and the spec-050 run picker.
    """

    def __init__(self, history_repo: RunHistoryRepository) -> None:
        if history_repo is None:
            raise ValueError("ReclusterService requires a non-None RunHistoryRepository.")
        self._history_repo = history_repo

    # ---- reads ---------------------------------------------------------

    def list_recent_runs(self, limit: int = 20) -> List[RunPickerEntry]:
        """Return up to ``limit`` recent ``fc_app_v2`` runs, newest first."""
        rows = self._history_repo.find(
            RunHistoryCriteria(producer="fc_app_v2", limit=limit)
        )
        entries: List[RunPickerEntry] = []
        for r in rows:
            if not r.output_dir or not r.run_id:
                continue
            out_dir = Path(r.output_dir)
            is_orphan = not (out_dir / "face_clustering.db").exists()
            entries.append(RunPickerEntry(
                run_id=r.run_id,
                output_dir=out_dir,
                album=r.source_album or "(unknown)",
                started_at=r.started_at or "",
                n_faces=int(r.n_faces or 0),
                n_clusters=int(r.n_clusters or 0),
                status=r.status or "",
                is_orphan=is_orphan,
            ))
        return entries

    # ---- compute -------------------------------------------------------

    def recluster(
        self,
        prior_run_dir: Path,
        params: FCParams,
        *,
        runs_base_dir: Optional[Path] = None,
    ) -> ReclusterResult:
        """Re-cluster ``prior_run_dir`` with ``params`` into a fresh snapshot.

        Sync compute — the heavy step is the 8-stage clustering chain over
        the prior run's face_records; producer steps (detect / align /
        embed) are skipped. Total wall-clock on the Budapest reference run
        is ~5-15 s; the tab wraps this call in ``st.spinner``.

        Args:
            prior_run_dir: A complete v5 run dir (RunStore.faces() must work).
            params: The new clustering parameters.
            runs_base_dir: Where to allocate the snapshot dir. Defaults to
                ``~/.sim_bench/runs``.

        Returns:
            ReclusterResult — snapshot_dir, n_clusters, n_faces, parent_run_id.

        Raises:
            ValueError: If ``prior_run_dir`` does not exist or is unreadable.
        """
        prior_run_dir = Path(prior_run_dir)
        if not prior_run_dir.exists():
            raise ValueError(
                f"prior_run_dir does not exist: {prior_run_dir}. "
                "Pick a run from the picker; the History tab won't show "
                "orphaned rows as loadable."
            )
        if not prior_run_dir.is_dir():
            raise ValueError(f"prior_run_dir is not a directory: {prior_run_dir}")

        base = Path(runs_base_dir) if runs_base_dir is not None else (
            sim_bench_data_dir() / "runs"
        )
        snapshot_dir, snapshot_run_id = allocate_run_dir(base, prior_run_dir.name)

        # Load the prior run's faces and re-run the unified clustering chain.
        # FCAppRunner.recluster() sets context.parent_run_id for us; we still
        # need to thread it through to RunExporter explicitly because the
        # exporter takes parent_run_id as a kwarg, not from context.
        store = RunStore(prior_run_dir)
        face_records = store.faces()
        context = PipelineContext()
        context.face_records = face_records
        context.parent_run_id = prior_run_dir.name

        started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        fc_result = FCAppRunner().run(context, step_configs=params.to_step_configs())
        finished_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        if not fc_result.success:
            raise RuntimeError(
                f"Recluster failed: {fc_result.error_message or 'unknown error'}"
            )

        base_cr = getattr(context, "cluster_result", None) or _empty_cluster_result(
            len(face_records)
        )
        merged_cr = getattr(context, "merged_cluster_result", None)

        # Preserve the parent's source_album for display continuity.
        prior_meta = store.metadata()
        RunExporter(snapshot_dir).export(
            faces=face_records,
            base_cluster_result=base_cr,
            merged_cluster_result=merged_cr,
            core_indices=getattr(context, "core_indices", []) or [],
            merge_log=getattr(context, "merge_log", None),
            merge_metadata=getattr(context, "merge_metadata", None),
            config=None,
            source_album=prior_meta.source_album,
            producer=_SNAPSHOT_PRODUCER,
            run_id=snapshot_run_id,
            started_at=started_at,
            finished_at=finished_at,
            parent_run_id=prior_run_dir.name,
            crop_source_dir=prior_run_dir / "crops",
            filters=getattr(context, "filters", None),
        )

        return ReclusterResult(
            snapshot_dir=snapshot_dir,
            n_clusters=fc_result.n_clusters,
            n_faces=len(face_records),
            parent_run_id=prior_run_dir.name,
        )


def _empty_cluster_result(n_faces: int):
    """Fallback when the chain produced no clustering — every face is noise."""
    import numpy as np
    from face_cluster.types import ClusterResult
    return ClusterResult(
        labels=np.full(n_faces, -1, dtype=int),
        clusters={},
        cluster_stats={},
        exemplars={},
        n_clusters=0,
        n_noise=n_faces,
    )


__all__ = ["ReclusterService", "ReclusterResult", "RunPickerEntry"]
