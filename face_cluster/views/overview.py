"""spec-066 — OverviewService: run-level dashboard aggregation.

Backend layer for the v2 Overview tab. Streamlit-free, deterministic,
unit-testable. Aggregates the global ``action_log`` for one producer
(``fc_app_v2``) into a typed :class:`DashboardMetrics`.

Design decisions (see specs/066-.../ARCHITECTURE_PLAN.md):

* **No wall-clock here.** ``last_run_at`` is returned raw; the tab turns
  it into "3h ago". Keeps ``compute_dashboard`` pure → trivially testable.
* **action_log only.** ``RunRow`` already carries n_faces / n_clusters /
  album / status, so we never open the 50 per-run DBs (Decision D2).
* **Profile is read defensively** from ``RunRow.payload['profile']``. No
  run records it yet (the writer change ships in this same spec); existing
  runs fall back to ``"(none)"``. The per-profile breakdown therefore
  lights up automatically once new runs carry the field.
"""
from __future__ import annotations

import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from face_cluster.repositories import (
    RunHistoryCriteria,
    RunHistoryRepository,
)
from face_cluster.run_history import RunRow

logger = logging.getLogger(__name__)

PRODUCER = "fc_app_v2"
NO_PROFILE = "(none)"


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AlbumStat:
    """One bar in the runs-per-album chart."""
    album: str
    n_runs: int


@dataclass(frozen=True, slots=True)
class StatusStat:
    """One bar in the runs-per-status chart (complete / failed / ...)."""
    status: str
    n_runs: int


@dataclass(frozen=True, slots=True)
class ProfileStat:
    """One bar in the runs-per-profile chart. ``(none)`` until runs record it."""
    profile: str
    n_runs: int


@dataclass(frozen=True, slots=True)
class RunPoint:
    """One point on the n_clusters time-series (oldest -> newest)."""
    run_id: int
    started_at: Optional[str]
    n_clusters: Optional[int]
    album: str


@dataclass(frozen=True, slots=True)
class DashboardMetrics:
    """Aggregate dashboard for all ``fc_app_v2`` runs in the window.

    Scalars are ``None`` when there is nothing to average (empty history).
    ``last_run_at`` is a raw ISO string — the tab computes the human "age".
    """
    total_runs: int
    total_faces_ever: int
    avg_n_clusters: Optional[float]
    median_n_clusters: Optional[float]
    last_run_at: Optional[str]
    gate_pass_rate: Optional[float]
    per_album: List[AlbumStat] = field(default_factory=list)
    per_status: List[StatusStat] = field(default_factory=list)
    per_profile: List[ProfileStat] = field(default_factory=list)
    timeseries: List[RunPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class OverviewService:
    """Typed read API for the Overview tab.

    Owns a :class:`RunHistoryRepository` via constructor injection (tests
    inject one wired to an in-memory DB; production gets the default).
    ``runs_dir`` is accepted for a future per-run deep-stats pass but is
    unused today — the headline metrics all come from ``action_log``.
    """

    def __init__(
        self,
        repo: Optional[RunHistoryRepository] = None,
        runs_dir: Optional[Path] = None,
    ) -> None:
        self._repo = repo or RunHistoryRepository()
        self._runs_dir = runs_dir

    def compute_dashboard(self, *, limit: int = 50) -> DashboardMetrics:
        """Aggregate the most recent ``limit`` ``fc_app_v2`` runs.

        Pure given DB state: no clock, no filesystem, no session_state.

        Args:
            limit: max number of runs to pull from ``action_log``.

        Returns:
            A :class:`DashboardMetrics`. Empty history yields zeros and
            ``None`` scalars (never raises).
        """
        rows: List[RunRow] = self._repo.find(
            RunHistoryCriteria(producer=PRODUCER, limit=limit)
        )
        total_runs = len(rows)
        total_faces_ever = sum(int(r.n_faces or 0) for r in rows)

        cluster_counts = [int(r.n_clusters) for r in rows if r.n_clusters is not None]
        avg_n_clusters = statistics.fmean(cluster_counts) if cluster_counts else None
        median_n_clusters = statistics.median(cluster_counts) if cluster_counts else None

        started = [r.started_at for r in rows if r.started_at]
        last_run_at = max(started) if started else None

        complete = sum(1 for r in rows if (r.status or "").lower() == "complete")
        gate_pass_rate = (complete / total_runs) if total_runs else None

        per_album = _counter_to_stats(
            Counter(r.display_album for r in rows),
            lambda name, n: AlbumStat(album=name, n_runs=n),
        )
        per_status = _counter_to_stats(
            Counter((r.status or "unknown") for r in rows),
            lambda name, n: StatusStat(status=name, n_runs=n),
        )
        per_profile = _counter_to_stats(
            Counter(_profile_of(r) for r in rows),
            lambda name, n: ProfileStat(profile=name, n_runs=n),
        )

        timeseries = [
            RunPoint(
                run_id=r.id,
                started_at=r.started_at,
                n_clusters=r.n_clusters,
                album=r.display_album,
            )
            for r in sorted(rows, key=lambda r: (r.started_at or ""))
        ]

        return DashboardMetrics(
            total_runs=total_runs,
            total_faces_ever=total_faces_ever,
            avg_n_clusters=avg_n_clusters,
            median_n_clusters=median_n_clusters,
            last_run_at=last_run_at,
            gate_pass_rate=gate_pass_rate,
            per_album=per_album,
            per_status=per_status,
            per_profile=per_profile,
            timeseries=timeseries,
        )

    def has_real_profiles(self, metrics: DashboardMetrics) -> bool:
        """True iff at least one run recorded a profile other than ``(none)``.

        The tab uses this to decide whether the per-profile chart is worth
        rendering yet (it stays hidden until the writer change starts
        populating profiles on new runs)."""
        return any(p.profile != NO_PROFILE for p in metrics.per_profile)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _profile_of(row: RunRow) -> str:
    """Profile name recorded for a run, or ``(none)``.

    Reads ``payload['profile']`` (written by run_v2_pipeline as of spec-066).
    Tolerant of the 40 pre-existing runs that predate the writer change."""
    prof = row.payload.get("profile")
    return str(prof) if prof else NO_PROFILE


def _counter_to_stats(counter: Counter, make):
    """Turn a Counter into a list of stat dataclasses, desc by count then name."""
    return [
        make(name, n)
        for name, n in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    ]


__all__ = [
    "AlbumStat",
    "StatusStat",
    "ProfileStat",
    "RunPoint",
    "DashboardMetrics",
    "OverviewService",
]
