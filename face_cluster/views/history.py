"""spec-042 H1 — HistoryService: typed access to the run history.

Backend layer for the v2 History tab. Streamlit-free.

Wraps the existing query helpers in ``face_cluster.run_history`` and
``face_cluster.run_history_db`` (which are sound — the legacy backend
layer was always typed; only the legacy *tab* code was hand-rolled).
Adds new typed dataclasses (``HistoryQuery``, ``RunDetail``, ``RunSummary``,
``LoadedRun``, ``ActionRow``) and a declarative column spec for the table
renderer.

All public methods take typed inputs and return typed outputs — no
``dict`` returns. Read methods (``list_*`` / ``get_*``) are pure given DB
state; mutations (``update_*``) name their side effects in the docstring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from face_cluster.config_diff import ConfigDelta, compute as _config_diff_compute
from face_cluster.repositories import (
    NotFoundError as _RepoNotFoundError,
    RunHistoryCriteria,
    RunHistoryRepoConfig,
    RunHistoryRepository,
)
from face_cluster.run_history import RunRow
from face_cluster.views._specs import ColumnSpec


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HistoryQuery:
    """User-supplied filters for the run history table.

    All fields optional. Empty/None means "no filter on this axis";
    combination is AND, not OR.
    """
    album: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    text: Optional[str] = None  # substring of album / run_name / comment, case-insensitive


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Parsed pipeline_run.json content. None-valued fields when absent
    (e.g., pre-spec-012 runs lack the funnel stats)."""
    n_faces: Optional[int] = None
    n_core: Optional[int] = None
    n_clusters_base: Optional[int] = None
    n_clusters_merged: Optional[int] = None
    n_noise: Optional[int] = None
    stage_durations: Dict[str, float] = field(default_factory=dict)
    merge_count: Optional[int] = None
    merge_candidate_threshold: Optional[float] = None


@dataclass(frozen=True, slots=True)
class RunDetail:
    """Full detail for a selected run.

    Joins:
    * the action_log row (``row``);
    * the parsed FCParams config dict from pipeline_run.json (``config``);
    * the parent run's row (``parent_row``, None if no parent);
    * the field-level config diff vs parent (``config_delta``, empty list when no parent);
    * the parsed run summary (``summary``, None if pipeline_run.json absent);
    * a flag indicating whether the canonical 3-artifact set exists on disk.
    """
    row: RunRow
    config: Dict[str, Any]
    parent_row: Optional[RunRow]
    config_delta: List[ConfigDelta]
    summary: Optional[RunSummary]
    has_required_artifacts: bool


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """Result of ``HistoryService.load_run(id)``.

    The tab layer takes this and writes its fields into
    ``st.session_state``. The service deliberately does NOT touch
    session_state itself — that keeps the service Streamlit-free
    and unit-testable.
    """
    run_id: int
    source_album: str
    output_dir: Path
    pipeline_result: Any  # face_cluster.pipeline.PipelineResult; kept Any to avoid a back-edge import


@dataclass(frozen=True, slots=True)
class ActionRow:
    """One row in the 'Recent Actions' sub-table.

    Covers non-pipeline action types: ``merge_apply``, ``profile_save``,
    ``ml_train``, ``model_load``. Pipeline runs go in the main table.
    """
    id: int
    started_at: Optional[str]
    action_type: str
    status: str
    duration_s: Optional[float]
    details: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Declarative table specs (consumed by the v2 components/run_table.py)
# ---------------------------------------------------------------------------

def _short_path(p: Any) -> str:
    """Last two path components, slash-joined; bare basename if no parent."""
    parts = Path(str(p)).parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return Path(str(p)).name or ""


def _iso_to_local(value: Any) -> str:
    """Strip ISO seconds + space-separate date / time for compact display."""
    s = str(value or "")
    return s[:16].replace("T", " ")


def _format_duration(seconds: Any) -> str:
    """e.g. 12.4 -> '12.4s'. None -> '-'."""
    if seconds is None or seconds == "":
        return "-"
    try:
        return f"{float(seconds):.1f}s"
    except (TypeError, ValueError):
        return "-"


RUN_COLUMNS: List[ColumnSpec] = [
    ColumnSpec(field="display_album", label="Album"),
    ColumnSpec(field="run_name", label="Run name", fallback_fields=("run_id",)),
    ColumnSpec(field="run_kind", label="Kind", fallback_fields=("action_type",)),
    ColumnSpec(field="output_dir", label="Output", formatter=_short_path),
    ColumnSpec(field="started_at", label="Created", formatter=_iso_to_local),
    ColumnSpec(field="n_faces", label="Faces"),
    ColumnSpec(field="n_core", label="Core"),
    ColumnSpec(field="n_clusters", label="Clusters"),
    ColumnSpec(field="status", label="Status"),
    ColumnSpec(field="comment", label="Comment"),
]


ACTION_COLUMNS: List[ColumnSpec] = [
    ColumnSpec(field="started_at", label="when", formatter=_iso_to_local),
    ColumnSpec(field="action_type", label="type"),
    ColumnSpec(field="status", label="status"),
    ColumnSpec(field="duration_s", label="duration", formatter=_format_duration),
    ColumnSpec(field="details", label="details"),
    ColumnSpec(field="error", label="error"),
]


# ---------------------------------------------------------------------------
# ActionTypeFormat — typed dispatch for the 'details' column
# ---------------------------------------------------------------------------

def _fmt_merge_apply(payload: dict, row: Dict[str, Any]) -> str:
    return (
        f"round={payload.get('round')}  "
        f"approved={payload.get('n_approved')}  "
        f"clusters_before={payload.get('clusters_before')} -> "
        f"{row.get('n_clusters')}"
    )


def _fmt_profile_save(payload: dict, row: Dict[str, Any]) -> str:
    return f"profile={payload.get('profile_name')}"


def _fmt_ml_train(payload: dict, row: Dict[str, Any]) -> str:
    acc = payload.get("accuracy", "")
    return f"model={payload.get('model_type')}  acc={acc}"


def _fmt_model_load(payload: dict, row: Dict[str, Any]) -> str:
    return f"model={payload.get('model_name')}"


def _fmt_default(payload: dict, row: Dict[str, Any]) -> str:
    return ""


_ACTION_FORMATTERS = {
    "merge_apply": _fmt_merge_apply,
    "profile_save": _fmt_profile_save,
    "ml_train": _fmt_ml_train,
    "model_load": _fmt_model_load,
}


class ActionTypeFormat:
    """Typed dispatch for one-line action summaries.

    Replaces the legacy dict-of-lambdas at
    ``app/face_clustering/tabs/history_tab.py:349``. Each formatter is a
    named module-level function so it shows up cleanly in stack traces
    and can be tested independently.
    """

    @staticmethod
    def format(action_type: str, payload: dict, row: Dict[str, Any]) -> str:
        """Return the one-line 'details' summary for an action_log row.

        Args:
            action_type: the row's ``action_type``.
            payload: the parsed ``payload_json`` dict (may be empty).
            row: the raw action_log row dict — provides ``n_clusters``
                and any other action-row-level fields formatters may
                want to read.

        Returns:
            A short human-readable string, or "" if no formatter is
            registered for ``action_type``.
        """
        formatter = _ACTION_FORMATTERS.get(action_type, _fmt_default)
        return formatter(payload, row)


# ---------------------------------------------------------------------------
# HistoryService
# ---------------------------------------------------------------------------

_DEFAULT_ACTION_TYPES: tuple[str, ...] = (
    "merge_apply",
    "profile_save",
    "ml_train",
    "model_load",
)

_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "faces.csv",
    "clusters.csv",
    "embeddings.npy",
)


class HistoryService:
    """Query and mutate the global ``action_log`` DB.

    Composes a :class:`RunHistoryRepository` via constructor injection.
    Streamlit-free; safe to call from CLI, pytest, or a FastAPI handler.

    Read methods (``list_*`` / ``get_*``) are pure given DB state.
    Mutation methods (``update_*`` / ``load_*``) name their side
    effects in the docstring.

    spec-043 migration: previously held a ``self._db_path`` and threaded
    it through every legacy free-function call. Now holds one Repository
    instance and delegates; ``db_path`` is a Repository concern.
    """

    def __init__(self, repo: Optional[RunHistoryRepository] = None):
        """Initialize with an optional Repository (tests inject; production
        gets the default).

        Args:
            repo: a configured :class:`RunHistoryRepository`. When None,
                a default Repository is created pointing at the global
                ``~/.sim_bench/sim_bench.db``.
        """
        self._repo = repo or RunHistoryRepository()

    # ------------------------------------------------------------------ read

    def list_runs(self, query: HistoryQuery) -> List[RunRow]:
        """Return runs matching the filter, newest-first.

        Args:
            query: filter combination. Empty fields = no filter on that axis.

        Returns:
            list of RunRow. Empty when no runs match.

        Side effects: none.
        """
        criteria = RunHistoryCriteria(
            album=query.album,
            date_from=query.date_from,
            date_to=query.date_to,
            text=query.text,
        )
        return self._repo.find(criteria)

    def list_albums(self) -> List[str]:
        """Distinct album names ever recorded, sorted.

        Returns: list of source_album strings (excluding NULL/empty).

        Side effects: none.
        """
        return self._repo.distinct_albums()

    def get_run_detail(self, run_id: int) -> RunDetail:
        """Full detail for one run.

        Args:
            run_id: action_log primary key.

        Returns:
            ``RunDetail`` joining the row, its parent (if any), the
            parsed pipeline_run.json config + summary, and a flag
            indicating whether required output artifacts exist.

        Raises:
            ValueError: if ``run_id`` is not present in the DB.

        Side effects: none. Reads pipeline_run.json from the run's
        output_dir when present.
        """
        row = self._repo.get_by_id(run_id)
        if row is None:
            raise ValueError(f"No run with id={run_id}")

        # Config: prefer the typed config_json on the row; fall back to
        # parsing pipeline_run.json from disk for legacy runs that
        # predate the config_json column.
        config = row.config or {}
        summary: Optional[RunSummary] = None
        has_required = False
        if row.output_dir:
            out_dir = Path(row.output_dir)
            prun_path = out_dir / "pipeline_run.json"
            if prun_path.exists():
                try:
                    prun = json.loads(prun_path.read_text(encoding="utf-8"))
                    if not config:
                        config = prun.get("config") or prun.get("summary", {}).get("config") or {}
                    summary = _summary_from_pipeline_run(prun)
                except Exception:
                    summary = None
            has_required = out_dir.exists() and all(
                (out_dir / artifact).exists() for artifact in _REQUIRED_ARTIFACTS
            )

        parent_row: Optional[RunRow] = None
        config_delta: List[ConfigDelta] = []
        if row.parent_run_id is not None:
            parent_row = self._repo.get_by_id(row.parent_run_id)
            if parent_row is not None:
                config_delta = _config_diff_compute(parent_row.config or {}, config)

        return RunDetail(
            row=row,
            config=config,
            parent_row=parent_row,
            config_delta=config_delta,
            summary=summary,
            has_required_artifacts=has_required,
        )

    def list_other_actions(
        self,
        action_types: Sequence[str] = _DEFAULT_ACTION_TYPES,
        limit: int = 100,
    ) -> List[ActionRow]:
        """List recent non-pipeline actions, newest-first.

        Args:
            action_types: filter by these ``action_type`` values; defaults
                to the four canonical non-pipeline types.
            limit: max number of rows to return.

        Returns:
            list of ``ActionRow``. The ``details`` field is formatted by
            ``ActionTypeFormat`` from each row's ``payload_json``.
        """
        rows = self._repo.find(RunHistoryCriteria(
            action_types=list(action_types),
            limit=limit,
        ))
        out: List[ActionRow] = []
        for row in rows:
            payload = row.payload
            # ActionTypeFormat expects a dict-like row for n_clusters access;
            # build a minimal dict view from the RunRow so the formatter
            # signature (kept stable) still works.
            row_view = {"n_clusters": row.n_clusters}
            details = ActionTypeFormat.format(row.action_type, payload, row_view)
            out.append(ActionRow(
                id=row.id,
                started_at=row.started_at,
                action_type=row.action_type,
                status=row.status,
                duration_s=row.duration_s,
                details=details,
                error=row.error,
            ))
        return out

    def get_action_payload(self, action_id: int) -> Dict[str, Any]:
        """Return the parsed ``payload_json`` dict for an action_log row.

        Args:
            action_id: action_log primary key.

        Returns:
            Parsed payload dict, or empty dict if the row is missing,
            payload_json is NULL, or JSON parsing fails.
        """
        row = self._repo.get_by_id(action_id)
        if row is None:
            return {}
        return row.payload

    # ------------------------------------------------------------- mutation

    def update_comment(self, run_id: int, comment: str) -> None:
        """Persist a free-text comment for ``run_id``.

        Args:
            run_id: action_log primary key.
            comment: free-text, max 2048 chars (enforced by the Repository).

        Raises:
            ValueError: when comment exceeds 2048 characters or the run
                does not exist. (Translated from the Repository's
                ``ValidationError`` / ``NotFoundError`` to preserve the
                pre-spec-043 contract.)

        Side effects: UPDATEs ``action_log.comment`` for ``run_id``.
        Idempotent — same comment twice = same final state.
        """
        from face_cluster.repositories import ValidationError as _RepoValErr
        try:
            self._repo.update_comment(run_id, comment)
        except (_RepoValErr, _RepoNotFoundError) as e:
            # Preserve the pre-migration contract: callers catch
            # ``ValueError``. Once spec-042 B2 (ServiceError hierarchy)
            # lands, this translation can go away — Service methods will
            # propagate Repository errors directly.
            raise ValueError(str(e)) from e

    def load_run(self, run_id: int) -> LoadedRun:
        """Load a completed run's pipeline result into a typed container.

        Args:
            run_id: action_log primary key.

        Returns:
            ``LoadedRun`` ready for the tab layer to write into
            ``st.session_state``.

        Raises:
            ValueError: if the run is missing, incomplete, or required
                output artifacts are absent.

        Side effects: none. Reads files from the run's ``output_dir``.
        Does NOT touch ``st.session_state`` — the tab layer owns that.
        """
        from face_cluster.loader import load_pipeline_result

        detail = self.get_run_detail(run_id)
        if detail.row.status != "complete":
            raise ValueError(
                f"Run {run_id} is not complete (status={detail.row.status!r})"
            )
        if not detail.row.output_dir:
            raise ValueError(f"Run {run_id} has no output_dir recorded")
        if not detail.has_required_artifacts:
            missing = [
                artifact for artifact in _REQUIRED_ARTIFACTS
                if not (Path(detail.row.output_dir) / artifact).exists()
            ]
            raise ValueError(
                f"Run {run_id} is missing required artifacts: {missing}"
            )

        output_dir = Path(detail.row.output_dir)
        pipeline_result = load_pipeline_result(output_dir)
        return LoadedRun(
            run_id=run_id,
            source_album=detail.row.display_album,
            output_dir=output_dir,
            pipeline_result=pipeline_result,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _summary_from_pipeline_run(prun: Dict[str, Any]) -> RunSummary:
    """Parse the ``pipeline_run.json`` dict into a typed ``RunSummary``.

    Tolerates missing fields and old shapes — every value is optional.
    """
    summary_block = prun.get("summary") or {}
    stages_block = prun.get("stages") or {}
    stage_durations: Dict[str, float] = {}
    for stage_name, info in stages_block.items():
        if isinstance(info, dict):
            elapsed = info.get("elapsed_s") or info.get("duration_s")
            if isinstance(elapsed, (int, float)):
                stage_durations[stage_name] = float(elapsed)

    # Merge stats may be embedded under "merge_metadata" or summary.
    merge_meta = prun.get("merge_metadata") or {}
    merge_log = prun.get("merge_log")
    merge_count = None
    if isinstance(merge_log, list):
        merge_count = sum(1 for entry in merge_log if isinstance(entry, dict) and entry.get("actually_merged"))

    merge_candidate_threshold = (
        merge_meta.get("merge_candidate_threshold")
        if isinstance(merge_meta, dict) else None
    )

    return RunSummary(
        n_faces=summary_block.get("n_faces"),
        n_core=summary_block.get("n_core"),
        n_clusters_base=summary_block.get("n_clusters"),
        n_clusters_merged=summary_block.get("n_clusters_merged"),
        n_noise=summary_block.get("n_noise"),
        stage_durations=stage_durations,
        merge_count=merge_count,
        merge_candidate_threshold=merge_candidate_threshold,
    )


__all__ = [
    "HistoryQuery",
    "RunRow",
    "RunSummary",
    "RunDetail",
    "LoadedRun",
    "ActionRow",
    "ConfigDelta",
    "RUN_COLUMNS",
    "ACTION_COLUMNS",
    "ActionTypeFormat",
    "HistoryService",
]
