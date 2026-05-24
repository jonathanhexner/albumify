# History Tab — PRD and Implementation Plan

Sub-spec of [spec-042](spec.md) · Pilot tab for the v2 rebuild
Status: **Draft**
Created: 2026-05-25

This document is the detailed plan for the History tab. It's the **pilot** for the architectural pattern that every other tab in spec-042 will follow, so the level of detail here is intentionally higher than the per-tab summaries in the parent spec.

---

## 1. Problem & Objectives

### 1.1 What the History tab is for

A user comes back to the v2 FC App and wants to:

- See the runs they've done in the past (filtered by album, date, free-text)
- Click into one to inspect what params it used, what it produced, and what's in its log
- Load a past run's data into the analysis tabs without re-running the pipeline
- See ancillary actions taken on those runs (manual merges, profile saves, ML model training)

### 1.2 Why it's the pilot

Three reasons it goes first:

1. **It's the worst code in the legacy app** — 365 LOC mixing 7 responsibilities, 16 hand-rolled `cfg.get('field', '?')` literals duplicating FCParams field names, three section-divider comment blocks substituting for function decomposition. If the v2 rebuild can transform *this* into the spec-042 layered shape, every other tab is easier.
2. **It exercises every architectural piece** that spec-042 introduces: backend service class, typed dataclasses, `ColumnSpec` declarative table, `widget_factory.render_field(readonly=True)`, three test layers, full UI orchestration. Once it works, the others are template applications.
3. **It has zero algorithmic risk.** No clustering math, no pipeline invocation, no equivalence test needed. Pure data layer + UI. If it doesn't work cleanly, the architecture is wrong — there's no other variable.

### 1.3 Objectives

- **Behavior-preserving** with the legacy History tab — every feature the user uses today must work the same way.
- **Half the LOC** of the legacy implementation, achieved through component decomposition.
- **Zero `cfg.get('field')` literals**; the run-config display reads from `FCParams` via `widget_factory.render_field(readonly=True)`.
- **Service unit tests with synthetic data** ≥10 cases covering every method's contract.
- **Streamlit-free backend** — `face_cluster/views/history.py` testable in pytest without any UI framework.

---

## 2. User Stories & Acceptance Criteria

### US-1 — List & filter runs (P1)

**As** a user
**I want to** see my past runs filtered by album, date, and free text
**So that** I can find a specific run quickly

**Acceptance criteria**:
- Given the action_log DB has runs across 3 albums, when I select album "Budapest", then only Budapest runs appear.
- Given runs span a 30-day window, when I set date_from = T-7d and date_to = today, then only runs in that window appear.
- Given I type "merge experiment" in the search field, then runs whose `album`, `run_name`, or `comment` contains that substring (case-insensitive) appear.
- Given multiple filters are set, then results match ALL filters (AND, not OR).
- Given no filters, then all runs appear, newest first.
- Given an empty DB, then the table shows "No runs match the current filters" — no exception.

### US-2 — Select & inspect a run (P1)

**As** a user
**I want to** click a run and see its config, summary, and files
**So that** I can understand what that run was and what it produced

**Acceptance criteria**:
- Given I click a row, then a detail panel appears below the table.
- The panel shows: album/run name header, optional parent link, config (read-only widgets from `FCParams`), quality funnel, cluster counts, merge candidates, stage timeline, log viewer, payload viewer, data files list.
- The config view shows **all** FCParams fields the run used, **read-only**, grouped by `UI_SPEC.ui_group`. No field-name string literals in the rendering code.
- If the run has a parent, then the config delta vs parent is shown.

### US-3 — Edit comment (P2)

**As** a user
**I want to** add or edit a free-text comment on any run
**So that** I can annotate runs with context (e.g., "best merge so far")

**Acceptance criteria**:
- Given a selected run, when I type in the comment field and the value changes, then the comment persists to the DB on the next interaction (no explicit Save button).
- Bulk-edit via `st.data_editor` in an expander: I can edit multiple comments and click "Save comments" to persist.
- Comment length capped at 2048 characters; longer values raise a user-visible error.

### US-4 — Load a run into analysis tabs (P1)

**As** a user
**I want to** click "Load into analysis tabs" on a completed run
**So that** the Cluster Analysis / Face Analysis / etc. tabs show that run's data

**Acceptance criteria**:
- Given a complete run, the Load button is enabled.
- Given an incomplete run (status ≠ "complete" or missing artifacts), the Load button is disabled and shows which files are missing.
- Given a run is already loaded, the button is replaced by a success message.
- After Load, `st.session_state["pipeline_result"]` holds the typed LoadedRun and `st.session_state["current_source_album"]` is set.
- Clicking Load triggers a rerun so the analysis tabs pick up the new state.

### US-5 — View ancillary actions (P3)

**As** a user
**I want to** see non-pipeline actions (manual merges, profile saves, ML training)
**So that** I have a complete audit trail

**Acceptance criteria**:
- A "Recent Actions" sub-table appears at the bottom of the tab listing the last 100 actions whose `action_type` is in `{"merge_apply", "profile_save", "ml_train", "model_load"}`.
- Each row shows: timestamp, type, status, duration, summary details, error.
- Clicking a row shows the full payload as JSON in an expander.

---

## 3. Visual Specification

```
┌─────────────────────────────────────────────────────────────────────┐
│ # History                                                            │
├─────────────────────────────────────────────────────────────────────┤
│ FILTER BAR (3 columns, widths 2:2:3)                                 │
│ [Album ▼ (all)]  [Date range ▦]    [🔍 Search album/name/comment]    │
├─────────────────────────────────────────────────────────────────────┤
│ Pipeline Runs (N)                                                    │
│ ┌──────────┬──────────┬──────┬────────┬──────────────┬──────────┐   │
│ │ Album    │ Run name │ Kind │ Output │ Created      │ Clusters │   │
│ ├──────────┼──────────┼──────┼────────┼──────────────┼──────────┤   │
│ │ Budapest │ k5-merge │ full │ runs/… │ 2026-05-24…  │ 33       │ ← selected
│ │ Budapest │ k3-tight │ recl │ runs/… │ 2026-05-23…  │ 47       │   │
│ │ Paris    │ exp-1    │ full │ runs/… │ 2026-05-20…  │ 12       │   │
│ └──────────┴──────────┴──────┴────────┴──────────────┴──────────┘   │
│ ▾ Edit comments inline (data_editor)                                 │
├─────────────────────────────────────────────────────────────────────┤
│ ── SELECTED RUN DETAIL (only when a row is selected) ────────────    │
│ **Album:** Budapest  |  **Run:** k5-merge                            │
│ Parent: k3-baseline   ▾ Config Delta (vs parent)                     │
│ Comment: [Best merge so far                                    ]     │
│                                                                      │
│ ▾ Run Summary (expanded)                                             │
│   ▾ Run Configuration  ← uses widget_factory.render_field(readonly)  │
│   Quality funnel: detected=340  core=282  noise=58                   │
│   Cluster counts: faces=340  base=33  merged=33                      │
│   Stage durations: detect_persons=54s  embed=62s  cluster=2s  …      │
│                                                                      │
│ ┌──────────────┬────────────────┐                                    │
│ │ ▸ View Log   │ ▸ Full Payload │                                    │
│ └──────────────┴────────────────┘                                    │
│ ▾ Data Files (file picker, downloads)                                │
│                                                                      │
│ [ Load into analysis tabs ]  ← primary button                        │
├─────────────────────────────────────────────────────────────────────┤
│ Recent Actions                                                       │
│ ┌─────────────┬──────────────┬──────────┬──────────┐                 │
│ │ when        │ type         │ status   │ details  │                 │
│ ├─────────────┼──────────────┼──────────┼──────────┤                 │
│ │ 2026-05-25  │ profile_save │ complete │ k5-merge │                 │
│ │ 2026-05-24  │ ml_train     │ complete │ acc=0.84 │                 │
│ └─────────────┴──────────────┴──────────┴──────────┘                 │
│ Inspect action payload: [select ▾]  ▸ Full Payload                   │
└─────────────────────────────────────────────────────────────────────┘
```

Order matches the legacy app for muscle-memory continuity. **No new visual elements** — the rebuild is purely architectural.

---

## 4. Data Contracts (Backend Dataclasses)

All defined in `face_cluster/views/history.py`. All frozen + slotted dataclasses. No `dict` returns from any service method.

```python
@dataclass(frozen=True, slots=True)
class HistoryQuery:
    """User-supplied filters for the run history table.

    All fields optional. Empty/None means 'no filter on this axis'.
    Combination is AND, not OR.
    """
    album: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    text: Optional[str] = None  # matches album / run_name / comment substring, case-insensitive


@dataclass(frozen=True, slots=True)
class RunRow:
    """One row in the runs table. Lean — only the columns we display.

    Populated by HistoryService.list_runs() from the action_log row +
    JOIN to a per-run metadata source. Pure data, no behavior.
    """
    id: int
    display_album: str
    run_id: Optional[str]
    run_name: Optional[str]
    run_kind: Optional[str]                # "full" | "recluster" | "remerge" | ...
    action_type: str                        # "fc_app_v2_run" | etc.
    output_dir: Optional[str]
    started_at: Optional[str]               # ISO-8601 string
    finished_at: Optional[str]
    status: str                             # "complete" | "running" | "failed" | ...
    n_faces: Optional[int]
    n_core: Optional[int]
    n_clusters: Optional[int]
    parent_run_id: Optional[int]
    comment: Optional[str]
    log_file: Optional[str]


@dataclass(frozen=True, slots=True)
class ConfigDelta:
    """One field difference between two FCParams configs."""
    field: str
    parent_value: Any
    child_value: Any


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Parsed pipeline_run.json content. None if file is absent (pre-spec-012 runs)."""
    n_faces: Optional[int]
    n_core: Optional[int]
    n_clusters_base: Optional[int]
    n_clusters_merged: Optional[int]
    n_noise: Optional[int]
    stage_durations: dict[str, float]      # stage_name -> seconds
    merge_count: Optional[int]             # number of merges actually applied
    merge_candidate_threshold: Optional[float]


@dataclass(frozen=True, slots=True)
class RunDetail:
    """Full detail for a selected run. Joins row + parent + parsed JSONs."""
    row: RunRow
    config: dict[str, Any]                  # raw FCParams dict from pipeline_run.json
    parent_row: Optional[RunRow]
    config_delta: list[ConfigDelta]         # [] if no parent
    summary: Optional[RunSummary]
    has_required_artifacts: bool            # faces.csv, clusters.csv, embeddings.npy all exist


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """Result of HistoryService.load_run(id). Tab layer writes this to session_state."""
    run_id: int
    source_album: str
    output_dir: Path
    pipeline_result: Any                    # PipelineResult from face_cluster.loader.load_pipeline_result


@dataclass(frozen=True, slots=True)
class ActionRow:
    """Row in the 'Recent Actions' sub-table (non-pipeline actions)."""
    id: int
    started_at: Optional[str]
    action_type: str                        # "merge_apply" | "profile_save" | "ml_train" | "model_load"
    status: str
    duration_s: Optional[float]
    details: str                            # action-type-specific one-line summary
    error: Optional[str]
```

---

## 5. Service Contract

```python
# face_cluster/views/history.py

class HistoryService:
    """Query and mutate the global action_log DB.

    Owns: a connection path to ~/.sim_bench/sim_bench.db (or a test override).
    Streamlit-free: no st.* calls anywhere in this module.

    Read methods (list_*, get_*) are pure given DB state. Mutation methods
    (update_*, load_*) name their side effects in the docstring.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize with an optional DB path override (used in tests)."""

    def list_runs(self, query: HistoryQuery) -> list[RunRow]:
        """Return runs matching the filter, newest-first.

        Args:
            query: filter combination. Empty fields = no filter on that axis.

        Returns:
            list of RunRow. Empty when no runs match.

        Side effects: none.
        """

    def list_albums(self) -> list[str]:
        """Distinct album names ever recorded, sorted.

        Returns: list[str]. Empty when DB is empty.

        Side effects: none.
        """

    def get_run_detail(self, run_id: int) -> RunDetail:
        """Full detail for one run.

        Args:
            run_id: action_log primary key.

        Returns:
            RunDetail joining the row, its parent (if any), the parsed
            pipeline_run.json config + summary, and an
            `has_required_artifacts` flag computed against the run's
            output_dir.

        Raises:
            ValueError if run_id is not found in the DB.

        Side effects: none.
        """

    def update_comment(self, run_id: int, comment: str) -> None:
        """Persist a new comment for the given run.

        Args:
            run_id: action_log primary key.
            comment: free-text, max 2048 chars.

        Raises:
            ValueError if comment exceeds 2048 chars.

        Side effects: UPDATEs action_log.comment for run_id. Idempotent
        (same comment twice = same final state).
        """

    def load_run(self, run_id: int) -> LoadedRun:
        """Load a completed run's pipeline result into a typed container.

        Args:
            run_id: action_log primary key.

        Returns:
            LoadedRun ready for the tab to write into st.session_state.

        Raises:
            ValueError if run_id missing, run incomplete, or required
            artifacts missing.

        Side effects: none. Does NOT touch st.session_state — the tab
        layer is responsible for that.
        """

    def list_other_actions(
        self,
        action_types: Sequence[str] = ("merge_apply", "profile_save", "ml_train", "model_load"),
        limit: int = 100,
    ) -> list[ActionRow]:
        """List recent non-pipeline actions, newest-first."""

    def get_action_payload(self, action_id: int) -> dict[str, Any]:
        """Return the raw payload_json dict for inspection."""
```

---

## 6. Declarative Specs

```python
# face_cluster/views/history.py

RUN_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(field="display_album", label="Album"),
    ColumnSpec(field="run_name", label="Run name", fallback_fields=("run_id", "id")),
    ColumnSpec(field="run_kind", label="Kind", fallback_fields=("action_type",)),
    ColumnSpec(field="output_dir", label="Output", formatter=_short_path),
    ColumnSpec(field="started_at", label="Created", formatter=_iso_to_local),
    ColumnSpec(field="n_faces", label="Faces"),
    ColumnSpec(field="n_core", label="Core"),
    ColumnSpec(field="n_clusters", label="Clusters"),
    ColumnSpec(field="status", label="Status"),
    ColumnSpec(field="comment", label="Comment"),
]


ACTION_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(field="started_at", label="when", formatter=_iso_truncate_seconds),
    ColumnSpec(field="action_type", label="type"),
    ColumnSpec(field="status", label="status"),
    ColumnSpec(field="duration_s", label="duration", formatter=_format_duration),
    ColumnSpec(field="details", label="details"),
    ColumnSpec(field="error", label="error"),
]
```

### Action-type formatter — typed dispatch, not a dict of lambdas

```python
# face_cluster/views/history.py

class ActionTypeFormat:
    """Typed dispatch for non-pipeline action one-line summaries.

    Replaces the legacy dict-of-lambdas at history_tab.py:349.
    """

    @staticmethod
    def format(action_type: str, payload: dict[str, Any], action_row: ActionRow) -> str:
        """Return the one-line 'details' summary for the actions table."""
        formatter = _FORMATTERS.get(action_type, _format_default)
        return formatter(payload, action_row)


def _format_merge_apply(payload, action_row) -> str:
    return (f"round={payload.get('round')}  "
            f"approved={payload.get('n_approved')}  "
            f"clusters_before={payload.get('clusters_before')} -> {action_row.n_clusters_after}")


def _format_profile_save(payload, action_row) -> str:
    return f"profile={payload.get('profile_name')}"


def _format_ml_train(payload, action_row) -> str:
    acc = payload.get("accuracy", "")
    return f"model={payload.get('model_type')}  acc={acc}"


def _format_model_load(payload, action_row) -> str:
    return f"model={payload.get('model_name')}"


def _format_default(payload, action_row) -> str:
    return ""


_FORMATTERS: dict[str, Callable[..., str]] = {
    "merge_apply": _format_merge_apply,
    "profile_save": _format_profile_save,
    "ml_train": _format_ml_train,
    "model_load": _format_model_load,
}
```

The dict still exists, but it dispatches to **named module-level functions**, not lambdas. Each function is independently testable and traceable in a stack trace.

---

## 7. Streamlit Components

Five components, each in its own file, each ≤ 60 LOC.

### 7.1 `app/face_clustering_v2/components/run_filter_bar.py`

```python
def render_filter_bar(service: HistoryService) -> HistoryQuery:
    """Three side-by-side filter widgets (album, date range, free-text).

    Reads from: HistoryService.list_albums() to populate the dropdown.
    Returns: the HistoryQuery the user has selected.
    Side effects: none.

    Column widths 2:2:3 keep the wider text search anchored on the right.
    """
```

### 7.2 `app/face_clustering_v2/components/run_table.py`

```python
def render_run_table(
    rows: list[RunRow],
    columns: list[ColumnSpec],
) -> Optional[int]:
    """Render a Streamlit dataframe with single-row selection.

    Args:
        rows: typed RunRow list.
        columns: declarative column spec.

    Returns:
        The selected row's id, or None if nothing selected.

    Side effects: writes session_state via st.dataframe's selection callback.
    """
```

### 7.3 `app/face_clustering_v2/components/run_detail.py`

```python
def render_run_detail(detail: RunDetail) -> None:
    """Render the full detail panel for one selected run.

    Layout:
        - Header (album, run name, parent link, config delta expander, comment input)
        - Run Summary expander (config view + quality funnel + cluster counts + stage timeline)
        - Two-column row: Log viewer expander | Payload expander
        - Data Files expander

    Reads from: detail (already populated by the service).
    Side effects:
        - Comment edits trigger HistoryService.update_comment() via the
          session_state-watching pattern.
    """


def _render_run_config_section(config: dict[str, Any]) -> None:
    """Render the run's FCParams config as read-only widgets, grouped by UI_SPEC group.

    Replaces 16 lines of `st.text(f"K: {cfg.get('K', '?')}")` in the legacy
    tab. Iterates over UI_SPEC fields and calls
    widget_factory.render_field(readonly=True, value_override=...) for each.
    Zero field-name string literals in this function.
    """
    from app.face_clustering_v2.widget_factory import render_field
    from app.face_clustering_v2.ui_spec import UI_SPEC, fields_by_group

    for group_name in ("cluster", "quality", "merge", "cap"):
        group_fields = fields_by_group().get(group_name, [])
        present_fields = [f for f in group_fields if f in config]
        if not present_fields:
            continue
        with st.expander(f"Stage · {group_name.title()}", expanded=False):
            for field_name in present_fields:
                render_field(field_name, readonly=True, value_override=config[field_name])


def _render_run_summary_section(summary: RunSummary) -> None:
    """Render the parsed pipeline_run.json summary (funnel, counts, durations)."""


def _render_log_viewer(log_file: Optional[str]) -> None:
    """Tail of the run's log file (last 200 lines) in an st.code block."""


def _render_payload_viewer(action_id: int, service: HistoryService) -> None:
    """JSON dump of the action_log payload for debug."""


def _render_data_files_panel(output_dir: Path) -> None:
    """File picker + downloads for the run's output dir."""
```

### 7.4 `app/face_clustering_v2/components/load_button.py`

```python
def render_load_button(detail: RunDetail, service: HistoryService) -> None:
    """Render the 'Load into analysis tabs' button with three states.

    States:
        - Already loaded: green success message, no button.
        - Incomplete run: disabled button + warning naming missing files.
        - Complete + not loaded: enabled primary button.

    On click:
        - Calls service.load_run(detail.row.id) → LoadedRun
        - Writes LoadedRun fields to st.session_state["pipeline_result"]
          and st.session_state["current_source_album"]
        - Invalidates run caches
        - Triggers st.rerun()
    """
```

### 7.5 `app/face_clustering_v2/components/actions_table.py`

```python
def render_actions_table(actions: list[ActionRow]) -> Optional[int]:
    """Render the 'Recent Actions' sub-table.

    Returns: the selected action_id (from selectbox below the table), or None.
    """
```

### 7.6 `app/face_clustering_v2/tabs/history_tab.py` (the orchestrator)

```python
def render_history_tab() -> None:
    """Render the History tab.

    Layout: filter bar → runs table → selected-run detail → recent actions sub-table.

    Reads: HistoryService (queries the action_log DB)
    Writes: st.session_state when the user clicks Load
    """
    st.header("History")
    service = HistoryService()

    query = render_filter_bar(service)
    runs = service.list_runs(query)

    st.subheader(f"Pipeline Runs ({len(runs)})")
    if not runs:
        st.info("No runs match the current filters.")
    else:
        selected_id = render_run_table(runs, RUN_COLUMNS)
        if selected_id is not None:
            detail = service.get_run_detail(selected_id)
            render_run_detail(detail)
            render_load_button(detail, service)

    st.divider()
    st.subheader("Recent Actions")
    actions = service.list_other_actions()
    if not actions:
        st.info("No other actions recorded yet.")
        return
    selected_action_id = render_actions_table(actions)
    if selected_action_id is not None:
        with st.expander("Full Payload (debug)", expanded=False):
            st.json(service.get_action_payload(selected_action_id))
```

**Target: ≤ 50 LOC** for the tab orchestrator. The legacy version is 365.

---

## 8. Test Plan

Three layers, in order of execution cost.

### 8.1 Service unit tests (synthetic data) — `tests/face_clustering/views/test_history_service_synthetic.py`

**Method: `list_runs`** (≥6 cases)

| # | Scenario | Setup | Assertion |
|---|---|---|---|
| 1 | Empty DB → empty list | Empty action_log | `list_runs(HistoryQuery()) == []` |
| 2 | No filters → all runs | 5 runs across 2 albums | Returns all 5, newest-first |
| 3 | Filter by album | 5 runs, 3 Budapest + 2 Paris | `query.album="Budapest"` → 3 rows |
| 4 | Filter by date range | runs spanning 30 days | `date_from=T-7d` → only last-7-day runs |
| 5 | Free-text filter — album substring | Budapest + Paris + Budapest_v2 | `text="Buda"` → 2 rows |
| 6 | Free-text filter — comment substring | one run with comment "best merge" | `text="merge"` → that 1 row |
| 7 | Multi-filter is AND not OR | Budapest run today + Paris run today + Budapest old | `album="Budapest" + date_from=today` → 1 row |
| 8 | Newest-first ordering | 5 runs with random `started_at` | Returned list sorted DESC by `started_at` |

**Method: `list_albums`** (≥2 cases)

| # | Scenario | Assertion |
|---|---|---|
| 9 | Distinct + sorted | Returns sorted unique album names |
| 10 | Empty DB → empty list | `list_albums() == []` |

**Method: `get_run_detail`** (≥3 cases)

| # | Scenario | Assertion |
|---|---|---|
| 11 | Returns full shape on existing run | All fields populated; `config_delta` is empty list when no parent |
| 12 | Includes parent details | Run with `parent_run_id` set → `parent_row` is not None; `config_delta` has the differing fields |
| 13 | Raises `ValueError` on missing id | `service.get_run_detail(999999)` raises |

**Method: `update_comment`** (≥3 cases)

| # | Scenario | Assertion |
|---|---|---|
| 14 | Persists | After `update_comment(id, "foo")`, `get_run_detail(id).row.comment == "foo"` |
| 15 | Idempotent | Calling twice with same value → no change |
| 16 | Rejects > 2048 chars | Raises `ValueError` |

**Method: `load_run`** (≥3 cases)

| # | Scenario | Assertion |
|---|---|---|
| 17 | Returns typed LoadedRun on complete run | `result.run_id == id`; `result.source_album` populated; `result.pipeline_result` not None |
| 18 | Raises on incomplete run | run with `status != "complete"` → `ValueError` |
| 19 | Does NOT touch session_state | After the call, `st.session_state` is unchanged (would need a monkeypatch or skipif streamlit) |

**Method: `list_other_actions`** (≥2 cases)

| # | Scenario | Assertion |
|---|---|---|
| 20 | Filters by action_type | DB has 3 `merge_apply` + 5 `fc_app_v2_run` → only 3 returned with default types |
| 21 | Honors limit | DB has 200 actions → returns 100 |

**`ActionTypeFormat.format`** (1 case per action_type, ≥5)

| # | Action type | Assertion |
|---|---|---|
| 22 | `merge_apply` | Output contains `round=` and `approved=` substrings |
| 23 | `profile_save` | Output contains `profile=` |
| 24 | `ml_train` | Output contains `model=` and `acc=` |
| 25 | `model_load` | Output contains `model=` |
| 26 | unknown type | Returns empty string (default formatter) |

**Total: ≥26 synthetic-data test cases.** All run in <100ms.

### 8.2 Service integration test (real fixture) — `tests/face_clustering/views/test_history_service_real.py`

Uses the `v2_budapest_run_dir` session fixture (skips if absent).

| # | Test | Assertion |
|---|---|---|
| 1 | `list_runs(HistoryQuery())` returns ≥1 row | At least one row with `action_type` in the v2 set |
| 2 | At least one row has `producer='fc_app_v2'` | Verifies the action_log has been populated by the v2 app this week |
| 3 | `get_run_detail` on a known Budapest run | `config["K"]` matches the saved value; `summary.n_clusters_base` is 33 (or 22 depending on which run is picked) |
| 4 | `list_albums()` contains "Budapest2025_Google" | (or whatever the source dir's `name` was) |

### 8.3 UI smoke — `tests/manual/_v2_history_smoke.py`

Playwright headless against Streamlit on port 8889.

```python
def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page()
        page.goto("http://localhost:8889", wait_until="networkidle", timeout=30000)
        page.wait_for_selector("h1:has-text('Face Clustering')", timeout=15000)

        # Open History tab.
        page.get_by_text("History", exact=True).click()
        page.wait_for_timeout(1000)

        body = page.locator("body").inner_text()
        # Section header visible
        assert "Pipeline Runs" in body, "Pipeline Runs heading missing"
        # No Streamlit error markdown
        for marker in ("StreamlitAPIException", "ValidationError", "Traceback"):
            assert marker not in body, f"Streamlit error: {marker}"
        # Filter bar widgets rendered
        for label in ("Album", "Date range", "Search"):
            assert label in body, f"filter widget {label!r} missing"

        page.screenshot(path="tests/manual/_v2_history_smoke.png", full_page=True)
        b.close()
    print("OK — History tab renders without errors")
    return 0
```

---

## 9. Implementation Plan

Phased so each step is independently shippable. Each ends with a checkpoint.

### Phase H0 — Prerequisites from spec-042

These are already required by spec-042 Phase 0 — listed here for visibility:

- [ ] `face_cluster/views/__init__.py` and `face_cluster/views/_specs.py` exist (`ColumnSpec`, `ActionTypeFormat`)
- [ ] `widget_factory.render_field(name, readonly=True, value_override=...)` mode exists
- [ ] Session fixtures `v2_budapest_run_dir`, `synthetic_action_log_db` exist
- [ ] `tests/face_clustering/views/_seed.py` provides row factories
- [ ] Arch tests for layering and docstrings are wired

If any of these is missing when starting the pilot, do it first.

### Phase H1 — Service backend (~2 hours)

- [ ] **H1.1** Add `face_cluster/views/history.py` with all dataclasses (`HistoryQuery`, `RunRow`, `ConfigDelta`, `RunSummary`, `RunDetail`, `LoadedRun`, `ActionRow`) — pure data, no methods.
- [ ] **H1.2** Add `RUN_COLUMNS` and `ACTION_COLUMNS` declarative specs.
- [ ] **H1.3** Add `ActionTypeFormat` class + per-action-type formatter functions.
- [ ] **H1.4** Add `HistoryService` class with all 7 methods implemented. Each method has full docstring per the contract.
- [ ] **H1.5** Add `tests/face_clustering/views/test_history_service_synthetic.py` covering the 26 cases in §8.1. **All must pass.**
- [ ] **H1.6** Add `tests/face_clustering/views/test_history_service_real.py` with the 4 cases in §8.2.

**Checkpoint H1**: `pytest tests/face_clustering/views/test_history_service_*.py` → all green. Service is fully tested without any UI dependency.

### Phase H2 — Components (~2 hours)

- [ ] **H2.1** Add `app/face_clustering_v2/components/__init__.py`.
- [ ] **H2.2** Add `app/face_clustering_v2/components/run_filter_bar.py` (`render_filter_bar`). Self-contained, no service-method calls inside the widget block (gathers values; calling code wires the result into a query).
- [ ] **H2.3** Add `app/face_clustering_v2/components/run_table.py` (`render_run_table`). Generic over `ColumnSpec` — also reusable by Cluster Analysis, History, Actions sub-table.
- [ ] **H2.4** Add `app/face_clustering_v2/components/run_detail.py` with sub-functions for header, config view (uses `widget_factory.render_field(readonly=True)`), summary, log viewer, payload viewer, files panel.
- [ ] **H2.5** Add `app/face_clustering_v2/components/load_button.py`.
- [ ] **H2.6** Add `app/face_clustering_v2/components/actions_table.py`.

**Checkpoint H2**: Each component file exists, ≤ 80 LOC, has docstring, no `cfg.get()` patterns. Arch tests still pass (no Streamlit in views, no DB access in components).

### Phase H3 — Tab orchestrator (~30 min)

- [ ] **H3.1** Add `app/face_clustering_v2/tabs/history_tab.py` per §7.6.
- [ ] **H3.2** Wire into `app/face_clustering_v2/main.py`'s `st.tabs([...])`.

**Checkpoint H3**: Tab loads in Streamlit without error against `v2_budapest_run_dir`.

### Phase H4 — UI smoke + manual verification (~30 min)

- [ ] **H4.1** Add `tests/manual/_v2_history_smoke.py` per §8.3. Run it; commit the result screenshot path to `.gitignore`.
- [ ] **H4.2** Manual walkthrough:
   - Filter by album → results narrow.
   - Filter by date range → results narrow.
   - Free-text filter → results narrow.
   - Click a Budapest run → detail panel renders.
   - Detail shows config grouped by stage, read-only.
   - Click Load → analysis tabs (when they exist) would pick up the loaded run.
   - Add a comment, change a tab and come back → comment persists.
   - Bulk-edit comments via the inline data_editor.

**Checkpoint H4**: All 8 manual steps pass; screenshot captured.

### Phase H5 — Close-out (~30 min)

- [ ] **H5.1** Run the full v2 test surface (`pytest tests/face_clustering/views/ tests/architecture/`) — green.
- [ ] **H5.2** Append `CHANGES_LOG.md` entry: "feat(spec-042 H1-H5): History tab pilot — re-implemented on the spec-041 contracts, ~50 LOC tab orchestrator + 5 reusable components + 6 services methods + 30 synthetic tests".
- [ ] **H5.3** Update `docs/architecture/classes.html` — add `HistoryService`, `RunRow`, etc.
- [ ] **H5.4** Update `docs/architecture/data_flow.html` — tab → service → DB flow.
- [ ] **H5.5** Mark all spec-042 Phase 1 tasks (T020–T050) as `[x]` in `specs/042-fc-app-v2-tab-parity/tasks.md`.

---

## 10. Estimates

| Phase | Effort | Description |
|---|---|---|
| H0 (prereqs) | 0h | Done as part of spec-042 Phase 0 |
| H1 (service) | ~2h | 7 methods + 26 unit tests |
| H2 (components) | ~2h | 5 components, 80 LOC each |
| H3 (tab orchestrator) | ~0.5h | One 50-LOC file |
| H4 (UI smoke + manual) | ~0.5h | Playwright + manual walkthrough |
| H5 (close-out) | ~0.5h | Tests + docs + CHANGES_LOG |
| **Total** | **~5.5h** | |

**Per spec-042 estimate**: History pilot was budgeted at ~6 hours. The detailed plan lands at 5.5h — within budget. The synthetic test suite is the biggest chunk; if it grows past 30 cases, allocate 7h.

---

## 11. Definition of Done

- [ ] `face_cluster/views/history.py` exists; `HistoryService` has 7 methods; all have docstrings per §4.6 of spec-042.
- [ ] `face_cluster/views/history.py` does not import `streamlit` (arch test enforces).
- [ ] ≥26 synthetic-data unit tests pass.
- [ ] ≥4 real-fixture integration tests pass (skipped cleanly when `v2_budapest_run_dir` is absent).
- [ ] `app/face_clustering_v2/tabs/history_tab.py` is ≤ 50 LOC of orchestration.
- [ ] No `cfg.get('field')` or `st.text(f"…{cfg.get(…)…}")` patterns anywhere in v2 History code.
- [ ] No `fc1, fc2, fc3` placeholder variable names; column variables are descriptive (`col_album`, `col_date`, `col_search`).
- [ ] No ASCII section comments (`# ---- xxx ----`).
- [ ] Playwright smoke (`_v2_history_smoke.py`) passes; screenshot captured.
- [ ] Manual walkthrough (§H4.2, 8 steps) passes against the Budapest fixture.
- [ ] CHANGES_LOG entry exists.
- [ ] All spec-042 Phase 1 tasks marked complete.

---

## 12. Open Questions

1. **`HistoryService` injection vs singleton?** Recommended: instance class with optional `db_path` in `__init__` (for tests). Tab creates one per render call; cheap enough. Avoids global singleton.
2. **`RunSummary` parsed from `pipeline_run.json` or from DB?** Recommended: from `pipeline_run.json` (matches legacy behavior). DB-side `run_metadata` table doesn't yet hold the funnel/timeline data.
3. **Where does `RunRow.run_kind` come from?** Recommended: parse from `action_type` if not explicitly stored. v2 writes `action_type='fc_app_v2_run'`; map to `"full"`. Legacy stores `run_kind` explicitly in some rows.
4. **Should the inline `st.data_editor` for bulk comment editing stay?** Recommended: yes, parity with legacy. It's a clear power-user feature.
5. **`load_run` returning `PipelineResult` — typed how?** Recommended: keep it as `Any` in the dataclass for now (avoid a dep on `face_cluster.pipeline.PipelineResult` in views/). Tab layer knows to write it directly to session_state. Tightening this type is a separate cleanup.
