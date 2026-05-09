# Implementation Plan: Run History & Run Annotations (013)

**Branch**: `013-run-history-annotations` | **Date**: 2026-04-23
**Spec**: [spec.md](spec.md) | **Data model**: [data-model.md](data-model.md) | **Research**: [research.md](research.md)

---

## Summary

Extend the `action_log` DB table with album-identity, lineage, and annotation columns. Add a `allocate_run_dir` helper for safe unique output-path reservation. Rewrite the History tab as a searchable, filterable table. Add a run header with parent link and config diff. Add a Run Summary panel (depends on spec 012 outputs).

---

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: SQLite (via `sqlite3` stdlib), Streamlit ≥ 1.29, existing `face_cluster/run_history_db.py`
**Storage**: `~/.sim_bench/sim_bench.db` — `action_log` table
**Testing**: pytest (`tests/face_clustering/`)
**Target Platform**: Windows 10 (local desktop app)
**Constraints**: No new top-level packages; all new modules go in `face_cluster/`. No try/except except at system boundaries.

---

## Constitution Check

| Rule | Status |
|---|---|
| Two public methods on `FaceClusteringPipeline` | Not affected — no new methods added |
| Algorithms only in `face_cluster/` | ✓ — new helpers (`run_naming`, `config_diff`, `run_history`) go in `face_cluster/` |
| No algorithm code in `app/` | ✓ — app calls helpers, never implements logic |
| No test uses real album path (except e2e) | ✓ — all new tests use `tmp_path` |
| Writer-reader contracts | ✓ — `action_log` schema is owned by `run_history_db.py`; all readers use its loader |
| Non-blocking UI | ✓ — History search query is fast (SQL); no new background threads needed |
| Tests use production-default config | ✓ — no config relaxation in tests |
| ASCII-only CLI/log output | ✓ — no new print/log with Unicode |

---

## Project Structure

### Documentation (this feature)

```text
specs/013-run-history-annotations/
├── spec.md
├── plan.md            (this file)
├── research.md
├── data-model.md
├── tasks.md
└── checklists/
    └── requirements.md
```

### Source Code Changes

```text
face_cluster/
├── run_history_db.py       MODIFY — schema migration, new columns, search query
├── run_naming.py           NEW    — allocate_run_dir, RunDirSpec
├── config_diff.py          NEW    — compute(parent_config, child_config) -> List[ConfigDelta]
├── run_history.py          NEW    — RunRow, HistoryFilters, search()
└── __init__.py             MODIFY — export new public types

app/
└── face_clustering.py      MODIFY — History tab rewrite, run header, Run Summary panel

tests/face_clustering/
├── test_run_naming.py      NEW
├── test_config_diff.py     NEW
├── test_run_history.py     NEW
└── test_history_migration.py  NEW
```

---

## Phase Plan

### Phase 0 — DB Migration (foundational, blocks everything)

1. Add new columns to `action_log` via `ALTER TABLE … ADD COLUMN` in `run_history_db.init_table()`. Use `IF NOT EXISTS` guard so repeated calls are idempotent.
2. Add `source_album` and `n_core` to `_HOT_FIELDS` set.
3. Update `start_action` and `complete_action` to accept and write `source_album`, `run_name`, `parent_run_id`, `run_kind`, `config_json`, `n_core`, `comment`.
4. Add `update_comment(run_id, comment, db_path)` helper.
5. Add new indexes: `idx_action_log_source_album`, `idx_action_log_comment`.

**Test**: `test_history_migration.py` — create DB without new columns, call `init_table()`, assert new columns exist and old data is unchanged.

---

### Phase 1 — `run_naming.py` (US2 — overwrite protection)

`allocate_run_dir(spec: RunDirSpec, results_root: Path, db_path: Path | None) -> Path`

- Constructs candidate path: `results_root / spec.source_album / f"{spec.kind}_{n}"`.
- Increments `n` (starting from 1) until neither the path exists on disk NOR a reservation row exists in `action_log` for that `output_dir`.
- Inserts a reservation row (`status='reserved'`) atomically via `INSERT OR IGNORE` (unique constraint on `output_dir`).
- Returns the reserved path. Callers must call `start_action` or `complete_action` to upgrade the row's status.

**Test**: `test_run_naming.py` — sequential calls produce incrementing paths; two rapid calls never return the same path.

---

### Phase 2 — `config_diff.py` (US5 — config diff)

`compute(parent: dict, child: dict) -> list[ConfigDelta]`

- Iterates keys present in either dict.
- Returns `ConfigDelta(field, parent_value, child_value)` only for keys where values differ.
- Missing key treated as `None` (not as "changed" vs a key that is explicitly `None`).

**Test**: `test_config_diff.py` — identical configs produce empty list; one changed field produces one delta; key present only in child produces one delta.

---

### Phase 3 — `run_history.py` (US1, US4 — search)

`search(filters: HistoryFilters, db_path: Path | None) -> list[RunRow]`

- Builds a parameterised SQL query with optional WHERE clauses for `source_album`, date range, and LIKE text search.
- Returns list of `RunRow` dataclass instances.

`distinct_albums(db_path: Path | None) -> list[str]` — returns sorted list of distinct non-NULL `source_album` values for the filter dropdown.

`get_run_by_id(run_id: str, db_path: Path | None) -> RunRow | None`

**Test**: `test_run_history.py` — insert test rows covering 3 albums; verify each filter criterion independently; verify text search on comment; verify NULL `source_album` rows are returned with fallback label in caller.

---

### Phase 4 — App: History tab rewrite (US1, US4)

In `app/face_clustering.py`, replace `render_history_tab()`:

- Filter bar: `st.selectbox` (Album), `st.date_input` (date range), `st.text_input` (search).
- Build `HistoryFilters` from widget values; call `run_history.search()`.
- Render `st.dataframe` with column config. Display `source_album or "(unknown)"` in Album column.
- `on_select="rerun"` captures selected row; loads that run into session state.

---

### Phase 5 — App: Source-album propagation (US1 — fixes P0)

Update every action-dispatch path in `app/face_clustering.py`:

- Apply + Remerge: pass `source_album` from current session to `start_action` payload.
- Recluster: same.
- Manual merge: same.

Update `FaceClusteringPipeline.run()` and `recluster()` to accept and propagate `source_album` through to `start_action`.

---

### Phase 6 — App: `allocate_run_dir` integration (US2)

Replace all local output-path computations in `app/face_clustering.py` with calls to `allocate_run_dir`. Display the resolved path in the UI before dispatching.

---

### Phase 7 — App: Comment field (US3)

- In History table: add editable `st.text_input` per row (use `st.data_editor` with a comment column).
- On change: call `run_history_db.update_comment(run_id, text)`.
- In run header: same `st.text_input` bound to current run's comment.

---

### Phase 8 — App: Run header + config diff (US5)

New `render_run_header(run: RunRow)` component in `app/face_clustering.py`:

- Displays: Album • Run name • Parent (clickable) • `st.expander("Config Δ")`.
- Config Δ: load `config_json` for current row and parent row; call `config_diff.compute()`; render as `st.dataframe`.
- Called at the top of every tab render when a run is loaded.

---

### Phase 9 — App: Run Summary panel (US6, depends on spec 012)

New `render_run_summary(run_dir: Path)` component:

- Reads `pipeline_run.json`, `merge_metadata.json`, `merge_log.json` from `run_dir`.
- Renders quality funnel, cluster formation counts, stage timeline.
- When `merge_log.json == []`: renders "0 candidates at threshold {merge_candidate_threshold}".
- When spec 012 outputs are absent: renders "Data not available (pre-012 run)".

---

## Resolved Decisions (from research.md)

| # | Decision |
|---|---|
| 1 | Extend `action_log` with ADD COLUMN (not a new table) |
| 2 | `source_album` from `session.json` or inherited; never parsed from output path |
| 3 | `allocate_run_dir` uses DB INSERT OR IGNORE as atomic reservation |
| 4 | Comment in `action_log.comment`, 2048-char limit enforced in Python |
| 5 | Config diff: shallow dict comparison, differing keys only |
| 6 | History table: `st.dataframe` with `on_select` |
| 7 | `run_kind` enum: base/recluster/remerge/manual_merge |
| 8 | `n_core` as hot column in `action_log` |

---

## Dependencies / Ordering

- Phase 0 (migration) blocks all other phases.
- Phases 1–3 (helpers) can run in parallel after Phase 0.
- Phase 4–8 (app) require Phase 0–3 to be complete.
- Phase 9 (Run Summary) requires spec 012 to be deployed.
- Phases 4–8 are independent of spec 012.
