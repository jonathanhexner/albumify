# Tasks: Run History & Run Annotations (013)

**Date**: 2026-04-23
**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

---

## Phase 1 — Setup

- [ ] T001 Create `specs/013-run-history-annotations/` directory structure (already done — confirm all artifacts present)
- [ ] T002 Update `.specify/feature.json` to point to `specs/013-run-history-annotations`

---

## Phase 2 — Foundational: DB Migration

*Blocks all other phases. Must complete and pass tests before any app changes.*

- [ ] T003 Extend `face_cluster/run_history_db.py`: add `ALTER TABLE … ADD COLUMN` for `source_album`, `run_name`, `parent_run_id`, `run_kind`, `comment`, `config_json`, `n_core`; guard with column-exists check for idempotency
- [ ] T004 Update `_HOT_FIELDS` set and `start_action` / `complete_action` signatures in `face_cluster/run_history_db.py` to accept and write new columns
- [ ] T005 Add `update_comment(run_id, comment, db_path)` write helper to `face_cluster/run_history_db.py`; enforce 2048-char limit
- [ ] T006 Add `idx_action_log_source_album` and `idx_action_log_comment` indexes in `face_cluster/run_history_db.py`
- [ ] T007 Write `tests/face_clustering/test_history_migration.py`: create DB without new columns, call `init_table()`, assert all new columns exist and pre-existing data is unchanged

---

## Phase 3 — User Story 2: Overwrite Protection

*Prerequisite: Phase 2 complete.*

- [ ] T008 [US2] Create `face_cluster/run_naming.py` with `RunDirSpec` dataclass and `allocate_run_dir(spec: RunDirSpec, results_root: Path, db_path: Path | None) -> Path`; use DB INSERT OR IGNORE as atomic reservation
- [ ] T009 [US2] Write `tests/face_clustering/test_run_naming.py`: sequential calls produce incrementing paths; two rapid calls return different paths; existing directory causes counter increment

---

## Phase 4 — User Story 5 (config diff helper)

*Can run in parallel with Phase 3.*

- [ ] T010 [P] [US5] Create `face_cluster/config_diff.py` with `ConfigDelta` dataclass and `compute(parent: dict, child: dict) -> list[ConfigDelta]`
- [ ] T011 [P] [US5] Write `tests/face_clustering/test_config_diff.py`: identical dicts → empty list; one changed field → one delta; key only in child → one delta with parent_value=None

---

## Phase 5 — User Story 1 & 4: `run_history.py` Search Helper

*Prerequisite: Phase 2 complete.*

- [ ] T012 [US1] Create `face_cluster/run_history.py` with `RunRow`, `HistoryFilters` dataclasses and `search(filters: HistoryFilters, db_path) -> list[RunRow]`; `distinct_albums(db_path) -> list[str]`; `get_run_by_id(run_id, db_path) -> RunRow | None`
- [ ] T013 [US1] Write `tests/face_clustering/test_run_history.py`: insert rows across 3 albums with known dates and comments; verify each filter (album, date range, text) independently; verify NULL source_album rows are returned without crash

---

## Phase 6 — User Story 1 & 4: History Tab App Rewrite

*Prerequisite: Phases 2, 5 complete.*

- [ ] T014 [US4] Rewrite `render_history_tab()` in `app/face_clustering.py`: filter bar (album selectbox, date range, text input) + `st.dataframe` with column config; display `source_album or "(unknown)"` in Album column
- [ ] T015 [US4] Wire `on_select="rerun"` on the History dataframe so clicking a row loads that run into session state and navigates to Clusters tab

---

## Phase 7 — User Story 1: Source-Album Propagation

*Prerequisite: Phase 2 complete.*

- [ ] T016 [US1] Update `FaceClusteringPipeline.run()` in `face_cluster/pipeline.py` to accept `source_album: str` and pass it to `start_action` payload
- [ ] T017 [US1] Update `FaceClusteringPipeline.recluster()` in `face_cluster/pipeline.py` to inherit `source_album` from parent run row (look up via `parent_run_id`)
- [ ] T018 [US1] Update all action-dispatch paths in `app/face_clustering.py` (Apply + Remerge, recluster, manual merge) to supply `source_album` from current session state

---

## Phase 8 — User Story 2: allocate_run_dir Integration

*Prerequisite: Phases 3, 7 complete.*

- [ ] T019 [US2] Replace all local output-path computations in `app/face_clustering.py` with calls to `allocate_run_dir`; display resolved path in UI before dispatching each action

---

## Phase 9 — User Story 3: Comment Field

*Prerequisite: Phase 2 complete.*

- [ ] T020 [US3] Add editable comment column to History `st.data_editor` in `app/face_clustering.py`; on change call `run_history_db.update_comment()`
- [ ] T021 [US3] Add `st.text_input` for comment in the run header component (Phase 10); bound to current run's comment field; same `update_comment` call

---

## Phase 10 — User Story 5: Run Header + Config Diff

*Prerequisite: Phases 2, 4, 5 complete.*

- [ ] T022 [US5] Add `render_run_header(run: RunRow)` component in `app/face_clustering.py`: displays Album • Run name • Parent (clickable) • `st.expander("Config Delta")`
- [ ] T023 [US5] Implement Config Delta rendering: load `config_json` for current and parent rows; call `config_diff.compute()`; render as two-column `st.dataframe` with "field | old → new" columns
- [ ] T024 [US5] Call `render_run_header()` at the top of every tab render when a run is loaded in `app/face_clustering.py`

---

## Phase 11 — User Story 6: Run Summary Panel (spec 012 dependent)

*Prerequisite: spec 012 deployed; Phases 2, 5 complete.*

- [ ] T025 [US6] Add `render_run_summary(run_dir: Path)` in `app/face_clustering.py`: read `pipeline_run.json`, `merge_metadata.json`, `merge_log.json`; render quality funnel, cluster counts, stage timeline
- [ ] T026 [US6] Handle `merge_log.json == []` case: display "0 candidates at threshold {merge_candidate_threshold}" from `merge_metadata.json`
- [ ] T027 [US6] Handle missing spec 012 outputs gracefully: display "Data not available (pre-012 run)" when files are absent

---

## Phase 12 — Polish

- [ ] T028 Export `RunRow`, `RunDirSpec`, `HistoryFilters`, `ConfigDelta` from `face_cluster/__init__.py`
- [ ] T029 Update `docs/architecture.md` to reflect new helpers (`run_naming`, `config_diff`, `run_history`) and extended `action_log` schema
- [ ] T030 Append `[FEATURE]` entry to `CHANGES_LOG.md` covering all modified files
- [ ] T031 Mark SIGHTING-025 as resolved in `docs/SIGHTINGS.md` (overwrite protection delivered by T008, T019)

---

## Dependencies

```
Phase 2 (migration) → Phase 3, 5, 7, 9
Phase 3 (run_naming) → Phase 8
Phase 4 (config_diff) → Phase 10
Phase 5 (run_history) → Phase 6, 10, 11
Phase 7 (source_album propagation) → Phase 8
Spec 012 → Phase 11
```

## Parallel Opportunities

- T010, T011 (config_diff) can run in parallel with T008–T009 (run_naming).
- T012–T013 (run_history) can run in parallel with T008–T011.
- T014–T021 (app changes) can be batched once Phases 2–5 are done.
