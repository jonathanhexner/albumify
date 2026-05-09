# Research: Run History & Run Annotations (013)

**Date**: 2026-04-23
**Feature**: 013-run-history-annotations

---

## Decision 1 — Extend `action_log` vs new table

**Decision**: Extend `action_log` with new columns via `ALTER TABLE … ADD COLUMN`.

**Rationale**: All existing history infrastructure (writers, readers, migration scripts) targets `action_log`. A new table would require a join on every History query and a dual-write in every caller. Additive `ALTER TABLE` is safe in SQLite for NULL-default columns and requires no data migration beyond the column additions.

**Alternatives considered**:
- New `run_metadata` table with 1:1 join — rejected: unnecessary complexity, join overhead, dual-write burden.
- Replace `album` column with `source_album` — rejected: `ALTER TABLE … RENAME COLUMN` is SQLite 3.25+ and risky on prod. Instead add `source_album` alongside `album`; deprecate `album` in a follow-up.

---

## Decision 2 — `source_album` population strategy

**Decision**: `source_album` is written at run-registration time from `session.json.source_album` (for base runs) or inherited from the parent row's `source_album` (for all derived runs). It is never derived by parsing the output path.

**Rationale**: `session.json` already carries `source_album` as the canonical identity (spec 009). Inheriting from the parent row guarantees all variants of an album remain grouped regardless of output path structure.

**Alternatives considered**:
- Parse output dir name to infer album — rejected: brittle, order-dependent, breaks with custom folder names (SIGHTING-025 root cause).
- Require user to name the album manually at run start — rejected: friction, error-prone.

---

## Decision 3 — `allocate_run_dir` atomicity

**Decision**: Use a SQLite `INSERT OR IGNORE` reservation row to atomically claim a path before creating the directory.

**Rationale**: `os.path.exists` + `mkdir` is a TOCTOU race. By inserting a placeholder row in the DB (unique constraint on `output_dir`) before `os.makedirs`, the DB becomes the lock. If the insert fails the counter increments and retries. Single-process local app makes this sufficient; no file-system locking needed.

**Alternatives considered**:
- `mkdir` with `exist_ok=False` then catch `FileExistsError` — rejected: no DB record of the reservation; directory could be created by a parallel app action without a DB row.
- File-system lock file — rejected: adds cleanup burden and doesn't integrate with DB history.

---

## Decision 4 — Comment storage

**Decision**: `comment TEXT` column on `action_log` with a 2 048-character hard limit enforced in the write helper (not the DB schema — SQLite has no VARCHAR length enforcement).

**Rationale**: Free-text comments need no relational structure. Keeping them in `action_log` avoids a separate table. The 2 048-character limit is enforced in Python before the INSERT/UPDATE.

**Alternatives considered**:
- Separate `run_comments` table with versioned history — rejected: over-engineered for a single editable field.
- Store in `pipeline_run.json` only — rejected: no searchability without reading every JSON file.

---

## Decision 5 — Config diff computation

**Decision**: Load `config_json` from both parent and child `action_log` rows. Compute symmetric diff: keys present in either config with different values. Render only differing keys in the UI expander.

**Rationale**: Shallow dict diff is sufficient for the pipeline config (flat YAML-derived dict). Deep nested diffs would require a recursive walk with path notation — disproportionate complexity for the use case.

**Alternatives considered**:
- Deep recursive diff — deferred: can be added later if users request it.
- Store config hash only — rejected: cannot reconstruct the actual delta without the full config.

---

## Decision 6 — History table rendering (Streamlit)

**Decision**: `st.dataframe` with `on_select="rerun"` and `selection_mode="single-row"` for click-to-load. Filter bar rendered above with `st.selectbox` (album), `st.date_input` (date range), `st.text_input` (free-text).

**Rationale**: `st.dataframe` supports column config, sorting, and selection callbacks natively in recent Streamlit versions. Avoids building a custom HTML table.

**Alternatives considered**:
- `st.table` — rejected: no sorting, no selection.
- `AgGrid` component — rejected: external dependency, not already in the project.

---

## Decision 7 — `run_kind` vocabulary

**Decision**: Allowed values: `base`, `recluster`, `remerge`, `manual_merge`. Stored as `TEXT`, validated in Python before insert.

**Rationale**: Maps directly to existing `action_type` values in `action_log`; no new vocabulary needed, just a normalized enum column.

---

## Decision 8 — `n_core` column

**Decision**: Add `n_core INTEGER` column to `action_log`. Populated from `pipeline_run.json.summary.n_core` at run completion.

**Rationale**: The History table column "n_faces / n_core" requires `n_core` to be a hot field for sorting/display without reading JSON files.

---

## Known Constraints

- SQLite 3.x: `ALTER TABLE … ADD COLUMN` only supports columns with NULL default or constant default. All new columns default to NULL. ✓
- Streamlit `st.dataframe` `on_select` requires Streamlit ≥ 1.29. Confirm version in `pyproject.toml` before implementation.
- `pipeline_run.json` extensions (adding `source_album`, `run_name`, `comment`) are additive-only; existing readers ignore unknown keys. ✓
