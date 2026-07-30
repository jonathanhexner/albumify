# Tasks: Column Registry for RunHistoryRepository (044)

Predecessor: spec-043 (Repository pattern)
Standards reference: [`docs/architecture/architecture_standards.md`](../../docs/architecture/architecture_standards.md) §A2 (declarative specs), §B0 (Repository pattern), §A6 (per-migration discipline)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped (rationale required)

---

## Design notes (binding for every task)

- **D1**: **Build → test → migrate**, never out of order. Phase 1 adds the registry alongside existing code (no behavior change). Phase 2 proves the registry is byte-equivalent to the hand-written constants (temporary parity tests). Phase 3 swaps the constants. Phase 4 swaps the methods. Each step preserves behavior.
- **D2**: Repository **external interface is unchanged**. The 33 existing synthetic + 3 real-fixture Repository tests are the contract guard. They must pass after every phase.
- **D3**: Service-layer tests (31 synthetic + 4 real) must also pass after every phase — they compose the Repository; if its behavior changes, they break.
- **D4**: Temporary verification tests added in Phase 2 are explicitly marked as such (`# TEMPORARY — remove in Phase 5`). They get deleted once the constants they verify against are themselves deleted.
- **D5**: The drift-guard arch tests (`test_run_history_repo_column_registry.py`) are PERMANENT — they survive the refactor and catch future drift.
- **D6**: Adding a column to `action_log` after this spec lands requires editing **exactly 2 places**: the `_COLUMNS` list + the `RunRow` dataclass. The arch tests catch any other place that would need editing.

---

## Phase 1 — Build the registry (~45 min)

Adds the new types and registry alongside existing constants. No method changes; no behavior change. Existing 74 tests stay green.

- [ ] **T001** Add `ColumnDef` frozen-slotted dataclass to `face_cluster/repositories/run_history_repo.py` (top of file, near other module constants). Fields per spec §4.1: `name`, `sql_type`, `initial`, `nullable`, `default_sql`, `primary_key`, `hot`, `filterable`.
- [ ] **T002** Add `_COLUMNS: list[ColumnDef]` per spec §4.2. 24 entries.
- [ ] **T003** Add `_create_table_sql() -> str` function generating the CREATE TABLE statement from `_COLUMNS` where `initial=True`. Don't call it yet — the existing `_CREATE_SQL` is still in use.
- [ ] **T004** Add `_MIGRATION_COLUMNS_NEW: list[tuple[str, str]]` derived from `_COLUMNS` where `initial=False`. Named "_NEW" to coexist with the existing `_ALTER_COLUMNS`.
- [ ] **T005** Add `_HOT_FIELDS_NEW: tuple[str, ...]` derived from `_COLUMNS` where `hot=True`. Named "_NEW" to coexist with existing `_HOT_FIELDS`.
- [ ] **T006** Add `_FILTERABLE_FIELDS: tuple[str, ...]` derived from `_COLUMNS` where `filterable=True`. (New constant — no equivalent existed before.)
- [ ] **T007** Verify imports + run the 33 Repository tests: `pytest tests/face_clustering/repositories/ -q`. All pass.

**Checkpoint**: Registry exists; generated constants are computed; no existing code path uses them yet; all 74 tests still pass.

---

## Phase 2 — Parity test (~30 min)

Add the permanent drift-guard tests + temporary equivalence assertions. The temporary ones prove the generated constants match the hand-written ones; once green, Phase 3 swaps the constants.

- [ ] **T010** Create `tests/architecture/test_run_history_repo_column_registry.py`.
- [ ] **T011** Add **PERMANENT** test `test_runrow_fields_match_columns_registry`: assert every `_COLUMNS[i].name` appears as a field on `RunRow`. (Direction: registry ⊆ RunRow; RunRow may have extras.)
- [ ] **T012** Add **PERMANENT** test `test_filterable_columns_have_matching_criteria_fields`: for each column with `filterable=True`, assert either (a) `RunHistoryCriteria` has a same-named field, OR (b) the column is in an allowlist of documented special cases (e.g., `source_album` ↔ `criteria.album`, `action_type` ↔ `criteria.action_types`).
- [ ] **T013** Add **PERMANENT** test `test_initial_and_migration_partition_is_complete`: every column has its `initial` flag set explicitly; no column appears in both the initial CREATE list and the migration list.
- [ ] **T014** Add **PERMANENT** test `test_only_nullable_hot_fields`: every column with `hot=True` is `nullable=True`. A NOT NULL hot field would crash on insert when a payload key is missing.
- [ ] **T015** Add **TEMPORARY** test `test_generated_create_table_matches_legacy_create_sql`: open two in-memory SQLite DBs (one with `_CREATE_SQL`, one with `_create_table_sql()`); assert `PRAGMA table_info` returns identical column lists.
- [ ] **T016** Add **TEMPORARY** test `test_migration_columns_new_equals_legacy`: assert `_MIGRATION_COLUMNS_NEW == _ALTER_COLUMNS`.
- [ ] **T017** Add **TEMPORARY** test `test_hot_fields_new_equals_legacy`: assert `set(_HOT_FIELDS_NEW) == set(_HOT_FIELDS)`.
- [ ] **T018** Run: `pytest tests/architecture/test_run_history_repo_column_registry.py -v`. All 7 (4 permanent + 3 temporary) pass.

**Checkpoint**: Generated constants are PROVEN equivalent to hand-written constants. Phase 3 is safe to start.

---

## Phase 3 — Migrate constants (~15 min)

Replace the hand-written constants with the generated ones. The 7 arch tests + 74 existing tests stay green.

- [ ] **T020** Delete the hand-written `_CREATE_SQL` string constant. Replace its usage sites with `_create_table_sql()` (function call). Likely 1–2 sites.
- [ ] **T021** Delete the hand-written `_ALTER_COLUMNS` list. Rename `_MIGRATION_COLUMNS_NEW` → `_MIGRATION_COLUMNS`. Update `_migrate(conn)` to iterate the new name.
- [ ] **T022** Delete the hand-written `_HOT_FIELDS` constant. Rename `_HOT_FIELDS_NEW` → `_HOT_FIELDS`.
- [ ] **T023** Remove the 3 temporary parity assertions (T015 / T016 / T017) from the arch test file — the constants they referenced no longer exist. Leave the 4 permanent tests.
- [ ] **T024** Run: `pytest tests/face_clustering/repositories/ tests/face_clustering/views/ tests/architecture/test_run_history_repo_column_registry.py -v`. All 33 + 35 + 4 = 72 pass.

**Checkpoint**: The constants are now generated. Repository behavior unchanged (proven by the 33 Repository tests).

---

## Phase 4 — Migrate methods (~45 min)

Rewrite `start_action`, `complete_action`, `_build_where` to iterate `_COLUMNS`. This is the meat of the refactor — where the 9 → 2 touch-point reduction shows up.

- [ ] **T030** Rewrite `start_action` per spec §4.4:
  - Build INSERT column list from `[c.name for c in _COLUMNS if not c.primary_key]`.
  - Build placeholders and SQL string from the column list.
  - Compute values via `_start_action_value(col, action_type, payload)` helper that handles the 4 lifecycle special cases (`action_type`, `status`, `started_at`, `payload_json`) and the "all-None" group (`ended_at`, `duration_s`, `error`); rest get `payload.get(col.name)`.
- [ ] **T031** Add `_start_action_value(col, action_type, payload)` module-level helper. Pure function; easily unit-testable in isolation if needed.
- [ ] **T032** Rewrite `complete_action`'s UPDATE construction per spec §4.4:
  - Hardcoded SET parts: `status='complete'`, `ended_at=?`, `duration_s=?`.
  - Hot-field SET parts: iterate `_COLUMNS` where `c.hot`; emit `{name}=COALESCE(?,{name})`.
  - Trailing SET part: `payload_json=?`.
  - Values list mirrors the SET part order.
- [ ] **T033** Rewrite `_build_where` per spec §4.4:
  - Cross-field validation unchanged.
  - Generic equality loop: iterate `_FILTERABLE_FIELDS`; emit `col = ?` when criteria value is non-None.
  - Special cases preserved verbatim:
    - `action_types` (list filter on `action_type`)
    - `album` (criteria field name ↔ `source_album` column)
    - `text` + `text_fields` (multi-column LIKE)
    - `date_from` / `date_to` (range)
    - `ids` (IN clause)
- [ ] **T034** Run full surface: `pytest tests/face_clustering/repositories/ tests/face_clustering/views/ tests/architecture/ -q`. All 74 pass.

**Checkpoint**: The 9 → 2 touch points goal is reached. Adding a column now requires editing only `_COLUMNS` + `RunRow`.

---

## Phase 5 — Final cleanup (~5 min)

The migration-parity test introduced in Phase 2 is now obsolete (its referent — the hand-written constants — no longer exists). Already deleted in T023 if I followed the plan; verify.

- [ ] **T040** Confirm the temporary parity tests are gone from `test_run_history_repo_column_registry.py`. Only the 4 permanent drift guards remain.
- [ ] **T041** Final test sweep: `pytest tests/ -q`. All pass.

**Checkpoint**: Test surface stable; drift guards in place.

---

## Phase 6 — Docs + close-out (~30 min)

- [ ] **T050** Update `face_cluster/repositories/run_history_repo.py` module docstring: add a "Column Registry" section pointing at `_COLUMNS` as the single source of truth + cross-reference to the spec / architecture standards.
- [ ] **T051** Update `docs/architecture/architecture_standards.md` §B0: add a "Column Registry" sub-section per spec §4.5 explaining the 2-touch-point rule + the arch tests.
- [ ] **T052** Update `specs/042-fc-app-v2-tab-parity/ARCHITECTURE_STANDARDS.html` §B0 with the same content.
- [ ] **T053** Append `CHANGES_LOG.md` entry.
- [ ] **T054** Flip spec status `Draft` → `Code Review` → `Implemented`.
- [ ] **T055** Commit + push.

**Checkpoint**: Pattern documented as a binding standard. Future Repositories follow this shape.

---

## Out of scope (explicit deferrals)

- **Auto-generating `RunRow`** from `_COLUMNS` via `make_dataclass()`. Loses IDE autocomplete; arch test catches drift instead.
- **Extracting `ColumnDef` to a shared module** for use by future Repositories. Premature — wait for a second consumer.
- **Replacing the idempotent ALTER migration with Alembic-style versioned migrations**. Different concern; not in scope.
- **Generalizing `RunHistoryCriteria` to be code-generated** from filterable columns. The typed dataclass + manual special cases (text, date_from, ids) is acceptable.
- **Migrating to SQLAlchemy**. Repeatedly evaluated; rejected at current scale. The column registry closes the main pain point that SQLAlchemy would address.
