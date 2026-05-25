# spec-044 — Column Registry for RunHistoryRepository

Status: **Draft**
Author: Jonathan Hexner
Created: 2026-05-25
Predecessors: [spec-043](../043-repository-pattern/spec.md) (Repository pattern)
Architectural standards: [`docs/architecture/architecture_standards.md`](../../docs/architecture/architecture_standards.md) §A2 (declarative specs over repetitive code), §B0 (Repository pattern), §A6 (per-tab migration discipline)

---

## 1. Objective

Replace the 9 hardcoded references to the `action_log` column list inside `face_cluster/repositories/run_history_repo.py` with a **single column registry** (`_COLUMNS: list[ColumnDef]`). Generate all derived constants (`_CREATE_SQL`, `_MIGRATION_COLUMNS`, `_HOT_FIELDS`, `_FILTERABLE_FIELDS`) and all per-column code paths (INSERT column list, UPDATE SET clauses, WHERE composition) from it.

After: adding a column to the table touches **2 places** (the `_COLUMNS` registry + the `RunRow` dataclass), down from 9. The 9-→-2 reduction is enforced by an architecture test asserting `RunRow` field set matches `_COLUMNS` names.

**Repository's external interface is unchanged.** This is a pure internal refactor — every existing test (33 synthetic + 3 real-fixture + 35 service + 3 arch) continues to pass without modification.

### Non-goals

- Migrating away from raw `sqlite3` to SQLAlchemy. Repeatedly evaluated; rejected at current scale. See [ARCHITECTURE_STANDARDS.html §B0 discussion](../042-fc-app-v2-tab-parity/ARCHITECTURE_STANDARDS.html) — the column registry closes the main pain point that SQLAlchemy would address.
- Generalizing `ColumnDef` to other Repositories (future `ClusterAnalysisRepository`, etc.). Per-Repository registries are fine for now; if a second Repository proves the pattern, extract a shared helper later.
- Reducing `RunRow`'s field declaration to be auto-generated. `RunRow` stays a hand-declared dataclass; an arch test enforces parity with `_COLUMNS`.

---

## 2. Why now

### 2.1 The problem you named

> "It just feels the repository has a really lot of hardcoded fields in the code. I can understand that some need to be hardcoded… but everything? if we ever change anything I worry we'll have a great deal of work."

Audit of `face_cluster/repositories/run_history_repo.py` confirms: adding a new column to `action_log` today requires 9 separate edits — and missing one is silently broken until a specific code path fires. The user's concern is well-founded.

### 2.2 Aligns with established standards

A2 of the architecture standards says: *"when you see a list of 'field name + how to display it' appearing as code, replace with a list of typed spec objects + one renderer."* The Repository today has 5 such lists (CREATE columns, ALTER columns, HOT fields, INSERT columns, UPDATE columns) that should all derive from one source. spec-044 is A2 applied to the persistence layer.

### 2.3 Cost is low; payoff compounds

- ~3 hours of focused work.
- No consumers change (Repository external interface preserved).
- Every future column-addition (and there will be many — schema_v6, new spec-013-style ALTERs) saves ~30 minutes of cross-file editing and removes a class of "forgot to update X" bugs.
- Future Repositories (`ClusterAnalysisRepository`, etc.) reuse the pattern.

---

## 3. Scope

### In scope

| Concern | What ships |
|---|---|
| **New types** | `ColumnDef` dataclass in `face_cluster/repositories/run_history_repo.py` (or a sibling `_columns.py` if it grows). One `_COLUMNS: list[ColumnDef]` registry. |
| **Generated constants** | `_create_table_sql()`, `_MIGRATION_COLUMNS`, `_HOT_FIELDS`, `_FILTERABLE_FIELDS` all derived from `_COLUMNS` instead of hand-written. |
| **Refactored methods** | `start_action`, `complete_action`, `_build_where` rewritten to iterate `_COLUMNS` instead of repeating column names inline. ~80 LOC removed; ~50 LOC added (the registry). Net: −30 LOC. |
| **Parity test (NEW)** | `tests/architecture/test_run_history_repo_column_registry.py` — asserts `RunRow` field set matches `_COLUMNS` names + a few self-consistency invariants (hot fields are nullable in CREATE, initial fields aren't in MIGRATION_COLUMNS, etc.). |
| **Migration parity test (NEW)** | Same file: assert generated `_create_table_sql()` produces a schema-equivalent DB to the hand-written `_CREATE_SQL` (Phase 2 verification — once green, the hand-written version is replaced). |
| **Docs** | Update `docs/architecture/architecture_standards.md` §B0 with a Column Registry note. Update the Repository class docstring to reference `_COLUMNS`. |

### Out of scope

| Item | Reason |
|---|---|
| Auto-generating `RunRow` from `_COLUMNS` | Dataclass field declarations are static at class def time. Auto-generation via `make_dataclass()` is ugly and breaks IDE autocomplete. Hand-declare `RunRow`; an arch test catches drift. |
| Extracting `ColumnDef` to a shared `face_cluster/repositories/_column_def.py` for use across Repositories | Premature — no second Repository exists yet. Lifting happens when the pattern proves itself on a second consumer. |
| Replacing the idempotent ALTER migration with a versioned migration framework (Alembic-style) | Out of scope; current pattern is fine at our schema complexity. |
| Generalizing `RunHistoryCriteria` filterable-field mapping into a code-generated criteria dataclass | Premature; the typed dataclass + manual WHERE composition for special cases (text, date_from, ids) is acceptable. |

---

## 4. Design

### 4.1 The `ColumnDef` dataclass

```python
@dataclass(frozen=True, slots=True)
class ColumnDef:
    """One column on the action_log table — single source of truth.

    Attributes:
        name: SQL column name. Also matches the RunRow field name.
        sql_type: "INTEGER" | "TEXT" | "REAL".
        initial: True if this column is in the initial CREATE TABLE.
            False = added later via ALTER TABLE (idempotent migration).
        nullable: True (default) for all except id / NOT NULL columns.
        default_sql: SQL DEFAULT expression for CREATE TABLE (e.g. "'running'").
        primary_key: True for the id column.
        hot: True if this column is writable via start_action's payload
            and complete_action's result_fields (i.e., participates in
            the INSERT and the COALESCE UPDATE).
        filterable: True if this column has a matching field on
            RunHistoryCriteria; _build_where will emit a "col = ?"
            clause when the criteria value is non-None.
    """
    name: str
    sql_type: str
    initial: bool = True
    nullable: bool = True
    default_sql: Optional[str] = None
    primary_key: bool = False
    hot: bool = False
    filterable: bool = False
```

### 4.2 The `_COLUMNS` registry

The 24-row source of truth:

```python
_COLUMNS: list[ColumnDef] = [
    # Identity + lifecycle
    ColumnDef("id",            "INTEGER", primary_key=True, nullable=False),
    ColumnDef("action_type",   "TEXT", nullable=False, filterable=True),
    ColumnDef("status",        "TEXT", nullable=False, default_sql="'running'", filterable=True),
    ColumnDef("started_at",    "TEXT", nullable=False),
    ColumnDef("ended_at",      "TEXT"),
    ColumnDef("duration_s",    "REAL"),
    ColumnDef("error",         "TEXT"),

    # Hot fields (writable via start/complete)
    ColumnDef("run_id",        "TEXT",    hot=True),
    ColumnDef("source_dir",    "TEXT",    hot=True),
    ColumnDef("output_dir",    "TEXT",    hot=True),
    ColumnDef("album",         "TEXT",    hot=True),
    ColumnDef("n_faces",       "INTEGER", hot=True),
    ColumnDef("n_clusters",    "INTEGER", hot=True),
    ColumnDef("n_noise",       "INTEGER", hot=True),
    ColumnDef("log_file",      "TEXT",    hot=True),

    # Payload (special-case; not "hot" because special serialization)
    ColumnDef("payload_json",  "TEXT"),

    # Spec-013 (added via ALTER)
    ColumnDef("source_album",  "TEXT",    initial=False, hot=True, filterable=True),
    ColumnDef("run_name",      "TEXT",    initial=False, hot=True, filterable=True),
    ColumnDef("parent_run_id", "INTEGER", initial=False, hot=True, filterable=True),
    ColumnDef("run_kind",      "TEXT",    initial=False, hot=True),
    ColumnDef("comment",       "TEXT",    initial=False, hot=True, filterable=True),
    ColumnDef("config_json",   "TEXT",    initial=False, hot=True),
    ColumnDef("n_core",        "INTEGER", initial=False, hot=True),

    # Spec-040 T4 (producer column)
    ColumnDef("producer",      "TEXT",    initial=False, hot=True, filterable=True),
]
```

### 4.3 Generated constants

```python
def _create_table_sql() -> str:
    """Emit CREATE TABLE statement from the columns whose `initial=True`."""
    cols = []
    for c in _COLUMNS:
        if not c.initial:
            continue
        parts = [c.name, c.sql_type]
        if c.primary_key:
            parts.append("PRIMARY KEY AUTOINCREMENT")
        elif not c.nullable:
            parts.append("NOT NULL")
        if c.default_sql:
            parts.append(f"DEFAULT {c.default_sql}")
        cols.append(" ".join(parts))
    return f"CREATE TABLE IF NOT EXISTS {_TABLE} (\n  " + ",\n  ".join(cols) + "\n)"


# Replaces _ALTER_COLUMNS
_MIGRATION_COLUMNS: list[tuple[str, str]] = [
    (c.name, c.sql_type) for c in _COLUMNS if not c.initial
]

# Replaces _HOT_FIELDS
_HOT_FIELDS: tuple[str, ...] = tuple(c.name for c in _COLUMNS if c.hot)

# New — drives _build_where's generic equality clauses
_FILTERABLE_FIELDS: tuple[str, ...] = tuple(c.name for c in _COLUMNS if c.filterable)
```

### 4.4 Refactored methods

#### `start_action`

```python
def start_action(self, action_type: str, *, payload: Optional[Dict] = None) -> int:
    """INSERT a new 'running' row; return its id. Side effects: one row written."""
    self._check_writable()
    payload = payload or {}

    # Columns to populate (everything except id, which is autoincrement).
    cols = [c for c in _COLUMNS if not c.primary_key]
    col_names = [c.name for c in cols]
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO {_TABLE} ({', '.join(col_names)}) VALUES ({placeholders})"

    values = [_start_action_value(c, action_type, payload) for c in cols]
    with self._connect() as conn:
        cur = conn.execute(sql, values)
        conn.commit()
        return int(cur.lastrowid)


def _start_action_value(col: ColumnDef, action_type: str, payload: dict) -> Any:
    """Compute the INSERT value for one column on start_action."""
    if col.name == "action_type":   return action_type
    if col.name == "status":        return "running"
    if col.name == "started_at":    return _now_iso()
    if col.name == "payload_json":  return json.dumps(payload)
    if col.name in ("ended_at", "duration_s", "error"):  return None
    return payload.get(col.name)  # hot fields + remaining initial fields
```

#### `complete_action`

```python
def complete_action(self, action_id, *, result_fields=None, payload_update=None):
    """UPDATE row to 'complete'; merge result_fields + payload_update."""
    self._check_writable()
    result_fields = result_fields or {}
    ended = _now_iso()

    with self._connect() as conn:
        row = conn.execute(
            f"SELECT started_at, payload_json FROM {_TABLE} WHERE id=?",
            (action_id,),
        ).fetchone()
        if row is None:
            raise NotFoundError(...)

        started = datetime.fromisoformat(row["started_at"])
        duration = (datetime.fromisoformat(ended) - started).total_seconds()
        payload = json.loads(row["payload_json"] or "{}")
        if payload_update:
            payload.update(payload_update)

        # SET clause: hardcoded ones first, then COALESCE for every hot field.
        set_parts = ["status='complete'", "ended_at=?", "duration_s=?"]
        for col in _COLUMNS:
            if col.hot:
                set_parts.append(f"{col.name}=COALESCE(?,{col.name})")
        set_parts.append("payload_json=?")

        values = [ended, duration]
        values.extend(result_fields.get(c.name) for c in _COLUMNS if c.hot)
        values.extend([json.dumps(payload), action_id])

        conn.execute(
            f"UPDATE {_TABLE} SET {', '.join(set_parts)} WHERE id=?",
            values,
        )
        conn.commit()
```

#### `_build_where`

```python
def _build_where(self, criteria: RunHistoryCriteria) -> tuple[str, tuple]:
    """Translate criteria into WHERE clause + params.

    For filterable columns: emit `col = ?` when criteria has a non-None value.
    Special-case clauses (text, date_from/to, ids, action_types) still hand-written.
    """
    # ... cross-field validation unchanged ...

    clauses = ["status != 'reserved'"]
    params: list = []

    # Generic equality clauses for every filterable field.
    for col_name in _FILTERABLE_FIELDS:
        if col_name == "action_type":
            # special: action_types is a list filter
            if criteria.action_types:
                placeholders = ",".join("?" for _ in criteria.action_types)
                clauses.append(f"action_type IN ({placeholders})")
                params.extend(criteria.action_types)
            continue
        if col_name == "source_album":
            # special: criteria field is named "album" (legacy)
            if criteria.album:
                clauses.append("source_album = ?")
                params.append(criteria.album)
            continue
        value = getattr(criteria, col_name, None)
        if value is not None:
            clauses.append(f"{col_name} = ?")
            params.append(value)

    # Special cases: text, date range, ids — unchanged
    # ...
    return " AND ".join(clauses), tuple(params)
```

### 4.5 Drift guard test

```python
# tests/architecture/test_run_history_repo_column_registry.py

def test_runrow_fields_match_columns_registry():
    """Every column in _COLUMNS has a matching RunRow field (and vice versa
    for the columns RunRow exposes)."""
    from face_cluster.repositories.run_history_repo import _COLUMNS
    from face_cluster.run_history import RunRow

    column_names = {c.name for c in _COLUMNS}
    runrow_fields = {f.name for f in dataclasses.fields(RunRow)}

    missing_in_runrow = column_names - runrow_fields
    assert not missing_in_runrow, (
        f"_COLUMNS declares {sorted(missing_in_runrow)!r} but RunRow doesn't "
        "expose them. Add fields to RunRow."
    )


def test_filterable_columns_have_matching_criteria_fields():
    """Every filterable column has a corresponding RunHistoryCriteria field
    (or a documented special case)."""
    # Validates the _build_where generic loop doesn't reference a criteria
    # field that doesn't exist.
    ...


def test_initial_and_migration_partition_is_complete():
    """Every column is either initial=True or initial=False (no missing flag).
    Initial columns are in the CREATE; migration columns are in the ALTER list."""
    ...


def test_only_nullable_hot_fields():
    """Hot fields must be nullable — start_action sets them to NULL when
    they're absent from the payload, and complete_action COALESCEs them
    over NULL inputs. A NOT NULL hot field would crash."""
    ...
```

---

## 5. Test plan

Three layers per spec-042 §A6:

### 5.1 Repository layer (regression — 33 existing tests, unchanged)

The 33 cases in `tests/face_clustering/repositories/test_run_history_repo_synthetic.py` are the contract guard. They must continue to pass with zero modifications. If any test breaks, the refactor changed behavior — back out, find the bug.

### 5.2 Service layer (regression — 31 + 4 existing tests)

`tests/face_clustering/views/test_history_service_*.py` — same story. The Service composes the Repository; the Repository's external interface is unchanged; the Service tests must pass unchanged.

### 5.3 Architecture (NEW — 4+ tests)

`tests/architecture/test_run_history_repo_column_registry.py`:

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_runrow_fields_match_columns_registry` | `RunRow` field set ⊇ `_COLUMNS` names. Catches "added column to registry, forgot RunRow". |
| 2 | `test_filterable_columns_have_matching_criteria_fields` | Every `filterable=True` column has a criteria field (or documented special case). Catches "marked filterable, forgot to add to RunHistoryCriteria". |
| 3 | `test_initial_and_migration_partition_is_complete` | Every column has its `initial` flag set; migration list and CREATE list don't overlap. |
| 4 | `test_only_nullable_hot_fields` | All `hot=True` columns are `nullable=True`. Catches the "NOT NULL hot field crashes on insert" case. |

### 5.4 Migration parity test (Phase 2 only)

`test_generated_create_table_matches_legacy_create_sql` — temporary test that asserts the new `_create_table_sql()` produces a schema-equivalent DB to the old hand-written `_CREATE_SQL`. Once Phase 3 swaps the constants, this test becomes redundant and gets removed. Insurance against introducing schema drift during the refactor.

### 5.5 Total surface

- **Existing**: 33 Repository + 35 Service + 3 layering arch + 3 repository arch = **74 tests pass unchanged**.
- **NEW**: 4 column-registry arch tests.
- **Temporary** (Phase 2 only, removed in Phase 7): 1 migration-parity test.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The refactor introduces a schema drift (CREATE / ALTER generates different DDL than hand-written). | Phase 2 ships the migration-parity test BEFORE Phase 3 swaps the constants. New schema must be byte-equivalent or test fails. |
| `_build_where`'s generic loop introduces a SQL injection vector via column names. | `_FILTERABLE_FIELDS` is derived from `_COLUMNS`, which is module-internal. No user input reaches column names. Safe. |
| The arch test `test_runrow_fields_match_columns_registry` is too strict (catches legitimate cases where RunRow has extra fields). | Test asserts `RunRow ⊇ _COLUMNS`, not equality — RunRow can have extra fields. Direction matters. |
| `RunHistoryCriteria` field name doesn't match column name (e.g., `criteria.album` ↔ column `source_album`). | Document these as explicit special cases in `_build_where`. The arch test allows declared exceptions. |
| Iterating `_COLUMNS` in hot paths slows queries. | The list has 24 items; the iteration adds microseconds. Existing 33-test suite runs in 1.4s; will measure after refactor. |

---

## 7. Phases — build → test → migrate

Standard discipline per spec-043 / A6. Each phase ends with all 74 existing tests passing.

### Phase 0 — None

No prerequisites; everything lands in `face_cluster/repositories/run_history_repo.py`.

### Phase 1 — Build (~45 min)

- [ ] **T001** Add `ColumnDef` dataclass to `face_cluster/repositories/run_history_repo.py`.
- [ ] **T002** Add `_COLUMNS: list[ColumnDef]` registry (the 24-row source of truth).
- [ ] **T003** Add `_create_table_sql()` function (replaces hand-written `_CREATE_SQL`).
- [ ] **T004** Add `_MIGRATION_COLUMNS_NEW` (parallel to existing `_ALTER_COLUMNS`; named "NEW" so we can compare in Phase 2).
- [ ] **T005** Add `_HOT_FIELDS_NEW` and `_FILTERABLE_FIELDS` (`_FILTERABLE_FIELDS` is genuinely new).

At this point, the registry exists and the generated constants are computed BUT no Repository method uses them. The 33 existing tests continue to pass against the unchanged code paths.

**Checkpoint**: Module imports cleanly. `_create_table_sql()` returns a valid SQL string. No behavior change.

### Phase 2 — Parity test (~30 min)

- [ ] **T010** Write `tests/architecture/test_run_history_repo_column_registry.py` with the 4 drift-guard tests from §4.5.
- [ ] **T011** Add a TEMPORARY migration-parity test: assert `_create_table_sql()` produces a schema-equivalent DB to the hand-written `_CREATE_SQL` (open both, run PRAGMA table_info, compare).
- [ ] **T012** Add a TEMPORARY assertion: `_MIGRATION_COLUMNS_NEW == _ALTER_COLUMNS` (list-equal).
- [ ] **T013** Add a TEMPORARY assertion: `_HOT_FIELDS_NEW == _HOT_FIELDS` (frozenset-equal).
- [ ] **T014** Run: `pytest tests/architecture/test_run_history_repo_column_registry.py -v`. All 7 (4 + 3 temporary) pass.

**Checkpoint**: The generated constants are PROVEN equivalent to the hand-written ones. Safe to swap.

### Phase 3 — Migrate constants (~15 min)

- [ ] **T020** Delete the hand-written `_CREATE_SQL`. Replace `init_table` / `_migrate` references to call `_create_table_sql()`.
- [ ] **T021** Delete the hand-written `_ALTER_COLUMNS`. Rename `_MIGRATION_COLUMNS_NEW` → `_MIGRATION_COLUMNS`. Update `_migrate` to iterate the new list.
- [ ] **T022** Delete the hand-written `_HOT_FIELDS`. Rename `_HOT_FIELDS_NEW` → `_HOT_FIELDS`.
- [ ] **T023** Remove the 3 temporary parity assertions (T012-T013) from the arch test — they're now tautological.
- [ ] **T024** Run: `pytest tests/face_clustering/repositories/ tests/face_clustering/views/ tests/architecture/test_run_history_repo_column_registry.py -v`. All 33 + 35 + 4 + 1 = 73 pass.

**Checkpoint**: The constants are now generated. The Repository's behavior is unchanged (proven by the 33 existing tests).

### Phase 4 — Migrate methods (~45 min)

- [ ] **T030** Rewrite `start_action` to iterate `_COLUMNS` and compute INSERT values via `_start_action_value(col, action_type, payload)`.
- [ ] **T031** Rewrite `complete_action`'s UPDATE SET clause + values to iterate `_COLUMNS` where `c.hot`.
- [ ] **T032** Rewrite `_build_where` to iterate `_FILTERABLE_FIELDS` for the generic equality clauses. Keep the special cases (`text`, `date_from`/`date_to`, `ids`) hand-written.
- [ ] **T033** Run: full surface (`pytest tests/face_clustering/repositories/ tests/face_clustering/views/ tests/architecture/ -q`). All 74 pass.

**Checkpoint**: The 9 → 2 touch points goal is reached for new column additions.

### Phase 5 — Remove migration-parity temporary (~5 min)

- [ ] **T040** Delete the migration-parity test added in T011. With the constants gone, there's nothing to compare against.
- [ ] **T041** Final test sweep: `pytest tests/ -q`. All pass.

**Checkpoint**: Permanent arch tests remain (4 drift guards); temporary verifications retired.

### Phase 6 — Docs + close-out (~30 min)

- [ ] **T050** Update `face_cluster/repositories/run_history_repo.py` module docstring with a "Column Registry" section explaining the pattern.
- [ ] **T051** Update `docs/architecture/architecture_standards.md` §B0: add a "Column Registry" sub-section noting the 2-touch-point rule + arch test.
- [ ] **T052** Update `specs/042-fc-app-v2-tab-parity/ARCHITECTURE_STANDARDS.html` similarly.
- [ ] **T053** Append `CHANGES_LOG.md` entry.
- [ ] **T054** Flip spec status `Draft` → `Code Review` → `Implemented`.

**Checkpoint**: Pattern documented as a binding standard for future Repositories.

### Total effort

**~2.5–3 hours**. Each phase is independently revertible.

---

## 8. Definition of Done

- [ ] `_COLUMNS: list[ColumnDef]` registry is the single source of truth for action_log columns.
- [ ] `_CREATE_SQL`, `_ALTER_COLUMNS`, `_HOT_FIELDS` constants no longer exist as hand-written declarations.
- [ ] `start_action`, `complete_action`, `_build_where` iterate `_COLUMNS` instead of repeating column names inline.
- [ ] Adding a column requires editing exactly **2 places**: the `_COLUMNS` list + the `RunRow` dataclass.
- [ ] 4 new drift-guard arch tests pass.
- [ ] All 74 existing tests (33 Repository + 35 Service + 6 architecture) continue to pass with zero modification.
- [ ] `architecture_standards.md` §B0 references the Column Registry pattern as binding for new Repositories.
- [ ] Spec status → **Implemented**.

---

## 9. Open questions

1. **Should `ColumnDef` live in `face_cluster/repositories/_column_def.py` (sibling to `_errors.py`) for reuse by future Repositories, or stay inside `run_history_repo.py` until proven on a second consumer?** Recommended: stay inline. Extract when the second Repository materializes. Avoids premature abstraction.

2. **Should the criteria-field-name mapping (`criteria.album` ↔ column `source_album`) be data-driven via `ColumnDef.criteria_alias: Optional[str]`?** Recommended: not yet. One alias is fine as a special case in `_build_where`; if a third alias shows up, lift it.

3. **`payload_json` is not `hot=True` because it has special serialization (JSON-dumped from a dict, not passed through). Should it have its own `payload=True` flag?** Recommended: no special flag. Two SQL-special-cases (`payload_json` and the lifecycle columns `started_at`/`ended_at`/`duration_s`/`status`) are hand-written in `_start_action_value`. Worth it to avoid over-flexing the registry.

4. **Should `_create_table_sql()` also emit indexes?** Today the indexes are created via separate `CREATE INDEX IF NOT EXISTS` statements in `_migrate`. Recommended: keep that separation — `ColumnDef` is about columns, not index strategy. Indexes are sparse enough to stay hand-written.
