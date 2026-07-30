# Tasks: Per-run ORM models (058)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

## Phase 0 — Drift guard skeleton (~30 min)
- [ ] **T001** Create `tests/face_clustering/db/test_orm_matches_schema.py`. Two helper functions: `create_from_ddl(tmp_path)` (today's `executescript(SCHEMA_DDL)`) and `create_from_orm(tmp_path)` (placeholder; stub).
- [ ] **T002** Write the comparator: open both DBs, query `sqlite_master` for table/column/index definitions, normalize whitespace, assert equal. **Initially fails** — that's the gate Phase 1 closes.

## Phase 1 — Mirror the 9 tables (~3 h)
- [ ] **T010** Create `sim_bench/run_db/models/_base.py`: `class Base(DeclarativeBase)` + naming convention (`ix_%(column_0_label)s`, `uq_%(table_name)s_%(column_0_name)s`, etc.).
- [ ] **T011** `face.py` — `Face` model mirroring `FACES_DDL` (28 columns including spec-040 `*_ratio` additions). Run drift-guard test; iterate until columns match exactly.
- [ ] **T012** Repeat T011 for: `cluster.py`, `cluster_assignment.py`, `merge_decision.py`, `filter_decision.py`, `image.py`, `scene_cluster.py`, `scene_cluster_assignment.py`, `run_metadata.py`.
- [ ] **T013** Indexes from `INDEXES_DDL` move to `__table_args__` on the appropriate models.

**Gate**: drift-guard test green for all 9 tables + indexes.

## Phase 2 — Flip source of truth (~1 h)
- [ ] **T020** Rewrite `sim_bench/run_db/_schema.py`: `SCHEMA_DDL` is now generated from `Base.metadata.create_all()` (string-extracted via `CreateTable` compiler). DDL constants stay defined for backward compat but are derived, not authored.
- [ ] **T021** Verify spec-046's pattern works: `SCHEMA_VERSION = 5` constant stays hand-maintained; bump on breaking change.
- [ ] **T022** Update drift-guard test to assert: **if you edit `schema.py`'s DDL strings by hand instead of regenerating, the test fails.** Mutation test: introduce a typo in `FACES_DDL`, confirm test red.

**Gate**: `pytest tests/face_clustering/db/` green.

## Phase 3 — Audit existing callers (~30 min)
- [ ] **T030** grep for `from sim_bench.run_db._schema import` — list every caller that imports a DDL constant. Confirm they all keep working unchanged (the DDL strings are still exported, just derived now).
- [ ] **T031** `RunExporter` calls `executescript(SCHEMA_DDL)` to bootstrap. Verify: this still uses the derived DDL and produces the same tables.
- [ ] **T032** `RunStore`'s `PRAGMA user_version` check stays — no change needed; still asserts schema version match.

**Gate**: `pytest tests/face_clustering/ -q` → no regression. spec-057 equivalence test (golden hashes) still green.

## Phase 4 — Cleanup + close-out (~1 h)
- [ ] **T040** Update `docs/architecture/classes.html` — add §"ORM models (per-run DB)" subsection.
- [ ] **T041** Update `docs/architecture/db_schemas.html` — note that per-run schema is now derived from ORM models.
- [ ] **T042** Update `docs/architecture/architecture_standards.md` §B0 — distinguish (a) full SQLAlchemy + Alembic for long-lived DBs (spec-046), (b) SQLAlchemy ORM-only for ephemeral per-run DBs (this spec).
- [ ] **T043** CHANGES_LOG entry.
- [ ] **T044** `/code-review` → REVIEW.md.
- [ ] **T045** Spec status → `Implemented`. Commit + push.

## Total estimate
**~4-6 hours.** Phase 1 dominates (one ORM model per ~20 min, including drift-guard validation cycle).
