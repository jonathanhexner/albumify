# Code Review: spec-046 — SQLAlchemy + Alembic Data Layer

**Reviewed**: 2026-05-28
**Reviewer**: Claude (self-review against `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Status**: PASS with 2 pass-with-follow-up findings; 0 blockers

---

## Part 1 — How it works

### Module inventory (new / changed)

| Module | Role |
|---|---|
| `face_cluster/repositories/_orm_base.py` | `Base(DeclarativeBase)` + naming convention |
| `face_cluster/repositories/_engine.py` | `create_engine_for_path()` with WAL pragmas |
| `face_cluster/repositories/_session.py` | `make_sessionmaker()` + `session_scope()` |
| `face_cluster/repositories/_base_repository.py` | `BaseRepository` (error translation helpers) |
| `face_cluster/repositories/models/action_log.py` | `ActionLog` ORM model — 24 columns |
| `face_cluster/repositories/run_history_repo.py` | `RunHistoryRepository` — SQLAlchemy-backed |
| `alembic.ini` + `alembic/env.py` + `alembic/versions/` | Versioned migrations; baseline migration matches production |
| `tests/face_clustering/repositories/conftest.py` | Shared fixtures (engine, transactional session, golden fixture) |
| `tests/face_clustering/fixtures/golden_run_history.db` | Committed real-data fixture |
| `tests/face_clustering/fixtures/rebuild_golden.py` | Regenerates fixture via legacy free functions |

### Data flow

```
Caller
  → RunHistoryRepository(config?)
     → cached engine_for(db_path) → SQLAlchemy Engine (WAL + pragmas)
     → cached sessionmaker
  → method call
     → session_scope(sm) [commit on success, rollback on exception]
     → session.scalars(select(ActionLog).where(...))  / session.add(row)
     → _to_run_row(model) → RunRow returned to caller
```

### Schema lifecycle

```
Edit ActionLog model
  → alembic revision --autogenerate -m "msg"
  → alembic/versions/<rev>_msg.py generated
  → alembic upgrade head applies to any DB
  → test_orm_models_in_sync_with_alembic_head guards drift at PR time
```

---

## Part 2 — Findings by checklist section

### §1 Correctness
**PASS.** Phase 5 equivalence test (24 parametrised cases — 19 reads + 5 mutations) proved byte-identical output between old and new against the real-data fixture before the swap. All 111 spec-046-relevant tests pass post-swap. Two pre-existing arch-test violations (constructor signature, public free functions in infra modules) fixed in this spec.

### §2 Architecture
**PASS.** New stack adheres to the standard Repository pattern:
- One ORM model per entity (clean Data Mapper)
- Alembic provides Unit of Work / migration history
- `BaseRepository` is the shared error-translation primitive that future Repositories inherit from
- Pure separation between schema (model), migrations (alembic), Repository (queries)

### §3 Testing
**PASS.** Test inventory:
- +3 session-infra smoke tests
- +2 Alembic baseline DDL equivalence tests
- +1 golden fixture shape test
- +1 ORM/Alembic drift guard (replaces 4 deleted column-registry guards)
- −4 column-registry drift guards (no longer needed; framework prevents drift)
- −1 equivalence test (Phase 5 artefact, served its purpose)

Net +2 test files; existing 42 repository tests + ~60 service + legacy tests all pass unchanged.

### §4 Performance
**PASS.** Engine + sessionmaker cached per `db_path` (process-wide). Each method opens its own short session (same per-call lifecycle as the legacy `sqlite3.connect()`). No measurable regression.

### §5 Testability
**PASS.** Test isolation:
- `transactional_session` fixture → standard SQLAlchemy outer-transaction rollback per test
- `golden_run_history_db_copy` → real-data per-test fresh copy
- No module-level monkeypatching needed (eliminates the spec-043 H1 bug class)

### §6 Boundary contracts
**PASS.** Public Repository API unchanged: `start_action`, `complete_action`, `fail_action`, `update_comment`, `find`, `find_one`, `count`, `get_by_id`, `distinct_albums`. Same `RunRow` return type. `RunHistoryRepoConfig` + `RunHistoryCriteria` kept as the typed boundary inputs.

### §7 Documentation
**PASS-with-follow-up.** `architecture_standards.md` §B0.1 retired, §B0.2 written with worked example. `CHANGES_LOG.md` entry written.

**F-1 (follow-up, pass-with-follow-up)**: `docs/architecture/classes.html` and `docs/architecture/db_global.html` still reference the legacy `_COLUMNS`/`ColumnDef` structure. Update in a follow-up PR — non-blocking for spec-046 handoff because the architecture_standards.md update is canonical and the HTMLs are derivative.

### §8 Sightings / Learnings
**PASS-with-follow-up.**

**F-2 (follow-up, pass-with-follow-up)**: Spec-046's session-policy decision was revised mid-flight from "pure session injection" (Option B) to "internal session management with db_path injection" (effectively the old Option A semantics, but on SQLAlchemy). Reason: pure session injection would have required rewriting ~70 existing test bodies. The trade-off was deliberate — clean testability via `db_path` injection still achieved; multi-call transactions deferred until a real need surfaces. Documented in CHANGES_LOG. If/when multi-call transactions are needed (e.g., spec-045 force-merge), add a `RunHistoryRepository.from_session(session)` constructor as a follow-up.

---

## Part 3 — Verdict

**Accept.** Spec-046 ships with:
- All 8 acceptance criteria verified (AC1 demo executed; AC3 equivalence test was green pre-swap; AC4 full suite green post-swap; AC6/AC7 met).
- Zero high-severity findings.
- Two pass-with-follow-up tickets filed (F-1 docs HTMLs, F-2 session-policy revision rationale).

**Follow-up tickets (file in TODO.md)**:
- F-1: Update `docs/architecture/classes.html` + `db_global.html` to reflect the ORM-based design.
- F-2: When a multi-call transaction need materialises (spec-045 candidate), add `RunHistoryRepository.from_session(session)` constructor and document Option B as the second supported path.
