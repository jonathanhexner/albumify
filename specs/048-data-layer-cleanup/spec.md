# spec-048 — Data Layer Cleanup (spec-046 follow-ups)

**Created**: 2026-05-28
**Status**: Implemented
**Predecessors**: spec-046 (SQLAlchemy + Alembic data layer)
**Estimated effort**: 1 focused day

---

## Problem

spec-046 shipped a working SQLAlchemy/Alembic stack but carried six smells that the post-implementation audit (`CODE_AUDIT.html`) and a user code-walk surfaced. None block functionality; together they undermine the "boring, standard, safe-to-modify" goal that motivated spec-046 in the first place.

This spec fixes all six in one pass before they harden into permanent debt — in particular before the **2026-06-08** legacy-shim deletion, which would otherwise break the golden-fixture rebuild script.

---

## What's wrong today

| # | Smell | Where | Why it's wrong |
|---|---|---|---|
| S1 | `subprocess.run([".venv/Scripts/alembic.exe", "upgrade", "head"])` | `run_history_repo.py:_ensure_schema` | Hardcoded venv path; non-portable; spawns a subprocess for a pure-Python op; can't be mocked. Alembic exposes `alembic.command.upgrade(cfg, "head")`. |
| S2 | `_to_run_row()` — 24-line manual ORM→dataclass mapping | `run_history_repo.py` | Two parallel definitions of the same shape (`ActionLog` columns ≡ `RunRow` fields) with no link. Silent drift on any column add/rename. |
| S3 | `_HOT_FIELDS = ("run_id", "status", ...)` tuple | `run_history_repo.py` | Name means nothing; this is column metadata ("updated on action completion") that belongs *on the column*, not in a sibling tuple. |
| S4 | `rebuild_golden.py` imports legacy free functions | `tests/face_clustering/fixtures/rebuild_golden.py` | Locked to `face_cluster.run_history_db` which is scheduled for deletion **2026-06-08**. After that date the fixture can't be regenerated. |
| S5 | Module-level `_engine_cache: dict[Path, ...]` | `run_history_repo.py` | Reinvents what SQLAlchemy's connection pool already does. Exists only to amortize S1's subprocess cost. Once S1 is fixed, dispensable. Also shared mutable state that test isolation has to fight. |
| S6 | `repo_root` computed inline; `get_db_path()` duplicated | `run_history_repo.py`, `run_history_db.py`, `training_db.py` | Path resolution belongs in one place, not inlined in every module that needs it. |

---

## What we build

| Module | Role | New / changed |
|---|---|---|
| `face_cluster/_paths.py` | Single source of truth: `repo_root()`, `sim_bench_data_dir()`, `face_history_db_path()`, `alembic_ini_path()` | **NEW** |
| `face_cluster/repositories/_schema.py` | `ensure_schema(db_path)` — calls `alembic.command.upgrade()` in-process | **NEW** (replaces `_ensure_schema` block in repo) |
| `face_cluster/repositories/models/action_log.py` | Move "hot field" flag onto each `mapped_column(..., info={"updated_on_complete": True})`. Add `ActionLog.hot_field_names()` classmethod. | **CHANGED** |
| `face_cluster/repositories/run_history_repo.py` | Drop `_engine_cache`, `_HOT_FIELDS`, `_to_run_row`, inline `repo_root`, subprocess. Use `_paths`, `_schema`, `RunRow.from_orm()`. | **CHANGED** (net −60 LOC est.) |
| `face_cluster/run_history.py` | Add classmethod `RunRow.from_orm(model: ActionLog)` doing the 1-line `cls(**{c.name: getattr(model, c.name) for c in ActionLog.__table__.columns})` mapping, with `source_album` fallback preserved. | **CHANGED** |
| `tests/face_clustering/fixtures/rebuild_golden.py` | Use `RunHistoryRepository` instead of legacy free functions. | **CHANGED** |
| `tests/architecture/test_runrow_matches_action_log.py` | Asserts `set(RunRow.__dataclass_fields__) == set(ActionLog.__table__.columns.keys())`. Drift guard for S2. | **NEW** |
| `tests/architecture/test_paths_module_sole_owner.py` | Asserts no other module computes `repo_root` or `Path.home() / ".sim_bench"` inline. | **NEW** |

---

## What we don't build

- New centralized path module for **non-DB** paths (run outputs, config dirs). Out of scope; only the paths spec-046 inlined.
- Refactoring `training_db.py` to be SQLAlchemy-backed. We **do** consolidate its `get_db_path()` into `_paths`, but the rest of `training_db.py` stays untouched (separate spec if needed).
- Changes to `RunHistoryRepository`'s public API. Same methods, same signatures.

---

## Locked decisions

1. **Path module name**: `face_cluster/_paths.py` (underscore prefix matches the spec-046 convention for infrastructure modules: `_orm_base`, `_engine`, `_session`, `_base_repository`).
2. **`alembic.command.upgrade()` in-process** — no subprocess, no environment-variable handoff. Pass `sqlalchemy.url` programmatically via `cfg.set_main_option(...)`.
3. **`RunRow.from_orm()` is the one mapping point** — generated from `ActionLog.__table__.columns`. The `source_album` `_UNKNOWN_ALBUM` fallback is preserved as a 1-line post-processing step on the dataclass.
4. **Drop `_engine_cache` entirely** — `RunHistoryRepository.__init__` creates its own engine. SQLAlchemy's per-engine `QueuePool` (or `SingletonThreadPool` for SQLite) does the connection reuse. If profiling later shows engine creation is hot, revisit with a single `lru_cache`'d factory in `_engine.py` — but not pre-emptively.
5. **`info={"updated_on_complete": True}`** — SQLAlchemy's standard mechanism for column-level metadata. `ActionLog.hot_field_names()` reads back from `__table__.columns`. No parallel tuple.
6. **No public API changes** — all six fixes are internal. Consumers (`history.py` service, `pipeline.py`, tests) don't recompile.

---

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | No `subprocess` import in `face_cluster/repositories/` | Phase 1 + grep guard test |
| AC2 | No `.venv` or `alembic.exe` string literal anywhere in `face_cluster/` | grep guard in Phase 1 |
| AC3 | `RunRow.from_orm(action_log)` returns byte-identical output to old `_to_run_row(action_log)` across the 24-row golden fixture | Phase 4 parametrized test |
| AC4 | `set(RunRow.__dataclass_fields__) == set(ActionLog.__table__.columns.keys())` | Phase 4 arch test (new) |
| AC5 | `ActionLog.hot_field_names()` returns the same tuple `_HOT_FIELDS` did, in the same order | Phase 3 test |
| AC6 | `rebuild_golden.py` imports zero symbols from `face_cluster.run_history_db` or `face_cluster.run_history` | Phase 5 + grep guard |
| AC7 | `_engine_cache` deleted; no module-level mutable dict in `face_cluster/repositories/` | Phase 2 + arch test |
| AC8 | `repo_root`, `Path.home() / ".sim_bench"`, and `face_history.db` literals appear only inside `face_cluster/_paths.py` | Phase 0 arch test (new) |
| AC9 | Full test suite green: `pytest tests/face_clustering tests/architecture` | Phase 6 |
| AC10 | `/code-review` produces `REVIEW.md` with zero high-severity findings | Phase 7 |

---

## Risks

| Risk | Mitigation |
|---|---|
| `alembic.command.upgrade()` behaves differently from CLI (logging, config search) | Phase 1 test runs both against a fresh DB and asserts identical resulting schema |
| Dropping `_engine_cache` slows repeated repo construction | Benchmark in Phase 2 (one micro-bench, target < 5 ms construction). Acceptable on the History tab (one construction per page render). |
| `RunRow.from_orm` breaks if column types diverge from dataclass field types | Phase 4 byte-equivalence test catches this; runs on real golden data |
| Centralized path module shadows test monkey-patching | `_paths` functions are pure (no module-level constants); tests can pass an explicit `db_path` to the Repository (already the case) |

---

## Definition of Done

All 10 acceptance criteria green + `REVIEW.md` filed + `CHANGES_LOG.md` entry + spec-046's `CODE_AUDIT.html` SMELL/CONTRACT rows annotated "resolved by spec-048".
