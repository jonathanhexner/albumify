# Tasks: SQLAlchemy + Alembic Data Layer (046)

**Status**: Implemented
**Estimated effort**: 2 focused days

---

## Design notes

- **D1** — SQLAlchemy 2.0 typed API: `Mapped[T]`, `mapped_column()`, `select()`. No 1.x syntax.
- **D2** — Caller-injected `Session`. Factory `make_run_history_repo(db_path)` for callers that want a ready-to-use Repository.
- **D3** — Configure `MetaData(naming_convention=...)` on `Base` so Alembic autogenerate output is deterministic.
- **D4** — Golden fixture = real pipeline run against `D:\sim-bench\test_data\face_clustering`, committed to git (~50 KB).
- **D5** — `v2` suffix lives Phases 5–6 only. Gone in final state.
- **D6** — Legacy free-function modules become ≤30 LOC delegators routing to new Repository.
- **D7** — One arch test (`alembic check`) replaces four deleted drift-guard tests.

---

## Phase 0 — Dependencies

- [ ] T001 Add `sqlalchemy>=2.0.30,<2.1` and `alembic>=1.13,<2` to `pyproject.toml`; `pip install -e .`

**Check**: `.venv/Scripts/python -c "import sqlalchemy, alembic"` succeeds.

---

## Phase 1 — Foundation (engine, session, ORM base)

- [ ] T010 Create `_orm_base.py` — `Base(DeclarativeBase)` + naming convention
- [ ] T011 Create `_engine.py` — `create_engine_for_path(db_path)` with WAL + foreign_keys + busy_timeout pragmas
- [ ] T012 Create `_session.py` — `make_sessionmaker(engine)` + `session_scope(sm)` context manager (commit on success, rollback on exception)
- [ ] T013 Test `tests/face_clustering/repositories/test_session_infra.py`:
  - pragmas applied
  - session_scope commits on success
  - session_scope rolls back on exception

**Check**: 3/3 pass.

---

## Phase 2 — Alembic setup

- [ ] T020 `alembic init alembic`
- [ ] T021 Configure `alembic.ini` (script_location, file_template)
- [ ] T022 Configure `alembic/env.py` to import `Base.metadata`; read DB URL from `SIM_BENCH_DB_URL`
- [ ] T023 Create empty `face_cluster/repositories/models/__init__.py`

**Check**: `.venv/Scripts/alembic.exe history` runs without error.

---

## Phase 3 — ActionLog model + baseline migration

- [ ] T030 Create `models/action_log.py` — `ActionLog(Base)` with all 24 current columns
- [ ] T031 Import `ActionLog` in `models/__init__.py`
- [ ] T032 `alembic revision --autogenerate -m "baseline action_log"`. Inspect generated file.
- [ ] T033 Test `test_alembic_baseline.py`:
  - `alembic upgrade head` on empty DB → schema matches legacy `_create_table_sql() + _migrate()` output
  - `alembic stamp head` on populated DB is idempotent (no data change, version table added)

**Check**: 2/2 pass.

---

## Phase 4 — Golden fixture from real data

- [ ] T040 Write `tests/face_clustering/fixtures/rebuild_golden.py` — runs legacy pipeline against `D:\sim-bench\test_data\face_clustering`, seeds edge-case rows (NULL hot fields, unicode comment, parent/child pair), saves to `golden_run_history.db`
- [ ] T041 Execute rebuild; commit `golden_run_history.db` to git
- [ ] T042 Test `test_golden_fixture.py` — asserts row count, status diversity, edge cases present

**Check**: 1/1 passes; .db file committed.

---

## Phase 5 — New Repository (parallel) + equivalence test

- [ ] T050 Create `_base_repository.py` — constructor takes `Session`, helpers for criteria → `select()`, error translation (NoResultFound → NotFoundError, IntegrityError → ValidationError)
- [ ] T051 Create `run_history_repo_v2.py` — `RunHistoryRepositoryV2(BaseRepository)` with full public API
- [ ] T052 Add `make_run_history_repo_v2(db_path)` factory (context manager: engine + session + repo lifecycle)
- [ ] T053 Create `tests/face_clustering/repositories/conftest.py` — shared fixtures: `fresh_db_engine`, `transactional_session`, `golden_run_history_db_path`
- [ ] T054 Test `test_run_history_repo_equivalence.py` — parametrized:
  - **Reads**: `find`, `count`, `find_one`, `get_by_id` across 9 criteria combinations — assert `old(...) == new(...)`
  - **Mutations**: `start_action`, `complete_action`, `fail_action`, `update_comment` — each runs on a fresh copy, assert full-table dump equal

**Check**: all parametrized cases pass. **This is THE proof.**

---

## Phase 6 — Flip the public name

- [ ] T060 `repositories/__init__.py` re-exports `RunHistoryRepositoryV2 as RunHistoryRepository`
- [ ] T061 Update 6 consumer sites to use `make_run_history_repo(db_path)` factory:
  - `app/face_clustering_v2/pipeline.py` (×2)
  - `face_cluster/views/history.py`
  - 4 test files in `tests/face_clustering/{views,repositories}/`
- [ ] T062 Delete old `run_history_repo.py`
- [ ] T063 Rename `run_history_repo_v2.py` → `run_history_repo.py`; rename class `V2 → ` (drop suffix)

**Check**: `pytest tests/face_clustering tests/architecture` all green. **No new test added — existing tests are the oracle.**

---

## Phase 7 — Legacy shims + arch test swap

- [ ] T070 Rewrite `face_cluster/run_history_db.py` as ≤30 LOC of thin delegators
- [ ] T071 Rewrite `face_cluster/run_history.py` as ≤30 LOC of thin delegators (`RunRow` + `_row_to_run_row` stay)
- [ ] T072 Delete `tests/architecture/test_run_history_repo_column_registry.py` (4 drift guards no longer needed)
- [ ] T073 Add `tests/architecture/test_orm_models_in_sync_with_alembic.py` — single test running `alembic check`

**Check**: all legacy tests (`test_run_history_db.py`, `test_run_history.py`, `test_pipeline_history_hook.py`, `test_history_tab_data.py`) pass unchanged; new arch test passes.

---

## Phase 8 — Browser sanity

- [ ] T080 Add Playwright test that opens v2 History tab against golden fixture, asserts row count + one specific run_id visible
- [ ] T081 Manual verification: `scripts/restart_apps.bat` → open v2 History tab → confirm rows, filters, comment-edit all work

**Check**: Playwright test passes; manual smoke clean.

---

## Phase 9 — "Add a column" live demo (AC1)

- [ ] T090 Add `throwaway_demo: Mapped[str | None]` to `ActionLog`. Run `alembic revision --autogenerate`. Inspect migration. Run `alembic upgrade head`. Run `alembic downgrade -1`.
- [ ] T091 Revert: delete field + delete generated migration file. Repo state clean.

**Check**: demo confirms 2-file column add. AC1 satisfied.

---

## Phase 10 — Docs

- [ ] T100 `docs/architecture/architecture_standards.md`: retire §B0.1, add §B0.2 with worked example
- [ ] T101 `docs/architecture/classes.html`: replace `ColumnDef`/`RunHistoryRepository` rows with `ActionLog` ORM model + `BaseRepository` + `RunHistoryRepository`
- [ ] T102 `docs/architecture/db_global.html`: replace `_COLUMNS` references with Alembic-based migration story
- [ ] T103 `CHANGES_LOG.md` entry

**Check**: no stale `_COLUMNS`/`ColumnDef` references outside `specs/044-column-registry/` and CHANGES_LOG history.

---

## Phase 11 — Code review gate

- [ ] T110 Run `/code-review` → `REVIEW.md`
- [ ] T111 Resolve or waive high-severity findings
- [ ] T112 Flip spec status → Implemented

**Check**: REVIEW.md exists, no unresolved blockers.

---

## Test delta

| | Added | Deleted | Net |
|---|---|---|---|
| Phase 1 | 3 | 0 | +3 |
| Phase 3 | 2 | 0 | +2 |
| Phase 4 | 1 | 0 | +1 |
| Phase 5 | ~13 parametrized | 0 | +13 |
| Phase 7 | 1 | 4 | −3 |
| Phase 8 | 1 | 0 | +1 |
| **Total** | **~20** | **4** | **+17** |

Every new test covers a new layer (session, alembic), is the equivalence proof, or is a permanent drift guard. No redundant unit tests.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Phase 5 reveals undocumented legacy behavior | New Repository MUST match old. Behavior changes are a separate spec. |
| Production DB upgrade path | Phase 3 test #2 proves stamp-on-populated is idempotent. |
| Streamlit module cache hides errors | Phase 8 manual step uses `restart_apps.bat`. |
| Alembic env.py circular imports | env.py imports only leaf module `models/`. Phase 2 catches this. |
| Estimate slips past 2 days | Phases revertable individually. If Phase 5 blocks, ship the foundation (0–4) and re-spec the swap. |
