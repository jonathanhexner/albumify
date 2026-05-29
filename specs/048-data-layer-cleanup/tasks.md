# Tasks: spec-048 Data Layer Cleanup

**Status**: Implemented
**Estimated effort**: 1 focused day (8 phases)

Phases are ordered so each unlocks the next and tests stay green between phases.

---

## Phase 0 — Centralized paths module (fixes S6)

- [ ] T000 Create `face_cluster/_paths.py`:
  ```python
  from pathlib import Path
  from functools import lru_cache

  @lru_cache(maxsize=1)
  def repo_root() -> Path:
      return Path(__file__).resolve().parents[1]

  @lru_cache(maxsize=1)
  def sim_bench_data_dir() -> Path:
      d = Path.home() / ".sim_bench"
      d.mkdir(parents=True, exist_ok=True)
      return d

  def face_history_db_path() -> Path:
      return sim_bench_data_dir() / "face_history.db"

  def sim_bench_db_path() -> Path:
      return sim_bench_data_dir() / "sim_bench.db"

  def alembic_ini_path() -> Path:
      return repo_root() / "alembic.ini"
  ```
- [ ] T001 Update `face_cluster/run_history_db.py:get_db_path()` to delegate to `_paths.face_history_db_path()` (1-line body, kept for shim compat)
- [ ] T002 Update `face_cluster/training_db.py:get_db_path()` to delegate to `_paths.sim_bench_db_path()` (1-line body)
- [ ] T003 Add `tests/architecture/test_paths_module_sole_owner.py`:
  - Greps `face_cluster/**/*.py` for `Path.home() / ".sim_bench"` → only `_paths.py` allowed
  - Greps for `Path(__file__).resolve().parents` → only `_paths.py` allowed
  - Greps for hardcoded `"face_history.db"` / `"sim_bench.db"` → only `_paths.py` allowed

**Check**: arch test passes; existing tests still green.

---

## Phase 1 — In-process Alembic (fixes S1)

- [ ] T010 Create `face_cluster/repositories/_schema.py`:
  ```python
  from pathlib import Path
  from alembic.config import Config
  from alembic import command
  from face_cluster._paths import alembic_ini_path

  def ensure_schema(db_path: Path) -> None:
      cfg = Config(str(alembic_ini_path()))
      cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
      command.upgrade(cfg, "head")
  ```
- [ ] T011 Replace `_ensure_schema` block in `run_history_repo.py` with `ensure_schema(db_path)` call; delete `subprocess`, `os`, inline `repo_root` imports
- [ ] T012 Test `tests/face_clustering/repositories/test_schema_upgrade.py`:
  - Fresh empty DB → `ensure_schema()` → schema matches Alembic head
  - Already-at-head DB → `ensure_schema()` is idempotent (no error, no version-table change)
  - Compare resulting `sqlite_master` dump to a control DB upgraded via the legacy subprocess call → identical (proves API equivalence)
- [ ] T013 Add grep guard to existing `tests/architecture/test_repositories.py`:
  - No `subprocess` import in `face_cluster/repositories/**`
  - No `".venv"` or `"alembic.exe"` string in `face_cluster/**`

**Check**: 3 new tests pass; arch grep passes.

---

## Phase 2 — Drop `_engine_cache` (fixes S5)

- [ ] T020 Delete module-level `_engine_cache` dict in `run_history_repo.py`
- [ ] T021 Move engine + sessionmaker creation into `RunHistoryRepository.__init__`:
  ```python
  def __init__(self, config: Optional[RunHistoryRepoConfig] = None):
      self._config = config or RunHistoryRepoConfig()
      db_path = self._config.db_path or face_history_db_path()
      ensure_schema(db_path)
      self._engine = create_engine_for_path(db_path)
      self._sm = make_sessionmaker(self._engine)
  ```
- [ ] T022 Micro-benchmark in `tests/face_clustering/repositories/test_repo_construction_perf.py`:
  - Construct repo 50 times against fresh temp DB
  - Assert mean construction < 50 ms (loose ceiling; typical < 20 ms)
- [ ] T023 Add arch assertion: no module-level mutable dict named `_*_cache` in `face_cluster/repositories/**` (regex: `^_\w+_cache\s*[:=]\s*\{\}` at module scope)

**Check**: bench passes; arch test passes; existing test suite green.

---

## Phase 3 — Hot fields → column metadata (fixes S3)

- [ ] T030 In `models/action_log.py`, annotate each "hot" column with `info={"updated_on_complete": True}`:
  ```python
  status: Mapped[str | None] = mapped_column(String, info={"updated_on_complete": True})
  ended_at: Mapped[float | None] = mapped_column(Float, info={"updated_on_complete": True})
  duration_s: Mapped[float | None] = mapped_column(Float, info={"updated_on_complete": True})
  # ... same for the other 5 hot fields
  ```
- [ ] T031 Add `ActionLog.hot_field_names()` classmethod:
  ```python
  @classmethod
  def hot_field_names(cls) -> tuple[str, ...]:
      return tuple(
          c.name for c in cls.__table__.columns
          if c.info.get("updated_on_complete")
      )
  ```
- [ ] T032 In `run_history_repo.py`: delete `_HOT_FIELDS` tuple; replace all references with `ActionLog.hot_field_names()`
- [ ] T033 Test `tests/face_clustering/repositories/test_hot_fields.py`:
  - `ActionLog.hot_field_names()` returns exactly the legacy 8-element tuple, in the same order
  - Order matches column declaration order (stable)

**Check**: test passes; `_HOT_FIELDS` literal absent from codebase (grep).

---

## Phase 4 — Single source of truth: `RunRow.from_orm` (fixes S2)

- [ ] T040 In `face_cluster/run_history.py`, add classmethod:
  ```python
  @classmethod
  def from_orm(cls, model: ActionLog) -> "RunRow":
      values = {c.name: getattr(model, c.name) for c in ActionLog.__table__.columns}
      if not values.get("source_album"):
          values["source_album"] = _UNKNOWN_ALBUM
      return cls(**values)
  ```
- [ ] T041 In `run_history_repo.py`: delete `_to_run_row` function; replace 4 call sites with `RunRow.from_orm(model)`
- [ ] T042 Test `tests/architecture/test_runrow_matches_action_log.py`:
  - Asserts `set(RunRow.__dataclass_fields__.keys()) == set(ActionLog.__table__.columns.keys())`
  - Catches column add / field-name drift at PR time
- [ ] T043 Test `tests/face_clustering/repositories/test_runrow_from_orm_equivalence.py`:
  - Load all 24 rows from golden fixture as `ActionLog` instances
  - Assert `RunRow.from_orm(m) == _legacy_to_run_row(m)` for each
  - `_legacy_to_run_row` checked in as a frozen copy of the deleted function, just for this test (deleted in Phase 6)

**Check**: arch test green; 24 equivalence cases green.

---

## Phase 5 — Port `rebuild_golden.py` off legacy modules (fixes S4)

- [ ] T050 Rewrite `tests/face_clustering/fixtures/rebuild_golden.py`:
  - Replace `from face_cluster.run_history_db import start_action, complete_action, ...` with `from face_cluster.repositories.run_history_repo import RunHistoryRepository`
  - Same data seeded; same edge cases (NULL hot fields, unicode comment, parent/child pair)
- [ ] T051 Execute rebuild; diff resulting `.db` against the committed `golden_run_history.db`:
  - Schema identical
  - Row content identical (modulo timestamps for new rows — seed deterministic timestamps)
- [ ] T052 If diff is clean, commit the regenerated `.db`. If not, debug until clean (don't paper over).
- [ ] T053 Add grep guard to existing arch test: `rebuild_golden.py` imports nothing from `face_cluster.run_history_db` or `face_cluster.run_history` (except `RunRow` type, which is fine)

**Check**: regenerated fixture round-trips through `test_golden_fixture.py` green; arch guard passes.

---

## Phase 6 — Cleanup + test sweep (fixes AC9)

- [ ] T060 Delete the frozen `_legacy_to_run_row` from Phase 4's equivalence test (served its purpose)
- [ ] T061 Run full target test suite:
  ```
  .venv/Scripts/python -m pytest tests/face_clustering tests/architecture -v
  ```
- [ ] T062 Run unrelated suites that touch the data layer:
  ```
  .venv/Scripts/python -m pytest tests/views tests/test_pipeline_history_hook.py tests/test_history_tab_data.py -v
  ```
- [ ] T063 Manual smoke: `scripts/restart_apps.bat`, open v2 History tab against real data, confirm rows render, filters work, comment edits persist.

**Check**: all green; manual smoke clean.

---

## Phase 7 — Code review gate (fixes AC10)

- [ ] T070 Run `/code-review` against the spec-048 branch — produces `specs/048-data-layer-cleanup/REVIEW.md`
- [ ] T071 Walk every section (§1–§8) of `docs/guides/CODE_REVIEW_CHECKLIST.md`:
  - §1 Correctness: AC1-AC8 green, equivalence tests passed
  - §2 Architecture: confirm no new module-level mutable state, no new cross-layer leaks
  - §3 Testing: confirm new tests are drift guards (AC4) or equivalence proofs (AC3, AC5), not redundant unit tests
  - §4 Performance: confirm Phase 2 bench in REVIEW
  - §5 Testability: confirm test fixtures unchanged from spec-046 (no regression in isolation)
  - §6 Boundary contracts: confirm `RunHistoryRepository` public API byte-identical (one `gh pr diff` or equivalent against the public surface)
  - §7 Documentation: update `specs/046-sqlalchemy-data-layer/CODE_AUDIT.html` annotating each SMELL row "resolved in spec-048"; update §B0.2 of `docs/architecture/architecture_standards.md` with the `info={"updated_on_complete"}` pattern
  - §8 Sightings: file a `LEARNINGS.md` entry: "When a refactor leaves smells behind, schedule the cleanup spec immediately — don't trust that 'P2/P3 follow-ups' will get done"
- [ ] T072 Resolve any high-severity findings in REVIEW.md (block handoff until zero)
- [ ] T073 Flip spec-048 status: Draft → In Progress → Code Review → Implemented
- [ ] T074 `CHANGES_LOG.md` entry: `[REFACTOR] spec-048: data layer cleanup — fixed 6 smells from spec-046 audit`

**Check**: REVIEW.md exists; zero high-severity findings; spec status = Implemented.

---

## Test delta

| Phase | Added | Deleted | Net |
|---|---|---|---|
| Phase 0 | 1 (arch path-sole-owner) | 0 | +1 |
| Phase 1 | 1 (schema upgrade) | 0 | +1 |
| Phase 2 | 1 (repo construction perf) | 0 | +1 |
| Phase 3 | 1 (hot fields equivalence) | 0 | +1 |
| Phase 4 | 2 (RunRow-vs-ActionLog arch + from_orm equivalence) | 0 | +2 |
| Phase 5 | 0 (existing `test_golden_fixture.py` re-runs) | 0 | 0 |
| Phase 6 | 0 (frozen `_legacy_to_run_row` deleted, was Phase-4-internal) | 0 | 0 |
| **Total** | **6** | **0** | **+6** |

All 6 new tests are permanent drift guards or equivalence proofs against the old behavior. No redundant unit tests.

---

## Sequencing rationale

- **Phase 0 first** because every later phase needs `_paths`.
- **Phase 1 before 2** because dropping `_engine_cache` is only safe once `ensure_schema()` is cheap (no subprocess).
- **Phase 3 before 4** because both touch `run_history_repo.py`; doing hot-fields first keeps Phase 4's diff small.
- **Phase 5 last among code changes** because it depends on the new Repository being stable.
- **Phase 6 before 7** because the code review needs a green suite to review against.
- **Phase 7 mandatory** — a refactor spec without a code-review gate is how the original smells slipped through.
