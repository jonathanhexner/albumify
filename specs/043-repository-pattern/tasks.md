# Tasks: Repository Pattern for face_cluster Persistence (043)

Predecessor: spec-042 (architecture standards + History tab pilot)
Standards reference: [`docs/architecture/architecture_standards.md`](../../docs/architecture/architecture_standards.md) §B0, §A6, §B1

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped (rationale required)

---

## Design notes (binding for every task below)

- **D1**: Order is **build → test → migrate**, never out of order. Phase 1 builds the new code; Phase 2-3 test it in isolation; Phase 4-5 migrate consumers. No consumer touches the Repository until its tests are green.
- **D2**: Both apps stay runnable at every commit. Legacy `app/face_clustering/` and v2 `app/face_clustering_v2/` both work after every merge. If a commit breaks either, it doesn't ship.
- **D3**: Existing service tests (35 in `tests/face_clustering/views/test_history_service_*.py`) are the contract guard for the migration. They must continue to pass — only the test-setup line (how the Service is constructed) changes.
- **D4**: All public classes and methods on the Repository get docstrings per A5. Arch test enforces.
- **D5**: The Repository takes a typed Config in `__init__`; no growing kwarg list. Arch test enforces.
- **D6**: Query methods take typed `RunHistoryCriteria`; no dict-of-strings. Arch test enforces.
- **D7**: Old free functions in `face_cluster/run_history.py` + `run_history_db.py` stay through the burn-in. They emit `DeprecationWarning` but don't break callers. Removal is a separate follow-up spec.

---

## Phase 0 — Prerequisites

- [ ] **T001** Add `face_cluster/repositories/_errors.py` with `RepositoryError`, `NotFoundError`, `ValidationError`. Module-level docstring explains the hierarchy + the future merge with `face_cluster/views/_errors.py` (spec-042 B2 follow-up).
- [ ] **T002** Verify the spec-042 A6 standard is in `docs/architecture/architecture_standards.md` (already landed in commit `8d9cc1d` — confirm only).

**Checkpoint**: errors module imports cleanly. Nothing else changes. Both apps still run; full test suite still passes.

---

## Phase 1 — Build the Repository (new code; no consumers yet)

- [ ] **T010** Add `face_cluster/repositories/__init__.py`. Re-exports `RunHistoryRepository`, `RunHistoryRepoConfig`, `RunHistoryCriteria`.
- [ ] **T011** Add `face_cluster/repositories/run_history_repo.py` with:
  - `RunHistoryRepoConfig` (frozen dataclass; 5 fields with defaults per spec §4.2).
  - `RunHistoryCriteria` (frozen dataclass; 13 fields per spec §4.3).
  - `RunHistoryRepository` class skeleton — `__init__(self, config: Optional[RunHistoryRepoConfig] = None)` + 9 method stubs raising `NotImplementedError`.
  - All public symbols carry docstrings per A5.
- [ ] **T012** Implement query methods (`find`, `find_one`, `count`, `distinct_albums`, `get_by_id`). Each delegates to a parameterized SQL builder; criteria fields map to WHERE clauses; ordering / limit / offset come from `RunHistoryCriteria`.
- [ ] **T013** Implement mutation methods (`start_action`, `complete_action`, `fail_action`, `update_comment`). Raise `NotFoundError` when targeting a missing `action_id`. `update_comment` raises `ValidationError` on overlength input (>2048 chars).
- [ ] **T014** Confirm the Repository module doesn't import `streamlit` (would be a layering violation per A1).

**Checkpoint**: `from face_cluster.repositories import RunHistoryRepository` works in a fresh Python. No consumers exist yet — legacy app + v2 app + all existing tests untouched and green.

---

## Phase 2 — Repository synthetic tests (24+ cases)

- [ ] **T020** Add `tests/face_clustering/repositories/__init__.py`.
- [ ] **T021** Add `tests/face_clustering/repositories/_seed.py` — row factories. Likely a thin wrapper around the existing `tests/face_clustering/views/_seed.py` factories, OR move common helpers up if duplication appears.
- [ ] **T022** Add `tests/face_clustering/repositories/test_run_history_repo_synthetic.py` with the **24+** cases enumerated in `spec.md` §5.1. Each test constructs a fresh `RunHistoryRepository(RunHistoryRepoConfig(db_path=synthetic_action_log_db))` from the session fixture.
- [ ] **T023** Run: `pytest tests/face_clustering/repositories/test_run_history_repo_synthetic.py -v`. All 24+ pass.

**Checkpoint**: Repository works correctly against synthetic in-memory DBs. Each query method, each mutation method, every edge case enumerated in the PRD has a passing test.

---

## Phase 3 — Repository real-fixture smoke (3 cases)

- [ ] **T030** Add `tests/face_clustering/repositories/test_run_history_repo_real.py` with 3 cases per spec §5.2. Construct the Repository with explicit `db_path=Path.home() / ".sim_bench" / "sim_bench.db"` (bypasses the session monkeypatch the same way the existing real-fixture tests do).
- [ ] **T031** Run against the dev `~/.sim_bench/sim_bench.db`: all 3 pass.

**Checkpoint**: Repository's schema assumptions match the real DB.

---

## Phase 4 — Migrate `HistoryService` to use the Repository

- [ ] **T040** Update `face_cluster/views/history.py`:
  - `__init__(self, repo: Optional[RunHistoryRepository] = None)` replaces `__init__(self, db_path=None)`.
  - All method bodies that called `run_history.X(db_path=self._db_path)` or `run_history_db.X(db_path=self._db_path)` now call `self._repo.X(...)`.
  - `list_runs(query)` constructs a `RunHistoryCriteria` from the `HistoryQuery` and delegates to `self._repo.find(...)`.
  - The `_summary_from_pipeline_run` helper and any non-DB logic stays in `HistoryService` — Repository only owns persistence.
- [ ] **T041** Update `tests/face_clustering/views/test_history_service_synthetic.py`:
  - Test setup changes from `HistoryService(db_path=synthetic_action_log_db)` to `HistoryService(repo=RunHistoryRepository(RunHistoryRepoConfig(db_path=synthetic_action_log_db)))`.
  - Assertions unchanged (this is the contract guard).
- [ ] **T042** Update `tests/face_clustering/views/test_history_service_real.py` analogously.
- [ ] **T043** Run: `pytest tests/face_clustering/views/ tests/face_clustering/repositories/ -v`. **All 31 + 4 + 24 + 3 = 62 tests pass.**

**Checkpoint**: `HistoryService` now runs on top of the Repository. External contract unchanged — every existing service test still passes by virtue of the same return values. The shadowed-import trap is gone (Service holds a Repository, not free functions).

---

## Phase 5 — Migrate legacy app callers

- [ ] **T050** Audit: `grep -rE "from face_cluster\.(run_history|run_history_db)" app/`. List every file + line.
- [ ] **T051** For each call site, replace with:
  - `RunHistoryRepository()` instantiated inline (for one-off queries), OR
  - A Service wrapper if the caller is already conceptually a service-shaped consumer.
- [ ] **T052** Confirm legacy app launches: `.venv/Scripts/streamlit run app/face_clustering/main.py --server.port 8501 --server.headless true`.
- [ ] **T053** Manually click through the legacy History tab. Confirm filter / select-row / load-button still work.
- [ ] **T054** Confirm v2 app launches: `.venv/Scripts/streamlit run app/face_clustering_v2/main.py --server.port 8888 --server.headless true`.
- [ ] **T055** Run the Playwright smoke: `python tests/manual/_v2_history_smoke.py`. Green + screenshot.

**Checkpoint**: Both apps run against the new Repository. Functionality preserved end-to-end.

---

## Phase 6 — Arch tests + fixture cleanup

- [ ] **T060** Add `tests/architecture/test_repositories.py` with:
  - `test_repositories_take_typed_config()` — scan `face_cluster/repositories/` for classes named `*Repository`; assert each `__init__` accepts at most one positional/keyword arg and that arg is a frozen dataclass.
  - `test_repositories_have_no_module_level_db_functions()` — scan `face_cluster/repositories/*.py` AST-level for module-level functions that contain SQL strings or `sqlite3.connect`.
- [ ] **T061** Simplify `tests/face_clustering/conftest.py`'s `_isolate_run_history_db` autouse fixture:
  - The Repository owns `db_path` per-instance, so monkeypatching the source module is no longer required.
  - Either remove the fixture (preferred) or keep a simpler version that only catches any leftover legacy call paths.
- [ ] **T062** Run the entire test suite: `pytest tests/ -q`. Everything green.

**Checkpoint**: Drift caught at PR time going forward. Fragile import-shadowing workaround retired.

---

## Phase 7 — Deprecation signals (old free functions stay; warn on import)

- [ ] **T070** Add at top of `face_cluster/run_history.py`:
  ```python
  import warnings
  warnings.warn(
      DeprecationWarning(
          "face_cluster.run_history is deprecated. "
          "Use face_cluster.repositories.RunHistoryRepository instead."
      ),
      stacklevel=2,
  )
  ```
- [ ] **T071** Same for `face_cluster/run_history_db.py`.
- [ ] **T072** Surround the warning with `warnings.simplefilter("once", DeprecationWarning)` so production logs aren't flooded.
- [ ] **T073** Confirm `pytest tests/ -q` still green (warnings show in summary but don't fail).

**Checkpoint**: Old code still works (no behavior change); new code is the recommended path. Future PR introducing a new caller of the deprecated functions surfaces a visible warning in test output.

---

## Phase 8 — Close-out

- [ ] **T080** Run `/code-review` → `specs/043-repository-pattern/REVIEW.md`. Resolve high-severity findings.
- [ ] **T081** Update `docs/architecture/classes.html`: add `RunHistoryRepository` (Layer 1), `RunHistoryRepoConfig`, `RunHistoryCriteria` (Layer 2).
- [ ] **T082** Update `docs/architecture/data_flow.html`: show Service → Repository → DB.
- [ ] **T083** Update `docs/architecture/architecture_standards.md` §A6 audit table for History tab — Repository row goes from ⚠️ MISSING to ✅.
- [ ] **T084** Append CHANGES_LOG entry per landed commit.
- [ ] **T085** Flip spec status `Draft` → `Code Review` → `Implemented`.
- [ ] **T086** File a follow-up sighting "remove deprecated face_cluster.run_history / run_history_db after 2-week burn-in".

**Checkpoint**: spec marked Implemented; legacy free-function removal tracked.

---

## Out of scope (explicit deferrals)

- **Removal of `face_cluster/run_history.py` and `run_history_db.py`** — separate spec after 2-week burn-in. They keep working through this entire spec with only `DeprecationWarning` visibility.
- **Generalizing the Repository pattern to other persistence** (face_clustering.db, embeddings.npy, profile JSONs) — each per-tab spec adds its own Repository where needed.
- **`action_log` schema changes** — wire-compatible refactor.
- **Async / connection-pool variants of the Repository** — premature; SQLite per-call connections work fine today.
