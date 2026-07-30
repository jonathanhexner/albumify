# spec-043 — Repository pattern for face_cluster persistence

Status: **Draft**
Author: Jonathan Hexner
Created: 2026-05-25
Predecessors: [spec-040](../040-unified-pipeline-framework/spec.md), [spec-041](../041-fc-params-container/spec.md), [spec-042](../042-fc-app-v2-tab-parity/spec.md)
Architectural standards: [`docs/architecture/architecture_standards.md`](../../docs/architecture/architecture_standards.md) §B0 (Repository pattern), §A6 (per-tab migration discipline), §B1 (no shadowed imports)

---

## 1. Objective

Replace the ~600 LOC of module-level free functions in `face_cluster/run_history.py` and `face_cluster/run_history_db.py` with a typed Repository class — `RunHistoryRepository` — that:

1. Takes a typed Config dataclass in `__init__` (`RunHistoryRepoConfig`).
2. Exposes a small composable query API (`find`/`find_one`/`count`/`distinct_albums`/`get_by_id`) driven by typed criteria (`RunHistoryCriteria`).
3. Exposes mutation methods with named verbs (`start_action`/`complete_action`/`fail_action`/`update_comment`).
4. Carries `db_path` exactly once — at construction — instead of threading it through every call site.
5. Is the only module under `face_cluster/` that imports `sqlite3` for the action_log.

Migrate `HistoryService` and all legacy callers to use the Repository. Existing tests continue to pass; new Repository-layer tests are added per spec-042 A6's three-layer gate.

**Non-goals:**
- Removing the legacy free functions. They stay (with a `DeprecationWarning`) through a burn-in period; a follow-up spec removes them.
- Generalizing the pattern beyond `face_cluster/run_history*`. Other persistence (face_clustering.db, embeddings.npy) is a separate Repository in a future spec.
- Changing the `action_log` schema. Wire-compatible refactor — no migration needed.

---

## 2. Why now

### 2.1 The defect that surfaced this work

During spec-042 H1 (History tab pilot), the session fixture in `tests/face_clustering/conftest.py` monkeypatches `face_cluster.run_history_db.get_db_path` to redirect to a temp DB. `face_cluster.run_history.search()` reads through a `from face_cluster.run_history_db import get_db_path` binding captured at import time — the monkeypatch misses it. Result: `list_runs` hit the real DB while `list_actions` hit the temp DB. Three of four real-fixture tests passed for the wrong reason; one failed.

The shadowed-import trap is endemic to the module-level free function pattern. Constructor injection eliminates it.

### 2.2 The standard now requires it

`docs/architecture/architecture_standards.md` §B0 declares the Repository pattern as binding. spec-042 §4.2 declares "every backend service is a class with typed methods" — and `HistoryService` (which follows it) currently delegates to free functions (which don't). The lower layer is inconsistent with the standard we just established.

### 2.3 It unblocks the seven remaining tab migrations

Each future tab in spec-042 will have its own Service. Without the Repository pattern at the persistence layer, every Service repeats the same workarounds (threading `db_path`, hand-rolling test isolation). With it: each Service is ~30 LOC of composition; tests inject a test Repository instance; arch tests catch drift.

---

## 3. Scope

### In scope

| Concern | What ships |
|---|---|
| **Build** | NEW `face_cluster/repositories/__init__.py`, `face_cluster/repositories/run_history_repo.py` (RunHistoryRepoConfig + RunHistoryCriteria + RunHistoryRepository class). Includes Pandera-style validation of inputs where appropriate. |
| **Test (synthetic)** | NEW `tests/face_clustering/repositories/test_run_history_repo_synthetic.py` — ≥15 cases covering every public method + edge cases. |
| **Test (real-fixture smoke)** | NEW `tests/face_clustering/repositories/test_run_history_repo_real.py` — ≥3 cases against `~/.sim_bench/sim_bench.db`. |
| **Migrate HistoryService** | `face_cluster/views/history.py` — `__init__(self, repo: Optional[RunHistoryRepository] = None)`. Method bodies replace direct `run_history.search(...)` calls with `self._repo.find(...)`. All 31 existing synthetic tests + 4 real-fixture tests continue to pass; updates only the wiring, not the assertions. |
| **Migrate legacy callers** | Audit + update direct callers of `face_cluster.run_history.*` / `run_history_db.*` in `app/face_clustering/` (~6 sites). Each becomes a Repository instantiation or, where appropriate, a thin facade. |
| **Arch tests** | NEW `tests/architecture/test_repositories.py` — `test_repositories_take_typed_config()`, `test_repositories_have_no_module_level_db_functions()`. |
| **Deprecation signals** | Existing `run_history.py` / `run_history_db.py` free functions emit `DeprecationWarning` pointing at the Repository. They keep working (no removal in this spec). |
| **Docs** | Update `docs/architecture/classes.html` with the Repository entry; update `architecture_standards.md` audit row for History tab from "⚠️ missing Repository tests" to ✅. |

### Out of scope

| Item | Reason |
|---|---|
| Removing `face_cluster/run_history.py` and `face_cluster/run_history_db.py` | Burn-in required. Removed in a follow-up spec after 2 weeks of green CI. |
| Generalizing the Repository pattern to other persistence (face_clustering.db, embeddings.npy) | Separate spec per Repository. Each per-tab spec adds its own Repository where needed. |
| `action_log` schema changes | Wire-compatible refactor. The Repository wraps the existing schema. |
| Removing the session fixture monkeypatch in `tests/face_clustering/conftest.py` | Simplifies dramatically (or disappears) once the Repository is the only caller. The session fixture removal is a Phase 6 cleanup, not a separate spec. |
| Generalizing `RunHistoryCriteria` into a base "criteria" pattern for other Repositories | Premature — wait until a second Repository earns the abstraction. |

---

## 4. Design

### 4.1 Module layout

```
face_cluster/
  repositories/                       (NEW)
    __init__.py                       — re-exports RunHistoryRepository, RunHistoryRepoConfig, RunHistoryCriteria
    run_history_repo.py               — the Repository class + Config + Criteria dataclasses

  run_history.py                      (KEEP; emit DeprecationWarning on import)
  run_history_db.py                   (KEEP; emit DeprecationWarning on import)
```

### 4.2 The Config dataclass

```python
@dataclass(frozen=True, slots=True)
class RunHistoryRepoConfig:
    """Configuration for RunHistoryRepository.

    Every field has a default. Adding a new field is non-breaking for
    existing callers.
    """
    db_path: Optional[Path] = None
    auto_migrate: bool = True
    read_only: bool = False
    log_queries: bool = False
    connection_timeout_s: float = 5.0
```

Today only `db_path` is meaningfully used. The other fields are forward-looking; their defaults are no-ops. They're declared now so callers don't break when (e.g.) `read_only=True` is needed for a future safety feature.

### 4.3 The Criteria dataclass

```python
@dataclass(frozen=True, slots=True)
class RunHistoryCriteria:
    """Composable filter for run history queries.

    All fields optional. Empty/None = no filter on that axis.
    Combination is AND, not OR.
    """
    ids: Optional[list[int]] = None
    parent_run_id: Optional[int] = None
    album: Optional[str] = None
    status: Optional[str] = None
    producer: Optional[str] = None
    action_types: Optional[list[str]] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    text: Optional[str] = None
    text_fields: tuple[str, ...] = ("source_album", "run_name", "comment")
    limit: int = 500
    offset: int = 0
    order_by: Literal["started_at_desc", "started_at_asc"] = "started_at_desc"
```

### 4.4 The Repository class

Full signature in `face_cluster/repositories/run_history_repo.py`:

```python
class RunHistoryRepository:
    """Read + write access to the action_log table. Thin SQL wrapper.

    Owns: a connection path to ~/.sim_bench/sim_bench.db (or test override
    via RunHistoryRepoConfig). Knows the schema; knows nothing about views,
    services, Streamlit, or business logic.

    Composability: HistoryService and any future Service that needs run
    history data takes a RunHistoryRepository via constructor injection.
    The Service composes one or more Repositories; the Repository composes
    nothing — it's a leaf.
    """

    def __init__(self, config: Optional[RunHistoryRepoConfig] = None):
        """Initialize with a Config (or None for defaults)."""

    # --- composable queries ---
    def find(self, criteria: RunHistoryCriteria) -> list[RunRow]:
        """Return rows matching the criteria, ordered + paginated per criteria.

        Args:
            criteria: typed filter. Empty criteria = all rows.

        Returns: list of RunRow (typed). Empty when no match.
        Side effects: none.
        """

    def find_one(self, criteria: RunHistoryCriteria) -> Optional[RunRow]:
        """Return the first row matching the criteria, or None.

        Convenience: equivalent to find(criteria)[0] if matches else None,
        but issues a LIMIT 1 query.
        """

    def count(self, criteria: RunHistoryCriteria) -> int:
        """Total rows matching the criteria without fetching them."""

    def distinct_albums(self) -> list[str]:
        """Sorted list of distinct non-empty source_album values."""

    def get_by_id(self, run_id: int) -> Optional[RunRow]:
        """Single-id lookup. Convenience wrapper over find_one."""

    # --- mutations ---
    def start_action(
        self,
        action_type: str,
        *,
        payload: Optional[dict] = None,
        source_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        source_album: Optional[str] = None,
        run_name: Optional[str] = None,
        parent_run_id: Optional[int] = None,
        run_kind: Optional[str] = None,
        producer: Optional[str] = None,
    ) -> int:
        """Insert a new action_log row in 'running' status. Returns the id.
        Side effects: INSERTs one row.
        """

    def complete_action(
        self,
        action_id: int,
        *,
        result_fields: Optional[dict] = None,
    ) -> None:
        """Mark an action complete; merge result_fields into the row.
        Side effects: UPDATEs the action_log row.
        Raises: NotFoundError if action_id missing.
        """

    def fail_action(self, action_id: int, error: str) -> None:
        """Mark an action failed with an error message.
        Side effects: UPDATEs the action_log row.
        Raises: NotFoundError if action_id missing.
        """

    def update_comment(self, action_id: int, comment: str) -> None:
        """Set the comment on an action.
        Side effects: UPDATEs the action_log row.
        Raises: ValidationError if comment > 2048 chars; NotFoundError
                if action_id missing.
        """
```

### 4.5 Error model (aligns with §B2 of standards)

The Repository raises from the `face_cluster.views._errors` hierarchy. Even though we haven't formally introduced `ServiceError` yet (deferred from spec-042 B2), this Repository is the right time to start. Add a minimal version:

```python
# face_cluster/repositories/_errors.py
class RepositoryError(Exception):
    """Base for Repository-layer errors. Carries an optional user_message."""

class NotFoundError(RepositoryError): pass
class ValidationError(RepositoryError): pass
```

Services catching these have a clean migration path when `ServiceError` lands.

### 4.6 Migration of `HistoryService`

Before (today):

```python
class HistoryService:
    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path

    def list_runs(self, query: HistoryQuery) -> list[RunRow]:
        filters = run_history.HistoryFilters(album=query.album, ...)
        return run_history.search(filters, db_path=self._db_path)

    def update_comment(self, run_id: int, comment: str) -> None:
        run_history_db.update_comment(run_id, comment, db_path=self._db_path)
    # ... 5 more delegations, each threading self._db_path
```

After:

```python
class HistoryService:
    def __init__(self, repo: Optional[RunHistoryRepository] = None):
        self._repo = repo or RunHistoryRepository()

    def list_runs(self, query: HistoryQuery) -> list[RunRow]:
        criteria = RunHistoryCriteria(
            album=query.album, date_from=query.date_from,
            date_to=query.date_to, text=query.text,
        )
        return self._repo.find(criteria)

    def update_comment(self, run_id: int, comment: str) -> None:
        self._repo.update_comment(run_id, comment)
    # ... etc; no db_path threading anywhere
```

Net: -~30 LOC in `HistoryService`; same external behavior; existing 35 tests continue to pass with only the fixture-construction changing (`repo=RunHistoryRepository(RunHistoryRepoConfig(db_path=tmp))` instead of `db_path=tmp`).

### 4.7 Architecture tests

Two new tests in `tests/architecture/test_repositories.py`:

```python
def test_repositories_take_typed_config():
    """Every *Repository class's __init__ either takes no args or one
    positional/keyword arg that is a frozen dataclass instance. Forbids
    the 'grow a 5th kwarg' anti-pattern."""

def test_repositories_have_no_module_level_db_functions():
    """face_cluster/repositories/*.py exposes only classes, dataclasses,
    and module-level constants. No module-level functions that query DBs."""
```

Plus update the existing `test_v2_layering.py` to assert that `app/face_clustering_v2/tabs/*` doesn't import from `face_cluster.repositories.*` directly (services own that boundary).

---

## 5. Test plan — per the spec-042 A6 three-layer gate

### 5.1 Repository layer (NEW — 15+ synthetic tests)

`tests/face_clustering/repositories/test_run_history_repo_synthetic.py` — against `synthetic_action_log_db` fixture:

| Method / behavior | # cases | What they verify |
|---|---|---|
| `find(criteria=empty)` | 1 | Returns all rows |
| `find(criteria=album)` | 1 | Filters by album |
| `find(criteria=date_from + date_to)` | 2 | Inclusive bounds; reversed range = empty |
| `find(criteria=text)` | 3 | Substring match in album, run_name, comment |
| `find(criteria=ids=[…])` | 1 | Filters to specific ids |
| `find(criteria=multi)` | 1 | AND combination |
| `find(criteria=limit / offset)` | 1 | Pagination works |
| `find(criteria=order_by)` | 2 | Both order directions verified |
| `find_one` | 2 | Returns first match; None when no match |
| `count(criteria)` | 1 | Returns count without fetching all rows |
| `distinct_albums` | 2 | Sorted; empty DB returns empty list |
| `get_by_id` | 2 | Existing id returns row; missing id returns None |
| `start_action` | 1 | Inserts row in 'running' status, returns id |
| `complete_action` | 2 | Merges result_fields; raises NotFoundError on missing id |
| `fail_action` | 1 | Sets status='failed' + error text |
| `update_comment` | 3 | Persists; idempotent; raises ValidationError on overlength |

Total: **≥24 cases**. Above the ≥5-per-Repository A6 floor.

### 5.2 Repository real-fixture smoke (NEW — 3 tests)

`tests/face_clustering/repositories/test_run_history_repo_real.py` — against `~/.sim_bench/sim_bench.db`:

| # | Test |
|---|---|
| 1 | `find(criteria=empty)` returns ≥1 row on populated dev DB |
| 2 | `find(criteria=producer="fc_app_v2")` returns ≥1 row from spec-040 T4 |
| 3 | `distinct_albums()` contains "Budapest2025_Google" |

### 5.3 Service layer (regression — 35 existing tests unchanged)

`tests/face_clustering/views/test_history_service_synthetic.py` (31 tests) and `test_history_service_real.py` (4 tests) — re-run after Phase 4 migration. The contract of `HistoryService` is unchanged; only the construction (`HistoryService(repo=...)` instead of `HistoryService(db_path=...)`) needs updating in test setup.

### 5.4 Frontend layer (regression)

`tests/manual/_v2_history_smoke.py` — re-run after Phase 4 migration against live Streamlit on port 8889. Screenshot captured.

### 5.5 Architecture tests (NEW — 2)

`tests/architecture/test_repositories.py` — the two tests in §4.7.

### 5.6 Total test surface

- **NEW**: 24 Repository synthetic + 3 Repository real-fixture + 2 architecture = **29 new tests**
- **Regression**: 35 service tests + 1 UI smoke = unchanged in behavior, updated in setup

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Migrating `HistoryService` accidentally changes the contract. | Existing 35 service tests are the contract guard. They run before and after the migration; any behavior change shows up immediately. |
| Some legacy app file (`app/face_clustering/*`) imports a free function we miss. | Phase 5 starts with a grep audit (`grep -r "from face_cluster.run_history" app/`). The arch test `test_no_module_level_db_functions` would also catch new violations. |
| The Pandera / typed validation inside `find()` (e.g., on `criteria.date_from <= criteria.date_to`) over-constrains a legacy caller. | Cross-field validation is added incrementally. Phase 1 ships raw structure; cross-field rules added in a sub-phase only after all existing callers are surveyed. |
| Removing the import-shadowing trap breaks the session fixture in `tests/face_clustering/conftest.py`. | Phase 6 simplifies (or removes) the session fixture. Real-fixture tests that use explicit `db_path=` no longer need the workaround. |
| Two consumers (HistoryService + a legacy tab) construct a Repository with different `RunHistoryRepoConfig` values and the resulting connection settings conflict. | The Repository's config is per-instance, not global. Two Repositories = two independent connection paths. SQLite handles multi-process / multi-connection fine. |
| The `DeprecationWarning` from the old free functions floods production logs. | Emit once per import via `warnings.simplefilter("once", DeprecationWarning)` at top of the legacy modules. |

---

## 7. Phases — **build → test → migrate**, never out of order

Each phase ends with a checkpoint where (a) all tests pass, (b) the legacy app runs, (c) the v2 app runs. Strangler-fig discipline per §A6.

### Phase 0 — Prerequisites (~30 min)

- [ ] **T001** Add the `face_cluster.views._errors` module (or its Repository-layer minimal equivalent) with `RepositoryError`, `NotFoundError`, `ValidationError`. Used by Phase 1 onward.
- [ ] **T002** Confirm the spec-042 A6 standard is in `docs/architecture/architecture_standards.md`. (Already landed — verify only.)

**Checkpoint**: Errors module imports cleanly; no other code changes.

### Phase 1 — Build (new code, no consumers yet) (~2h)

- [ ] **T010** Create `face_cluster/repositories/__init__.py`. Re-exports the public names.
- [ ] **T011** Create `face_cluster/repositories/run_history_repo.py` with `RunHistoryRepoConfig`, `RunHistoryCriteria`, and `RunHistoryRepository` class skeleton — all methods raise `NotImplementedError`. Docstrings per §A5 of standards.
- [ ] **T012** Implement query methods (`find`, `find_one`, `count`, `distinct_albums`, `get_by_id`).
- [ ] **T013** Implement mutation methods (`start_action`, `complete_action`, `fail_action`, `update_comment`).
- [ ] **T014** Import the existing `RunRow` from `face_cluster.run_history` (we don't move it yet — the type lives there until the burn-in spec).

**Checkpoint**: `from face_cluster.repositories import RunHistoryRepository, RunHistoryRepoConfig, RunHistoryCriteria` works. No consumers are using it yet. Existing app + tests untouched and green.

### Phase 2 — Test the new code (synthetic) (~2h)

- [ ] **T020** Create `tests/face_clustering/repositories/__init__.py` + `_seed.py` (mirrors the views-layer seed factories).
- [ ] **T021** Write `tests/face_clustering/repositories/test_run_history_repo_synthetic.py` with ≥24 cases per §5.1. Each test instantiates `RunHistoryRepository(RunHistoryRepoConfig(db_path=synthetic_action_log_db))` and asserts on typed return values.
- [ ] **T022** Run the test suite: `pytest tests/face_clustering/repositories/ -v`. All ≥24 cases pass.

**Checkpoint**: Repository works correctly in isolation. Verified against synthetic data; no production code touched yet.

### Phase 3 — Real-fixture smoke (~30 min)

- [ ] **T030** Write `tests/face_clustering/repositories/test_run_history_repo_real.py` with 3 cases per §5.2. Pattern matches the existing `test_history_service_real.py` (explicit `db_path` via `Path.home() / ".sim_bench" / "sim_bench.db"` to bypass the session monkeypatch — once the migration completes, the monkeypatch goes away).
- [ ] **T031** Run: `pytest tests/face_clustering/repositories/test_run_history_repo_real.py -v`. All 3 pass against the dev `~/.sim_bench/sim_bench.db`.

**Checkpoint**: Repository works against real data. Schema assumptions verified.

### Phase 4 — Migrate `HistoryService` to use the Repository (~1h)

- [ ] **T040** Update `face_cluster/views/history.py`:
  - `HistoryService.__init__(self, repo: Optional[RunHistoryRepository] = None)` instead of `db_path`.
  - Every method that called `run_history.X(...)` or `run_history_db.X(...)` now calls `self._repo.X(...)`.
  - The body of `list_runs` builds a `RunHistoryCriteria` from the `HistoryQuery`.
- [ ] **T041** Update `tests/face_clustering/views/test_history_service_synthetic.py`:
  - Fixture changes from `HistoryService(db_path=tmp)` to `HistoryService(repo=RunHistoryRepository(RunHistoryRepoConfig(db_path=tmp)))`. Assertions unchanged.
- [ ] **T042** Update `tests/face_clustering/views/test_history_service_real.py` analogously.
- [ ] **T043** Run full surface: `pytest tests/face_clustering/views/ tests/face_clustering/repositories/ -v`. All 35 + 27 = 62 tests pass.

**Checkpoint**: `HistoryService` runs on top of the Repository. Behavior identical (contract test = the existing 35 tests).

### Phase 5 — Migrate legacy app callers (~1h)

- [ ] **T050** Audit: `grep -rE "from face_cluster\.(run_history|run_history_db)" app/`. List every call site.
- [ ] **T051** For each legacy site, replace direct free-function calls with either:
  - A Repository instantiation (`RunHistoryRepository()`) used inline, OR
  - For tabs that already conceptually wrap a Service, a thin wrapper that takes a Repository.
- [ ] **T052** Confirm legacy app still starts: `.venv/Scripts/streamlit run app/face_clustering/main.py --server.port 8501 --server.headless true`. Manually click through History tab.
- [ ] **T053** Update `app/face_clustering/tabs/history_tab.py` and its companion files specifically — these are the heaviest direct callers.

**Checkpoint**: Legacy app runs against the new Repository. Both apps are functionally identical to before; the persistence layer underneath is the new shape.

### Phase 6 — Arch tests + fixture cleanup (~1h)

- [ ] **T060** Write `tests/architecture/test_repositories.py` with the two tests in §4.7.
- [ ] **T061** Simplify or remove the session fixture in `tests/face_clustering/conftest.py`. The shadowed-import workaround is no longer needed — the Repository owns the `db_path` per-instance.
- [ ] **T062** Re-run the entire test surface: `pytest tests/ -q`. Everything green.

**Checkpoint**: Drift caught at PR time going forward. The fragile test-isolation workaround is gone.

### Phase 7 — Deprecation signals (~30 min)

- [ ] **T070** Add a module-level `warnings.warn(DeprecationWarning(...))` at the top of `face_cluster/run_history.py` pointing at `face_cluster.repositories.RunHistoryRepository`.
- [ ] **T071** Same for `face_cluster/run_history_db.py`.
- [ ] **T072** Use `warnings.simplefilter("once", DeprecationWarning)` inside the modules so production logs aren't flooded.
- [ ] **T073** Confirm pytest still passes (deprecation warnings show but don't fail tests).

**Checkpoint**: Old code still works (zero behavior change), but new code is the recommended path. Removal is a follow-up spec after burn-in.

### Phase 8 — Close-out (~1h)

- [ ] **T080** Run `/code-review` → `specs/043-repository-pattern/REVIEW.md`. Resolve high-severity findings.
- [ ] **T081** Update `docs/architecture/classes.html`: add `RunHistoryRepository`, `RunHistoryRepoConfig`, `RunHistoryCriteria`.
- [ ] **T082** Update `docs/architecture/data_flow.html`: show the Service → Repository → DB flow.
- [ ] **T083** Update `docs/architecture/architecture_standards.md` §A6 audit row for History tab: "Repository tests" goes from ⚠️ MISSING to ✅.
- [ ] **T084** Update `CHANGES_LOG.md` per landed commit.
- [ ] **T085** Flip spec status `Draft` → `Code Review` → `Implemented`.
- [ ] **T086** File a follow-up sighting for "remove deprecated free functions after burn-in (target: 2 weeks)".

**Checkpoint**: Spec marked Implemented; deprecation removal tracked.

### Total estimate

**~8 hours** spread across 8 phases. Each phase is independently shippable; the legacy app stays runnable throughout.

---

## 8. Definition of Done

- [ ] `face_cluster/repositories/run_history_repo.py` exists with `RunHistoryRepoConfig`, `RunHistoryCriteria`, and `RunHistoryRepository` (all 9 methods).
- [ ] No `cfg.get('field', '?')` / dict-of-strings patterns anywhere in the Repository.
- [ ] `RunHistoryRepository.__init__` takes a typed Config dataclass (enforced by arch test).
- [ ] ≥24 synthetic-data Repository tests pass.
- [ ] ≥3 real-fixture Repository tests pass.
- [ ] All 35 existing `HistoryService` tests continue to pass with the Repository underneath.
- [ ] Playwright UI smoke for History tab continues to pass.
- [ ] Two new arch tests pass.
- [ ] Session fixture in `tests/face_clustering/conftest.py` is simplified or removed.
- [ ] Both legacy app and v2 app launch + render History tab cleanly.
- [ ] Legacy free functions emit `DeprecationWarning` on import (once per process).
- [ ] Architecture HTMLs updated.
- [ ] REVIEW.md produced.
- [ ] Spec status → **Implemented**.

---

## 9. Open questions

1. **Should `RunHistoryRepository` consolidate both `run_history` (search-ish reads) AND `run_history_db` (CRUD writes) into one class?** Recommended: yes. They operate on the same table; splitting them into "Reader" + "Writer" classes is premature for a single-table Repository. If a future Repository handles multiple tables, that's the time to revisit.

2. **Where do the typed exceptions live — `face_cluster/repositories/_errors.py` or `face_cluster/views/_errors.py`?** Recommended: `face_cluster/repositories/_errors.py` for now; if the Service-layer error hierarchy in spec-042 §B2 lands later and converges, refactor.

3. **Does the Repository need a `__del__` / context-manager protocol for connection lifecycle?** Recommended: no, SQLite's per-call connection model (used today) is fine. If a future Repository wraps a connection pool, that's when this matters.

4. **Should `RunHistoryCriteria.date_from <= date_to` be validated at construction or at query time?** Recommended: at query time. A frozen dataclass shouldn't constrain field combinations; `find()` validates before issuing SQL and raises `ValidationError`.

5. **Backward compatibility for `face_cluster.run_history.RunRow`?** Recommended: `RunRow` stays in `face_cluster/run_history.py` (re-exported from `face_cluster.repositories`). Moving it would break too many imports. Removal happens in the deprecated-modules cleanup spec.
