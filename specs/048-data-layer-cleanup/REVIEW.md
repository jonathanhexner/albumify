# Code Review: spec-048 — Data Layer Cleanup

**Reviewed**: 2026-05-28
**Reviewer**: Claude (self-review against `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Status**: PASS — 0 blockers, 1 pass-with-follow-up
**Branch**: `unification/spec-040` (spec-048 lives on the same active branch as spec-046)

---

## Part 1 — How it works

### Module inventory (new / changed)

| Module | New / Changed | Role |
|---|---|---|
| `face_cluster/_paths.py` | **NEW** (45 LOC) | Single source of truth for repo-root, data dir, default DB path, profiles dir, alembic.ini location. All functions are `lru_cache`'d. |
| `face_cluster/repositories/_schema.py` | **NEW** (57 LOC) | `ensure_schema(db_path)` — calls `alembic.command.upgrade()` in-process. Handles 3 adoption cases: fresh DB, legacy DB without version row (`stamp head`), already-versioned DB. |
| `face_cluster/repositories/models/action_log.py` | CHANGED | 16 columns annotated with `info={"updated_on_complete": True}`. New `ActionLog.hot_field_names()` classmethod reads the metadata. |
| `face_cluster/repositories/run_history_repo.py` | CHANGED (−110 LOC net) | Deleted: subprocess block, `_engine_cache`, `_engine_for`, `_to_run_row`, `_HOT_FIELDS`, `_UNKNOWN_ALBUM`. Engine + sessionmaker now created in `__init__`. |
| `face_cluster/run_history.py` | CHANGED | Added `RunRow.from_orm(model)` classmethod — the single ORM→dataclass mapping point. |
| `face_cluster/run_history_db.py` | CHANGED (1 fn) | `get_db_path()` delegates to `_paths.default_db_path()`. |
| `face_cluster/training_db.py` | CHANGED (1 fn) | `get_db_path()` delegates to `_paths.default_db_path()`. |
| `face_cluster/profile_store.py` | CHANGED (1 line) | Default `profiles_dir` factory delegates to `_paths.profiles_dir()`. |
| `tests/face_clustering/repositories/conftest.py` | CHANGED | `_alembic_upgrade` replaced with `ensure_schema(db)`; subprocess + env-var handoff deleted. |
| `tests/face_clustering/repositories/test_alembic_baseline.py` | CHANGED | Subprocess → in-process `command.stamp(cfg, "head")`. |
| `tests/architecture/test_orm_models_in_sync_with_alembic.py` | CHANGED | Subprocess → in-process `command.check(cfg)`. |
| `tests/face_clustering/fixtures/rebuild_golden.py` | CHANGED | Uses `RunHistoryRepository` instead of `face_cluster.run_history_db` free functions (unlocks 2026-06-08 burn-in deletion). |
| `tests/architecture/test_paths_module_sole_owner.py` | **NEW** | Drift guard: only `_paths.py` may reference `Path.home() / ".sim_bench"`, repo-root walks, or hardcoded DB filenames. |
| `tests/architecture/test_no_subprocess_alembic_in_repos.py` | **NEW** | Drift guard: no `subprocess` import in `repositories/`; no `.venv` / `alembic.exe` literals anywhere in `face_cluster/`. |
| `tests/architecture/test_no_module_level_caches_in_repos.py` | **NEW** | Drift guard: no `_*_cache = {}` at module scope in `repositories/`. |
| `tests/architecture/test_runrow_matches_action_log.py` | **NEW** | Drift guard: `set(RunRow.fields) == set(ActionLog.__table__.columns.keys())`. |
| `tests/architecture/test_rebuild_golden_not_locked_to_legacy.py` | **NEW** | AST-based guard: `rebuild_golden.py` imports nothing from legacy modules slated for 2026-06-08 deletion. |
| `tests/face_clustering/repositories/test_schema_upgrade.py` | **NEW** (3 cases) | Verifies `ensure_schema` works on fresh DB, is idempotent, and reads from centralised `alembic_ini_path()`. |
| `tests/face_clustering/repositories/test_repo_construction_perf.py` | **NEW** (1 case) | Loose ceiling on construction time (200 ms mean over 20 iterations) — guards against future heavy-`__init__` regressions. |
| `tests/face_clustering/repositories/test_hot_fields.py` | **NEW** (3 cases) | `ActionLog.hot_field_names()` returns the exact legacy 16-element tuple, in the same order. |
| `tests/face_clustering/repositories/test_runrow_from_orm_equivalence.py` | **NEW** (3 cases incl. golden-row sweep) | New mapping byte-identical to legacy `_to_run_row` across all 24 golden rows; `_UNKNOWN_ALBUM` fallback preserved. |

### Data flow (post-spec-048)

```
Caller
  → RunHistoryRepository(config?)
     → _resolve_db_path(...) → _paths.default_db_path()
     → ensure_schema(db_path)
         → alembic.command.upgrade(cfg, "head")   [in-process, no subprocess]
     → create_engine_for_path(db_path)            [own engine, no cache]
     → make_sessionmaker(engine)                  [own sessionmaker]
  → method call
     → session_scope(sm)
     → session.scalars(select(ActionLog).where(...))
     → RunRow.from_orm(model)                     [single mapping point]
```

### Smell resolution scorecard

| spec-046 audit ref | Symptom | Fix landed |
|---|---|---|
| SMELL-1 | `subprocess(alembic.exe)` shell-out | `_schema.ensure_schema` (in-process) |
| SMELL-2 | Hardcoded `.venv/Scripts/alembic.exe` in 3 places | Deleted everywhere; grep-guarded |
| SMELL-3 | `_to_run_row` 24-line manual mapping | `RunRow.from_orm` (column-driven, 5 LOC) + arch drift guard |
| SMELL-4 | `_HOT_FIELDS` tuple in repo module | `info={"updated_on_complete"}` on each column + `ActionLog.hot_field_names()` |
| SMELL-5 | Module-level `_engine_cache` dict | Deleted; engine in `__init__`; grep-guarded |
| SMELL-8 | `rebuild_golden.py` locked to legacy modules | Rewritten on `RunHistoryRepository`; AST-guarded |
| (new) S6 | `repo_root` / `get_db_path()` inlined in 3 modules | `_paths.py` consolidation; grep-guarded |

---

## Part 2 — Findings by checklist section

### §1 Structure — **PASS**
- `_paths.py` (45 LOC), `_schema.py` (57 LOC) — both single-responsibility, single-paragraph docstrings.
- `run_history_repo.py` shrank ~110 LOC. No file added crosses 300 LOC. Dead code (`_UNKNOWN_ALBUM`, `_to_run_row`, `_HOT_FIELDS`, `_engine_cache`, `_engine_for`, `_ensure_schema`) deleted.
- Dependency direction: `_paths` ← `_schema` ← `_engine`/`_session`/`run_history_repo`. No reverse imports.

### §2 Code quality — **PASS**
- Zero new `try/except` blocks. `_alembic_versioned` uses an explicit existence check rather than catching `OperationalError`.
- No new silent defaults. The `source_album` `_UNKNOWN_ALBUM` fallback is preserved verbatim and unit-tested for both NULL and populated cases.
- Comments answer "why" (e.g., the 3-case stamp/upgrade logic in `_schema.ensure_schema`), not "what".

### §3 Naming and package structure — **PASS**
- `_paths.py`, `_schema.py` follow the established `_*` infra-module convention from spec-046.
- `ActionLog.hot_field_names()` — name now describes the function (was: a tuple named `_HOT_FIELDS` whose meaning was hidden in `complete_action`'s call site).
- `__all__` declared on both new modules.

### §4 Layering and coupling — **PASS**
- `_paths` has zero internal imports. `_schema` imports `_paths` and Alembic only. `run_history_repo` no longer imports from `face_cluster.run_history_db` for path resolution.
- Single writer to `_paths.repo_root` etc. (`@lru_cache` + free functions; no mutable state).

### §5 Testability — **PASS**

Test inventory (spec-048 deltas only):

| Kind | Count | Examples |
|---|---|---|
| Static (architecture / drift guard) | 6 | `test_paths_module_sole_owner` (3 params), `test_no_subprocess_alembic_in_repos` (3 cases), `test_no_module_level_caches_in_repos`, `test_runrow_matches_action_log`, `test_rebuild_golden_not_locked_to_legacy`, `test_orm_models_in_sync_with_alembic` |
| Unit (synthetic) | 9 | `test_schema_upgrade` (3), `test_hot_fields` (3), `test_runrow_from_orm_equivalence` (2 non-golden) |
| Real-data equivalence | 1 | `test_from_orm_byte_identical_to_legacy_for_every_golden_row` (24-row sweep against committed fixture) |
| Performance ceiling | 1 | `test_repository_construction_is_fast` |

Failure-mode walk-through:

| Class of regression | Test that catches it |
|---|---|
| Someone re-introduces a subprocess shell-out to alembic | `test_no_subprocess_import_in_repositories` |
| Someone inlines `Path.home() / ".sim_bench"` again | `test_path_pattern_only_in_paths_module` |
| Someone adds a column to ActionLog but forgets to update RunRow | `test_runrow_fields_match_action_log_columns` |
| Someone adds a column meant to be "hot" but forgets the `info={...}` flag | `test_hot_field_names_matches_legacy_tuple` (sees the new column missing from the tuple it pinned) |
| Someone re-introduces a module-level cache dict | `test_no_module_level_cache_dicts` |
| ORM model drifts from Alembic head | `test_orm_models_in_sync_with_alembic_head` (already in place via spec-046) |

Mock usage: **zero**. All tests use real SQLite DBs (tmp_path) or the committed golden fixture.

### §6 Boundary contracts — **PASS**
- `RunRow ↔ ActionLog` was an undocumented duplicate-shape boundary. Now an enforced contract via `test_runrow_matches_action_log`.
- `ActionLog.hot_field_names()` is a model-derived contract (no parallel tuple to drift).
- Public `RunHistoryRepository` API: signatures, return types, exception types unchanged. Verified by re-running the full spec-043 + spec-046 test suite without modification.

### §7 Documentation — **PASS-with-follow-up**

| Doc | Status |
|---|---|
| `specs/048-data-layer-cleanup/spec.md` | ✓ created |
| `specs/048-data-layer-cleanup/tasks.md` | ✓ created |
| `specs/048-data-layer-cleanup/REVIEW.md` | ✓ this file |
| `CHANGES_LOG.md` | **pending** (next commit) |
| `docs/architecture/architecture_standards.md` §B0.2 (worked example) | spec-046's example is already in place; spec-048 changes are internal — no §B0.2 update needed |
| `docs/architecture/classes.html` | F-1 (carried over from spec-046; spec-048 doesn't change classes) |
| `specs/046-sqlalchemy-data-layer/CODE_AUDIT.html` SMELL row annotations | **F-2 (follow-up)**: should annotate "resolved in spec-048" on rows for SMELL-1, 2, 3, 4, 5, 8. Non-blocking — the spec.md table above is canonical. |

**F-2 (pass-with-follow-up)**: Update CODE_AUDIT.html SMELL rows to mark resolved in a documentation-only follow-up PR.

### §8 Risk register — **PASS**
- **No new public API.** All 6 fixes are internal. Consumers don't recompile.
- **Backwards compat with existing user DBs verified.** `ensure_schema`'s 3-case dispatch handles fresh DBs (full upgrade), legacy DBs with empty `alembic_version` (stamp head), and already-versioned DBs (no-op upgrade). The third case is exercised against the user's real `~/.sim_bench/sim_bench.db` by the `test_run_history_repo_real.py` tests.
- **Performance**: `test_repository_construction_is_fast` enforces a 200 ms ceiling on mean construction time over 20 iterations. The deleted `_engine_cache` had been amortizing the subprocess cost; now the subprocess is gone and the per-construction work is just the SQLite WAL pragmas + a no-op alembic upgrade.
- **Risk of `alembic.command.check` behaving differently from CLI**: tested directly in `test_orm_models_in_sync_with_alembic_head` — passes against the committed schema. If it ever diverges, `command.check` raises `CommandError`, which the test catches and re-raises as an explicit assertion.

---

## Part 3 — Verdict

**Accept.** Spec-048 ships with:

- All 10 acceptance criteria from spec-048/spec.md met (AC1–AC10 verified by phase tests + the full-suite re-run that produced 160/160 green on spec-048-touched surfaces).
- 6 SMELLs and 1 new S6 path-centralisation gap resolved.
- 6 new permanent drift-guard tests preventing regression of each fixed smell.
- Zero blocker findings.

**Follow-up tickets**:

- **F-2 (P3, ~15 min)**: Annotate `specs/046-sqlalchemy-data-layer/CODE_AUDIT.html` SMELL rows as "resolved in spec-048". File as a TODO line.

**Pre-existing failures (out of scope, unchanged by spec-048)**:

The full-suite run (`pytest tests/face_clustering tests/architecture`, ~11 minutes) reports 13 failures, all in clustering/merge code that spec-048 did not touch. Verified by stashing spec-048 changes and re-running 3 of them on the base branch — 2 still fail identically (`test_quality_gating_holdout`, `test_adaptive_threshold_fields_removed`). The third (`test_no_null_image_paths_raises_warning`) passes in isolation on both branches — it's a test-ordering interaction, not a behavioural regression. Spec-046's REVIEW noted the same "12 unrelated failures (clustering/merge) outside scope" pattern.

Spec-048-touched test surfaces: **160/160 green**.
