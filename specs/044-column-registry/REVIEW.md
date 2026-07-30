# spec-044 — Code Review

Reviewer: Claude (Opus 4.7 / 1M)
Date: 2026-05-25
Branch: `unification/spec-040` @ `7400aac`
Base: `main`
Checklist: [`docs/guides/CODE_REVIEW_CHECKLIST.md`](../../docs/guides/CODE_REVIEW_CHECKLIST.md)

---

## Verdict (TL;DR)

**Accept with follow-ups.** 0 blockers, 3 `pass-with-followup` findings. The refactor delivers the 9-→-2 touch-point reduction promised in the spec; the Repository's external contract is provably unchanged (75/75 Repository + Service + new arch tests pass); the new drift-guard tests are load-bearing (each invariant has a documented failure mode). The follow-ups are doc-drift items (architecture HTMLs predate spec-043 and now spec-044) and a structural watch-item (the Repository module is 699 LOC and approaching the point where the spec's own footnote — "split if it grows" — applies).

The spec may flip to `Implemented` after the doc-drift follow-ups (F-1, F-2) are filed as tickets.

---

## Part 1 — How it works

### Module inventory (changed files)

| File | LOC delta | Role |
|---|---|---|
| `face_cluster/repositories/run_history_repo.py` | +298 / −111 (net +187; 699 total) | Repository class + Config + Criteria + **new** `ColumnDef` + `_COLUMNS` registry + generated constants + 3 rewritten methods |
| `tests/architecture/test_run_history_repo_column_registry.py` | +90 (new file) | 4 permanent drift-guard arch tests |
| `docs/architecture/architecture_standards.md` | +37 | New §B0.1 "Column Registry" sub-section + table-of-standards row |
| `CHANGES_LOG.md` | +27 | One entry, end-of-phase |
| `specs/044-column-registry/spec.md` | ±1 | Status `Draft` → `Code Review` |

### Dependency map

```
_COLUMNS  ─┬─►  _create_table_sql()       ─► _migrate(conn)
           ├─►  _MIGRATION_COLUMNS        ─► _migrate(conn)
           ├─►  _HOT_FIELDS               ─► complete_action(...)
           ├─►  _FILTERABLE_FIELDS        ─► _build_where(criteria)
           └─►  _INSERT_COLUMNS           ─► start_action(...)

_FILTERABLE_ALIASES ─► _build_where (column → criteria-field map)
_start_action_value(col, action_type, payload) ─► start_action

[arch test] ──► _COLUMNS              ──► invariant 3 (initial/migration partition)
            ──► _COLUMNS  ⊆ RunRow    ──► invariant 1 (RunRow parity)
            ──► _FILTERABLE_FIELDS    ──► invariant 2 (criteria field parity)
            ──► _COLUMNS where hot    ──► invariant 4 (nullable hot fields)
```

### Data flow — adding a column (the 2-touch-point claim)

Before spec-044, adding column `foo` required edits in 9 places:

1. `_CREATE_SQL` string
2. `_ALTER_COLUMNS` list (if added later) OR (1)
3. `_HOT_FIELDS` tuple (if writable from payload)
4. `start_action` INSERT column list
5. `start_action` placeholder count
6. `start_action` VALUES tuple
7. `complete_action` UPDATE SET clause
8. `complete_action` UPDATE values tuple
9. `_build_where` (if filterable)

Plus the `RunRow` dataclass.

After spec-044, the same column requires edits in exactly 2 places:

1. `_COLUMNS` (one new `ColumnDef(...)` row carrying all metadata)
2. `RunRow` dataclass (so `_row_to_run_row` exposes it to consumers)

The 4 drift-guard arch tests fail loudly if you skip either — the change is mechanical and the safety net is automated.

---

## Part 2 — Findings by section

### §1 Structure — **pass-with-followup**

| Criterion | Verdict | Notes |
|---|---|---|
| No module >300 LOC unless justified | `pass-with-followup` | `run_history_repo.py` is 699 LOC. Single named responsibility (RunHistoryRepository + its types). Spec.§3 footnote already pre-acknowledged the split option (`_columns.py`). See **F-3**. |
| No module mixes responsibilities | `pass` | One class (`RunHistoryRepository`), its Config + Criteria + ColumnDef. All directly serve the Repository's purpose. |
| No file holds multiple unrelated classes | `pass` | `RunHistoryRepoConfig`, `RunHistoryCriteria`, `ColumnDef`, `RunHistoryRepository` form a closed type family. |
| Module docstring summarizes in one sentence | `pass` | "RunHistoryRepository: typed persistence access to action_log." |
| No function >4 params | `pass` | All 4 new public/private signatures take typed objects or ≤3 args. |
| No dead code | `pass` | Removed `_CREATE_SQL`, `_ALTER_COLUMNS`, hand-written hot-field tuple — all unused after Phase 3. |
| Dependency direction respected | `pass` | No new imports across layers; all changes are internal to the Repository module. |

### §2 Code quality — **pass**

| Criterion | Verdict |
|---|---|
| try/except discipline | `pass` (no new try/except introduced) |
| if/else nesting ≤ 3 | `pass` |
| No silent defaults at boundaries | `pass` (`_start_action_value` raises naturally if `col.name` falls into an unknown bucket; `payload.get(name)` returns `None` for missing keys — same semantic as before) |
| No precedence-dependent expressions | `pass` |
| Comments answer "why" | `pass` (sparse, only where non-obvious — e.g. the `_FILTERABLE_ALIASES` rationale, the inclusive-end-of-day date filter) |

### §3 Naming and package structure — **pass**

| Criterion | Verdict | Notes |
|---|---|---|
| Module name matches contents | `pass` | `run_history_repo.py` is still 1:1 with `RunHistoryRepository`. |
| Naming consistency | `pass` | `_COLUMNS`, `_HOT_FIELDS`, `_FILTERABLE_FIELDS`, `_MIGRATION_COLUMNS`, `_INSERT_COLUMNS` follow the same `_THING_BUCKET` convention. |
| Subpackage when ≥3 files share a concern | N/A (no new files in `repositories/` package) |
| `__all__` declared | `pass` | `__all__` unchanged — `ColumnDef` is intentionally internal (not exported). Spec.§3-out-of-scope explicitly defers extraction to a shared module. |

### §4 Layering and coupling — **pass**

| Criterion | Verdict |
|---|---|
| No reverse imports | `pass` |
| No duplicated logic | `pass` — actively *removes* duplication (this was the whole spec) |
| Single writer per piece of state | `pass` — now one writer per column for INSERT (`_start_action_value`), one writer per hot field for UPDATE (the COALESCE loop in `complete_action`) |
| Two-way writers documented | N/A |

### §5 Testability — **pass** (load-bearing section)

**Test inventory** (per the checklist's mandatory listing):

| Kind | Count | Path | Status |
|---|---|---|---|
| Architecture (NEW) | 4 | `tests/architecture/test_run_history_repo_column_registry.py` | ✅ pass |
| Repository synthetic | 33 | `tests/face_clustering/repositories/test_run_history_repo_synthetic.py` | ✅ pass unchanged |
| Repository real-fixture | 3 | `tests/face_clustering/repositories/test_run_history_repo_real.py` | ✅ pass unchanged |
| Service synthetic | 31 | `tests/face_clustering/views/test_history_service_synthetic.py` | ✅ pass unchanged |
| Service real-fixture | 4 | `tests/face_clustering/views/test_history_service_real.py` | ✅ pass unchanged |
| **Total** | **75** | (sweep: `pytest tests/face_clustering/repositories/ tests/face_clustering/views/ tests/architecture/test_run_history_repo_column_registry.py -q`) | **75/75 ✅, 3.25s** |

| Criterion | Verdict |
|---|---|
| At least one real E2E test on realistic data | `pass` — 3 Repository + 4 Service tests run against `~/.sim_bench/sim_bench.db` |
| Each test has one responsibility | `pass` — each drift guard asserts exactly one invariant (RunRow parity / criteria parity / partition completeness / nullable hot fields) |
| Mock usage justified | `pass` — no mocks introduced; in-memory SQLite for synthetic tests |
| Failure-mode walk-through | `pass` — see table below |
| Test placement | `pass` — drift guards under `tests/architecture/`; behavior tests under `tests/face_clustering/` |

**Failure-mode walk-through** (each known bug class spec-044 was supposed to prevent):

| Failure class | Test that catches it | What fails |
|---|---|---|
| Added column to `_COLUMNS`, forgot `RunRow` field | `test_runrow_fields_match_columns_registry` | `AssertionError: _COLUMNS declares ['new_col'] but RunRow doesn't expose them` |
| Marked column `filterable=True`, forgot `RunHistoryCriteria` field | `test_filterable_columns_have_matching_criteria_fields` | `AssertionError: Column 'new_col' is marked filterable but criteria field 'new_col' doesn't exist` |
| Column put in both initial CREATE and ALTER migration | `test_initial_and_migration_partition_is_complete` | `AssertionError: Columns ['new_col'] appear in both initial and migration` |
| Marked column `hot=True` AND `nullable=False` (would crash on insert when payload omits the key) | `test_only_nullable_hot_fields` | `AssertionError: Hot columns ['new_col'] are NOT NULL` |
| Generated `_create_table_sql()` produces a different schema than the legacy `_CREATE_SQL` | (TEMPORARY in Phase 2; deleted Phase 5) — Phase 2 test `test_generated_create_table_matches_legacy_create_sql` covered this *during the refactor*; not a recurring concern after Phase 3 |

### §6 Boundary contracts — **pass**

| Criterion | Verdict | Notes |
|---|---|---|
| Each new boundary has runtime enforcement | `pass` | `ColumnDef` is an internal frozen-slotted dataclass — Python enforces immutability + slot membership at runtime |
| `extra="forbid"` on new Pydantic BaseModels | N/A | No Pydantic models in this spec |
| Pandera schemas | N/A | No DataFrames touched |
| Contract claimed but not invoked = fail | `pass` | `_FILTERABLE_ALIASES` is consulted at runtime in `_build_where` AND validated by `test_filterable_columns_have_matching_criteria_fields`. The new `ColumnDef` flags (`initial` / `hot` / `filterable`) are each consumed by at least one runtime code path. |
| Config-knob → producer check (SIGHTING-061 generalization) | `pass` | The 4 drift guards *are* this check, applied to columns: every flag is required to have a downstream consumer that uses it. |

### §7 Documentation deliverables — **pass-with-followup**

| Deliverable | Status |
|---|---|
| `spec.md` | ✅ present, status flipped to `Code Review` |
| `tasks.md` | ✅ present, 6 phases / D1–D6 design notes |
| `REVIEW.md` | ✅ this file |
| `EXECUTIVE_SUMMARY.html` | N/A (internal refactor; no cross-team surface) |
| `docs/architecture/db_schemas.html` | ✅ no DB schema *change* — refactor produces byte-identical schema (proven by the now-removed Phase 2 parity test) |
| `docs/architecture/classes.html` | ⚠️ **F-1** — does not mention `ColumnDef` (new dataclass) or `RunHistoryRepository` (spec-043 oversight). CLAUDE.md update mandate applies. |
| `docs/architecture/data_flow.html` | ✅ no pipeline-step change |
| `docs/architecture/index.html` | ✅ no new doc added |
| `docs/architecture/db_global.html` | ⚠️ **F-2** — `action_log` section still describes legacy module-level functions; missing the `producer` column row and the `RunHistoryRepository` writer-module update. Predates spec-044 (spec-043 doc drift), surfaced now. |
| `CHANGES_LOG.md` | ✅ entry added |
| `LEARNINGS.md` | N/A (no new failure class) |
| `docs/architecture/architecture_standards.md` | ✅ §B0.1 added; row added to standards table |
| Memory | N/A (no new cross-session guidance) |

### §8 Risk register — **pass**

| Criterion | Verdict | Notes |
|---|---|---|
| Known deferred work has tickets | `pass` | Spec §3 documents 4 out-of-scope items; this REVIEW.md files **F-1/F-2/F-3** for the new ones |
| Workarounds named and tested | N/A — no workarounds introduced |
| Backwards-compat surface | `pass` | Repository external contract unchanged; 33 + 35 + 3 existing tests pass unmodified — the contract guard |
| Hot-path performance measured | `pass-with-followup` | Spec §6 acknowledged the 24-row iteration adds microseconds. Test suite runtime (3.25s vs. legacy ~3.5s) is consistent — no measurable regression. Not a follow-up; the matrix is small enough. |

---

## Part 3 — Verdict per area + follow-up tickets

| Area | Verdict |
|---|---|
| Structure | `pass-with-followup` — F-3 |
| Code quality | `pass` |
| Naming | `pass` |
| Layering | `pass` |
| **Testability** | `pass` |
| **Boundary contracts** | `pass` |
| Documentation | `pass-with-followup` — F-1, F-2 |
| Risk register | `pass` |

### Follow-up tickets (3)

#### F-1 — `docs/architecture/classes.html` missing `ColumnDef` + `RunHistoryRepository`

**Severity**: docs drift (not a runtime blocker).
**Source**: §7 documentation mandate (CLAUDE.md).
**Disposition**: append to `TODO.md` as a doc-housekeeping item; can be batched with F-2.
**Specifics**:
- Add `RunHistoryRepository` section (constructor / public methods / Config / Criteria) — spec-043 oversight.
- Add `ColumnDef` row to the dataclass section — spec-044 addition.
- Cross-link to `architecture_standards.md` §B0 + §B0.1.

#### F-2 — `docs/architecture/db_global.html` `action_log` section is stale

**Severity**: docs drift (not a runtime blocker, but misleads new contributors).
**Source**: §7 documentation mandate.
**Disposition**: append to `TODO.md`; can be batched with F-1 in one doc-update PR.
**Specifics** (line numbers from current file):
- Line 285: "Writer module / functions: `face_cluster.run_history_db` — module-level functions (no class wrapper today)" → replace with `face_cluster.repositories.RunHistoryRepository` (spec-043 outcome).
- Line 297 column table: add `producer` column row (spec-040 T4 + spec-044 column-registry entry).
- Line 283 `<span class="src">`: `run_history_db.py` → `repositories/run_history_repo.py`.

#### F-3 — `run_history_repo.py` is approaching the "split when it grows" threshold

**Severity**: structural watch-item; not actionable yet.
**Source**: §1 module size + spec §3 out-of-scope footnote.
**Disposition**: file as a `[~]` deferred TODO; revisit when the next column-registry consumer (e.g., `ClusterAnalysisRepository`) materializes — that's the natural trigger for extracting `ColumnDef` to `face_cluster/repositories/_column_def.py`.
**Specifics**: file is now 699 LOC (up from ~590 after spec-043). Sub-700 is still defensible given the single named responsibility and the module docstring summarizes it cleanly, but a third growth phase will push past 800 LOC. The split target — per spec.§3 — is a sibling `_columns.py` for `ColumnDef` + `_COLUMNS` + the generated constants + `_start_action_value`. ~120 LOC carve-out; mechanical.

---

## Spec status disposition

- **No §1–§7 criterion is marked `fail`.** No blockers to handoff.
- F-1, F-2, F-3 are all `pass-with-followup` items per the checklist's grading.
- Recommend flipping `specs/044-column-registry/spec.md` from `Code Review` → `Implemented` once F-1 and F-2 are filed as TODO items (no need to *fix* them in this spec; the rule is "filed, not fixed").

Awaiting user decision on whether to (a) file the 3 follow-ups and flip status now, or (b) batch F-1/F-2 into a doc-update PR before flipping.
