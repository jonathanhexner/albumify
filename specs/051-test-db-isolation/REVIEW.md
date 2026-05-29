# Code Review: spec-051 — Test DB isolation + orphan UX

**Reviewed**: 2026-05-28
**Reviewer**: Claude (self-review against `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Status**: PASS — 0 blockers, 0 follow-ups
**Branch**: `unification/spec-040`

---

## Part 1 — How it works

### Module inventory

| Module | New / Changed | Role |
|---|---|---|
| `tests/conftest.py` | CHANGED (+30 LOC) | Session-scoped autouse fixture `isolate_action_log_db` redirects `_paths.default_db_path` to a per-session tmp file. |
| `tests/architecture/test_action_log_db_isolation.py` | NEW (40 LOC) | Meta-guard: fixture exists, is session-scoped + autouse, targets the right module. |
| `app/face_clustering_v2/components/run_picker.py` | CHANGED | `RunPickerEntry.is_orphan` flag; `_all_entries_from_repo` (replaces `_entries_from_repo`, alias kept); `_partition_entries`; `_format_label` adds `[missing]` prefix; `render_run_picker` shows orphan footnote. |
| `app/face_clustering_v2/tabs/clusters_tab.py` | CHANGED (+6 LOC) | Blocks orphan selection with `st.warning(...)`; returns before constructing RunStore. |
| `scripts/cleanup_orphan_action_log.py` | NEW (120 LOC) | `--dry-run` (default) / `--apply --yes-i-counted N` CLI. Cross-platform tmp-path filter. |
| `tests/face_clustering/test_v2_run_picker.py` | CHANGED (+2 tests) | Partition function correctness; `[missing]` label prefix. |
| `tests/face_clustering/test_cleanup_orphan_action_log.py` | NEW (5 cases) | Dry-run no-op; apply deletes only orphans; wrong-count aborts; idempotent; cross-platform pattern coverage. |

### Data flow

```
Production app:
  RunHistoryRepository()
    → _resolve_db_path(None)
    → _paths.default_db_path()     ← real ~/.sim_bench/sim_bench.db

Pytest session:
  conftest.isolate_action_log_db (autouse, session)
    monkeypatches _paths.default_db_path → tmp_path_factory dir
  → every RunHistoryRepository() in tests targets the tmp DB
  → production DB never touched

Cleanup script:
  default_db_path()
    → SELECT ... WHERE output_dir LIKE '%pytest%' OR ...
    → print (dry-run) | DELETE (with --yes-i-counted N matching)
```

---

## Part 2 — Findings by checklist section

### §1 Structure — **PASS**
- `cleanup_orphan_action_log.py` 120 LOC, single CLI responsibility.
- `run_picker.py` grew by ~30 LOC (orphan partition + footnote); still under 150 LOC, single component.
- No file > 200 LOC. Dependency direction: tests → conftest → `_paths`; script → `_paths` + `sqlite3` (no Repository roundtrip — direct SQL because the script is administrative).

### §2 Code quality — **PASS**
- No try/except suppressing errors. Script uses argparse return codes; tab uses `st.warning` to surface the orphan condition explicitly.
- `--apply` requires `--yes-i-counted N` matching the real count — no silent destructive default.
- Comments explain *why* (e.g., the autouse fixture's docstring spells out the opt-out pattern for tests that genuinely need the real DB).

### §3 Naming and package structure — **PASS**
- `isolate_action_log_db` — verb + noun, names the contract.
- `_partition_entries` / `_all_entries_from_repo` — clear roles. Old `_entries_from_repo` kept as alias for the 4 existing spec-050 tests that import it.
- Script name describes what it does.

### §4 Layering and coupling — **PASS**
- No reverse imports. Script reaches into `_paths` (intentional — it's an admin tool).
- Single writer for the autouse fixture (only conftest.py).

### §5 Testability — **PASS**

Test inventory:

| Kind | Count | File |
|---|---|---|
| Arch (meta-guard) | 2 | `test_action_log_db_isolation.py` |
| Unit (picker partition + label) | 2 | `test_v2_run_picker.py` (additions) |
| Integration (cleanup CLI) | 5 | `test_cleanup_orphan_action_log.py` |
| **Total new** | **9** | |

Failure-mode walk-through:

| Class of regression | Test that catches it |
|---|---|
| Someone removes `isolate_action_log_db` from conftest | `test_isolate_action_log_db_fixture_is_session_autouse` |
| Someone downgrades the fixture to `function`-scope or non-autouse | Same test (asserts scope + autouse) |
| Someone retargets the fixture to a non-`_paths` attribute | `test_isolate_action_log_db_targets_paths_module` |
| Picker mis-classifies an orphan as loadable | `test_partition_separates_orphan_from_loadable` |
| Cleanup script deletes a non-orphan | `test_apply_deletes_only_orphans` |
| Cleanup `--apply` runs without the count guard | `test_apply_with_wrong_count_aborts` |
| Tmp-path filter misses a platform | `test_filter_covers_cross_platform_tmp_patterns` |

Zero mocks. All tests use real SQLite tmp DBs.

### §6 Boundary contracts — **PASS**
- `RunPickerEntry` gains `is_orphan: bool` with default `False` — non-breaking for existing call sites.
- Script CLI is typed via argparse with explicit return codes (0 / 2 / 3) — no exception-leak surface.
- Autouse fixture's contract is documented in the docstring including the opt-out pattern.

### §7 Documentation — **PASS**
- `spec.md` + `tasks.md` + `REVIEW.md` (this file) all present.
- `CHANGES_LOG.md` entry queued for Phase 5 close.
- SIGHTING-076 to be filed and immediately closed (cross-referencing the user-reported orphan-row symptom from the previous session).
- `docs/architecture/` — no architecture-HTML updates required (no schema / class shape changes).

### §8 Risk register — **PASS**
- Production DB pollution: stopped at the source (autouse fixture); verified by snapshot delta = 0 in Phase 1.
- Historical orphan rows: cleanup script ready; user runs `--apply` explicitly.
- UX surface for orphans created outside test pollution (user deletes a real run dir): handled identically — `[missing]` marker + warning + no traceback.

---

## Part 3 — Verdict

**Accept.** Spec-051 ships with:
- All 11 acceptance criteria green (AC10 verified by Phase 4 full sweep — placeholder until sweep completes; AC11 = this document).
- The reported user symptom (orphan rows surfacing in the v2 picker) resolved at three independent layers: (1) source fix via autouse fixture; (2) UX safety net in the picker + Clusters tab; (3) cleanup helper for historical rows.
- 9 new tests; zero deleted.
- Zero blockers, zero follow-ups.

**Note**: the user must run `scripts/cleanup_orphan_action_log.py --apply --yes-i-counted 18` themselves to remove the 18 historical orphan rows from their real DB. The dry-run output is captured in the session log for review.

---

## Phase 4 sweep result

Full `pytest tests/face_clustering tests/architecture`:

| | Pre-spec-051 baseline | Post-spec-051 | Delta |
|---|---|---|---|
| Passed | 683 | **699** | +16 |
| Failed | 8 | **5** | −3 |
| Errors | 4 | **0** | −4 |
| Skipped | 9 | 9 | 0 |

**Zero new failures introduced.** Net improvement of 7 tests (the 4 errors in `test_fc_app_v2_e2e.py` from spec-050 wiring were already fixed by spec-050 Phase 6; the autouse fixture didn't regress anything). The remaining 5 failures are all pre-existing and tracked in spec-049 / SIGHTING-071/072/073/074.

AC10 satisfied.

**Production DB pollution check** (AC1): snapshot delta after running the new spec-051 isolated tests = **0 new rows** in `~/.sim_bench/sim_bench.db`. The autouse fixture works.
