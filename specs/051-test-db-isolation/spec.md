# spec-051 — Test isolation for the production action_log DB + orphan UX

**Created**: 2026-05-28
**Status**: Implemented
**Predecessors**: spec-050 (v2 picker), spec-048 (data layer)
**Linked sightings**: SIGHTING-076 (filed below as part of this spec)

---

## Problem

The test suite has been writing rows into the user's real
`~/.sim_bench/sim_bench.db` action_log table for an unknown number of
days. As of today, **18 orphan rows** point at pytest temp directories
that no longer exist. The user discovered this when spec-050's run
picker auto-selected one of those rows and the Clusters tab failed to
load the (deleted) `face_clustering.db`.

This is a class-of-bug, not an instance: any test that constructs
`RunHistoryRepository()` with no `db_path` writes to the real DB. There
is no guardrail.

---

## Locked decisions (per user "go ahead")

1. **Session-wide autouse fixture** in `tests/conftest.py` redirecting `_paths.default_db_path` to a per-session tmp file. No test can hit the real DB without explicit opt-out.
2. **No architecture test** — the session-wide fixture makes a "you forgot the fixture" arch test redundant.
3. **Picker shows orphans with a `[missing]` marker** plus a footnote count below the dropdown. Selection of an orphan is blocked at the Clusters tab with a clear "this run's directory no longer exists" message — no traceback.
4. **One-shot cleanup helper** at `scripts/cleanup_orphan_action_log.py` with `--dry-run` (default) and `--apply` modes. Prints the rows it would delete; deletes only with `--apply`. Documented in the spec; not auto-run.
5. **Cross-platform tmp filter** for the cleanup: matches output_dir containing any of `pytest`, `Temp`, `/tmp/`, `tmpfs`, `AppData\Local\Temp`, or having `tmp` as a path component.

---

## What we build

| Module | Role | New / changed |
|---|---|---|
| `tests/conftest.py` | Add session-scoped autouse fixture `isolate_action_log_db` that points `_paths.default_db_path` at `tmp_path_factory.mktemp("isolated_action_log") / "sim_bench.db"` for the whole test session | **CHANGED** |
| `app/face_clustering_v2/components/run_picker.py` | `_entries_from_repo` partitions rows into `loadable` + `orphan`; `render_run_picker` shows orphans with `[missing]` prefix and a footnote count | **CHANGED** |
| `app/face_clustering_v2/tabs/clusters_tab.py` | When user picks an orphan entry, show `st.warning(...)` and return early — no `RunStore` construction attempt | **CHANGED** |
| `scripts/cleanup_orphan_action_log.py` | CLI tool: `--dry-run` (default) prints orphan rows; `--apply` deletes them. Filter per Locked decision §5. | **NEW** |
| `tests/face_clustering/test_v2_run_picker.py` | Add 2 cases covering orphan partitioning + label formatting | **CHANGED** |
| `tests/face_clustering/test_cleanup_orphan_action_log.py` | 4 cases: dry-run produces no DB changes; `--apply` deletes only the matching rows; cross-platform filter coverage; idempotency (re-run after `--apply` finds nothing) | **NEW** |
| `tests/architecture/test_action_log_db_isolation.py` | Single test asserting the autouse fixture is wired in tests/conftest.py — a meta-guard against someone removing the isolation by accident | **NEW** |

---

## What we don't build

- A general "tests must not touch any user file" sandbox. Out of scope — this spec covers the action_log DB specifically.
- An app-startup orphan check / auto-cleanup. The user must run the cleanup script explicitly. Auto-deleting rows from a user DB at app launch is too risky.
- Migration of historical orphan rows from other tables (`pipeline_runs`, etc.) — same isolation problem may exist there but is out of scope; tracked as follow-up if surfaced.

---

## Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| AC1 | Running the full pytest suite produces zero new rows in `~/.sim_bench/sim_bench.db`. | Phase 1 manual verification: snapshot row count before + after `pytest tests/`; assert delta = 0 |
| AC2 | The autouse fixture is session-scoped and applies to every test file without opt-in. | Phase 1 — arch test asserts the fixture exists with `autouse=True, scope="session"` |
| AC3 | A test explicitly opting out (a deliberate "use real DB" test, if any) can still do so via a marker or fixture override. | Phase 1 — documented escape hatch in the fixture's docstring |
| AC4 | Picker rendering: orphan rows show `[missing]` prefix; loadable rows do not. | Phase 2 — unit test on `_format_label` + partition function |
| AC5 | Picker footnote shows accurate orphan count when orphans exist; hidden when count is zero. | Phase 2 — AppTest |
| AC6 | Clusters tab blocks selection of an orphan with a `st.warning` and no traceback. | Phase 2 — AppTest |
| AC7 | `scripts/cleanup_orphan_action_log.py --dry-run` lists the 18 known orphans without touching the DB. | Phase 3 — integration test against a seeded DB |
| AC8 | `--apply` deletes only the matching rows; non-orphan rows are preserved. | Phase 3 — integration test |
| AC9 | Cleanup is idempotent: second `--apply` finds zero orphans. | Phase 3 — integration test |
| AC10 | `pytest tests/face_clustering tests/architecture` — no regressions vs. pre-spec-051 baseline. | Phase 4 — full sweep |
| AC11 | `/code-review` produces REVIEW.md with zero high-severity findings. | Phase 5 |

---

## Risks

| Risk | Mitigation |
|---|---|
| Session-wide autouse fixture breaks tests that expect to read the real DB (e.g., `test_run_history_repo_real.py`). | Those tests construct `RunHistoryRepository` with an **explicit `db_path`** pointing at the real DB — they bypass `default_db_path()` entirely. The fixture only redirects the default-path code path. Verified in Phase 1. |
| `--apply` deletes a row the user actually wanted (e.g., a run they manually moved to a different drive). | Default is `--dry-run`. `--apply` shows the SQL it would run and requires the user to type the row count to confirm (`--yes-i-counted N`). |
| Existing test fixtures that already monkeypatch `_paths.default_db_path` collide with the autouse fixture. | The autouse fixture sets a base path; per-test monkeypatches override it (pytest's `monkeypatch.setattr` runs after the autouse setup). Verified in Phase 1 with `test_fc_app_v2_e2e.py` which still has its own per-test monkeypatch. |

---

## Definition of Done

All 11 acceptance criteria green + the 18 known orphan rows cleaned up (run the script once after AC9 passes) + REVIEW.md filed + CHANGES_LOG entry + SIGHTING-076 closed.
