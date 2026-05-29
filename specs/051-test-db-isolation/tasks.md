# Tasks: spec-051 Test DB isolation + orphan UX

**Status**: Implemented
**Estimated effort**: ~2 hours

5 phases. Phase 1 alone fixes the production-pollution bug; Phases 2-3 cover the UX and historical cleanup; Phases 4-5 verify and gate.

---

## Phase 1 — Session-wide DB isolation (fixes the class of bug)

- [ ] T010 In `tests/conftest.py`, add a session-scoped autouse fixture:
  ```python
  @pytest.fixture(scope="session", autouse=True)
  def isolate_action_log_db(tmp_path_factory):
      """Redirect _paths.default_db_path to a per-session tmp file so
      no test can accidentally write to the user's real
      ~/.sim_bench/sim_bench.db. Tests that need a different path
      can still monkeypatch in their own fixture (overrides this one).
      Tests that need the REAL DB construct RunHistoryRepository with
      an explicit db_path argument — they bypass default_db_path entirely."""
      import face_cluster._paths as _paths
      fake_db = tmp_path_factory.mktemp("isolated_action_log") / "sim_bench.db"
      orig = _paths.default_db_path
      _paths.default_db_path = lambda: fake_db
      yield fake_db
      _paths.default_db_path = orig
  ```
- [ ] T011 Add `tests/architecture/test_action_log_db_isolation.py` asserting the fixture exists in `tests/conftest.py` with `autouse=True, scope="session"` and the right name. (Meta-guard: if someone removes the fixture, this test fails.)
- [ ] T012 Manual verification:
  - Snapshot `SELECT COUNT(*) FROM action_log` on the real DB.
  - Run `pytest tests/face_clustering tests/architecture -q`.
  - Re-snapshot. Delta MUST be 0.
- [ ] T013 Remove now-redundant per-fixture monkeypatches that targeted `_paths.default_db_path` (the autouse covers them). Keep ones that target `run_history_db.get_db_path` (those test the legacy shim directly).

**Check**: arch test passes; manual delta is 0.

---

## Phase 2 — Picker orphan UX

- [ ] T020 In `app/face_clustering_v2/components/run_picker.py`:
  - Split `_entries_from_repo` into `_all_entries_from_repo` (returns everything) + `_partition_entries(entries)` returning `(loadable, orphan)`.
  - An entry is `orphan` when `output_dir / "face_clustering.db"` does not exist.
  - `_format_label` prefixes `[missing] ` to orphan entries.
  - `RunPickerEntry` gains a `is_orphan: bool` field.
  - `render_run_picker` renders the combined list (loadable first, orphans after) and, when orphans exist, shows a footnote: `st.caption(f"{N} orphan run(s) hidden from default. Run scripts/cleanup_orphan_action_log.py --apply to remove them.")`.
- [ ] T021 In `app/face_clustering_v2/tabs/clusters_tab.py`:
  - When `entry.is_orphan` is true, show `st.warning("This run's directory no longer exists at <path>. It may have been deleted or moved. Pick another run or use Advanced override.")` and return — do not construct RunStore.
- [ ] T022 Add 2 cases to `tests/face_clustering/test_v2_run_picker.py`:
  - Partition function correctly classifies orphan vs loadable.
  - `_format_label` adds `[missing]` prefix for orphans only.

**Check**: 6/6 picker tests + 3/3 picker AppTests green.

---

## Phase 3 — One-shot cleanup helper

- [ ] T030 Create `scripts/cleanup_orphan_action_log.py`:
  - Args: `--dry-run` (default true), `--apply`, `--yes-i-counted N`.
  - Connects to `face_cluster._paths.default_db_path()`.
  - Selects rows from `action_log` whose `output_dir` matches the cross-platform tmp filter (Locked decision §5).
  - Dry-run: prints count + first 20 rows in a readable table.
  - `--apply`: requires `--yes-i-counted N` matching the actual count; deletes; prints the deletion count.
  - Idempotent: second `--apply` finds zero and exits 0.
- [ ] T031 Create `tests/face_clustering/test_cleanup_orphan_action_log.py` (4 cases):
  - Dry-run produces no DB changes.
  - `--apply --yes-i-counted N` deletes only the matching rows.
  - Cross-platform filter coverage (pytest, Temp, /tmp/, tmpfs, AppData\Local\Temp).
  - Re-run after `--apply` finds zero (idempotency).
- [ ] T032 Run the script in dry-run mode against the user's real DB. Confirm the 18 orphan rows are listed and nothing else. Show the user the output before they run `--apply`.

**Check**: 4/4 cleanup tests green; dry-run output matches expectation.

---

## Phase 4 — Full sweep

- [ ] T040 Run `pytest tests/face_clustering tests/architecture -q`. No regressions vs the pre-spec-051 baseline (currently 8 failing + 4 errors, all pre-existing — they should stay the same count, no new failures introduced).
- [ ] T041 Manual smoke of the v2 app:
  - Run a v2 pipeline → new run appears in picker (no `[missing]`).
  - Manually delete the run dir on disk → restart app → that run now shows with `[missing]` and selecting it shows the warning.
  - Run cleanup script `--dry-run` → it lists the manually-deleted run as an orphan.

**Check**: full suite no worse; manual smoke clean.

---

## Phase 5 — Code review gate

- [ ] T050 Run `/code-review` → produces `specs/051-test-db-isolation/REVIEW.md`.
- [ ] T051 Walk all 8 sections of `docs/guides/CODE_REVIEW_CHECKLIST.md`. Verdict per section.
- [ ] T052 If any high-severity finding, resolve before flipping spec to Implemented.
- [ ] T053 Flip spec.md + tasks.md status: Draft → In Progress → Code Review → Implemented.
- [ ] T054 `CHANGES_LOG.md` entry under `[BUGFIX]`.
- [ ] T055 SIGHTING-076 (filed during spec authoring) marked RESOLVED with cross-link to spec-051.

**Check**: REVIEW.md exists; zero blockers; spec is Implemented.

---

## Test delta

| Phase | Added | Type |
|---|---|---|
| 1 | 1 | arch (fixture-exists guard) |
| 2 | 2 | unit (picker partition + label) |
| 3 | 4 | integration (cleanup CLI) |
| **Total** | **7** | |

No tests deleted. Per-fixture monkeypatches that targeted `_paths.default_db_path` removed in T013 (now redundant under the session-wide autouse), but the tests themselves keep running.

---

## Sequencing rationale

- Phase 1 first because it's the only one that stops new pollution. Even if Phases 2-5 slip, the bleeding stops.
- Phase 2 before 3 because the picker UX is the user-visible recovery surface; cleanup is optional housekeeping.
- Phase 3 before 4 because the cleanup script needs to run before the manual smoke (otherwise the picker still shows the 18 historical orphans).
- Phase 5 mandatory per the new CLAUDE.md §Implementation gate.
