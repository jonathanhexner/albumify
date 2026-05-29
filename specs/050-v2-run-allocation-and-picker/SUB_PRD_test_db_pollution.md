# Sub-PRD — Test pollution of the production action_log DB

**Created**: 2026-05-28
**Status**: Draft — awaiting decision on plan
**Parent spec**: spec-050 (v2 app run allocation + picker)
**Surfaced by**: user-reported error from a real v2 app run after spec-050 shipped.

---

## What happened, in order

1. User restarts the v2 app, clicks the Clusters tab.
2. Picker auto-selects the most-recent `fc_app_v2` row.
3. That row's `output_dir` column points at `C:\Users\Jonathan Hexner\AppData\Local\Temp\pytest-of-Jonathan Hexner\pytest-413\test_params_path_does_not_emit0\out\` — a directory that hasn't existed since pytest tore it down.
4. Clusters tab tries to load `face_clustering.db` from that path and shows the error.

The picker is doing exactly what spec-050 said it would: read recent `fc_app_v2` rows from `action_log` and let the user pick one. The row it picked is real — it's just **a row that should never have been there.**

## Why orphan rows exist

`~/.sim_bench/sim_bench.db` has **18 rows pointing at pytest temp directories**. Sample:

```
id=438  status=failed     album='kwargs_test_album'        dir=.../pytest-413/test_params_path_does_not_emit0/out
id=437  status=failed     album='kwargs_test_album'        dir=.../pytest-413/test_no_kwargs_runs_without_cr0/out
id=435  status=failed     album='cli_save_profile_test'    dir=.../pytest-412/test_save_profile_round_trip0/out2
id=434  status=complete   album='e2e_test_album'           dir=.../pytest-411/v2_e2e_out0
id=432  status=complete   album='e2e_test_album'           dir=.../pytest-410/v2_e2e_out0
id=431  status=complete   album='cli_full_run_fixture'     dir=.../pytest-410/test_full_run_against_fixture0/out
...
```

These come from three test files that invoke `run_v2_pipeline` and (transitively) `RunHistoryRepository()`:

| Test file | Has DB-isolation fixture? | Polluted? |
|---|---|---|
| `tests/face_clustering/test_run_v2_pipeline_kwargs.py` | **No** — never had one | Yes, every run |
| `tests/face_clustering/test_fc_app_v2_e2e.py` | Had one, but targeting `run_history_db.get_db_path` | Polluted for **every run between spec-048 Phase 7 (2026-05-28 AM) and spec-050 Phase 6 (2026-05-28 PM)** |
| `tests/face_clustering/test_run_v2_script.py` | Same — targeted the wrong attribute | Same window |

The `test_fc_app_v2_e2e.py` / `test_run_v2_script.py` window opened because **spec-048 Phase 7** changed `RunHistoryRepository._resolve_db_path` from importing `face_cluster.run_history_db.get_db_path` to importing `face_cluster._paths.default_db_path` directly. Existing monkeypatches kept targeting the now-bypassed name. They went silently ineffective. I caught and fixed them in spec-050 Phase 6, but new orphan rows had already been written.

## Root cause: a class of bug, not an instance

**Any test in this repo that constructs `RunHistoryRepository()` with no `db_path` argument writes to the user's real DB unless the test explicitly monkeypatches `face_cluster._paths.default_db_path`.**

There is no guardrail today. Specifically:

- No `conftest.py` autouse fixture that redirects the default DB for the test session.
- No arch test that asserts every test calling `run_v2_pipeline` (or `RunHistoryRepository()`) sits inside an isolation fixture.
- The Repository's `_resolve_db_path` happily reaches into the user's filesystem with zero guard against "are we inside a test process?"

The next contributor to add a test that uses the v2 pipeline will repeat the same mistake unless they happen to read the existing tests carefully and copy the right monkeypatch. That's not a guardrail; that's an attractive nuisance.

## What the picker should and shouldn't do

The picker is **not** the bug — it's the surface that revealed the bug. But because orphan rows can plausibly exist for non-test reasons too (user deleted a run dir manually, ran on a network drive that's now disconnected, etc.), the picker should handle orphans gracefully.

Two reasonable approaches:

- **A. Silent filter**: don't show rows whose `output_dir / face_clustering.db` no longer exists. Pro: clean UX. Con: hides the fact that the user has a polluted DB; they can't tell anything was filtered.
- **B. Visual marker**: show the row with a `[missing]` suffix and disable selection (or warn on click). Pro: visible, debuggable. Con: clutters the dropdown with rows the user can't act on.

I think the right answer is **B with an explicit count** ("18 orphan runs hidden — see History tab" or similar), so it's discoverable without being noisy. This is a UX call you should make, not me.

## Decision points (please confirm before I plan)

1. **DB isolation guardrail**: Add a session-wide autouse fixture in `tests/conftest.py` that redirects `_paths.default_db_path` to a per-session tmp file? Or scope it more narrowly (e.g., per-package conftest, only for tests that import `RunHistoryRepository`)?
   - **My recommendation**: session-wide. The risk of accidentally hitting the production DB in any test is too high for an opt-in design. A test that genuinely needs the real DB (we have none today) can opt out explicitly.
2. **Architecture test**: Add a test that asserts no test in `tests/**` calls `RunHistoryRepository()` outside an isolation fixture? Or trust the session-wide fixture and skip the arch test?
   - **My recommendation**: skip the arch test if we have the session-wide fixture. The fixture makes the arch test redundant.
3. **Picker UX for orphans**: A (hide silently), B (show with `[missing]` marker), or C (something else)?
   - **My recommendation**: B, with a footnote count.
4. **Cleanup of the existing 18 orphan rows**: Should I write a one-shot cleanup script (`scripts/cleanup_orphan_action_log_rows.py`), document a SQL query in the sub-PRD for you to run by hand, or both?
   - **My recommendation**: SQL query in the spec + a small `scripts/` helper that you run once. Don't auto-run on app startup — that's silent surgery on a user DB.
5. **Pollution recovery audit**: Beyond the 18 obvious pytest-tmp rows, should we check for other suspicious patterns (e.g., rows with `source_album` matching test fixture names like `e2e_test_album`)? Same `LIKE '%pytest%' OR LIKE '%Temp%'` filter would catch them all on Windows; needs widening for cross-platform tmp paths.
   - **My recommendation**: include the wider filter in the cleanup query (`pytest`, `Temp`, `/tmp/`, `tmpfs`).

---

## Out of scope for this sub-PRD

- Spec-049 (broader test-suite-recovery PRD). Spec-049's failures are different — backend tests asserting wrong things. This sub-PRD is about test side-effects on the user's environment.
- A general "tests must not touch any user-owned file" guardrail. We have other tests that write to `~/.sim_bench/profiles/` etc.; those would need a broader sandboxing pattern. Scope here is the action_log DB specifically.

---

## Recommended next step

Confirm the 5 decision points above; I'll then write the **fix plan** (probably spec-051) covering: (a) test isolation guardrail, (b) picker orphan UX, (c) one-shot cleanup script, (d) arch test or not, (e) cross-platform tmp-path filter.

Until that lands, the current behavior is: every `pytest tests/face_clustering/test_run_v2_pipeline_kwargs.py` run adds 3 new orphan rows to your real action_log. The fix-forward path is to land (a) before running any v2 tests again.
