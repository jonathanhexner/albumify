---
title: "spec-114 REVIEW - Compare two jobs in the app"
---

# spec-114 REVIEW — Compare two jobs in the app

*2026-10-17. Reviewer: Claude (against `docs/guides/CODE_REVIEW_CHECKLIST.md`). Scope: the uncommitted spec-114
change set on `spec/106-albumify-web-showcase` (control DB columns + reaper, `JobComparer`, compare endpoint,
Compare runs page, Jobs page filter, regression script on the main job list, HTML report removal).*

**Verdict: ACCEPT with follow-ups.** 0 `fail`, 5 `pass-with-followup` (all ticketed). No V2 (face-clustering app) tab
changed, so the Budapest gate does not apply to this spec.

## Part 1 — How it works

| Area | Module | Role |
|---|---|---|
| Schema | `sim_bench/control/models.py` | `Job.tag`, `Job.worker` (nullable) |
| Schema upgrade | `sim_bench/control/db.py` `_add_missing_columns` | adds missing nullable model columns to an existing DB; refuses NOT NULL |
| Dead jobs | `sim_bench/control/jobs.py` `this_worker`, `worker_alive`, `JobRepo.reap_orphans`, `set_tag` | `host:pid:start-time`; fail live jobs whose worker is gone |
| Job flow | `sim_bench/serving/job_service.py` | `SubmitInputs.tag`; worker recorded at STAGING / load |
| Comparison | `sim_bench/control/compare.py` `JobComparer` | both albums via `AlbumBuilder` (edits applied) -> `AlbumComparator`; per-photo decisions; people + face crop |
| API | `sim_bench/api/routers/jobs.py` `GET /jobs/{b}/compare?with={a}`; `schemas/jobs.py` `CompareOut` (+4 models) | 404 / 409 / 422 refusals; reaper on list/read |
| App | `app/streamlit/pages/compare.py` (new), `pages/jobs.py`, `api_client.py`, `main.py`, `components/sidebar.py` | Compare runs page; regression runs hidden on Jobs page |
| Regression | `scripts/albumify_regression.py` | main job list, tags, `mark_baseline`; report removed |

```
compare.py (page) -> api_client -> GET /jobs/{b}/compare -> JobComparer -> AlbumBuilder (control DB)
                                                                       -> AlbumComparator (sim_bench.album.regression, pure)
albumify_regression.py -> JobService(tag) -> JobRepo -> control DB <- reap_orphans (on GET /jobs, /jobs/{id})
```

Dependency direction: `sim_bench.control.compare` -> `sim_bench.album.regression` (pure, stdlib only) and
`sim_bench.control.*`; the router -> `sim_bench.control.compare`; the page -> `api_client` only. No reverse imports
(grep). The Modal worker and the LocalRunner child do not import `sim_bench.control`, so the new `psutil` import
stays on the control-plane side; the CI fast lane does not import it either.

## Part 2 — Findings

### §1 Structure — pass-with-followup
- Sizes: `compare.py` (control) 109, `jobs.py` 159, `db.py` 102, page `compare.py` 210, router 182, script 164 LOC.
- Removed: `sim_bench/album/regression_report.py` and its test (O3); no orphan references (grep).
- **F1 (low)** `scripts/albumify_regression.py` `check(name, ds, runner, album_path, job_id)` has 5 parameters
  (limit 4): `album_path` / `job_id` are two ways to give an existing run. -> TODO: one `source` argument.

### §2 Code quality — pass-with-followup
- New try/except only around process lookup (`psutil`), a legitimate failure point.
- **F2 (low)** `sim_bench/control/jobs.py` `worker_alive`: `worker.rsplit(":", 2)` unpacks outside the `try`; a
  malformed `worker` string would raise and break `GET /jobs`. Only this code writes the column, so not reachable
  today. -> TODO: treat unparsable as dead.
- **F3 (low)** `JobRepo.set_tag` on an unknown id raises `AttributeError` instead of `KeyError` like `transition`. -> TODO.

### §3 Naming and package structure — pass
- `compare.py` in `sim_bench/control/` (jobs live there) and `pages/compare.py` in the app: names match contents.
- `__all__` declared in `sim_bench/control/compare.py` and extended in `jobs.py`. Pages follow the app convention (none).

### §4 Layering and coupling — pass-with-followup
- No duplicated logic: the page and the script share `AlbumComparator`; the API result equals the script result on
  all 4 real albums (verified 2026-10-17).
- Single writer: `jobs.tag` and `jobs.worker` written only through `JobRepo` (`create`, `transition`, `set_tag`).
- **F4 (medium)** two baseline references: the script's `check` compares with `baselines/<name>.json`; the page
  preselects the job tagged `:baseline`. `capture` sets both, so they agree, except after `capture --album` (no job
  to tag). -> TODO: `capture --album` refuses, or the script reads the baseline job from the DB.

### §5 Testability — pass-with-followup

| Kind | Tests | Count |
|---|---|---|
| Unit | `tests/control/test_jobs.py` `ut_OrphanJobs` (4), `ut_ControlDB::test_older_db_gets_the_new_nullable_columns` (1) | 5 |
| Unit (real DB + fake pipeline runner) | `tests/control/test_compare.py` | 6 |
| API (real router + DB) | `tests/api/test_jobs_router.py::ut_CompareRouter` | 3 |
| E2E real data | `tests/test_compare_page_e2e.py` (real API + Streamlit + the 8 regression jobs, Playwright) | 5 |
| Real-data cross-check | API vs script on 4 albums, all equal | script |
| Removed | `test_report_shows_status_failures_and_moved_picks_without_embedding_photos` (report gone) | −1 |

Full suite: **1,600 passed, 2 skipped, 0 failed**.

- Mocks: `reap_orphans(alive=lambda w: False)` in one test injects the liveness check, so "process gone" is
  deterministic; `worker_alive` itself is tested against real processes (this one, a non-existent pid, a reused pid).
  The fake pipeline runner is the existing spec-112 fixture (no models in unit tests).
- **F5 (low)** `tests/test_compare_page_e2e.py` names the 2026-10-16 job ids; a re-captured baseline makes it
  skip (ids gone) or fail (`[baseline]` moved). -> TODO: pick the jobs by tag.

Failure-mode walk-through:

| Bug class | Test that fails if it recurs |
|---|---|
| Killed job stays `running` (SIGHTING-137) | `ut_OrphanJobs::test_job_of_a_dead_worker_is_failed_as_interrupted`, `ut_CompareRouter::test_tag_is_listed_and_a_dead_jobs_status_is_corrected` |
| Reused pid taken for the old worker | `ut_OrphanJobs::test_worker_alive_rules` |
| New column breaks an existing `control.db` | `ut_ControlDB::test_older_db_gets_the_new_nullable_columns` |
| Page numbers differ from the regression check | real-data cross-check (4 albums); `test_compare.py` uses the same comparator |
| Owner edits ignored by the comparison | `test_an_owner_edit_in_b_shows_as_a_moved_pick` |
| Comparing runs on different photos | `test_jobs_on_different_photos_are_refused` (unit), e2e `..._refused_with_a_message` |
| Clickable photos regress to dataframe row-select | e2e `test_swapped_pair_opens_both_runs_decisions` clicks a real button |
| Regression runs clutter the owner's job list | e2e `test_jobs_page_hides_regression_runs_by_default` |

### §6 Boundary contracts — pass
- `CompareOut`, `CheckOut`, `PickDiffOut`, `PhotoDecisionOut`, `ComparePersonOut`: `extra="forbid"` (the `_Strict`
  base); FastAPI validates every response against `response_model`.
- `with` query parameter required (422 when missing, tested); unfinished jobs 409, unknown 404, different photos 422.
- Schema upgrade enforces its own rule (NOT NULL missing -> RuntimeError), not just documentation.

### §7 Documentation — pass
- `spec.md`, `tasks.md`, `updates.md` (verification table), this `REVIEW.md`.
- `db_schemas.html`: new control-DB `jobs` section (the control DB was missing from the docs; added with `tag`/`worker`).
- `classes.html`: `JobComparer`, `JobRepo` rows; `RegressionReport` row removed. `data_flow.html`: spec-114 flow.
- CHANGES_LOG (2 entries), LEARNINGS (SIGHTING-137), SIGHTINGS 137 (fixed) / 138 (open), ROADMAP row.

### §8 Risk register — pass-with-followup
- Backwards compatibility: existing `control.db` gains two NULL columns on first open (done live, 2026-10-17);
  jobs with `worker = NULL` in a live status are failed as interrupted (intended: nothing can be running them).
  A worker on another host is assumed alive (no false kills; its dead jobs need a manual fail).
- Hot path: `reap_orphans` runs on every `GET /jobs` and `GET /jobs/{id}`: one indexed-status query + one `psutil`
  lookup per live job (normally 0–1). Compare response: 304 KB for Slovakia (715 photos), 0.2 s.
- **F6 (medium)** the page caches each comparison in the session; an edit made on the Jobs page after a comparison is
  not reflected until the browser session restarts. -> TODO: drop the cached pair when either job's edits change
  (or a Refresh button).
- **F7 (medium, existing SIGHTING-138)** scene -1 photos share one giant `beat` list, so pick matching inside scene
  -1 is lenient: drift there cannot show as Missing/Extra. Ticketed.
- Leftovers: `~/.sim_bench/regression/` (old control.db, the killed Slovakia run, 95 MB). -> TODO (owner: delete).

## Part 3 — Verdict and follow-ups

| Area | Verdict |
|---|---|
| §1 Structure | pass-with-followup (F1) |
| §2 Code quality | pass-with-followup (F2, F3) |
| §3 Naming | pass |
| §4 Layering | pass-with-followup (F4) |
| §5 Testability | pass-with-followup (F5) |
| §6 Boundary contracts | pass |
| §7 Documentation | pass |
| §8 Risk register | pass-with-followup (F6, F7) |

| # | Ticket | Where |
|---|---|---|
| F1 | `albumify_regression.check`: one `source` argument instead of `album_path` + `job_id` | TODO.md |
| F2 | `worker_alive`: unparsable worker string = dead, not an exception | TODO.md |
| F3 | `JobRepo.set_tag`: `KeyError` on unknown job | TODO.md |
| F4 | One baseline reference (refuse `capture --album`, or read the baseline job from the DB) | TODO.md |
| F5 | Compare e2e test: choose jobs by tag, not fixed ids | TODO.md |
| F6 | Compare page: invalidate the cached comparison after edits (or Refresh button) | TODO.md |
| F7 | Scene -1 giant near-duplicate group | SIGHTING-138 (existing) |
| — | Delete `~/.sim_bench/regression/` leftovers | TODO.md (owner) |
