---
title: "spec-114 updates - Compare two jobs in the app"
---

# spec-114 updates

**Status:** Implemented (2026-10-17)

## 2026-10-17 — Code review: ACCEPT with follow-ups -> Implemented

`REVIEW.md`: 0 fails; F1–F6 ticketed in TODO.md (small code/test hygiene; F6 = stale cached comparison after an
edit), F7 = SIGHTING-138. Verification evidence: the 2026-10-17 entry below.

## 2026-10-17 — Built: Compare runs page, regression runs in the app's job list

Owner: "go with your recommendations, implement spec-114" (O1 a, O2 a, O3 b, O4 a).

- `jobs.tag`, `jobs.worker`; `ControlDB` adds missing nullable columns to an existing DB (the live `control.db`
  got both on first open). Deviation from the draft: no Alembic for the control DB (Alembic here targets
  `sim_bench.db` only); the additive-column rule is explicit and refuses anything else.
- `JobComparer` + `GET /jobs/{b}/compare?with={a}`; per-photo decisions travel in the same response (no extra endpoint).
- **Compare runs** page; Jobs page hides `regression:*` runs behind a toggle.
- `albumify_regression.py` uses the app's job list, tags runs, `capture` marks the baseline job. HTML report removed (O3).
- O4: the 8 finished regression jobs moved from `~/.sim_bench/regression/jobs` to `~/.sim_bench/jobs` and loaded with
  their tags (4 baselines, 4 checks). The killed Slovakia run (no album) and the old `regression/control.db` were left
  in place, unused.
- Found: SIGHTING-138 (scene -1 photos form one giant near-duplicate group; also makes the regression pick match
  lenient inside scene -1). Not fixed here.

| Claim | Evidence |
|---|---|
| Dead job failed as interrupted, live/queued/finished untouched, pid reuse caught | `tests/control/test_jobs.py::ut_OrphanJobs` 4 passed |
| Older DB gains `tag`/`worker` without losing rows | `tests/control/test_jobs.py::ut_ControlDB::test_older_db_gets_the_new_nullable_columns` |
| Comparison: match, owner edit shows as moved pick, every photo decided in both, people, refusal | `tests/control/test_compare.py` 6 passed |
| API: compare, 404/409/422 refusals, tag listed, dead job corrected on list | `tests/api/test_jobs_router.py::ut_CompareRouter` 3 passed |
| API numbers == script numbers on the 4 real albums (AC2) | script check 2026-10-17: quick, budapest, austria, slovakia all `api==script: True` |
| Page on real data: verdict + baseline preselected, Swapped dialog with both decisions, People with faces, different photos refused, Jobs page hides regression runs (AC1–AC5) | `tests/test_compare_page_e2e.py` 5 passed (real API + app + jobs); screenshots in `%TEMP%/albumify_compare_e2e/` (not git) |
| Nothing else broke | full suite 1,600 passed, 2 skipped, 0 failed (10 min) |

## 2026-10-16 — Drafted

Owner, after the spec-113 R1.5 HTML report: "hard to tell without the app … we need to figure out how to use the app
to view the results", and asked for a comparison view to keep going forward. Checked first: the Jobs page works
when pointed at the regression job list (9 jobs, 604/604 thumbnails loaded on the Slovakia album). Found on the way:
a killed job stays "running"; one Slovakia pick "beats 88 near-duplicates" (photos without a scene look grouped as
one burst). Both to be filed as sightings.

| Claim | Evidence |
|---|---|
| Jobs page shows regression albums when pointed at their DB | Playwright, `DATABASE_URL` + `ALBUMIFY_JOBS_ROOT` set; screenshot on disk |
