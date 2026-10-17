---
title: "spec-114 - Compare two jobs in the app"
---

# spec-114 — Compare two jobs in the app

**Status:** Implemented (2026-10-17) — REVIEW ACCEPT with follow-ups (F1–F6 in TODO, F7 = SIGHTING-138); owner decisions O1 (a), O2 (a), O3 (b), O4 (a)
**Owner:** Jonathan / Claude
**Priority:** P1 (regression results can only be read as PASS/FAIL lines or a static HTML page; owner: "we need to
figure out how to use the app to view the results")
**Relation:** builds on [[spec-112]] (control DB, jobs API, Jobs page) and [[spec-113]] R1 (regression suite,
`AlbumComparator`). Replaces the R1.5 HTML report as the main way to read a check.

## 1. Problem

The regression suite (spec-113) says PASS/FAIL, and since R1.5 writes a static HTML page. Neither lets you do what
you need to judge a change: open a photo that moved and see *why* each run treated it as it did. The app can show
one album at a time (Jobs page), but:

- regression runs live in a separate job list, so the app does not see them unless restarted against it;
- there is no view of two albums side by side;
- a job killed mid-run stays "running" forever (found 2026-10-16; SIGHTING to file).

The same need comes up outside regression: CPU vs L4, before vs after a settings change.

## 2. Goal

One page in the Albumify app, **Compare**, that takes any two jobs on the same photos (A = reference, B = new) and
shows what changed, with every photo clickable to see both runs' decisions. Regression checks become "compare the
new run with the baseline job" on this page.

## 3. Design

```
scripts/albumify_regression.py check  --> JobService (same control DB, job tagged "regression:<dataset>")
                                                |
app: Compare page --GET /jobs/{b}/compare?with={a}--> API --> AlbumComparator(album A, album B)
     |  pick A (baseline job of a dataset, or any job)              --> CompareResult + PickDiff + people
     |  pick B (later run of the same photos)
     v
  verdict + checks table | Missing | Extra | Swapped (A photo | B photo) | More from a group | People
  click a photo -> both runs' decisions for it (kept / near-dup of X / not selected / removed by filter)
```

| Piece | Where | Note |
|---|---|---|
| Comparison | `sim_bench/album/regression.py` (`AlbumComparator`, `PickDiff`) | exists |
| Two jobs | `sim_bench/control/compare.py` (`JobComparer`): both albums as the DB builds them (edits applied) + every photo's decision | new |
| API | `GET /jobs/{job_id}/compare?with={other_id}` in `sim_bench/api/routers/jobs.py` | refuses jobs on different photos |
| Job tag | `Job.tag` column (nullable): `regression:<dataset>` for checks, `regression:<dataset>:baseline` for the baseline job. Added to existing DBs by `ControlDB` (additive nullable columns; Alembic covers `sim_bench.db` only) | O1 |
| Dead jobs | `Job.worker` = `host:pid:start-time` of the process running it; a live-status job whose process is gone is failed as interrupted when jobs are listed | AC6 |
| Baseline | `tests/regression/albumify/baselines/<name>.json` keeps the summary (in git, no photos) **and** the baseline `job_id` | the job's photos stay on disk |
| Page | `app/streamlit/pages/compare.py` | real `st.button` grids (never dataframe row-select) |
| Script | `albumify_regression.py` writes to the main control DB + job store with the tag; HTML report removed (O3) | |

### What the page shows

| Section | Content |
|---|---|
| Header | A and B (date, git, runner, settings), PASS/FAIL against the tolerances when A is a baseline |
| Checks | counts and top-5 people: A, B, difference, allowed (as in R1.5) |
| Missing | A kept the moment, B kept nothing from it |
| Extra | B kept a moment A did not keep |
| Swapped | A's pick next to B's pick from the same near-duplicate group |
| More from a group | B kept a second shot of a moment A also kept |
| People | A's and B's people, largest first, with face count and a face crop; phase 2: match people across runs by shared faces |
| Photo detail | the photo large; for A and for B: kept / near-duplicate of X / not selected / removed by filter (+ reason, score, scene) |

## 4. Acceptance

| # | Criterion | Evidence |
|---|---|---|
| AC1 | Regression `check` jobs appear in the app without restarting it against another DB | Jobs page lists `regression:*` jobs; filter hides them by default |
| AC2 | Compare page shows the R1.5 breakdown for the 4 regression albums, numbers equal to `AlbumComparator` | Playwright on real data, screenshots on disk |
| AC3 | Clicking any photo in any section shows both runs' decision for it | Playwright click test |
| AC4 | Comparing jobs on different photos is refused with a clear message | API test |
| AC5 | Works for non-regression pairs (e.g. two runs of the same folder with different runners) | API test + one real pair |
| AC6 | A job whose process died is not shown as "running" forever | SIGHTING fix + test |

## 5. Out of scope

Matching people across runs (phase 2), scene-by-scene comparison, comparing jobs on different photo sets,
editing from the Compare page, the public product site (spec-106).

## 6. Open decisions (owner)

| # | Question | Options | Recommendation |
|---|---|---|---|
| O1 | Where regression jobs live | (a) main control DB + `tag`, hidden by default; (b) separate DB, app switches between them | (a): one app, one place; no restart |
| O2 | What can be compared | (a) any two jobs on the same photos; (b) only regression run vs baseline | (a): same page serves CPU vs L4 and settings changes |
| O3 | Keep the R1.5 HTML report | (a) keep as an offline file; (b) drop once the page lands | (b): one way to read results |
| O4 | Move today's regression jobs into the main DB | (a) import the 8 finished jobs with their tag; (b) re-run | (a): `load_existing` already does this, no compute |
