# spec-113 updates — Remove the deprecated `cluster_people` step

**Status:** Implemented (2026-10-18)

## 2026-10-18 — Implemented

Owner: "yes, flip spec-113 and commit, open PR 2". All acceptance criteria have evidence: review ACCEPT (REVIEW.md),
regression suite PASS on 4 real albums (2026-10-16/17), Budapest gate 11/11 (entry below), full default suite
**1,605 passed, 2 skipped, 0 failed** (2026-10-18, 7 min 37 s; first run caught a test-order leak from the new
`test_v2_profile_bar.py`, fixed, see LEARNINGS). T4.1 (optional `people_clustering` group in `PipelineBuilder`) deferred to TODO.

## 2026-10-18 — Budapest V2 gate green (SIGHTING-136 fixed)

The gate failed on main for reasons older than this spec (stale selectors, two click races, a 15-cluster expectation
made before the pose/blur gates worked, a run with no filter_decisions). Fixed under SIGHTING-136, plus two product
fixes the owner asked for: recluster keeps stored blur scores, and the Recluster tab can load a profile.

| Claim | Evidence |
|---|---|
| V2 gate green on this working tree (spec-113 + spec-114 + SIGHTING-136 changes) | `pytest -m budapest tests/face_clustering/e2e_budapest/`: 11 passed, 5 min 55 s |
| New fresh-run expectation is reviewed, not guessed | owner reviewed `reports/2026-10-17_sighting136_budapest_gate/before_after.html` (local): "looks okay" |
| G race fixed | `test_scenario_g_gallery.py` 10/10 after the fix (was 3/5) |

## 2026-10-16 (late) — HTML comparison report; Budapest gate red (SIGHTING-136)

**Report (R1.5).** Owner asked to see the comparison, not just PASS/FAIL. Each `check` now writes an HTML page to
`~/.sim_bench/regression/reports/` (outside git; it shows photos via file links, nothing embedded) and an `index.html`.
Today's four checks re-reported from their finished jobs (`check <name> --job <id>`, no re-run): all PASS, same numbers.

| Album | Same photo | Swapped for a sibling | Missing | Extra | More from a baseline group |
|---|---|---|---|---|---|
| quick | 23 | 0 | 0 | 0 | 0 |
| Budapest | 48 | 1 | 0 | 0 | 0 |
| Austria | 141 | 10 | 0 | 0 | 5 |
| Slovakia | 242 | 5 | 0 | 0 | 4 |

**Budapest V2 gate: 8 passed, 3 failed (A, C, F).** The same 3 fail identically on the last commit `d7c6ffa`
(worktree run), so not caused by this spec; filed SIGHTING-136. Spec stays In Progress until the gate is green or
the owner waives it.

| Claim | Evidence |
|---|---|
| Pick breakdown + report render, ids escaped, no embedded photos | `tests/album/test_regression.py` 10 passed (+3) |
| Breakdown totals reconcile with album counts | Austria 151/151, Slovakia 247 + 4 = 251, Budapest 49/49 (script check) |
| Thumbnails load | Playwright on the Slovakia page: 10/10 images loaded |
| Gate failures pre-date spec-113 | `pytest -m budapest` A, C, F on `d7c6ffa`: same 3 failures |

## 2026-10-16 — Removal done; regression suite captured and green

Removed the step, its config class, `face_cluster_bridge.py`, two unused yaml files, 4 old-vs-new tests; renamed the
settings block to `people_clustering` (old key still read, wins over yaml defaults: O1); spec-079 scripts archived
(O3). Explore's face-decision block removed, not fixed: no current step records per-face decisions (AC4 amended).

**Regression suite (R1).** Baselines captured on the pre-removal code (a frozen copy, run on CPU, same `JobService`
path as the product), then checked on the post-removal code, same machine:

| Album | Photos | In album | Picks | Scenes | People | Pick recall / precision | Result |
|---|---|---|---|---|---|---|---|
| quick (Slovakia first 48) | 48 | 46 = 46 | 23 = 23 | 9 = 9 | 9 = 9 | 1.0 / 1.0 | PASS |
| Budapest | 122 | 96 = 96 | 49 = 49 | 13 = 13 | 9 = 9 | 1.0 / 1.0 | PASS |
| Austria | 474 | 374 = 374 | 151 = 151 | 55 → 57 | 55 → 54 | 1.0 / 1.0 | PASS |
| Slovakia | 715 | 626 = 626 | 247 → 251 | 93 → 95 | 79 = 79 | 1.0 / 1.0 | PASS |

Small scene/people/pick drift on the two large albums is run-to-run variation (the removed step no longer ran;
cf. SIGHTING-130). CPU time: 10 min / 23 min / 86 min / 136 min per check.

**Effect of the SIGHTING-132 fix itself (before this spec):** Slovakia went from 118 people / 147 picks (Modal run,
2026-10-11, old step active) to 79 people / 247 picks: people now come from the spec-079 chain, and `select_best`
now receives the per-person sub-groups, so a scene can keep one best shot per person. Owner to judge the album size.

Found on the way: SIGHTING-133 (job path ignores people settings), SIGHTING-134 (broken opt-in test), a
LocalRunner output-encoding bug (fixed, +1 test).

| How it was verified | Evidence |
|---|---|
| Removal changes no album beyond tolerance | `scripts/albumify_regression.py check quick budapest austria slovakia`: 4 PASS (table above); baselines in `tests/regression/albumify/baselines/` |
| Comparator logic | `tests/album/test_regression.py`: 7 passed |
| Settings reach the 8 steps; old key honoured | `tests/pipeline/test_fc_analysis_export.py` (+2), `tests/architecture/test_config_parity.py` (+1) |
| No stale dependency; people steps before `select_best` | `tests/pipeline/test_builder.py::TestRealRegistry` (2) |
| Light suites touched by the change | 158 passed |

## 2026-10-16 — Drafted after the SIGHTING-132 fix

SIGHTING-132 showed the deprecated step still ran in every default pipeline. The immediate fix (three
`depends_on` lists) landed first as a bug fix; this spec removes the step so it cannot come back.

| How it was verified | Evidence |
|---|---|
| Prerequisite fix works | `tests/pipeline/test_builder.py::TestDefaultPipelinePeopleClustering` (3 tests): fail before the fix, pass after |
| No regressions from the prerequisite | 430 passed (`tests/pipeline`, `tests/architecture`, `tests/album`, `tests/serving`, `tests/control`, jobs API) + 12 passed (`cluster_people`-related face-clustering tests) |
