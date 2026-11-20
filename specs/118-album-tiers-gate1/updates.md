# spec-118 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap **Gate 1** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Draft 2026-11-13 — awaiting owner approval

## 2026-11-13 — Decisions

Owner: "bar is fine, park the sharpness filter". Gate 1 passes at Best 30 ≥ 90% kept and Best 10 all kept; the
sharpness filter stays as is and moves to the parked backlog.

## 2026-11-13 — Drafted from the Slovakia26 picks review

The owner reviewed every pick of run `52776cc5` (K=51) and said: near-duplicates are "logic we explicitly defined ...
we have control over it"; non-interesting photos, bad faces and time/geo scenes matter but "are more optimizations, so I
want to make sure we're proceeding toward productization ... Perhaps it would make sense to show best 10 images/30
images/all?". Then: "open a PRD with all the relevant information. we can store everything there."

| Claim | Evidence |
|---|---|
| The top of the ranking is already good | `picks_review.json` ranked by `composite_score`: Best 10 10/10, Best 20 20/20, Best 30 28/30, all 219/299 |
| Near-duplicates are the main drop reason | 47 of 80 drops (41 in scenes, 7 in no scene) |
| The sharpness filter removes wanted photos | 20 of 20 on the reviewed "removed" page; `step_decisions`: `filter_quality` "Sharpness x < threshold 0.2" |
| No ranking across scenes exists today | code map: `ResultService.get_images` returns stored order; `select_best` concatenates clusters; `album.json` is chronological |
| SIGHTING-150 | `select_best.py` `_select_from_cluster` counts `rank` with `enumerate(scored_images, 1)` over the unsorted list |

Report: [reports/2026-11-13_slovakia26_picks_review/report.html](../../reports/2026-11-13_slovakia26_picks_review/report.html).
