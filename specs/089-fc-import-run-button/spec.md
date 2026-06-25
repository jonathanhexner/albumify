# spec-089 — Import an existing run in the FC app Run tab

**Created**: 2026-06-25 · **Status**: In Progress · **Priority**: P2
**Source**: user — no way to load an Albumify export (e.g. `results/<album>/face_clustering_<ts>/`)
into the FC app; the History "Load into analysis tabs" button only appears for runs already in
the FC history DB, not external exports.

## What we build
A small **"Import existing run"** control at the top of the Run tab: a path text input + an
**Import** button. On click it calls the existing `face_cluster.loader.load_pipeline_result(dir)`
and wires the result into session state exactly like a finished run does, then points the user at
the Clusters tab.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | Entering an export dir + Import loads it into the analysis tabs | manual |
| 2 | Bad/missing path shows an error, doesn't crash | code (try/except) |
| 3 | Uses the existing loader + session wiring (no new load logic) | code review |

## Notes
- Reuses `load_pipeline_result` (already tested) + `_invalidate_run_caches` /
  `_create_session_from_result` (the same calls the run-complete path uses).
- Point the path at the `face_clustering_<ts>` dir (loader finds `_v4/face_clustering.db`).
