# spec-090 — Run picker in Albumify (view any run, not just the latest)

**Created**: 2026-06-25 · **Status**: In Progress · **Priority**: P2
**Source**: user — Albumify only shows the latest run of an album; older runs are unreachable,
forcing a "new album per run" (`run1…run15`) workaround.

## Problem
The frontend resolves album → results by always taking `results[0]` (latest completed run);
`get_people(album_id)` passes no `run_id`, so the backend defaults to latest. The backend
ALREADY accepts `run_id` (people/results) — the UI just never sends one. "Previous Runs" in
Results is display-only.

## What we build
A **Run** dropdown (shared component) on the Results and People pages that lists the album's
runs and stores the choice in `session_state.current_run_id`; all run-specific data then uses
that run instead of the hardcoded latest.

1. `session.py`: `current_run_id` on `SessionState` + `get/set_current_run_id`; reset it when
   the album changes.
2. NEW `components/run_selector.py`: pure `resolve_run_id(results, current)` (returns current if
   still valid, else latest) + `render_run_selector(album_id)` (selectbox labeled by
   date/people/selected; stores the choice; returns the run_id).
3. Wire `results.py` (use the picked run, not `results[0]`), `people.py` (pass `run_id` to
   `get_people`), `face_management._get_active_run_id` (prefer the picked run).

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | `resolve_run_id`: valid current → current; stale/None → latest; empty → None | unit test |
| 2 | Picking a run shows that run's results/people (not always latest) | manual |
| 3 | Switching album resets the run selection to that album's latest | code/manual |
| 4 | No run history → graceful "run the pipeline first" | code |

## Notes
- Backend unchanged (already supports `run_id`).
- Lets the user stop creating `run1…run15` albums — one album, many runs.
