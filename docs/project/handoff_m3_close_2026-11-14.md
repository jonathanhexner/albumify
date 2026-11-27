# Handoff: close specs 116 + 117 (M3), 2026-11-14

From the owner via Telegram (13:42): "go ahead" on part A below. Source: `docs/project/m3_closeout_2026-11-14.html`.
Branch `spec/117-neon-r2` (contains 116), worktree `D:\sim-bench-117`. Nothing is pushed; push and PR only after the owner's OK.

## Do now (no owner decision needed)

| # | Task | Done when |
|---|---|---|
| 1 | **117-H1**: `detached_modal.status()` turns a lost connection to Modal into "failed", so the watchdog fails a healthy GPU run and `worker_persister` refuses its results. A connection error is "unknown", not "failed"; the watchdog fails a job only on a real end state or a heartbeat gap past the grace. Log it as a SIGHTING | Test that raises the real exception type (not the fake that returns "failed"); `tests/serving/test_run_watchdog.py` green |
| 2 | **Heartbeat test** (SIGHTING-155): `test_db_progress.py::test_heartbeat_thread_beats_at_least_every_interval` polls until a deadline instead of sleeping 0.7 s | Passes on SQLite and Neon; re-run the parity set: 297/297 |
| 3 | **117-M1**: the worker stops at once when the Modal secret has no `DATABASE_URL`, instead of writing to SQLite inside the container (`runners.py:281`) | Test for the missing-URL case |
| 4 | **117-M6**: spend cap estimates with the slowest/dearest GPU in the fallback list (A10 ran 1.7x slower than the L4 estimate) (`job_service.py:142`) | Test: estimate uses the A10 rate |
| 5 | **117-H2 + M4**: architecture docs (display_path, new endpoints, RunDisplayCopies, autocommit read sessions, parallel uploads); CHANGES_LOG entries for dd33596, c0a2606, 1b7d747, c56d6db | Checklist section 7 met |
| 6 | Full suite + Budapest gate, then `/code-review` for 116 and 117 again | No High findings left except the owner decisions below |

## Waiting for the owner (do not decide)

- SIGHTING-144: accept 768 px results for the sharpness gate, or keep full size.
- Definition of done: merge with R11 met on People only and SIGHTING-154 open (as follow-ups), or fix them first.
- Then: one PR 117 -> main (closes 116, 117 and 112 Phase B). Push only on the owner's OK.

## Also proposed (separate, not part of this)

- `specs/119-repository-layer/` — Proposal written 2026-11-14 from the Trip Notebook session; number provisional. Review after 117 merges.
- Old "In Progress" specs to close or park: 102, 103, 108, 109, 110, 111 (owner to confirm).
