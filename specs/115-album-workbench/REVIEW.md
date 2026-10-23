---
title: "spec-115 REVIEW - Album workbench"
---

# spec-115 REVIEW — Album workbench

*2026-10-23. Independent reviewer (subagent) against `docs/guides/CODE_REVIEW_CHECKLIST.md`, all 8 sections; fixes by
the implementer, verified by tests and real runs. Scope: branch `spec/112-a2-workbench` since `main`.*

**Verdict: ACCEPT** after fixes. 0 high. 7 medium: all fixed with tests. 8 low: 6 fixed, 2 recorded as risks.

## Part 1 — How it works

See the spec's plan and class diagram and `docs/architecture/data_flow.html` ("spec-115"). In short: photos are
resized once into a content-addressed store; a job is a manifest of keys; the runner (local child or Modal L4)
returns results only; the server renders displays and crops itself; one Modal job runs at a time so the feature
cache on its Volume has one writer.

## Part 2 — Findings by section

| # | Section | Verdict | Notes |
|---|---|---|---|
| 1 | Structure | pass-with-followup | `modal_app.py` (≈370) and `pages/jobs.py` (≈330) exceed 300 lines → TODO split |
| 2 | Code quality | pass | try/except only at real failure points (unreadable photo, bad box, Modal I/O) |
| 3 | Naming / packages | pass | `__all__` everywhere; names match contents |
| 4 | Layering | pass-with-followup | Compare page imports `sim_bench.serving.run_settings` (pure, no heavy deps) → TODO move diff behind the API |
| 5 | Testability | pass | behaviour tests incl. real child processes, threads, fake Volume; real-run evidence for the Modal paths |
| 6 | Boundary contracts | pass | `extra="forbid"`; `start >= 0`; 404/409/422/502 mapped; spend cap on every L4 path |
| 7 | Docs | pass | `classes.html`, `data_flow.html`, spec/tasks/updates, CHANGES_LOG, LEARNINGS updated |
| 8 | Risk register | pass-with-followup | risks below are in the spec |

| Id | Severity | Finding | Resolution |
|---|---|---|---|
| M1 | medium | cancel of a job waiting for the L4 lock ignored until the lock frees | `_wait_for_modal` polls cancel; `ut_ReviewFixes::test_cancel_of_a_job_waiting...` |
| M2 | medium | purge could delete just-uploaded / unrecorded / damaged-record photos; non-atomic record | upload writes a `.used.json` claim; missing/damaged → folder time; atomic writes; `ut_InputRetention` + real upload check |
| M3 | medium | cancel just before spawn / before the child starts still ran | re-check after spawn and after Popen; tests |
| M4 | medium | concurrent Modal jobs lose cache rows (copy back = last writer wins) | every Modal runner under the gate; per-process limit recorded as a risk |
| M5 | medium | missing results folder hid the real run error | caught; `ut_ModalVolumeRemoteDownload` |
| M6 | medium | one bad box failed the load; half-written crop kept forever | per-face skip (`bad_boxes`), atomic writes; test |
| M7 | medium | delete-photos allowed on a live job; Modal errors → 500 | 409 unless finished; `RemoteError` → 502; test |
| L1 | low | cancel during LOADING accepted then ignored | refused (409); test |
| L2 | low | negative `start`; job-id-only L4 path skipped the cap | `ge=0`; cap on every L4 path; tests |
| L3 | low | queued-cancel race could fail an already-failed job | guarded; `_cancel_requested` cleared |
| L4 | low | job ids collide within a second | random suffix; test |
| L5 | low | warm containers keep the previous call's env | reset per call |
| L6 | low | estimate / submit reveal server folders | risk: owner-only API, before anything is public (spec-106) |
| L7 | low | E2E tests depended on order; cancel could race the job | own finished job per test; 60-photo cancel input; 4/4 pass |
| L8 | low | `scripts/modal_curate_driver.py` bypasses the shared store and the gate | risk recorded; its staged copies are purged after 7 days |

## Part 3 — Follow-ups (TODO.md)

- Split `modal_app.py` (worker functions vs retention) and `pages/jobs.py` (run tools vs album view).
- Serve the settings diff from the API instead of importing `run_settings` in the page.
- Retire or port `scripts/modal_curate_driver.py` to the shared store.
- spec-112 TA6.3 promote/rename round-trip on the new jobs (the API tests cover it; no real-data click yet).
