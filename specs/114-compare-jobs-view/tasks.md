---
title: "spec-114 tasks - Compare two jobs in the app"
---

# spec-114 tasks

Decisions 2026-10-16: O1 (a), O2 (a), O3 (b), O4 (a).

## T1 — Data
- [x] T1.1 `Job.tag` column, added to existing DBs by `ControlDB`; `JobService.submit` / `load_existing` take `tag`. (AC1)
- [x] T1.2 `albumify_regression.py`: main control DB, tag `regression:<dataset>`; baseline JSON keeps `job_id`. (AC1)
- [x] T1.3 Import today's finished regression jobs with their tags (O4). (AC1)
- [x] T1.4 Dead job not left "running" (SIGHTING-137, fixed, 5 tests). (AC6) | 2026-10-17

## T2 — API
- [x] T2.1 `GET /jobs/{id}/compare?with={other}`: checks, metrics, pick diff, people; refuse different photo sets. (AC2, AC4, AC5)
- [x] T2.2 Per-photo decision for both jobs (kept / near-dup of X / not selected / removed by filter), inside the compare response (no extra endpoint). (AC3)

## T3 — App
- [x] T3.1 Jobs page: tag filter, regression jobs hidden by default. (AC1)
- [x] T3.2 Compare page: choose A and B (dataset -> baseline preselected), header, checks table. (AC2)
- [x] T3.3 Sections Missing / Extra / Swapped / More from a group / People as `st.button` grids. (AC2)
- [x] T3.4 Photo detail with both runs' decisions. (AC3)

## T4 — Close-out
- [x] T4.1 Playwright on the 4 regression albums; screenshots on disk (not git). (AC2, AC3)
- [x] T4.2 `docs/architecture/` (classes, data_flow, db_schemas: control `jobs` table added, it was missing) updated; O3 applied (report module removed).
- [x] T4.3 `/code-review` -> REVIEW.md (ACCEPT, 0 fails); updates.md; status Implemented. | 2026-10-17
