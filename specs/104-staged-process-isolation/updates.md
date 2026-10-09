# spec-104 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · [REVIEW.md](REVIEW.html) · roadmap stage: **B enabler** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Implemented v1 (flag-gated, default off) — merged to `main` 2026-09-28.

## How it was verified
| Claim | Evidence |
|---|---|
| Per-step process isolation reclaims model memory | Probe: parent RSS +480 MB in-process vs +1 MB isolated (~100% reclaimed) |
| Child crash does not kill the run; context marshals in/out; progress relays | `tests/pipeline/test_step_isolation.py` (6 tests) |
| No regression in executor logging / architecture guards | `test_executor_step_io_logging.py` + `tests/architecture` — 142 passed, 0 failed |
| Capacity envelope | `reports/2026-09-21_spec104_capacity/`: ~2.5 GB floor + 0.67 MB/img; ~1.6 s/img CPU |
| Code review | REVIEW.md verdict ACCEPT (v1) with follow-ups |

## Known gaps (open follow-ups in TODO.md)
- No real-model E2E of the isolated path asserting output equivalence vs in-process (§5 finding).
- SIGHTING-122: face counts differ by 1–2 between isolated and in-process at N=300/500. First test
  is whether two in-process runs even agree.
- Phases 2–4 (stage grouping, memory gates, wire into Albumify/API) not started.

## Timeline
- 2026-09-28 — v1 merged to main (`--no-ff`), spec flipped to Implemented.
- 2026-09-21 — capacity report + SIGHTING-122 filed.
- 2026-09-15 — Phase 1 built; REVIEW.md ACCEPT-with-followups.
- 2026-07-30 — Phase 0: missing `release()` on face embeddings found (~360 MB); SIGHTING-117 re-diagnosed
  as stacked torch model memory, not detections/crops.

## Reports & artifacts
| Artifact | What it shows |
|---|---|
| [Capacity report](../../reports/2026-09-21_spec104_capacity/report.html) | Germany 100/300/500 imgs, real `faces` pipeline, cold cache: peak ≈ 2.5 GB model floor + 0.4–0.7 MB/img (2,845 MB isolated at N=500 vs 2,532 in-process); isolation costs ~+300 MB / +10% time; 16 GB fits ~20k imgs so runtime (~1.6 s/img) binds first. ±1–2 face-count divergence → SIGHTING-122 |
| [Architecture](architecture.html) · [Engineering design](engineering_design.html) | isolation design of record (spawn child per step, marshal context, relay progress) |
| [REVIEW](REVIEW.html) | ACCEPT (v1, flag-gated) with follow-ups |
