# spec-095 Code Review — Geo-Vision Studio

**Date**: 2026-06-30 · **Reviewer**: self (vs `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Scope**: `app/geo_vision/{geo_view.py,main.py}`, tests. Built on the spec-094 engine.

| # | Section | Verdict | Notes |
|---|---|---|---|
| 1 | Correctness | PASS | EXIF→map→accuracy→CSV verified end-to-end; haversine accuracy matches known Budapest/Vienna distances. Live Playwright run rendered summary, Budapest map, thumbnail. |
| 2 | Tests | PASS | 7 unit tests on pure helpers (haversine, accuracy hit/miss, EXIF-less skip, map points, csv). |
| 3 | Error handling | PASS | Missing/empty folder → warning, no crash; engine skips a failing method; no GPS → point/accuracy skipped; cache handler best-effort (None → re-infer, logged). |
| 4 | Naming/style | PASS | Pure helpers separated from UI; full imports; logging; mirrors app/album path bootstrap. |
| 5 | No dead/dup code | PASS | Reuses 094 engine + AnalysisColumn; NO duplicate scoring engine (the whole point of the build decision). |
| 6 | Security | N/A | No secrets. Reads local images; weights cached under ~/.cache. |
| 7 | Docs | PASS | Spec updated (build decision), tasks, CHANGES_LOG. |
| 8 | Windows/ASCII | PASS | Path bootstrap fixes Streamlit sys.path; forward-slash paths; ran on win32. |

## High-severity findings
None.

## Medium / notes
- **M1 — Live UI not exercised with the heavy CLIP/BLIP models** (StreetCLIP/GeoCLIP/
  BLIP download ~1.6GB + slow CPU). The engine code path is identical to EXIF (same
  `run_methods`), covered by spec-094's suite + smoke; confidence-bar rendering is
  trivial. EXIF path was driven live (Playwright) with the real Budapest examples.
  Follow-up: one manual full-model run for a screenshot.
- **M2 — StreetCLIP city accuracy not computed.** `geoclip_accuracy` uses haversine
  vs EXIF (cleanly defined). StreetCLIP returns a city *label* with no coords, and
  we have no EXIF city ground-truth, so its accuracy is intentionally omitted (shown
  as a guess only) rather than fudged. Documented in `geo_view`.
- **M3 — Couples to in-progress spec-094 engine** (uncommitted at build time). The
  consumed surface (`available_methods`, `run_methods`, `AnalysisColumn`,
  `discover_images`) is stable; if 094 renames these, 095 needs a matching update.

## Verdict
**PASS** — no high-severity findings. App implemented, unit-tested, and visually
verified on real data via Playwright.
