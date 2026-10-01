# spec-107 — Tasks (Input Triage & Discard Review)

**Status:** Draft — no implementation until spec.md approved. Full rationale: `spec.md`.

## P0 — Metadata tier (cheapest, highest-precision)  ✅ DONE 2026-10-01
- [x] T0.1 `sim_bench/triage/metadata.py`: `calc(TriageInputs)->TriageResult` (filename / camera-EXIF / format / screen-dims → keep|review|set_aside, conservative) + `classify_file(path)`.
- [x] T0.2 `tests/triage/test_metadata_triage.py` (9 tests, pure) — green.
- **Gate:** ✅ met — 715-roll validation: **8/8 screenshots set_aside, 707 keep, 0 review (0 false positives)**.

## P1 — Content tier (RAM++)
- [ ] T1.1 `triage_content` using RAM++ junk tags (`screenshot, document, poster, receipt, map, menu, logo…`); runs as its own isolated step (spec-104).
- [ ] T1.2 Threshold tuning + tests; measure added runtime/RSS.
- **Gate:** catches photographed docs/posters metadata misses; cost bounded.

## P2 — Triage step + context contract
- [ ] T2.1 Pipeline step that runs before scoring, writes `context.set_aside = [{image, reason, confidence}]`; curation runs on survivors only.
- [ ] T2.2 Surface `set_aside` (+ reasons, counts) in the result schema the UI reads.
- [ ] T2.3 Slovakia26 **re-run**: assert 0 junk in the curated album (vs 6/20).
- **Gate:** album is junk-free; set_aside populated with reasons.

## P3 — Behavioral tier (optional, needs full roll)
- [ ] T3.1 Singleton (scene-cluster-of-one) + no-person signal → "possible note-to-self".
- [ ] T3.2 Consume theme-relevance from spec-108 (if available) to extend set-aside to low-interest.
- **Gate:** catches ≥ the Slovakia utilitarian leaks (car/CDs/ladder) with low false-positives on the full contiguous roll.

## P4 — Review/restore UX contract + ship
- [ ] T4.1 Define the "Set aside (N)" review contract for spec-106 (reason chips, one-click restore). Never silent-delete.
- [ ] T4.2 CHANGES_LOG + `/code-review` → REVIEW.md; flip to Implemented.

## Out of scope
Theme/interestingness *scoring* (spec-108 produces it) · the review UI build (spec-106) · hard delete.
