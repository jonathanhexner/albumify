# spec-107 — Tasks (Input Triage & Discard Review)

**Status:** Draft — no implementation until spec.md approved. Full rationale: `spec.md`.

## P0 — Metadata tier (cheapest, highest-precision)  ✅ DONE 2026-10-01
- [x] T0.1 `sim_bench/triage/metadata.py`: `calc(TriageInputs)->TriageResult` (filename / camera-EXIF / format / screen-dims → keep|review|set_aside, conservative) + `classify_file(path)`.
- [x] T0.2 `tests/triage/test_metadata_triage.py` (9 tests, pure) — green.
- **Gate:** ✅ met — 715-roll validation: **8/8 screenshots set_aside, 707 keep, 0 review (0 false positives)**.

## P1 — Content tier (RAM++)  ✅ DONE 2026-10-01
- [x] T1.1 `sim_bench/triage/content.py`: pure `calc(tags)->TriageResult` (strong/soft junk taxonomy + moment-override: a printed item within a real scene -> review, not set_aside) + `RamTagger` (lazy RAM++ live tag source, transformers shim).
- [x] T1.2 `tests/triage/test_content_triage.py` (8 tests). Validated on the 75 real-tag subset: caught the 8 screenshots + **3 junk metadata missed** (poster `…105436` that leaked into the album, packaging `…155313`, re-shared screen `…153154`); signs/posters-in-scene -> review (kept); no false positives. Live `RamTagger` smoke green.
- **Gate:** ✅ met — catches photographed docs/posters metadata misses; cost = one isolated RAM++ step (spec-104).

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
