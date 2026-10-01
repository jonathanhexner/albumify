# spec-108 — Tasks (Album Semantic Layer)

**Status:** Draft — no implementation until spec.md approved. Rationale: `spec.md` + agentic_layer_design.html.

## P0 — Cheap content tags on all (Tier-1 semantic)
- [ ] T0.1 RAM++ content-tag step (reuse the validated tagger), isolated step (spec-104); emit per-image tags.
- [ ] T0.2 Post-filter tags to a noun allow-list (drop RAM++ verb/fragment noise observed 2026-10-01).
- **Gate:** clean per-image semantic tags on a real trip, cost bounded.

## P1 — Theme inference
- [ ] T1.1 Pick the VLM (local open: Qwen3-VL / MiniCPM-V vs API) — decision + thin wrapper (shared w/ spec-105).
- [ ] T1.2 Theme-inference call on a representative sample → structured `{type, place, motifs}`.
- **Gate:** correct theme on the 3 trips + Slovakia26 (human-judged).

## P2 — Captioning (the captioning request)
- [ ] T2.1 Per-keeper captions (~20 survivors) → "why picked" text + scene labels.
- [ ] T2.2 Album-level caption from theme + keeper captions.
- **Gate:** captions accurate/useful on spot-check; cost ≈ pennies/album.

## P3 — Theme-relevance / interestingness (feeds spec-107)
- [ ] T3.1 Per-photo theme-relevance score (VLM on ambiguous "real-but-maybe-utilitarian" subset).
- [ ] T3.2 Wire low-relevance → spec-107 `set_aside` (review, not delete).
- **Gate:** flags ≥ the Slovakia utilitarian leaks (car/CDs/ladder) with low false-positives.

## P4 — Optional semantic scene-clustering boost
- [ ] T4.1 Add tags/semantic signal as an opt-in auxiliary to the visual clusterer.
- [ ] T4.2 Measure vs visual-only duplicate-survival; ship only if no regression.
- **Gate:** opt-in, no dedup regression, documented.

## P5 — Harden + ship
- [ ] T5.1 Cost/latency budget (sample + survivors + ambiguous only); measured $/album.
- [ ] T5.2 CHANGES_LOG + `/code-review` → REVIEW.md; flip to Implemented.

## Out of scope
Replacing visual dedup · per-image VLM on the whole roll · auto-delete · over-tuning to a VLM (reference/judge only).
