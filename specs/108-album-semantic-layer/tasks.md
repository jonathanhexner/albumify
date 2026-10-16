# spec-108 — Tasks (Album Semantic Layer)

**Status:** Draft — no implementation until spec.md approved. Rationale: `spec.md` + agentic_layer_design.html.

## P0 — Cheap content tags on all (Tier-1 semantic)  ✅ DONE 2026-10-01
- [x] T0.1 Reuses `context.ram_tags` already cached by spec-107 `triage_content` — **no second RAM++ pass** (cost = 0 beyond triage). `derive_semantic_tags` step (depends_on triage_content).
- [x] T0.2 `sim_bench/semantic/tags.py`: pure `clean()` (drops the observed verb/fragment + vague/color noise) + `categorize()` → coarse facets (people/nature/urban/food/animal/activity), token-match for compound tags. 10 tests. Validated on the real Slovakia subset: sensible family-trip facet mix (34 people / 20 nature / 11 urban / 1 food / 9 other).
- **Gate:** ✅ met — clean per-image semantic tags + facets on a real trip, zero added model cost.

## P1 — Theme inference
- [x] T1.1 VLM decision: **MiniCPM-V 4.5 local** (2026-10-01, $0/token + privacy). VLM seam built:
      `sim_bench/vlm/base.py` `VLMProvider` protocol. **Live backend blocked on env** (needs
      transformers==4.51 vs 4.57; 8B on CPU impractical) → runs on spec-109 GPU worker / `.venv-vlm` / ollama.
- [x] T1.2 **Tag-based theme baseline (no VLM)** DONE: `sim_bench/semantic/theme.py` `infer_theme()` +
      `infer_album_theme` step → `context.album_theme {album_type, facet_distribution, top_tags, motifs}`.
      7 tests; validated on real Slovakia → "family trip — towns & nature" (people 34/nature 20/urban 11).
- [ ] T1.3 (deferred) VLM-refined theme from a representative image sample → overwrite with `source="vlm"`.
- **Gate:** tag-based theme correct on the real trip ✅; VLM refinement pending worker env.

## P2 — Captioning (the captioning request)  — BUILT 2026-10-01 (live validation pending model load)
- [x] T2.0 Isolated VLM runtime: `.venv-vlm` (transformers 4.51) + `vlm_worker.py` (MiniCPM-V, bf16 CPU)
      + `SubprocessVLMProvider` over JSON-lines; seam extended with `generate()` (text-only). Plumbing
      proven end-to-end with `--stub` on a real image.
- [x] T2.1 `caption_keepers` step: per-keeper caption via `provider.caption` → `context.image_captions`
      (keepers only; opt-in `enabled=false`). Pure helpers (`KEEPER_PROMPT`, `clean_caption`) + step
      tested with a fake provider.
- [x] T2.2 `caption_album` step: ONE text-only `provider.generate` synthesizing the headline from keeper
      captions + theme → `album_theme.caption` (`source="vlm"`). Tested (one call, not per-image).
- **Gate:** steps + helpers green (26 tests), in default_pipeline gated off, build-ordered after
      select_best.
- **Live MiniCPM-V on this CPU box — PROVEN TO LOAD, inference impractical.** All 4 shards cached on D:
      (cache relocated off a full C:); worker loads the 8B model in `.venv-vlm` (`{"ready": true}`,
      bf16). But a single CPU caption did not complete inside the task runner's limit (8B VLM on CPU is
      minutes/call, swapping on 32 GB). **Live caption + latency deferred to the spec-109 GPU worker**
      (the designed home). Plumbing fully proven via `--stub`; backend proven to download + load.

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
