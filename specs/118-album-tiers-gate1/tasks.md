# spec-118 — Tasks (Album tiers and Gate 1)

**Status:** Draft 2026-11-13
**Spec:** [spec.md](spec.html) · [updates](updates.html) · roadmap **Gate 1** · [ROADMAP.md](../../ROADMAP.html)

Design lives in the spec (§3 plan, §4 class diagram, §5-§6 file tables). This file orders the work. Builds on
spec-117 (branch `spec/117-neon-r2`); starts after the owner approves.

## W1 — Rank and tiers
- [ ] T1.1 `sim_bench/album/tiers.py` `AlbumTiers.calc(TierInputs) -> TierResult`: rank by score, ties by time;
      Best N with at most `max_per_scene` per scene, then fill by score. Scene -1 photos count as their own scene each.
      Test `tests/album/test_tiers.py` (ranks across scenes, spread, fewer than N). (R1 R2 R6)

## W2 — API `top=N`
- [ ] T2.1 `ResultService.get_images(..., top)`: build `TierInputs` from stored `composite_score` and
      `scene_clusters`; return the first N in rank order. (R3 R7)
- [ ] T2.2 `routers/results.py` `get_result_images`: `top: Optional[int] = Query(None, ge=1, le=1000)`. (R4)
- [ ] T2.3 Test `tests/api/test_results_tiers.py` (ranked, unchanged without top, 422, old run). (R3 R4 R7)

## W3 — Results page
- [ ] T3.1 `ApiClient.get_images(..., top)`. (R5)
- [ ] T3.2 `pages/results.py` `_render_image_views`: Best 10 · Best 30 (default) · All picks · All processed · By
      cluster; rank badge on cards; "all 7 picks" label when short. (R5 R6)
- [ ] T3.3 Playwright `tests/test_results_tiers_e2e.py` on the real app, screenshots on disk. (R5 R6)

## W4 — Honest reasons
- [ ] T4.1 `run_output.py` `build_reason_by_path`: a `rejected` decision wins over detection notes; `select_best` still
      last. Test `tests/serving/test_run_output.py::test_a_rejection_reason_wins_over_detection`. (R9)
- [ ] T4.2 `select_best.py` `_select_from_cluster`: count rank over the sorted list. Test
      `tests/pipeline/test_select_best_reasons.py`. (R10)

## W5 — Review yardstick
- [ ] T5.1 `scripts/score_review.py` `ReviewScorer`: read a run (DB) + `picks_review.json`; print keep rate at 10 / 30 /
      all and wanted photos now picked. Test `tests/scripts/test_score_review.py`; real: `52776cc5` gives 10/10, 28/30,
      219/299. (R8)

## W6 — Gate 1
- [ ] T6.1 Owner judges Best 30 of Slovakia26 (the spec-117 T8.1 run) and of Budapest in the app; verdict + keep rate in
      updates.md. (R11)
- [ ] T6.2 Full suite, Budapest gate, `/code-review` → REVIEW.md. (R12)
