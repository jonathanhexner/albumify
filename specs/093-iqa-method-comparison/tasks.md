# Spec 093 — Tasks

## Slice 0 — Dependency spike  ✅ DONE 2026-06-29
- [x] T0.1 Installed `pyiqa==0.1.15.post2` holding `numpy<2`. protobuf 3.20.3 +
      torch 2.3.1 untouched. **Required pin: `numpy<2`** (unpinned pyiqa pulls
      numpy 2.4.6 → ABI break). pyiqa also added a 3rd opencv variant → SIGHTING-111.
- [x] T0.2 Smoke OK on CPU: all 6 metrics (maniqa/musiq/hyperiqa/brisque/niqe/
      clipiqa) present; brisque/niqe/maniqa scored the finger-occlusion examples.
      Weights download to `~/.cache/torch/hub/pyiqa/`; maniqa cold-load ≈7s.
      `lower_better`: brisque/niqe = True, maniqa = False → direction-flip confirmed.

## Slice 1 — PyIQA scorer
- [ ] T1.1 `sim_bench/quality_assessment/pyiqa_quality.py`: `PyIQAQuality`
      (lazy `create_metric`, `assess_image`, direction-normalize via
      `metric.lower_better`, `is_available`).
- [ ] T1.2 Register under maniqa/hyperiqa/brisque/niqe/clipiqa; import in
      whatever module the registry auto-loads (mirror musiq/cnn_methods).
- [ ] T1.3 `ut_PyIQAQuality` unit test (availability + normalization on mock).
- [ ] T1.4 CHANGES_LOG entry [FEATURE].

## Slice 2 — Generic pipeline step
- [ ] T2.1 Add `method_scores: Dict[str, Dict[str, float]]` to `PipelineContext`.
- [ ] T2.2 `sim_bench/pipeline/steps/score_quality.py` `ScoreQualityStep`
      (thin; dispatch via registry; per-(image,method) cache).
- [ ] T2.3 Register in `all_steps.py`.
- [ ] T2.4 `test_score_quality_step`: dispatch + cache-hit on 2nd run, over
      `examples/finger_occlusion/`.
- [ ] T2.5 Update `docs/architecture/` (classes + pipeline step list).
- [ ] T2.6 CHANGES_LOG entry.

## Slice 3 — Standalone comparison app  ❌ SUPERSEDED BY spec-094
The standalone IQA app is replaced by spec-094 (Image Analysis Studio), whose
`image_quality` family consumes 093's `ScoreQualityStep`. **Do not build this
slice.** 093 ships backend-only (Slices 1–2). Retained here for traceability.
- [~] T3.1 `app/iqa_compare/main.py` — superseded by `app/image_studio/` (094).
- [~] T3.2 Run path — 094's `engine.run_methods` calls `ScoreQualityStep`.
- [~] T3.3 Table — 094's `view.py` renders the `image_quality` tab.
- [~] T3.4–T3.6 — owned by 094.

## Close-out
- [ ] T4.1 Run `/code-review` → `specs/093-iqa-method-comparison/REVIEW.md`.
- [ ] T4.2 Resolve high-severity findings; flip spec to Implemented.
