# Spec 093 — Tasks

## Slice 0 — Dependency spike  ✅ DONE 2026-06-29
- [x] T0.1 Installed `pyiqa==0.1.15.post2` holding `numpy<2`. protobuf 3.20.3 +
      torch 2.3.1 untouched. **Required pin: `numpy<2`** (unpinned pyiqa pulls
      numpy 2.4.6 → ABI break). pyiqa also added a 3rd opencv variant → SIGHTING-111.
- [x] T0.2 Smoke OK on CPU: all 6 metrics (maniqa/musiq/hyperiqa/brisque/niqe/
      clipiqa) present; brisque/niqe/maniqa scored the finger-occlusion examples.
      Weights download to `~/.cache/torch/hub/pyiqa/`; maniqa cold-load ≈7s.
      `lower_better`: brisque/niqe = True, maniqa = False → direction-flip confirmed.

## Slice 1 — PyIQA scorer  ✅ DONE 2026-06-29
- [x] T1.1 `sim_bench/image_quality_models/pyiqa_model_wrapper.py`: `PyIQAModel`
      (BaseQualityModel; lazy `create_metric`; `score_image` direction-normalized
      via `lower_better`; `raw_score`; `is_available`). NOTE: integration point
      corrected to `image_quality_models` (legacy QualityAssessor archived).
- [x] T1.2 Registered maniqa/musiq/hyperiqa/brisque/niqe/clipiqa in
      `model_factory.MODEL_REGISTRY` (all front PyIQAModel).
- [x] T1.3 `tests/image_quality_models/test_pyiqa_model.py` (6 tests:
      registry, availability, direction both branches, from_config, real BRISQUE).
- [x] T1.4 CHANGES_LOG entry.

## Slice 2 — Generic pipeline step  ✅ DONE 2026-06-29
- [x] T2.1 Added `method_scores: dict[str, dict[str, float]]` to `PipelineContext`.
- [x] T2.2 `sim_bench/pipeline/steps/score_quality.py` `ScoreQualityStep`
      (overrides process() for per-method `quality_<method>` cache namespace).
- [x] T2.3 Registered in `all_steps.py`.
- [x] T2.4 `tests/pipeline/test_score_quality_step.py` (dispatch, cache-hit,
      unknown-method skip, empty album). Plus real brisque+niqe run on examples.
- [x] T2.5 Updated `docs/architecture/data_flow.html` (step + method_scores field).
- [x] T2.6 CHANGES_LOG entry.

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
