# Tasks: InsightFace Blur Scoring Step (037)

## Design Notes

- **D1**: Laplacian variance, single-source helper. If FC App already has `_compute_blur` in `face_cluster.quality`, lift it to a shared module. If not, write once and import from both sides.
- **D2**: Step uses BaseStep template method (cacheable) keyed on `(image_path, "insightface_blur", model_name="laplacian_v1")`. Re-running the same image is a cache hit.
- **D3**: Reverting the SIGHTING-061 pin is in the same PR as adding the step. The whole point is to keep "the gate has a producer" invariant.

## Phase 1: Shared blur computation

- [ ] T001 Audit `face_cluster/quality.py` for existing blur computation; extract to `face_cluster/blur.py::laplacian_variance(crop_rgb: np.ndarray) -> float` if so.
- [ ] T002 Unit test `tests/face_clustering/test_blur.py`: synthetic crops of varying sharpness produce monotonically ranked scores.

**Checkpoint**: `pytest tests/face_clustering/test_blur.py` passes.

## Phase 2: New step

- [ ] T010 `sim_bench/pipeline/steps/configs/insightface_score_blur.py` — Pydantic `InsightFaceScoreBlurConfig(min_face_size: int = 50)`.
- [ ] T011 Register in `sim_bench/pipeline/steps/configs/__init__.py::STEP_CONFIG_MODELS`.
- [ ] T012 `sim_bench/pipeline/steps/insightface_score_blur.py` — `InsightFaceScoreBlurStep(BaseStep)`:
  - `_metadata.produces_fields = {"blur_score"}` (spec-036 hook).
  - `_get_cache_config` keyed on path + "blur_score" + "laplacian_v1".
  - `_process_uncached` reads `face_data["faces"][i]["aligned_face"]` (or recomputes from `bbox` + image if absent); writes `scores["blur_score"]`.
- [ ] T013 Unit test: step processes a synthetic context and writes blur_score onto every face above min_face_size.

**Checkpoint**: Step runs in isolation against a fixture; assertions pass.

## Phase 3: Pipeline registration

- [ ] T020 Update `configs/pipeline.yaml::default_pipeline` — insert `insightface_score_blur` after `align_faces`, before `filter_quality`.
- [ ] T021 Add `insightface_score_blur: {min_face_size: 50}` to the step configs section of `configs/pipeline.yaml`.

**Checkpoint**: `python -c "from sim_bench.api.services.pipeline_service import PipelineService; print(PipelineService().pipelines['default_pipeline'])"` includes the new step.

## Phase 4: Lift the SIGHTING-061 pin + waiver

- [ ] T030 `sim_bench/pipeline/steps/face_cluster_bridge.py::build_fc_config` — revert `blur_min=0.0` pin; read from config.
- [ ] T031 Update test `tests/architecture/test_config_parity.py::test_bridge_pose_and_det_gates_read_from_config` — assert `config.get("blur_min", ...)` is now present; remove the docstring-pin assertion.
- [ ] T032 Update `face_cluster/quality_contracts.py::GATE_WAIVERS` (spec-036) — remove the `cluster_people.blur_min` entry.
- [ ] T033 Update SIGHTING-061 status to `RESOLVED — pin lifted, blur step landed in spec-037`.

**Checkpoint**: spec-035 E2E test still passes; spec-036 `test_gate_has_producer.py` passes with no waivers for blur_min.

## Phase 5: Cross-pipeline parity

- [ ] T040 Pick one image; run it through FC App standalone and Albumify; compare blur scores. Within 5% per US3.
- [ ] T041 Document the helper location and the comparison result in `docs/architecture/blur_scoring.md` (1 page).

**Checkpoint**: Comparison documented; spec-033's CHANGES_LOG entry on blur unblocking can be removed when the next code-review runs.
