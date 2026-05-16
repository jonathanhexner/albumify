# Tasks: Config-to-Producer Graph Contract (036)

## Design Notes

- **D1**: The producer side gets a typed declaration (extend `StepMetadata`) rather than AST-scraping step source. Source-scraping breaks the moment someone writes `setattr(context, ...)` or assigns inside a helper.
- **D2**: The gate side gets a flat map in `face_cluster/quality_contracts.py`. The map is small (~5 gates today) and explicit; auto-discovery isn't worth the complexity.
- **D3**: Waivers are first-class. A pin without a waiver is a fail. A waiver without a target ticket is a fail. A waiver whose target ticket is `Implemented` is a fail. This forces pin removal when its underlying work lands.

## Phase 1: Producer declaration

- [ ] T001 Extend `sim_bench/pipeline/base.py::StepMetadata` with a `produces_fields: set[str]` attribute (default empty set).
- [ ] T002 Annotate the 4 face-level producer steps with the fields they write to `FaceRecord` / `insightface_faces[path]["faces"][i]`:
  - `insightface_detect_faces` → `{"confidence", "landmarks", "bbox"}`
  - `insightface_score_pose` → `{"pose_score"}` (not the 3-tuple — note in docstring)
  - `align_faces` → `{"aligned_face"}`
  - `extract_face_embeddings` → `{"embedding"}`
- [ ] T003 Tests: assert each annotated step's metadata round-trips through `to_dict()`.

**Checkpoint**: `.venv/Scripts/python -c "from sim_bench.pipeline.steps.insightface_detect_faces import InsightFaceDetectFacesStep; print(InsightFaceDetectFacesStep()._metadata.produces_fields)"` prints the declared set.

## Phase 2: Gate contract map + waiver mechanism

- [ ] T010 Create `face_cluster/quality_contracts.py` with:
  - `GATE_PRODUCERS: dict[str, str]` — gate config path → required FaceRecord/insightface_faces field
  - `GATE_WAIVERS: dict[str, tuple[str, str]]` — gate → (reason, target_spec)
  - Initial waiver: `"cluster_people.blur_min": ("InsightFace pipeline has no blur step yet", "specs/037-insightface-blur-step")`
- [ ] T011 Populate `GATE_PRODUCERS` for: `blur_min`, `yaw_max`, `pitch_max`, `roll_max`, `det_score_min`.

**Checkpoint**: Reading the map and the StepMetadata produces a consistent picture of "which gates have producers in `default_pipeline`."

## Phase 3: Architecture test

- [ ] T020 Create `tests/architecture/test_gate_has_producer.py`:
  - For each gate in `GATE_PRODUCERS`: assert the required field is in the union of `produces_fields` across steps in `default_pipeline` from `configs/pipeline.yaml`, OR is waived.
  - For each waiver: assert the gate exists in `GATE_PRODUCERS`; assert the target spec exists; assert the target spec's status is not `Implemented` (overdue check).
- [ ] T021 Run: confirms `blur_min` is waived with target spec-037, all other gates have producers.

**Checkpoint**: `.venv/Scripts/python -m pytest tests/architecture/test_gate_has_producer.py -v` passes with one waived gate and four enforced gates.

## Phase 4: Wire into CODE_REVIEW_CHECKLIST

- [ ] T030 Update `docs/guides/CODE_REVIEW_CHECKLIST.md` §6 — "Config knob → producer check" now references this test as the enforcement.
- [ ] T031 Add the test to the architecture test list in spec-033's REVIEW.md (historical record).

**Checkpoint**: A reviewer running `/code-review` against a future PR that adds a new gate but no producer gets a clean fail.
