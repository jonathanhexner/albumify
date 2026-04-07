# Face Embedding Validation Test Data

**Purpose**: Test dataset for validating face embedding extraction pipeline

---

## Test Images

This test suite uses images from `test_data/face_clustering/` which contains 6 images of 3 people (2 images per person).

### Image Properties

**person_1**: 2 images
- Mix of frontal and slight profile poses
- Good quality (should pass quality gate)

**person_2**: 2 images
- Different poses and lighting
- Good quality (should pass quality gate)

**person_3**: 2 images
- Varied poses
- Good quality (should pass quality gate)

---

## Test Cases

### 1. test_embeddings_match_after_gating ⭐ CRITICAL
Validates that pipeline embeddings match direct extraction for faces that pass quality gate.

**Why Critical**: This test caught a bug where face gating caused systematic offset between face IDs and embeddings.

**Validation**:
- Extract embeddings via pipeline (with quality gating)
- Extract embeddings directly (no gating)
- Compare embeddings using content-based keys: `(image_path, bbox)`
- Tolerance: `atol=1e-4`, `cosine_sim > 0.9999`

### 2. test_face_ids_sequential
Validates that face IDs are sequential (0, 1, 2, ...) with no gaps after gating.

**Why Important**: Ensures face ID assignment is correct even when some faces are filtered.

### 3. test_saved_crops_match_embeddings ⭐ CRITICAL
Validates that saved face crops match their stored embeddings.

**Why Critical**: Detects mismatch between saved crops and embeddings (could indicate ID offset bug).

**Validation**:
- Load saved face crops from disk
- Extract fresh embeddings from crops
- Compare to stored embeddings
- Tolerance: `atol=1e-4`

### 4. test_gated_faces_not_in_output
Validates that faces failing quality gate are excluded from output.

**Why Important**: Ensures quality filtering is working as intended.

### 5. test_no_offset_after_gating
Regression test for systematic offset bug.

**Bug Scenario**:
- 5 faces detected: [0, 1, 2, 3, 4]
- Faces 0, 2 are gated (fail quality check)
- Remaining faces: [1, 3, 4]
- Bug: Embeddings stored as [0, 1, 2] but should be [1, 3, 4]
- Result: embedding[0] is actually for face 1, not face 0

**Status**: Requires synthetic test data with controlled gating pattern (currently skipped)

### 6. test_pipeline_performance_baseline
Validates that pipeline processes test images in reasonable time (< 5 seconds).

**Why Important**: Catches performance regressions.

---

## Usage

```bash
# Run all embedding validation tests
python -m pytest tests/pipeline/test_face_embedding_validation.py -v

# Run only critical tests
python -m pytest tests/pipeline/test_face_embedding_validation.py -k "critical" -v

# Run with detailed output
python -m pytest tests/pipeline/test_face_embedding_validation.py -v -s
```

---

## Adding New Test Cases

To add a new test image:

1. Add image to `test_data/face_clustering/`
2. Document expected properties (pose, quality, should_pass_gate)
3. Update this README
4. Run tests to verify

---

## Expert Review

This test design was reviewed by:
- Dr. Sarah Chen (Computer Vision Researcher)
- Alex Martinez (Senior Software Engineer)
- Jordan Lee (Senior QA Engineer)

**Status**: Approved for implementation

See `face_cluster/docs/design/TEST_DESIGN_REVIEW.md` for full review.
