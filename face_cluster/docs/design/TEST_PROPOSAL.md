# Test Proposal: Embedding-Face-Image Validation

**Date**: 2026-03-28
**Proposed By**: User
**Context**: Previously encountered bug where face gating caused mismatch between faces and embeddings

---

## Problem Statement

In past implementations, when faces were filtered out during quality gating, this caused systematic mismatches between:
- Face records (image + bbox)
- Face crops (aligned images)
- Face embeddings (512-dim vectors)

This led to incorrect clustering where embeddings didn't match the faces they were supposed to represent.

---

## Proposed Test Design

### Test Dataset
Create **2-3 test images** where:
1. **Image 1**: High-quality face (will PASS quality gate)
2. **Image 2**: Low-quality face (will FAIL quality gate - extreme pose, blur, or small size)
3. **Image 3**: Multiple faces, mixed quality (some pass, some fail)

### Test Strategy

For each test image, run **two parallel paths**:

**Path A: Pipeline Extraction** (with quality gating)
```python
# Run full pipeline
context = run_pipeline(test_images)
pipeline_embeddings = context.face_embeddings  # From pipeline
pipeline_faces = context.face_records  # Face metadata
```

**Path B: Direct Extraction** (no filtering)
```python
# Direct extraction without gating
embedder = InsightFaceEmbedder()
all_faces = embedder.detect_and_embed(test_images)
direct_embeddings = {face.face_id: face.embedding for face in all_faces}
```

### Validation Checks

For faces that **passed** the quality gate:
1. **Embedding match**:
   ```python
   assert np.allclose(pipeline_embeddings[face_id], direct_embeddings[face_id])
   ```

2. **Image path match**:
   ```python
   assert pipeline_faces[face_id].image_path == all_faces[face_id].image_path
   ```

3. **Bbox match**:
   ```python
   assert pipeline_faces[face_id].bbox == all_faces[face_id].bbox
   ```

4. **Sequential IDs**: Face IDs should be sequential (0, 1, 2, ...) after filtering
   ```python
   assert list(pipeline_embeddings.keys()) == list(range(len(pipeline_embeddings)))
   ```

For faces that **failed** the quality gate:
5. **Not in pipeline output**:
   ```python
   assert face_id not in pipeline_embeddings
   ```

### Additional Test: Clustering Output Validation

After clustering:
```python
# Load exported data
faces_df = pd.read_csv("faces.csv")
clusters_df = pd.read_csv("clusters.csv")

# For each face in a cluster, verify crop matches embedding
for face_id in faces_df['face_id']:
    crop_path = f"face_crops/face_{face_id:04d}_aligned.jpg"
    crop = cv2.imread(crop_path)

    # Extract embedding from saved crop
    crop_embedding = embedder.extract_single(crop)

    # Should match stored embedding
    stored_embedding = get_embedding_from_metadata(face_id)
    assert np.allclose(crop_embedding, stored_embedding, atol=1e-5)
```

---

## Expected Outcomes

✅ **PASS**: All embeddings from pipeline match direct extraction
✅ **PASS**: Filtered faces are NOT in pipeline output
✅ **PASS**: Face IDs are sequential and match across all data structures
✅ **PASS**: Saved crops match their embeddings

❌ **FAIL**: Any mismatch indicates pipeline bug (face gating offset, reordering, etc.)

---

## Test Location

Suggested: `tests/pipeline/test_face_embedding_validation.py`

---

## Questions for Expert Review

1. **Test coverage**: Is 2-3 images sufficient, or should we test more edge cases?
2. **Tolerance**: What tolerance should we use for `np.allclose()` checks? (currently 1e-5)
3. **Failure scenarios**: What other failure modes should we test?
4. **Integration**: Should this be a unit test (isolated) or integration test (full pipeline)?
5. **Regression**: Should this test run on every pipeline change, or only during face clustering development?

---

**Status**: Awaiting expert review before implementation.
