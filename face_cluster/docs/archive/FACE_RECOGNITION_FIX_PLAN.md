# Face Recognition Fix Plan

## Problem Summary

**Symptom**: People tab shows all faces assigned to one person (Person 0 with 412 faces)

**Root Cause**: 93.8% of face embeddings are zero vectors (410 out of 437)
- Zero vectors are identical → HDBSCAN clusters them together
- Embedding extractor returns zero vectors when face crops are invalid/empty
- Something is broken in the cropping → embedding extraction pipeline

## Evidence

From database analysis:
```
Total face embeddings: 437
Zero vectors: 410 (93.8%)
Valid vectors: 27 (6.2%)
```

In `insightface_native.py`, zero vectors are returned when:
1. Face image is None or empty (lines 73-76)
2. Face image has unexpected shape (lines 86-88)
3. Fallback path can't extract embedding (lines 124-126)

## Investigation Tasks

### Task 1: Create Face Recognition Benchmark Test
**Purpose**: Verify the embedding model works correctly on known-good face crops (CASIA WebFace data)

**Test Data**: `D:\sim-bench\test_data\casia_webface`
- 2 folders (00000, 00001) = 2 different people
- Pre-cropped face images (no detection needed)

**Tests**:
1. Load images directly, extract embeddings
2. Verify all embeddings are valid (non-zero, 512-dim, normalized)
3. Check intra-person similarity (same person should be > 0.6)
4. Check inter-person distance (different people should be < 0.4)
5. Run clustering, verify 2 clusters with correct assignments

**Success Criteria**:
- If benchmark PASSES: Embedding model works, bug is in cropping/data flow
- If benchmark FAILS: Embedding model or its configuration is broken

### Task 2: Add Diagnostic Logging to Pipeline
**Purpose**: Trace why face crops are invalid

**Add logging to `extract_face_embeddings.py`**:
- Log face image shape before passing to extractor
- Log whether image data is None/empty
- Log crop coordinates and source image dimensions

**Add logging to `insightface_native.py`**:
- Log which path triggered zero vector (null image, bad shape, extraction failure)
- Log actual image shape received

### Task 3: Investigate Cropping Logic
**Files to check**:
- `extract_face_embeddings.py` lines 131-176: Cropping from InsightFace bbox
- Check if bbox coordinates are valid (not out of bounds)
- Check if image cache returns valid images

### Task 4: Fix the Bug
Based on findings from Tasks 1-3, implement fix.

## Implementation Plan

### Phase 1: Benchmark Test (Do First)
Create `tests/pipeline/test_face_recognition_benchmark.py`

```python
# Test structure:
def test_embedding_extraction_valid():
    """Verify embeddings are non-zero for valid face crops."""

def test_intra_person_similarity():
    """Same person's faces should have high similarity."""

def test_inter_person_distance():
    """Different people should have low similarity."""

def test_clustering_accuracy():
    """HDBSCAN should produce correct clusters."""
```

### Phase 2: Diagnostic Run
1. Add logging to identify which faces produce zero vectors
2. Run pipeline on small test set
3. Trace the specific failure point

### Phase 3: Fix
Apply targeted fix based on diagnostic findings.

### Phase 4: Verify
1. Clear cache: Delete face_embedding entries from universal_cache
2. Re-run pipeline
3. Verify embeddings are now valid
4. Verify People tab shows correct clustering

## Files to Modify

### New Files
- `tests/pipeline/test_face_recognition_benchmark.py`

### Files to Add Logging
- `sim_bench/pipeline/steps/extract_face_embeddings.py`
- `sim_bench/pipeline/face_embedding/insightface_native.py`

### Potential Bug Locations
- `sim_bench/pipeline/steps/extract_face_embeddings.py` (cropping logic)
- `sim_bench/pipeline/utils/image_cache.py` (image loading)
- Bbox coordinate handling (relative vs pixel coordinates mismatch)

## Commands

```bash
# Run benchmark test
python -m pytest tests/pipeline/test_face_recognition_benchmark.py -v -s

# Clear face embedding cache (after fix)
python -c "
import sqlite3
conn = sqlite3.connect(r'C:\Users\Jonathan Hexner\.sim_bench\sim_bench.db')
conn.execute(\"DELETE FROM universal_cache WHERE feature_type = 'face_embedding'\")
conn.commit()
print('Cleared face embedding cache')
conn.close()
"
```

## Expected Outcome

After fix:
- Face embeddings should have ~0% zero vectors
- HDBSCAN should produce distinct clusters per person
- People tab should show correct face groupings
