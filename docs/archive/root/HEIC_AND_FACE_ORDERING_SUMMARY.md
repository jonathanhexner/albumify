# HEIC Support and Face Ordering Implementation

**Date**: 2026-03-31
**Status**: PARTIALLY COMPLETE

## Summary

Implemented HEIC image format support and deterministic face ordering utilities. However, discovered that ground truth face crops don't match the detected faces at specified indices.

## Completed Work

### 1. HEIC Image Format Support ✓

**Files Modified**:
- `face_cluster/embedding.py`

**Implementation**:
- Replaced `cv2.imread` with PIL `Image.open` in `detect_and_embed` method
- Added `from pillow_heif import register_heif_opener` and `register_heif_opener()`
- Applied EXIF transpose to ensure correct orientation
- Handles RGB conversion for grayscale and RGBA images

**Result**: All 15/15 ground truth faces extracted successfully (6 HEIC files now load correctly)

### 2. Face Ordering Utilities ✓

**Files Created**:
- `sim_bench/utils/face_ordering.py`
- `scripts/analyze_face_detection_order.py`

**Implementation**:
- **`sort_faces_reading_order(faces)`**: Spatial ordering (top-to-bottom, left-to-right)
  - Groups faces into horizontal rows (50% Y-overlap threshold)
  - Sorts left-to-right within each row
  - Sorts rows top-to-bottom

- **`sort_faces_by_area(faces)`**: Sorts by bounding box area (largest first)

- **`sort_faces_by_confidence(faces)`**: Sorts by detection score (InsightFace default)

- **`get_face_ordering_index(faces, ordering)`**: Returns index mapping for conversions

**Analysis Results**:
- InsightFace default ordering: **confidence score** (highest first)
- Order is **consistent** across multiple runs
- Order is **NOT spatially intuitive** (user-requested reading order)

### 3. Configurable Face Ordering in InsightFaceEmbedder ✓

**Files Modified**:
- `face_cluster/embedding.py`

**Changes**:
- Added `face_ordering` parameter to `__init__` (default: 'detection_order')
- Options: 'detection_order', 'reading_order', 'area', 'confidence'
- Implemented `_sort_faces` method that applies selected ordering after detection
- Integrated with `sim_bench.utils.face_ordering` utilities

**Usage**:
```python
# Use confidence ordering (InsightFace default)
embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='detection_order')

# Use spatial reading order
embedder = InsightFaceEmbedder(model_name='buffalo_l', face_ordering='reading_order')
```

### 4. Full Pipeline Test Updates ✓

**Files Modified**:
- `tests/test_face_pipeline_full.py`

**Changes**:
- Replaced `cv2.imread` with PIL image loading (HEIC support)
- Updated embedder to use `detection_order` to match ground truth mapping
- All 15/15 faces now extract successfully

**Test Results**:
- ✓ `test_pipeline_extracts_all_faces`: PASSED
- ✗ `test_pipeline_embeddings_match_ground_truth`: FAILED
- ✗ `test_pipeline_preserves_identity_structure`: FAILED
- ✗ `test_distance_matrix_correlation`: FAILED

## Remaining Issue: Ground Truth Mapping Mismatch

### Problem

The ground truth face crops **do not match** the faces detected at the specified indices in `ground_truth_mapping.csv`.

### Evidence

**Diagnostic Script**: `scripts/verify_face_index_mapping.py`

**Similarity Scores** (should be >0.95 for correct matches):
```
Face 545: 0.715  (moderate - possible match)
Face 546: 0.295  (low - wrong face)
Face 550: 0.612  (moderate)
Face 551: 0.623  (moderate)
Face 557: 0.015  (very low - wrong face)
Face 558: -0.002 (negative - completely different)
Face 562: -0.012 (negative - completely different)
Face 569: -0.027 (negative - completely different)
Face 573: -0.003 (negative - completely different)
Face 580: 0.670  (moderate)
Face 584: 0.393  (low)
Face 587: 0.005  (very low - wrong face)
```

**Visual Verification**: Comparison images saved to `test_data/face_index_verification/`

### Root Cause Analysis

Possible explanations:

1. **Different Face Detection Run**: Ground truth crops may have been extracted from a different face detection run with different results

2. **Face Ordering Changed**: InsightFace detection order may have changed between creating ground truth and current run (though unlikely - confidence ordering should be stable)

3. **Manual Crop Creation**: Ground truth crops may have been manually created or cropped differently than automated pipeline

4. **Image Version Mismatch**: Source images in `test_data/source_images_ground_truth/` may differ from images used to create crops

## Recommended Next Steps

### Option 1: Regenerate Ground Truth (RECOMMENDED)

Create new ground truth using current detection + chosen ordering:

1. **Choose ordering convention**: `reading_order` (user-requested) or `detection_order` (backward compatible)

2. **Run detection on source images** with chosen ordering

3. **Save face crops** with corresponding `face_id → (image_path, face_index)` mapping

4. **Manually label** each crop with person identity

5. **Generate new mapping CSV** with correct indices

**Advantages**:
- Clean, reproducible ground truth
- Guaranteed correspondence with current detection
- Can use preferred reading_order convention

### Option 2: Reverse-Engineer Existing Crops

Try to match existing ground truth crops back to detected faces:

1. For each ground truth crop, compute embedding
2. For each source image, detect all faces and compute embeddings
3. Find best match using cosine similarity
4. Update mapping CSV with discovered indices

**Advantages**:
- Preserves existing manual labels
- No re-labeling required

**Disadvantages**:
- May fail if crops were manually created
- Heuristic matching could introduce errors

### Option 3: Use Isolated Crops Test Only

Keep only the working test (`test_face_embeddings_ground_truth.py`) that uses isolated crops:

**Advantages**:
- Already works (all tests pass)
- Tests core embedding extraction

**Disadvantages**:
- Doesn't test full pipeline (detect → align → embed)
- Doesn't verify face indexing correctness

## Technical Debt Items

1. **Ground Truth Versioning**: Add version/timestamp to ground truth data to track creation method

2. **Metadata Documentation**: Document which face ordering was used for each ground truth set

3. **Automated Ground Truth Generation**: Create script to generate ground truth with explicit ordering parameter

4. **Visual Verification Tool**: Build interactive tool to verify face crops match detection results before labeling

## Files Created/Modified

### Created:
- `sim_bench/utils/face_ordering.py` - Face ordering utilities
- `scripts/analyze_face_detection_order.py` - Detection order analysis
- `scripts/verify_face_index_mapping.py` - Ground truth verification
- `HEIC_AND_FACE_ORDERING_SUMMARY.md` (this file)

### Modified:
- `face_cluster/embedding.py` - HEIC support + configurable ordering
- `tests/test_face_pipeline_full.py` - HEIC support + detection_order
- `CHANGES_LOG.md` - Logged all changes

## Conclusion

✓ **HEIC Support**: Complete and working (all 15 faces extract successfully)

✓ **Face Ordering Utilities**: Complete and tested

✗ **Full Pipeline Test**: Blocked by ground truth mapping mismatch

**Blocker**: Ground truth crops don't correspond to face indices in mapping CSV

**Recommendation**: Regenerate ground truth with current detection + chosen ordering convention
