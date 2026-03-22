# SIGHTING-006: Root Cause Analysis - Complete Findings

**Date**: 2026-03-23
**Status**: ✅ ROOT CAUSE IDENTIFIED

---

## Executive Summary

**Problem**: Embeddings array has systematic +2 offset where `stored[N]` matches `fresh[N+2]`

**Root Cause**: `benchmark_face_clustering.py` uses incremental counter for filenames instead of face_id from metadata

**Impact**: When first 2 faces fail validation, all subsequent faces are saved with shifted filenames, creating permanent mismatch between metadata indices and crop filenames

**Severity**: Critical - affects all downstream processing that assumes filename index = metadata index

---

## The Bug

### Location
`scripts/benchmark_face_clustering.py`, lines 328-347

### Code
```python
def save_face_crops(metadata: List[Dict[str, Any]], config: CropConfig) -> List[int]:
    """Save face crops with full debug artifacts."""
    saved_count = 0  # ← Problem: Counter for filenames

    for i, face_meta in enumerate(metadata):  # i = metadata index (0,1,2,...)
        # ❌ WRONG: Passes saved_count (not i, not face_id)
        if save_single_face_crop(face_meta, saved_count, config):
            saved_indices.append(i)
            saved_count += 1  # Only increments on success

# Line 222 in save_single_face_crop():
prefix = f'face_{index:04d}'  # Uses index parameter (which is saved_count)
```

### What Happens

| Metadata Index | Face Valid? | saved_count | Filename Saved | Problem |
|----------------|-------------|-------------|----------------|---------|
| 0 | ❌ Failed validation | 0 | (not saved) | Skip |
| 1 | ❌ Failed validation | 0 | (not saved) | Skip |
| 2 | ✅ Valid | 0 | `face_0000.jpg` | ❌ Should be `face_0002.jpg` |
| 3 | ✅ Valid | 1 | `face_0001.jpg` | ❌ Should be `face_0003.jpg` |
| 4 | ✅ Valid | 2 | `face_0002.jpg` | ❌ Should be `face_0004.jpg` |
| ... | ... | ... | ... | All shifted by -2 |

**Result**:
- `face_0000.jpg` contains the face from metadata[2]
- `face_0569.jpg` contains the face from metadata[571]
- Systematic -2 offset in filenames

### Why This Creates +2 Offset in Embeddings

1. **During crop save** (benchmark_face_clustering.py):
   - metadata[571] → saved as `face_0569.jpg`

2. **During embedding regeneration** (regenerate_embeddings_from_crops.py):
   ```python
   for i, crop_file in enumerate(sorted_crops):  # i = 569
       embedding = extract_from_crop(crop_file)  # Extracts from face_0569.jpg
       embeddings[i] = embedding  # Stores at index 569
   ```
   - `face_0569.jpg` (contains metadata[571]'s face) → `fresh[569]`

3. **During original save**:
   - metadata[569] → `stored[569]`

4. **Result**:
   - `stored[569]` = metadata[569]'s face
   - `fresh[569]` = metadata[571]'s face (from `face_0569.jpg`)
   - Distance(stored[569], fresh[569]) = 0.78 (different people!)
   - Distance(stored[569], fresh[571]) = 0.0 (same person!)
   - **+2 offset confirmed**

---

## Verification

### Test 1: Pattern Verification
```python
# Test 20 random samples
for idx in sample_indices:
    dist_n = distance(stored[idx], fresh[idx])
    dist_n2 = distance(stored[idx], fresh[idx+2])

# Result: 100% of samples show stored[N] matches fresh[N+2] better
```

### Test 2: File Existence
```bash
ls results/Google_Germany/face_crops/face_0000_aligned.jpg  # Exists
ls results/Google_Germany/face_crops/face_0001_aligned.jpg  # Exists
# No gap in files - but content is shifted
```

### Test 3: Trace Save Logic
```bash
python scripts/debug_sighting_006/trace_crop_save_logic.py
# Shows: First 2 metadata entries failed validation
# Result: All subsequent faces saved with shifted filenames
```

---

## Why Only First 2 Faces?

**Likely reasons faces 0 and 1 failed validation**:

1. **Invalid bounding box** (line 214):
   ```python
   if not is_valid_bbox(w_px, h_px):  # w < 20 or h < 20
       return False
   ```

2. **Missing landmarks** (line 238):
   ```python
   if not landmarks or len(landmarks) < 5:
       # Can't do 5-point alignment without landmarks
       return False
   ```

3. **Invalid crop coordinates** (line 231):
   ```python
   if not is_valid_crop_coords(left, top, right, bottom):
       return False  # Crop extends outside image bounds
   ```

**Why specifically the first 2?**
- Face detection often struggles with edge cases (partial faces, occluded faces, small faces)
- First faces in dataset might be from first image in album
- First image might have poor quality, edge crops, or unusual composition
- Not suspicious - just bad luck that first 2 faces happened to fail validation

---

## Impact Analysis

### Affected Components

1. ✅ **`prepare_embeddings.py`** - SAFE
   - Uses `face.face_id` directly (line 89)
   - No offset possible

2. ❌ **`benchmark_face_clustering.py`** - AFFECTED
   - Uses `saved_count` for filenames
   - Creates offset when faces are skipped

3. ✅ **`regenerate_embeddings_from_crops.py`** - SAFE (but affected by upstream bug)
   - Correctly reads crops in order
   - But inherits shifted filenames from benchmark script

4. ✅ **Pipeline steps** - SAFE
   - Use `face_index` from context
   - Preserve identity throughout

### Data Affected

**Google_Germany dataset**:
- 727 total faces in metadata
- 725 saved crops (2 skipped)
- All 725 crops have shifted filenames (-2 offset)

**Other datasets**: Check if they were processed with `benchmark_face_clustering.py`

---

## Fix Implementation

### Option 1: Use face_id (RECOMMENDED)

```python
# Line 340: Change from
if save_single_face_crop(face_meta, saved_count, config):
    saved_indices.append(i)
    saved_count += 1

# To
face_id = face_meta.get('face_id', i)  # Fallback to index if no face_id
if save_single_face_crop(face_meta, face_id, config):
    saved_indices.append(i)
```

**Pros**:
- Preserves face identity across all processing
- No gaps in filenames (skipped faces = skipped IDs)
- Aligns with other scripts (prepare_embeddings.py)

**Cons**:
- None

### Option 2: Use metadata index directly

```python
# Line 340: Change from
if save_single_face_crop(face_meta, saved_count, config):

# To
if save_single_face_crop(face_meta, i, config):
```

**Pros**:
- Simple change
- Preserves alignment metadata[i] → face_i.jpg

**Cons**:
- Creates gaps in filenames (face_0000, face_0001 won't exist)
- Might confuse tools that expect sequential filenames

### Option 3: Pre-filter metadata

```python
# Before save loop
valid_metadata = [m for m in metadata if is_valid_face(m)]

# Then save with sequential indices
for i, face_meta in enumerate(valid_metadata):
    save_single_face_crop(face_meta, i, config)
```

**Pros**:
- Sequential filenames guaranteed
- Clean separation of validation and saving

**Cons**:
- Loses information about which original faces were skipped
- Harder to trace back to source metadata

---

## Prevention Measures

### 1. Add Assertion
```python
# After save_face_crops()
for i, saved_idx in enumerate(saved_indices):
    expected_file = crops_dir / f"face_{metadata[saved_idx]['face_id']:04d}_aligned.jpg"
    assert expected_file.exists(), f"Filename mismatch for metadata[{saved_idx}]"
```

### 2. Add Unit Test
```python
def test_crop_filename_matches_metadata_face_id():
    """Verify crop filenames match metadata face_ids."""
    metadata = load_metadata()
    crops_dir = Path("face_crops")

    for entry in metadata:
        face_id = entry['face_id']
        expected_file = crops_dir / f"face_{face_id:04d}_aligned.jpg"

        if expected_file.exists():
            # Verify embedding from crop matches metadata
            crop_embedding = extract_embedding(expected_file)
            meta_embedding = entry['embedding']
            assert distance(crop_embedding, meta_embedding) < 0.1
```

### 3. Documentation
Add to LEARNINGS.md:
- Never use loop counters for filenames that reference external data
- Always use stable identifiers (face_id, image_hash, etc.)
- Add assertions to verify filename → content mappings

---

## Tools Created

### 1. Systematic Debug Framework
`scripts/debug_sighting_006/run_debug.py`
- 6 hypothesis tests
- Structured reporting
- Actionable recommendations

### 2. Trace Save Logic
`scripts/debug_sighting_006/trace_crop_save_logic.py`
- Simulates crop saving to find skipped faces
- Shows metadata → filename mapping
- Identifies offset source

### 3. Verify Consistency
`scripts/debug_sighting_006/verify_crop_metadata_consistency.py`
- Checks alignment between crops, metadata, embeddings
- Detects gaps and offsets
- Quick sanity check for data integrity

---

## Lessons Learned

1. **Loop counters are dangerous for file IDs**
   - Never use `saved_count` or similar for filenames
   - Always use stable identifiers from source data

2. **Silent failures create offsets**
   - When validation fails silently, downstream assumes success
   - Log skipped items explicitly

3. **Modular debug tools are essential**
   - Interactive notebooks good for exploration
   - Systematic scripts better for root cause analysis
   - One hypothesis per test method

4. **Offset patterns indicate indexing bugs**
   - +2 offset too consistent to be random
   - Systematic offset → look for loop counter misuse

---

## Next Steps

1. ✅ ROOT CAUSE IDENTIFIED
2. ⏳ Implement fix (Option 1 recommended)
3. ⏳ Add unit test
4. ⏳ Add assertion
5. ⏳ Regenerate affected datasets
6. ⏳ Update LEARNINGS.md
7. ⏳ Close SIGHTING-006
