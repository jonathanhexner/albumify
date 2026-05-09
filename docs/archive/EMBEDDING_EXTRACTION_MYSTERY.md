# Embedding Extraction Mystery - 2026-03-30

## Summary

Face embedding extraction produces **different results** depending on the number of faces processed, despite using identical code and identical input images.

## The Problem

- **Small batch (7 faces)**: Extracts CORRECT embeddings
- **Large batch (727 faces)**: Extracts CORRUPTED embeddings (identical to old file from 2026-03-23)
- **Critical test case**: Face 545 vs 546
  - Correct: distance = 0.943 (different people)
  - Corrupted: distance = 0.082 (incorrectly similar)

## What We Verified

### ✓ Code is NOT the issue
- Tested with custom `InsightFaceEmbedder` wrapper → corrupted
- Tested with direct InsightFace API (no wrappers) → **still corrupted**
- Same code, different batch sizes, different results

### ✓ Face crop images are identical
- MD5 hashes match between test and original directories
- Pixel data is byte-for-byte identical (`np.array_equal = True`)
- Face crops are real files, not symlinks

### ✓ Not loading from old .npy file
- Renamed old corrupted embeddings file → **still produces corrupted results**
- Extraction happens before any comparison code runs
- No code path that loads old embeddings during extraction

### ✓ Not ONNX Runtime session caching
- Tested extracting face 545 five times → consistent results
- Processed 100 other faces then re-extracted 545 → still consistent
- ONNX Runtime does NOT cache embeddings internally

## The Mystery

**Why does the same face crop produce different embeddings based on batch size?**

```
Test directory (7 faces):
  Face 545 embedding: [-0.00793, 0.01506, -0.01154, ...] ← CORRECT

Original directory (727 faces, includes same face 545):
  Face 545 embedding: [-0.00609, 0.02404, -0.00006, ...] ← CORRUPTED
```

Both use identical face_0545_aligned.jpg (MD5: 5a981500b4e2364c8632164cb104af04)

## Evidence

1. **Minimal reproduction script** (`extract_embeddings_direct_insightface.py`)
   - Pure InsightFace code, no custom wrappers
   - Still produces corrupted embeddings for 727 faces

2. **ONNX caching test** (`test_onnx_caching.py`)
   - Proved ONNX Runtime is NOT caching results
   - Embeddings remain consistent across multiple extractions

3. **Debug comparison** (`debug_embedding_extraction.py`)
   - Both methods produce identical correct results in same script
   - Problem only occurs when processing full 727-face batch

## Hypotheses (Unverified)

1. **Sequence-dependent corruption**: Processing faces in a specific order triggers bug in InsightFace/ONNX
2. **File system issue**: Windows file system returns wrong data for large batch reads
3. **Memory corruption**: Buffer overflow or memory issue at ~700+ faces
4. **Face crop files themselves**: Despite identical MD5/pixels, some metadata or file structure is different

## Next Steps

1. Extract face 545 ALONE from original directory (no batch)
2. Extract faces in REVERSE order (729→0 instead of 0→729)
3. Process faces in small batches of 10, check if corruption appears gradually
4. Check Windows Event Viewer for file system or memory errors
5. Try extraction on Linux machine to rule out Windows-specific issue

## Files Created During Investigation

- `scripts/verify_embeddings_isolated.py` - Proved small batch works correctly
- `scripts/extract_embeddings_clean.py` - Failed to extract cleanly despite correct code
- `scripts/extract_embeddings_direct_insightface.py` - Bypassed all wrappers, still corrupted
- `scripts/test_onnx_caching.py` - Ruled out ONNX caching
- `scripts/debug_embedding_extraction.py` - Side-by-side method comparison
- `docs/EMBEDDING_CORRUPTION_ROOT_CAUSE_ANALYSIS.md` - Full architecture analysis

## Related Issues

- LEARNINGS.md entry: "Embedding corruption - regeneration script loaded cached data instead of computing fresh"
- SIGHTING-XXX: TBD (should file this as a sighting)

## Temporary Workaround

**None available.** The extraction process is fundamentally producing wrong results for the full dataset.

## Status

**OPEN - CRITICAL** - Blocking all face clustering work on Germany dataset.

---

Last updated: 2026-03-30 22:15
Investigated by: Claude Sonnet 4.5
