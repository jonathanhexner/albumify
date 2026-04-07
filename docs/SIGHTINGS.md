# Sightings

This file tracks issues that need investigation and resolution.

---

<!-- Format:
### SIGHTING-XXX: Brief title
**Status**: OPEN / IN PROGRESS / RESOLVED
**Severity**: Critical / High / Medium / Low
**Reported**: YYYY-MM-DD
**Persona**: Who should fix this (e.g., Senior SW Engineer, ML Engineer)

**Problem Description**:
What is the issue?

**Symptoms**:
- Observable behavior

**Suspicion**:
Possible root cause

**Steps to Reproduce**:
1. Step 1
2. Step 2

**Resolution**:
(filled when resolved)

**Findings**:
(what was learned - also add to LEARNINGS.md)
-->

### SIGHTING-015: Live-run clusters contain random faces — graph-local index not remapped to face-list index
**Status**: RESOLVED
**Severity**: Critical — all live-run cluster assignments are wrong
**Reported**: 2026-04-07
**Persona**: Senior SW Engineer

**Problem Description**:
When running the pipeline from the app (not loading from history), `cluster_result.clusters` maps cluster_id to graph-local node indices (0..n_core-1) instead of indices into the full `faces` list. The app and all analysis views (`ClusterView`, `FaceView`, `ClusterDebugView`) treat these as face-list indices, causing every cluster to display the wrong faces.

**Symptoms**:
- Cluster 6 (user-reported example): shows face_0292 and face_0026 with distance 0.96 — impossible given threshold 0.35
- Face Analysis tab shows neighbors from wrong clusters
- Cluster metrics (diameter, distances) are computed on wrong faces
- Bug only affects live runs; history loads are correct because `loader.py` remaps correctly

**Root Cause**:
`ConnectedComponentsClusterer.cluster()` stores graph-node indices in `clusters` dict. `export.py` converts via `core_indices[graph_node]` for CSV output (correct). `loader.py` rebuilds from CSV (correct). But `pipeline.py` returns the raw `ClusterResult` to the app without converting — the two code paths (live vs loaded) produce different index semantics.

**Fix**:
In `pipeline.py`, after exemplar selection (which needs graph-local indices for the distance matrix) and before export, remap:
- `cluster_result.clusters`: `core_indices[node]` for each node
- `cluster_result.exemplars`: same remapping
- `cluster_result.labels`: rebuild as full-length array indexed by face-list position

Export receives a separate copy with raw indices since it does its own `core_indices` mapping.

**Reproduction**:
1. Run pipeline from app (not history load)
2. Open Cluster Analysis for any cluster
3. Check if face crops match — they won't (random faces from other identities)

**Verification**:
After fix, `cluster_result.clusters[cid]` contains face-list indices matching what `loader.py` produces. Both live-run and history-load paths now use the same index convention.

---

### SIGHTING-014: ModuleNotFoundError face_cluster in Streamlit despite venv active [CLOSED]
**Status**: CLOSED
**Severity**: High — blocks all app usage
**Reported**: 2026-04-04
**Persona**: Senior SW Engineer

**Problem Description**:
`streamlit run app/face_clustering.py` raises `ModuleNotFoundError: No module named 'face_cluster'` even with the venv activated. `python -c "import face_cluster"` succeeds from the project root.

**Root Cause**:
The editable install finder (`__editable___sim_bench_0_1_0_finder.py`) was generated when `face_cluster/` did not yet exist (or `pip install -e .` was not re-run after `face_cluster/` was added). Its `MAPPING` only contains `{'sim_bench': '...'}`. `face_cluster` is absent.

`python` from the project root works because Python adds the current working directory to `sys.path`. Streamlit adds the script directory (`app/`) to `sys.path`, not the project root — so `face_cluster` is not found.

**Fix**:
1. Add `face_cluster` explicitly to `setup.cfg` `packages` so it survives future `find:` misses
2. Re-run `pip install -e .` to regenerate the editable finder with updated MAPPING
3. Add post-install check to CI / dev setup docs

**Resolution**:
`setup.cfg` updated to list packages explicitly; `pip install -e .` re-run; CLAUDE.md updated with rule: re-run `pip install -e .` whenever a new top-level package directory is added.

---

### SIGHTING-013: App Crashes on Cluster Analysis — crop_manifest Format Assumption Wrong [CLOSED]
**Status**: CLOSED
**Severity**: High
**Reported**: 2026-04-04
**Persona**: Senior SW Engineer

**Problem Description**:
`_crop_for_face()` in `app/face_clustering.py` called `entry.get("crop_path", "")` on the manifest entry, assuming the format was `{face_id: {"crop_path": "..."}}`. The actual format written by `face_cluster/crops.py` is `{face_id: "relative/path/string"}`. This caused `AttributeError: 'str' object has no attribute 'get'` on every render of the Cluster Analysis and Face Analysis tabs.

**Symptoms**:
- App crashed with `AttributeError` immediately on switching to Cluster Analysis after a run
- Traceback pointed to `_crop_for_face` → `entry.get("crop_path", "")`

**Root Cause**:
`_crop_for_face` was written by assuming the manifest schema without reading `face_cluster/crops.py`. The two modules had an implicit, undocumented contract. No test exercised the actual read path from manifest → PIL Image.

**Resolution**:
- Fixed `_crop_for_face`: `return _load_crop(str(output_dir / entry))`
- Fixed same assumption in `_load_result_from_dir` (History tab)
- Added `test_crop_manifest_format_is_flat_string` regression test — asserts entry is `str`, resolves to existing file
- 19/19 tests passing

---

### SIGHTING-012: face_id Not Globally Unique — Crops Overwrite, Browse Shows Wrong Images [CLOSED]
**Status**: CLOSED
**Resolved**: 2026-04-03
**Severity**: Critical
**Reported**: 2026-04-03
**Persona**: Senior SW Engineer

**Problem Description**:
`face_id` resets to 0 on every call to `InsightFaceEmbedder.detect_and_embed()`. The pipeline calls this method once per image. Face IDs are therefore only unique within a single image — not across the run. With 1230 images the maximum face_id equals the maximum number of faces detected in any single image (11 in Google_Germany). Result: 1103 faces share only 12 unique IDs, crops overwrite each other, and the cluster browser shows the same 12 images across all 125 clusters.

**Symptoms**:
- `crop_manifest.json` has 12 entries for a 1230-image album
- `faces.csv` has 1103 rows but `face_id` only ranges 0–11
- Cluster browser shows repetitive images (same 12 crops across all clusters)
- `face_id` value counts are heavily skewed: face_id=0 appears 543 times

**Root Cause**:
`face_id_counter = 0` is declared as a local variable inside `detect_and_embed()` in `face_cluster/embedding.py:70`. It resets to 0 on every invocation. The pipeline at `face_cluster/pipeline.py:188` calls `embedder.detect_and_embed([str(img_path)])` once per image in a loop.

**Why E2E Test Didn't Catch It**:
- Test data is 9 solo-portrait images (1 face per image) — all face_ids=0
- Clustering uses list index, not face_id, so clusters are correct
- Purity/completeness tests use `image_path` for identity lookup, not face_id
- No test asserts face_id uniqueness

**Fix**:
In `face_cluster/pipeline.py`, after the embed loop, reassign face_ids sequentially:
```python
for new_id, face in enumerate(faces):
    face.face_id = new_id
```
Add a test asserting `len({f.face_id for f in result.faces}) == len(result.faces)`.

**Resolution**: Fixed in `face_cluster/pipeline.py` — after embed loop, reassign face_ids sequentially with `for new_id, face in enumerate(faces): face.face_id = new_id`. Added `test_face_ids_are_globally_unique` regression test in `tests/face_clustering/test_pipeline_e2e.py`. 24/24 tests passing.

**Steps to Reproduce**:
1. Run `FaceClusteringPipeline` on any album with more than 1 image
2. Check `faces.csv` — face_id column has repeated values

---

### SIGHTING-011: Playwright E2E Tests Connect to Wrong Port [OPEN]
**Status**: 🔴 OPEN
**Severity**: Low
**Reported**: 2026-04-02
**Persona**: Senior SW Engineer

**Problem Description**: Playwright tests hardcode `http://localhost:8501` but Streamlit picks the next free port on each launch. When another Streamlit session is already running (or port 8501 is in use), the app starts on 8502 and tests fail with `ERR_CONNECTION_REFUSED`.

**Symptoms**: `playwright._impl._errors.Error: Page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:8501/`

**Suspicion**: Hardcoded `APP_URL = "http://localhost:8501"` in `tests/face_clustering/test_streamlit_e2e.py`.

**Steps to Reproduce**: Start any Streamlit app on 8501, then run `pytest tests/face_clustering/test_streamlit_e2e.py -m e2e`.

**Resolution**: Fixed — `APP_PORT = 8502` updated. Long-term: make configurable via `--base-url` or env var `STREAMLIT_TEST_PORT`.

---

### SIGHTING-010: Empty clusters.csv — Core Index Mapping Bug in export.py [OPEN]
**Status**: 🔴 OPEN
**Severity**: Critical
**Reported**: 2026-04-02
**Persona**: Senior SW Engineer

**Problem Description**: Running the pipeline on `D:\Google_Germany` produces an empty `clusters.csv`. Even when clustering succeeds, all faces in `faces.csv` show `cluster_id = -1`.

**Root Cause Identified**: `cluster_result.clusters` stores **graph-node indices** (0 to n_core-1), but `export.py` maps them directly against the full `faces` list index. When not all faces pass quality gating, `core_indices = [3, 7, 11, ...]` — graph node 0 corresponds to `faces[3]`, not `faces[0]`. The export uses `face_to_cluster.get(i, -1)` where `i` is the full-list index, so all real cluster assignments are missed.

**Why E2E test passed**: The test used `blur_min=10.0`, causing all/nearly-all faces to be core, making `core_indices ≈ [0,1,2,...]`. Graph node index i == face list index i by coincidence. The bug was invisible.

**Steps to Reproduce**: Run pipeline on any dataset where quality gating rejects some faces (default `blur_min=50.0`). Check clusters.csv — empty despite valid clustering output in logs.

**Fix**: Pass `core_indices` to `export_results()`. Map graph node index `g` → `core_indices[g]` → full face list index.

---

### SIGHTING-009: CLI Script Crashes on Windows with Unicode Progress Bar [OPEN]
**Status**: 🔴 OPEN
**Severity**: High
**Reported**: 2026-04-02
**Persona**: Senior SW Engineer

**Problem Description**: `scripts/run_face_clustering.py` crashes immediately at the discover stage with `'charmap' codec can't encode characters in position 15-34`.

**Root Cause**: `on_progress()` in the script uses `█` (U+2588) and `░` (U+2591) for a progress bar. Windows console uses `cp1252` (charmap) encoding by default, which cannot encode these Unicode block characters. The exception propagates into the `try/except` block inside `pipeline.run()`, which wraps it as `PipelineStageError("discover", ...)`.

**Steps to Reproduce**: `python scripts/run_face_clustering.py --images test_data/face_clustering/source_images --output results/test` on Windows.

**Fix**: Replace Unicode block chars with ASCII `#` and `-` in the progress bar.

---

### SIGHTING-008: Face Clustering Has No Cohesive Pipeline — Collection of Disconnected Scripts [IN PROGRESS]
**Status**: 🟡 IN PROGRESS — core implementation complete (17/17 tests pass), Playwright E2E on D:\Google_Germany pending
**Severity**: Critical
**Reported**: 2026-04-01
**Persona**: Senior SW Engineer + ML Engineer

**Problem Description**:
The face clustering subsystem is a collection of scripts that cannot be driven as a unified pipeline. The user cannot point at a directory of images and get clusters out end-to-end without manually chaining disconnected scripts. The Streamlit app has no A-to-Z run capability. There is no tested path from raw images → clusters that is guaranteed to work. The immediate trigger was `export_clustering_data.py` failing with `core=0, holdout=1094` (see Immediate Failure below).

**Symptoms**:
1. `scripts/export_clustering_data.py` requires PRE-EXTRACTED embeddings as input; there is no single entrypoint that accepts an image directory
2. Quality gating silently rejects ALL faces when SixDRepNet unavailable and landmark-based pose thresholds are strict — no fallback, no warning to user
3. `face_cluster/crops.py` and `face_cluster/export.py` referenced in RECOVERY_PLAN.md as required stages do NOT EXIST
4. No `face_cluster/pipeline.py` — no `FaceClusteringPipeline` class with a clean `run(image_dir, output_dir)` API
5. Streamlit apps (`face_clustering_labeling.py`, `face_clustering_debug/`) are read-only viewers with no ability to trigger the pipeline or see live progress
6. Zero A-to-Z tests covering the full detect → embed → quality-gate → cluster → export cycle on real test images
7. Immediate failure log:
   ```
   Computed pose for 0/1094 faces   ← SixDRepNet not available
   Quality gating results: core=0, holdout=1094
   Export failed: No faces passed quality gating!
   ```

**Immediate Failure Root Cause**:
`export_clustering_data.py --embeddings results/Google_Germany_FULL/embeddings.npy`:
- SixDRepNet not available → pose estimation returns 0 results for all 1094 faces
- Landmark-based heuristic poses already stored in FaceRecord; strict thresholds (`yaw_max=30°`, `pitch_max=25°`) filter out all faces in this large, varied dataset
- No fallback when external pose estimator unavailable: quality gate should gracefully degrade to blur+area only
- `blur_min=50.0` may also be too strict for faces loaded via the pre-computed embeddings path

**Structural Root Cause**:
`RECOVERY_PLAN.md` defines the correct architecture but it has NOT been implemented. Specifically missing:
- `face_cluster/crops.py` (Stage 3 in spec)
- `face_cluster/export.py` (Stage 5 in spec)
- `face_cluster/pipeline.py` (orchestrator with clean API)
- Tests in `tests/face_clustering/` for each stage

**Steps to Reproduce**:
```bash
python scripts/export_clustering_data.py \
  --embeddings results/Google_Germany_FULL/embeddings.npy \
  --output results/Google_Germany_FULL/clustering_export
# → Export failed: No faces passed quality gating!
```

**Resolution**:
See plan below — full re-implementation of face_cluster as cohesive sub-package.

**Findings**:
(to be filled after resolution)

---

### SIGHTING-007: Ground Truth Face Crop Mapping Mismatch [OPEN]
**Status**: 🔴 OPEN
**Severity**: Critical
**Reported**: 2026-03-31
**Persona**: ML/Testing Team

**Problem Description**:
The ground truth face crops in `test_data/face_crops_ground_truth/` do not correspond to the faces detected at the specified indices in `test_data/ground_truth_mapping.csv`. This breaks the full pipeline test and makes ground truth data unreliable for validation.

**Symptoms**:
1. **Full pipeline test fails with very low embedding similarity**:
   - Expected: >0.95 similarity between pipeline and ground truth embeddings
   - Actual: Many faces show <0.3 similarity, some negative
   - Examples: Face 558: -0.002, Face 557: 0.015, Face 546: 0.295

2. **Distance matrix correlation near zero**:
   - Expected: >0.90 correlation between pipeline and ground truth distance matrices
   - Actual: 0.0064 correlation (essentially random - no relationship)

3. **Visual verification confirms wrong faces**:
   ```
   Face 545: similarity 0.715 (moderate - possible match)
   Face 546: similarity 0.295 (low - wrong face)
   Face 550: similarity 0.612 (moderate)
   Face 551: similarity 0.623 (moderate)
   Face 557: similarity 0.015 (very low - completely different face)
   Face 558: similarity -0.002 (negative - completely different person)
   Face 562: similarity -0.012 (negative - completely different person)
   Face 569: similarity -0.027 (negative - completely different person)
   Face 573: similarity -0.003 (negative - completely different person)
   Face 580: similarity 0.670 (moderate - borderline)
   Face 584: similarity 0.393 (low - possibly wrong)
   Face 587: similarity 0.005 (very low - wrong face)
   Face 589: similarity 0.745 (moderate - borderline)
   Face 634: similarity 0.102 (very low - wrong face)
   Face 637: similarity 0.075 (very low - wrong face)
   ```

4. **Test results**:
   - ✓ `test_pipeline_extracts_all_faces`: PASSED (15/15 faces extracted with HEIC support)
   - ✗ `test_pipeline_embeddings_match_ground_truth`: FAILED (15/15 faces mismatched)
   - ✗ `test_pipeline_preserves_identity_structure`: FAILED (45 distance violations)
   - ✗ `test_distance_matrix_correlation`: FAILED (correlation 0.0064 instead of >0.90)

**Suspicion**:
1. **Different face detection run**: Ground truth crops may have been extracted from a different face detection run with different results (missed/extra faces causing index shift)

2. **Face ordering mismatch**: Mapping CSV may assume one ordering (e.g., confidence), but crops were created with different ordering or detection parameters

3. **Manual intervention**: Crops may have been manually created/edited/renamed without updating mapping CSV

4. **Source image version mismatch**: Source images in `test_data/source_images_ground_truth/` may differ from images used to create crops (different EXIF, different resolution, etc.)

5. **Database extraction bug**: The script (`scripts/query_face_lineage.py`) that created `ground_truth_mapping.csv` from database may have had a bug in index assignment

6. **SIGHTING-006 related**: Similar filename offset bug as SIGHTING-006 may have occurred during ground truth creation

**Steps to Reproduce**:
1. Run diagnostic script:
   ```bash
   python scripts/verify_face_index_mapping.py
   ```
   This creates side-by-side visual comparisons in `test_data/face_index_verification/`

2. Check visual comparisons - compare detected crops (left) vs ground truth crops (right)
   - Many show completely different faces or different orientations

3. Run full pipeline test:
   ```bash
   python -m pytest tests/test_face_pipeline_full.py -v -s
   ```
   - First test passes (15/15 faces extracted successfully with HEIC support)
   - Remaining 3 tests fail due to embedding mismatch

**Evidence Files**:
- `HEIC_AND_FACE_ORDERING_SUMMARY.md` - Full analysis and implementation summary
- `scripts/verify_face_index_mapping.py` - Diagnostic tool (creates visual comparisons)
- `test_data/face_index_verification/` - Visual comparisons (15 comparison images)
- `tests/test_face_pipeline_full.py` - Failing test suite
- `test_data/ground_truth_mapping.csv` - Suspect mapping file

**Impact**:

**BLOCKS**:
- Full pipeline end-to-end validation (cannot verify detect → align → embed pipeline works correctly)
- Ground truth expansion (cannot add more faces with confidence in mapping accuracy)
- Face ordering validation (cannot verify reading_order vs confidence_order produces correct results)
- Regression testing for face pipeline changes

**STILL WORKING**:
- ✅ Isolated crop test (`test_face_embeddings_ground_truth.py`) - All tests pass
- ✅ HEIC loading - All 15/15 faces extract successfully from HEIC files
- ✅ Face ordering utilities - Created and ready to use (`sim_bench/utils/face_ordering.py`)

**Resolution Options**:

**Option 1: Regenerate Ground Truth** (RECOMMENDED - cleanest solution)
1. Choose ordering convention: `reading_order` (user-requested, spatially intuitive) or `detection_order` (backward compatible)
2. Run face detection on all 15 source images with chosen ordering
3. Save aligned crops with deterministic filenames based on `(image_name, face_index)`
4. Manually label person identities (4 people, 15 faces)
5. Generate new `ground_truth_mapping.csv` with verified indices
6. Run diagnostic script to verify correspondence before committing
7. Document which ordering was used and creation timestamp

**Option 2: Reverse-Engineer Mapping** (preserves existing labels but risky)
1. For each of 15 ground truth crops, extract embedding
2. For each source image, detect all faces, extract all face embeddings
3. Find best match using cosine similarity (require threshold >0.95 to avoid ambiguity)
4. Update `ground_truth_mapping.csv` with discovered indices
5. Verify no ambiguous matches (multiple faces with high similarity)
6. Run full diagnostic to confirm correspondence

**Option 3: Investigate Original Creation Method** (forensic approach)
1. Check database `people.face_instances` for original creation metadata
2. Review `scripts/query_face_lineage.py` and `scripts/verify_ground_truth_traceability.py` for export logic
3. Check git history for when crops were created and what script was used
4. Determine if there's a systematic bug in database export or crop creation
5. Fix bug and regenerate if found

**Option 4: Manual Verification** (labor-intensive but thorough)
1. For each of 15 faces, manually open source image
2. Visually identify which detected face corresponds to ground truth crop
3. Update mapping CSV with correct indices
4. Re-run tests to verify

**Recommended Approach**: **Option 1** (Regenerate)
- Clean slate, no forensics needed
- Use deterministic `reading_order` as requested by user
- Documented creation process prevents future issues
- Can be completed in <1 hour

**Next Steps** (awaiting user decision):
1. User chooses resolution approach
2. If Option 1: Decide on ordering convention (reading_order vs detection_order)
3. Execute chosen approach
4. Verify with diagnostic script
5. Update tests and documentation
6. Add creation metadata to prevent recurrence

**Related Issues**:
- SIGHTING-006: Face Crop Filenames Don't Match Face IDs (+2 Offset) - similar filename mapping bug
- EMBEDDING_EXTRACTION_MYSTERY.md - unsolved 727-face corruption issue
- EMBEDDING_CORRUPTION_ROOT_CAUSE_ANALYSIS.md - caching and lineage issues

**Findings**:
(to be filled after resolution)

---

### SIGHTING-006: Face Crop Filenames Don't Match Face IDs (+2 Offset)
**Status**: ✅ RESOLVED
**Severity**: Critical
**Reported**: 2026-03-20
**Updated**: 2026-03-23
**Resolved**: 2026-03-23
**Persona**: Senior SW Engineer

**Problem Description**:
Regenerated embeddings from face crops have systematic +2 offset - `stored[N]` contains embedding for `face_{N+2}_aligned.jpg` instead of `face_{N}_aligned.jpg`.

**Symptoms**:
- Fresh vs stored embeddings show massive distances (0.78+)
- Comparing fresh embeddings: `stored[569]` matches `fresh[571]` (distance 0.0)
- Pattern confirmed for all test faces: `stored[N]` = `fresh[N+2]`
- FRESH metadata claims face_ids = [0,1,2,...] but crops don't match

**ROOT CAUSE IDENTIFIED** (2026-03-23):

**Location**: `scripts/benchmark_face_clustering.py:328-347`

**The Bug**:
```python
def save_face_crops(metadata: List[Dict[str, Any]], config: CropConfig) -> List[int]:
    saved_count = 0  # Counter for filenames
    for i, face_meta in enumerate(metadata):  # i = metadata index (0,1,2,...)
        if save_single_face_crop(face_meta, saved_count, config):  # ← PASSES saved_count
            saved_indices.append(i)
            saved_count += 1  # Only increments on SUCCESS
```

**Line 222**: `prefix = f'face_{index:04d}'` (uses `saved_count`, not metadata index)

**What Happens**:
1. Loop iterates metadata with index `i` (0, 1, 2, 3, ...)
2. But passes `saved_count` to `save_single_face_crop()` for the filename
3. `saved_count` only increments when crop saves successfully
4. If faces 0 and 1 fail validation (invalid bbox, missing landmarks) → NOT saved
5. `saved_count` stays at 0 when processing face 2
6. Face 2 (metadata index 2) gets saved as `face_0000.jpg` (should be `face_0002.jpg`)
7. All subsequent faces shifted: metadata[N] → saved as `face_{N-2}.jpg`

**Why +2 Specifically**:
First 2 faces in metadata failed validation checks:
- Face 0: Invalid bbox OR missing landmarks
- Face 1: Invalid bbox OR missing landmarks
- Face 2+: Valid, but saved with shifted filenames

**Impact**:
- `stored[569]` = embedding from `face_0569.jpg` = metadata[569]'s data
- But `face_0569.jpg` actually contains face from metadata[571] (because 2 faces were skipped)
- `fresh[571]` = embedding from `face_0571.jpg` = same physical face
- Distance(stored[569], fresh[571]) = 0.0 ← Perfect match!

**Verification Tools Created**:
1. `scripts/debug_sighting_006/run_debug.py` - Systematic hypothesis testing
2. `scripts/debug_sighting_006/trace_crop_save_logic.py` - Simulates save logic to find skipped faces
3. `scripts/debug_sighting_006/verify_crop_metadata_consistency.py` - Checks alignment

**Debugging Steps Completed**:
1. Created `notebooks/debug_embeddings_comparison.ipynb`
2. Compared 3 extraction methods → proved extraction code correct
3. Searched all 727 faces → found systematic +2 offset
4. Created modular debug framework with 6 hypothesis tests
5. Deep code analysis → pinpointed exact bug location (line 340)
6. Created trace script to identify which faces were skipped

**Steps to Reproduce**:
1. Run `benchmark_face_clustering.py` on dataset where first 2 faces fail validation
2. Observe: `face_0000.jpg` exists but corresponds to metadata[2], not metadata[0]
3. Run trace script: `python scripts/debug_sighting_006/trace_crop_save_logic.py --metadata <path> --crops <dir>`

**Resolution**:
✅ **FIXED** - Changed line 341 to use metadata index `i` instead of `saved_count`

**Fix Implemented** (Option 3 - Use metadata index):
```python
# Line 341: Changed from
if save_single_face_crop(face_meta, saved_count, config):

# To
if save_single_face_crop(face_meta, i, config):  # Use metadata index i
```

**Result**:
- Crop filename now matches metadata index
- If faces 0,1 fail → `face_0000.jpg` and `face_0001.jpg` don't exist (gaps OK)
- Face 2 saves as `face_0002.jpg` ✅ (not `face_0000.jpg`)
- Alignment preserved: `metadata[N]` ←→ `embeddings[N]` ←→ `face_{N:04d}.jpg`

**Validation Added**:
- `validate_crop_filenames()` function catches this bug immediately
- Runs after crop saving, before data is written to files
- Three test cases verify detection and prevention
- Tests pass: `pytest tests/test_crop_validation.py`

**Prevention**:
1. Add assertion after `save_face_crops()`:
   ```python
   assert all(crop_files[i] == f"face_{metadata[saved_indices[i]]['face_id']:04d}_aligned.jpg")
   ```
2. Use face_id from metadata for ALL filenames (never use loop counters)
3. Add test: `test_crop_filename_matches_metadata_face_id()`

**Testing Method Created**:
`tests/test_face_crop_integrity.py` - Verifies crop filenames match embeddings

---

### SIGHTING-005: Pre-clusters Contain Mixed People (Transitive Closure Problem)
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-03-06
**Persona**: ML Engineer

**Problem Description**:
Pre-merge clusters (from initial kNN + connected components clustering) already contain mixed people before any merging happens. Merge stage analysis revealed the real problem is in initial clustering, not merging.

**Specific Examples**:
- Pre-cluster 3: Contains multiple different people
- Pre-cluster 6: Contains multiple different people

**Symptoms**:
- User examining merge history to understand over-merged clusters
- Discovered that even PRE-MERGE clusters have wrong photos
- Individual faces in pre-clusters are far from exemplars (dist > 0.6 when threshold is 0.4)
- Some faces closer to OTHER clusters' exemplars than their own

**Root Cause Hypothesis**:
**Transitive Closure Problem** in kNN graph:
```
Face A → Face B → Face C
  │                  │
Person 1          Person 2

- Face A and B are similar (same person)
- Face B and C are similar (different person, but B is "bridge")
- A and C are NOT similar (different people)
- But connected components groups them together!
```

**Current Parameters**:
```python
K = 5                      # kNN neighbors
distance_threshold = 0.35  # Edge creation
min_cluster_size = 2

Process:
1. Build mutual kNN graph
2. Prune edges > 0.35
3. Connected components → clusters
```

**Why It Fails**:
1. **Threshold too loose**: 0.35 allows some cross-person connections
2. **Transitive connections**: A→B→C even if A≠C
3. **No outlier removal**: Bridge faces create wrong clusters
4. **No coherence check**: Cluster can have high diameter (0.6+)

**Investigation Needed**:
1. **Why are incorrect faces connected?**
   - What are their kNN neighbors?
   - Which edges connect them to the cluster?
   - Are they bridges between two identity groups?

2. **What metrics predict incorrect clustering?**
   - Distance to nearest exemplar?
   - Fraction of kNN neighbors outside cluster?
   - Alternative cluster assignments?

**Proposed Diagnostics**:
1. Pre-Cluster Explorer UI
2. Face-level kNN neighbor analysis
3. Graph path visualization
4. Alternative assignment analysis
5. Outlier detection metrics

**Resolution**:
Phase 1: Build diagnostics to understand problem (in progress)
Phase 2: Implement solution (TBD after analysis)

**Findings**:
(to be filled after diagnostic analysis)

---

### SIGHTING-004: Same Cluster Pair Merging Multiple Times in Merge Log
**Status**: RESOLVED
**Severity**: Medium (Design Confusion)
**Reported**: 2026-03-06
**Resolved**: 2026-03-06
**Persona**: ML Engineer

**Problem Description**:
Merge decisions log shows the same cluster pair (e.g., cluster 0 + 6) merging multiple times across different iterations. Once two clusters merge, one disappears and they become a single entity - they should never merge again. Either the merge logic has a bug (not actually merging), or the decision log is recording duplicates incorrectly.

**Symptoms**:
- User observing: "Pre-cluster 6 → Cluster 0" appearing multiple times in merge history
- Merge decisions CSV has 6,196 total decisions (1,454 merged, 4,742 rejected)
- For 727 faces going from 75→27 clusters (48 merges), why are there 1,454 "merged" decisions?
- Math doesn't add up: 48 merges should = 48 decisions, not 1,454

**Investigation Results** (via `debug_merge_duplicates.py`):
```
✅ actually_merged=True: 48 entries (correct!)
❌ actually_merged=False: 1,406 entries (valid candidates not chosen)
❌ 96 unique pairs across all iterations
❌ Example: Cluster 3 + 43 appeared 44 times (iterations 1-44)
   - Iterations 1-43: actually_merged=False (valid but not best)
   - Iteration 44: actually_merged=True (finally chosen)
```

**Root Cause**:
**Confusing terminology in decision logging.**

`action='merged'` means "this pair COULD merge" (passed validation checks), NOT "this pair DID merge".

In each iteration:
- 30-60 pairs are valid merge candidates → logged with `action='merged'`
- Only 1 pair actually executes → marked with `actually_merged=True`
- Same pair can be valid candidate in multiple iterations before finally being chosen

Example: Cluster 3 + 43
- Iterations 1-43: Valid candidate but another pair was better → `action='merged'`, `actually_merged=False`
- Iteration 44: Best candidate, actually executed → `action='merged'`, `actually_merged=True`

**Resolution**:
✅ Merge logic is CORRECT - clusters properly merge and disappear
✅ Decision logging is CORRECT - tracks all valid candidates for analysis
❌ UI was WRONG - filtered by `action='merged'` instead of `actually_merged=True`

**Fixes Applied**:
1. Updated UI to filter by `actually_merged=True` (shows only 48 actual merges)
2. Deduplicated rejected attempts (show last attempt per unique pair)
3. Added caption explaining duplicate candidates

**Findings**:
- `action='merged'` should be renamed to `action='valid_candidate'` for clarity
- Logging ALL candidates is useful for analysis (why wasn't this pair chosen?)
- UI must distinguish between "could merge" vs "did merge"
- This explains why merge analysis was unusable - showing 1,454 candidates instead of 48 merges!

---

### SIGHTING-003: pre_merge_cluster_id Column Not Being Added to faces.csv
**Status**: RESOLVED
**Severity**: High (User Error)
**Reported**: 2026-03-06
**Resolved**: 2026-03-06
**Persona**: Senior SW Engineer

**Problem Description**:
Export script runs without errors but fails to add `pre_merge_cluster_id` column to faces.csv, preventing diagnostic tabs from working in labeling app.

**Symptoms**:
- Export script completes successfully
- All diagnostic files created (merge_decisions.csv, pre_merge_clusters.csv, cluster_lineage.json)
- BUT faces.csv missing `pre_merge_cluster_id` column
- Labeling app shows: "Missing: pre_merge_cluster_id column in faces.csv"
- User has re-run export 3+ times with same result

**Suspicion**:
1. Code adds column during faces_data construction, but something prevents it from being included
2. Maybe pre_merge_result is None despite merge being enabled?
3. Maybe column is being added but saved to wrong file?
4. Maybe there's a conditional that's preventing the column from being added?

**Steps to Reproduce**:
1. Run: `python scripts/export_clustering_data.py --embeddings <path> --output <dir>`
2. Check faces.csv columns
3. Column `pre_merge_cluster_id` is missing

**Debug Process**:
Created `scripts/debug_export_issue.py` to systematically check files, columns, and logs.

**Resolution**:
**Root Cause**: User was loading OLD export directory in labeling app, not the NEW export directory.
- Export worked correctly: `results\Google_Germany\clustering_export` ✅ (column exists)
- App was loading: `results/face_clustering_training/diagnostic_test` ❌ (old data)

Debug script revealed:
```
✅ pre_merge_cluster_id column EXISTS
Sample values: [-1, -1, -1, -1, 0, -1, 1, 2, 0, -1]
Unique values: 76
```

**Solution**: User changed app to load correct directory, all tabs work.

**Findings**:
- Export code working correctly from the start
- Debug script invaluable for systematic diagnosis
- UI should show current loaded directory path more prominently
- Consider adding validation: if export_summary.json shows diagnostic files in metadata, but columns missing, show clear error

---

### SIGHTING-001: Face Alignment Not Working - Upside Down Faces Not Corrected
**Status**: IN PROGRESS
**Severity**: Critical
**Reported**: 2026-02-19
**Persona**: Senior SW Engineer
**Implementation Started**: 2026-02-19

**Problem Description**:
Face alignment is failing to properly orient faces. An upside-down face (requiring ~180° rotation) is only being rotated 6° instead of the correct angle. The entire face processing pipeline in `extract_face_embeddings.py` has severe architectural issues with mixed responsibilities.

**Symptoms**:
- Face #118 (20250822_194950.jpg) is upside down - eyes at bottom, mouth/nose at top
- System identifies landmarks correctly (knows eyes are down) but rotates only 6° instead of ~180°
- Aligned face crops show incorrectly oriented faces
- Landmarks on debug panel don't match actual facial features

**Suspicion**:
1. `roll_angle` calculation only considers eye-line angle (±90° max), not full orientation
2. 5-point alignment uses `estimateAffinePartial2D` which may not handle 180° flips
3. Mixed code paths: MediaPipe vs InsightFace detection handled in same step
4. No validation that alignment actually corrected orientation

**Architectural Issues Identified**:
1. `extract_face_embeddings.py` combines:
   - Face source detection (MediaPipe vs InsightFace)
   - Face cropping
   - Face alignment (5-point or roll-based)
   - Embedding extraction
   - Caching logic
2. No separation of concerns - can't test alignment independently
3. No unit tests for alignment correctness
4. Violates single responsibility principle
5. Multiple code paths for different face detection backends

**Steps to Reproduce**:
1. Run face clustering on album containing upside-down faces
2. Open Face Clustering Debug app
3. Go to Overview → Gallery
4. Click 🔍 on an upside-down face
5. Observe: landmarks don't match, face orientation is wrong

**Root Cause Analysis (2026-02-19)**:

1. **`compute_roll_angle()` only measures eye-line tilt, NOT face orientation**
   - Uses `atan2(dy, dx)` on eye positions → returns -180° to +180° but just for eye tilt
   - For upside-down face: eyes at bottom, but roll_angle ≈ 0° (eyes level, just upside down)
   - Missing: check if nose is BELOW eyes, mouth is BELOW nose

2. **5-point alignment can't fix upside-down faces**
   - `estimateAffinePartial2D` computes similarity transform (rotation + scale + translation)
   - For upside-down landmarks → tries to match to upright template → finds small rotation with high error
   - Would need 180° pre-rotation BEFORE 5-point alignment

3. **No validation that alignment worked**
   - No check that transformed landmarks are close to reference positions
   - Should fail loudly if alignment error is high

**Required Architecture Changes**:

```
CURRENT (broken):
insightface_detect_faces → filter_faces → score_face_frontal → extract_face_embeddings
                                              ↑                        ↑
                                         roll_angle only         alignment + crop + embed
                                         (no orientation)        (too many responsibilities)

PROPOSED (clean):
insightface_detect_faces
    → filter_faces
    → detect_face_orientation   [NEW - determines 0°/90°/180°/270° rotation needed]
    → align_faces               [NEW - applies rotation + 5-point alignment]
    → validate_alignment        [NEW - checks alignment quality, flags bad ones]
    → crop_faces                [NEW - just cropping, separate from alignment]
    → extract_face_embeddings   [SIMPLIFIED - just embedding extraction]
```

**Unit Tests Needed**:
1. `test_face_orientation_detection.py` - test with 0°, 90°, 180°, 270° rotated faces
2. `test_face_alignment.py` - test that alignment produces expected landmark positions
3. `test_alignment_validation.py` - test that bad alignments are detected

**Specific Test Case (Face #118)**:
```python
def test_upside_down_face_orientation():
    """Face with eyes at bottom, mouth at top should detect 180° rotation needed."""
    landmarks = [
        [100, 150],  # left_eye (at BOTTOM)
        [200, 150],  # right_eye (at BOTTOM)
        [150, 100],  # nose (ABOVE eyes - wrong!)
        [110, 50],   # left_mouth (at TOP - wrong!)
        [190, 50],   # right_mouth (at TOP - wrong!)
    ]
    orientation = detect_face_orientation(landmarks)
    assert orientation == 180, f"Expected 180° rotation, got {orientation}°"
```

**Resolution**:
Implementation in progress. New architecture implemented:

1. **Created `detect_face_orientation` step** (`sim_bench/pipeline/steps/detect_face_orientation.py`)
   - Analyzes 5-point landmarks to detect 0°/90°/180°/270° rotation
   - Checks vertical relationships (eyes above nose, nose above mouth)
   - Stores `orientation_angle` and `orientation_confidence` in face_info

2. **Created `align_faces` step** (`sim_bench/pipeline/steps/align_faces.py`)
   - Pre-rotates image by detected orientation angle
   - Transforms landmarks to rotated coordinates
   - Applies 5-point affine alignment to ArcFace template
   - Stores aligned crops in `context.aligned_faces`

3. **Created `validate_alignment` step** (`sim_bench/pipeline/steps/validate_alignment.py`)
   - Runs face detection on aligned crops
   - Verifies landmarks are near expected reference positions
   - Flags faces with high alignment error

4. **Created `crop_faces` step** (`sim_bench/pipeline/steps/crop_faces.py`)
   - Simple bbox cropping without alignment (for debug)

5. **Updated `extract_face_embeddings`**
   - Fixed imports (moved to top of file)
   - Now uses pre-aligned faces from `align_faces` step when available
   - Falls back to inline alignment for backward compatibility

6. **Added unit tests**
   - `tests/pipeline/test_face_orientation_detection.py` (15 tests)
   - `tests/pipeline/test_face_alignment.py` (16 tests)
   - Includes regression test for Face #118 (upside-down face)

7. **Updated pipeline.yaml**
   - Added `detect_face_orientation` and `align_faces` to default_pipeline
   - Added configuration sections for new steps

**Findings**:
- `compute_roll_angle()` measures eye-line tilt, NOT face orientation
- Face orientation requires checking spatial relationships (is nose below eyes?)
- 5-point alignment must be preceded by orientation correction for rotated faces
- Single-responsibility steps are testable and debuggable
- See `docs/LEARNINGS.md` for prevention guidelines

---

### SIGHTING-002: ML Cluster Merging Scripts Produce Different Results Than Working Notebook
**Status**: RESOLVED
**Severity**: High
**Reported**: 2026-02-28
**Resolved**: 2026-02-28
**Persona**: ML Engineer

**Problem Description**:
The ML cluster merging implementation (Phase 1 of PLAN_ML_CLUSTER_MERGING.md) has working code in `notebooks/debug_knn_graph_clustering.ipynb`, but the production scripts built to replicate this behavior produce completely different clustering results. A comparison script was also built but crashes when comparing the outputs.

**Symptoms**:
1. **Notebook works correctly**: Produces 274 faces, 25 clusters (11 after merge), 140 noise faces
2. **Export script produces wrong results**: Only 254 faces (missing 20), 24 clusters, 137 noise
3. **45.3% of faces have different cluster assignments** between notebook and script
4. **Comparison script crashes** with IndexError when trying to analyze differences (face IDs missing from one dataset)
5. Example: Notebook cluster 3 has 6 faces [12, 13, 14, 29, 35, 258], export cluster 3 has only 4 faces [12, 13, 14, 238]

**Suspicion**:
1. Export script may be using different quality gating parameters (why 20 fewer faces?)
2. Clustering pipeline may have subtle differences (order of operations, random seeds, tie-breaking)
3. ConservativeMerger may produce different results due to threshold calculation differences
4. Scripts were "reinvented" instead of extracting working code from notebook

**Steps to Reproduce**:
1. Run `notebooks/debug_knn_graph_clustering.ipynb` → exports `notebook_clustering_results.csv` (274 faces, 11 clusters after merge)
2. Run `python scripts/export_clustering_data.py --embeddings results/face_clustering_benchmark/embeddings_*.npy` → exports `faces.csv` (254 faces, different clustering)
3. Run `python scripts/compare_clustering_results.py` → crashes with IndexError

**Error Output**:
```
IndexError: single positional indexer is out-of-bounds
  File "D:\sim-bench\scripts\compare_clustering_results.py", line 68
    export_cluster = export_df[export_df['face_id'] == face_id]['cluster_id'].iloc[0]
```

**Root Cause**:
**False alarm - scripts were being compared on DIFFERENT input data.**

Investigation revealed:
1. Original comparison used `embeddings_2026-02-16_00-44-28.npy` (254 faces) for export script
2. Notebook automatically uses most recent embeddings: `embeddings_2026-02-20_10-19-23.npy` (274 faces)
3. When export script run on SAME embeddings as notebook: **0 mismatches (100% match)**

Secondary issue: Comparison script crashed with IndexError when face IDs were missing (fixed)

**Resolution**:
1. ✅ **Created `scripts/export_from_notebook_logic.py`** - Extracted exact notebook code for validation
2. ✅ **Fixed `scripts/compare_clustering_results.py`** - Handle missing face IDs gracefully
3. ✅ **Verified `scripts/export_clustering_data.py` is correct** - Produces identical results to notebook when given same input
4. ✅ **Created `scripts/compare_notebook_vs_export.py`** - Clean comparison on same embeddings

**Verification**:
```bash
# Run export on same embeddings as notebook
python scripts/export_clustering_data.py --embeddings results/face_clustering_benchmark/embeddings_2026-02-20_10-19-23.npy

# Compare results
python scripts/compare_notebook_vs_export.py
# Output: Mismatched assignments: 0 / 274 faces (0.0%)
```

**Findings**:
- Export script implementation is CORRECT - exact match with notebook
- Always compare on SAME input data when validating implementations
- Comparison scripts should handle edge cases (missing IDs, different row counts)
- See `docs/LEARNINGS.md` for prevention guidelines

**Prevention**:
When comparing notebook vs script outputs:
1. **Always use the same input data** - explicitly specify paths, don't rely on "most recent" logic
2. Document which embeddings file was used in comparison
3. Build comparison scripts that handle different-sized datasets gracefully
4. Compare metadata (file sizes, row counts) FIRST before diving into details
