# Learnings Log

This file tracks lessons learned from bugs and issues to prevent repeating past mistakes.

---

<!-- Add new entries at the top, newest first -->
<!-- Format:
### YYYY-MM-DD: Brief title
**Root cause**: What caused the issue
**Prevention**: How to avoid in future
-->

### 2026-04-07: Code added between stages must be inside error handling — or it becomes invisible on crash
**Root cause**: SIGHTING-015 fix added a remap block between exemplars and export stages, outside any try/except. It used `np.full()` but `numpy` was never imported. NameError killed the process; `pipeline_run.json` stayed `status: "running"`, `error: null`. No stage recorded the failure. Telemetry showed exemplars "done" and export never started — but no error.
**Prevention**: (1) Never put code between stage try/except blocks — either fold it into an adjacent stage or make it its own stage. (2) Use a single `_execute_stage()` runner so there's no way to forget wrapping. (3) Add a `_finalize()` safety net that marks any still-"running" record as "failed". (4) Declare all stages upfront as "pending" so you can see which never started.

### 2026-04-07: History tab must validate run completeness before allowing load
**Root cause**: Pipeline run Germany_6 crashed after exemplars stage but before export. `pipeline_run.json` existed with `status: "running"`, so History tab listed it. User clicked "Load" and got cryptic `FileNotFoundError: faces.csv not found`. No indication the run was incomplete.
**Prevention**: Always check both run status AND required output files before enabling load actions. Show explicit warning with missing file list and disable the button. Never trust `pipeline_run.json` presence alone as proof of a successful run.

### 2026-04-07: Two code paths producing the same data must use the same index convention
**Root cause**: `pipeline.py` returned `ClusterResult` with graph-local node indices (0..n_core-1). `loader.py` rebuilt it from CSV with face-list indices. Both paths feed the same analysis views, but one was wrong. All live-run cluster displays showed random faces; history loads were correct. Bug was invisible for months because most testing used history loads.
**Prevention**: When two code paths produce the same data structure, add an assertion that validates the invariant (e.g., `assert all(idx < len(faces) for nodes in clusters.values() for idx in nodes)`). The loader should be the reference implementation — test that `pipeline.run()` output matches `load_pipeline_result(output_dir)` in a round-trip test.

### 2026-04-05: Check what a model already produces before building a replacement
**Root cause**: InsightFace buffalo_l has always run `1k3d68` and returned `face.pose`. Instead of reading that field, embedding.py implemented a broken 5-point heuristic. The heuristic was so wrong that the quality gate was disabled entirely. The fix was one line.
**Prevention**: When integrating a model, print all attributes it returns before writing any downstream logic. Never assume a model only produces what the docs highlight.

### 2026-04-04: Persist model outputs at export time — never re-run a model to reload results
**Root cause**: `export.py` saved cluster assignments and crops but not embeddings. `_load_result_from_dir` in the app then had to re-run InsightFaceEmbedder (minutes of work) to reconstruct embeddings for analysis views. This was not an intentional design — it was an omission that was patched with a workaround instead of fixed at the root.
**Prevention**: Any output that a downstream consumer (app, analysis, training) will need must be written to disk by the export stage. If a consumer is re-computing what a producer already computed, that is a sign of a missing export. Rule added: export stage owns all persistent data; consumers load, never recompute.

### 2026-04-04: Session state must never be written from a background thread
**Root cause**: `_run_pipeline` wrote to `st.session_state` from inside a daemon thread. Streamlit's session state is tied to the render context; writes from background threads cause `ScriptRunContext` warnings and may silently fail under concurrent renders.
**Prevention**: Background thread stores result in `worker.result`. Render thread reads `worker.is_done` and applies results to session_state. This is now codified in the CLAUDE.md async pattern.

### 2026-04-04: Reader must be tested against writer — never assume a file format
**Root cause**: `_crop_for_face()` assumed `crop_manifest.json` had format `{id: {crop_path: ...}}`. Actual format written by `crops.py` is `{id: "path_str"}`. No test exercised the read path. The writer and reader were never co-tested: tests verified the writer (crops.py), and tests verified the analysis layer (analysis_views.py), but nothing tested that the app UI could actually load a crop image from the manifest.
**Prevention**: (1) For every file format shared between a writer module and a reader module, add a contract test that writes via the writer and reads via the reader in the same test. (2) Any function that reads a persistent file format is not tested until the test opens the actual file and validates the data type of each field — not just that the file exists. (3) CLAUDE.md rule: "Before writing any code that reads a file written by another module, read that module's writer first."

### 2026-04-03: Test data shape must match production data shape — solo portraits hide multi-face bugs
**Root cause**: `face_id_counter` is a local variable inside `detect_and_embed()`, resetting to 0 per call. Pipeline calls it once per image. Bug was invisible in 9 solo-portrait E2E tests because (a) 1 face/image means face_id=0 always, (b) clustering uses list index not face_id, (c) purity/completeness tests use `image_path` not face_id. 1230 real photos exposed it immediately — only 12 unique IDs across 1103 faces.
**Prevention**: E2E test data must include at least one group photo (multiple faces per image). Always add a `face_id uniqueness` assertion. After any pipeline fix, validate on real album data before declaring done — not just synthetic single-face data.

### 2026-04-01: Optional dependencies that fail silently can reject 100% of data
**Root cause**: `QualityGater` used SixDRepNet for pose estimation. When unavailable, all faces got `pose=None`. The pose check only ran when `pose is not None`, so it was silently skipped — but the bug was that landmark-based heuristic poses (stored earlier in FaceRecord) had inflated pitch/yaw values that failed strict thresholds, causing all 1094 faces to be rejected. No fatal error, just `core=0`.
**Prevention**: (1) When an optional dependency is unavailable, log a clear one-time WARNING stating which filter is being skipped. (2) Add a `require_pose` config flag — default False so the pipeline degrades gracefully. (3) After quality gate, assert `len(core_indices) > 0` immediately and raise with a helpful message listing which thresholds were active. Never let the pipeline silently produce zero core faces.

### 2026-04-01: A "library" without a single entrypoint is not a library — it's a collection of scripts
**Root cause**: face_cluster/ had 10+ modules with good algorithms but no `FaceClusteringPipeline` class. Every consumer (scripts, notebooks, apps) had to re-implement the orchestration sequence, each time slightly differently. Two required stages (`crops.py`, `export.py`) were referenced in the spec but never built. Result: working parts that don't compose.
**Prevention**: For any ML sub-package, define the public API first (`pipeline.py` with a single `run()` method) before implementing stages. The pipeline class is the integration test for the entire sub-package. If it can't be written in <150 lines, the stage APIs aren't clean enough.

### 2026-03-30: Embedding corruption - regeneration script loaded cached data instead of computing fresh
**Root cause**: User reported clustering showing wrong similarities (face 545 similar to 546 instead of 569/573). Investigation revealed stored embeddings in `.npy` file were corrupted (likely face ID offset during original extraction). Critical bug: `regenerate_embeddings_from_crops.py` script had hidden code path that loaded pre-existing embeddings instead of computing fresh from image pixels - ALL 3 regeneration attempts produced IDENTICAL corrupted output (100% match). Only an isolated test in clean directory (no pre-existing .npy files) produced correct embeddings, proving face crops were fine and extraction works correctly.
**Prevention**: (1) **Never trust "regenerate" scripts without verification** - check that output differs from input by comparing embeddings numerically. (2) **Add mandatory validation after extraction**: randomly sample 10 faces, re-compute embeddings fresh, assert cosine similarity > 0.95 between stored and fresh. (3) **Fix regeneration script**: remove ANY code paths that can load cached/pre-existing embeddings - force fresh computation from image pixels only. (4) **Content-based verification**: Store crop image hash alongside embedding, validate they match when loading. (5) **Visual inspection in release process**: Generate HTML report showing face crops + distance matrices for manual spot-checks. (6) **Run validation tests on production data**, not just synthetic test data. See detailed analysis: `docs/EMBEDDING_CORRUPTION_ROOT_CAUSE_ANALYSIS.md`

### 2026-03-23: Never use loop counters for file identifiers
**Root cause**: SIGHTING-006 - `save_face_crops()` used `saved_count` (incremental counter) for crop filenames instead of metadata index. When first 2 faces failed validation, all subsequent faces saved with -2 offset: metadata[2] → face_0000.jpg (should be face_0002.jpg). Created permanent mismatch between array indices and filenames.
**Prevention**: (1) Always use stable identifiers from source data (face_id, metadata index), never incremental counters that skip failures. (2) Add validation immediately after saving to verify filename → data correspondence. (3) Add unit tests that simulate partial failures. (4) Document in code: "Use index from metadata, not saved_count - preserves alignment even when some saves fail."

### 2026-03-06: Transitive closure in kNN graphs creates mixed clusters
**Root cause**: Connected components on kNN graph groups A-B-C even if A≠C. Face B acts as "bridge" between two different people if it's similar enough to both (within threshold). Initial clustering (kNN + connected components) created mixed-identity pre-clusters before any merging happened.
**Prevention**: (1) After connected components, check cluster coherence (diameter, outlier detection), (2) Add post-clustering pruning stage to remove bridges/outliers, (3) Build diagnostics showing kNN neighbors and graph paths to understand WHY incorrect faces clustered together, (4) Consider tighter initial threshold or two-stage clustering (strict → merge).

### 2026-03-06: Don't skip diagnostics when debugging complex ML pipelines
**Root cause**: When merge analysis showed over-merged clusters, initially assumed merge stage was the problem. Built merge diagnostics, only to discover the real issue was earlier (initial clustering). Wasted time optimizing wrong stage.
**Prevention**: For multi-stage ML pipelines, always validate EACH stage's output before proceeding. (1) Check pre-merge clusters first, (2) Then check merge decisions, (3) Build diagnostics for each stage, not just final output. When output is wrong, trace backward through pipeline to find where error originates.

### 2026-02-28: Always compare outputs on identical inputs before declaring a mismatch
**Root cause**: Notebook used most recent embeddings (auto-selected), export script was run manually on older embeddings file. Comparison showed 45% mismatch, concluded "export script is broken". Actually both scripts were correct - just different inputs.
**Prevention**: When validating that script replicates notebook: (1) Run both on EXACT same input file (explicitly specify path, don't rely on "most recent"), (2) Compare row counts and metadata FIRST before comparing content, (3) Document which input file was used in output metadata. Comparison scripts should gracefully handle missing face IDs and different dataset sizes.

### 2026-02-27: Merge analysis - users need cluster-to-cluster distances, not just failure criteria
**Root cause**: User asked "why didn't clusters merge?" I provided `get_merge_decisions_df()` showing which criteria failed (Exemplar, Support, Margin, Diameter). User said "useless". They actually needed: (1) distance matrix BETWEEN CLUSTERS, (2) exemplar distances for all cluster pairs, (3) min/mean/max pairwise distances between cluster pairs. The merge_decisions_df only shows proposed candidates (exemplar_dist < 0.45), not WHY pairs weren't even proposed.
**Prevention**: For "why didn't X happen?" questions, provide the input data first (distances), then decision logic second (criteria). Create `get_cluster_distances()` returning DataFrame with: C1, C2, Exemplar_Dist (for proposal threshold), Min_Dist, Mean_Dist, Max_Dist. This shows "Cluster 4 and 23: exemplar_dist=0.52 > 0.45 → not proposed" directly. Mark this as the PRIMARY method for merge analysis in docstring.

### 2026-02-20: kNN + threshold + connected components prevents over-merging
**Root cause**: HDBSCAN density-based clustering can over-merge when embeddings form one large dense region (no clear density valleys). Parameter tuning (epsilon, min_samples) had no effect because the "wrong" merges happened at high density.
**Prevention**: Add post-clustering split phase: for each large cluster, build kNN graph, prune edges below similarity threshold, find connected components. This ensures every face pair in a cluster is connected through strong similarity paths - weak transitive chains get broken.

### 2026-02-20: Landmark labels are PERSON-relative, not image-relative
**Root cause**: After rotating face image, landmarks were being swapped `[1,0,2,4,3]` to "maintain left/right semantics". But L_eye/R_eye labels refer to the PERSON's left/right eye, not image position. The affine transform correctly maps person's left eye to reference template's left eye position regardless of where it is in the image.
**Prevention**: Never swap landmark indices after rotation. Landmark labels = anatomical identity. Only transform COORDINATES, not LABELS.

### 2026-02-19: Eye-line angle ≠ face orientation
**Root cause**: `compute_roll_angle()` uses `atan2(dy, dx)` on eye positions. This measures eye-line tilt, NOT whether face is upside-down. An upside-down face with level eyes returns roll_angle ≈ 0°.
**Prevention**: Face orientation requires checking spatial relationships: is nose BELOW eyes? is mouth BELOW nose? Implement `detect_face_orientation()` that returns 0°/90°/180°/270°.

### 2026-02-19: Single Responsibility - alignment/cropping/embedding must be separate steps
**Root cause**: `extract_face_embeddings.py` does detection format conversion + cropping + alignment + embedding extraction. Can't test any piece independently. Bug in alignment can't be isolated.
**Prevention**: One step = one responsibility. Create separate `align_faces`, `crop_faces` steps. Each step should be independently testable with unit tests.

### 2026-02-19: Always build debug panels showing all pipeline stages
**Root cause**: Face landmarks didn't match aligned crops because coordinates were from different stages (original image vs transformed output).
**Prevention**: For any ML pipeline, build visualization showing: (1) original input, (2) each intermediate transform, (3) final output. Catches coordinate system mismatches immediately.

### 2026-02-19: Document coordinate systems explicitly
**Root cause**: Confusion between pixel coords, normalized coords, and reference template coords led to landmark-face mismatch.
**Prevention**: Every function that handles coordinates should document: input coord system, output coord system, and any transforms applied. Use type hints like `landmarks_px` vs `landmarks_norm`.

### 2026-02-19: No "backward compatibility" fallbacks in pre-deployment code
**Root cause**: After refactoring face alignment into separate steps, added excessive if/else fallback logic "for backward compatibility" in extract_face_embeddings. This violated single responsibility and added untested code paths.
**Prevention**: In pre-deployment, the pipeline is deterministic. If step A produces output for step B, step B should require that output - no fallbacks. Fallbacks create untested paths and hide integration bugs.

### 2026-02-19: Session context loss leads to incomplete fixes
**Root cause**: After context compaction, specific details about UI bugs (which tabs were broken, what errors were shown) were lost. Made changes to backend code but didn't verify the actual UI issues the user reported.
**Prevention**: Before making fixes, explicitly confirm the specific symptoms with user. After fixing, ask user to test and report results. Don't assume "error handling" fixes unknown bugs.

## 2026-02-27: Store Metadata Paths, Don't Guess Them

**Problem**: Face clustering labeling app couldn't find face_crops directory when export location differed from source location (e.g., `results/training/Budapest_merged` vs `results/Budapest/face_crops`).

**Failed Approaches** (3 iterations):
1. Heuristic path guessing (check parent, check parallel dirs)
2. String matching on "training" in path parts
3. More complex heuristics with name suffix removal

**Why They Failed**: 
- Export directory names are arbitrary (`_merged`, `_test`, `_v2`, etc.)
- Heuristics are fragile and impossible to cover all cases
- No way to guess the original source from modified export names

**Root Cause**: Missing metadata - export didn't store where it came from

**Correct Solution**: Store source paths in metadata
- Added `embeddings_source` and `embeddings_dir` to export_summary.json
- Streamlit app reads from metadata FIRST
- Fallback to heuristics only for legacy exports
- Clear error message with re-export instructions if metadata missing

**Key Learning**: 
- **Always store paths as metadata** rather than reconstructing via string manipulation
- **Fail fast with helpful errors** - tell user how to fix (re-export) vs silent failures
- **Test end-to-end** - trace through actual usage patterns, not just happy paths
- **Heuristics are tech debt** - they work until they don't, then debugging is painful

**Prevention**: When exporting/transforming data that references external files, always save:
1. Source path (where data came from)
2. Timestamp (for cache invalidation)
3. Version (for format compatibility)

**Related**: Similar pattern needed for cache keys, model checkpoints, dataset configs
