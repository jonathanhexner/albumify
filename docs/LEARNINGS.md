# Learnings Log

This file tracks lessons learned from bugs and issues to prevent repeating past mistakes.

---

<!-- Add new entries at the top, newest first -->
<!-- Format:
### YYYY-MM-DD: Brief title
**Root cause**: What caused the issue
**Prevention**: How to avoid in future
-->

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
