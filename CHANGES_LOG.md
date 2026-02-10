# Change Log

**Purpose**: Track all code modifications with timestamps for debugging and history.

**Format**: Each entry includes date/time (ISO 8601), files modified, change description, and reason.

---

## 2026-02-10 03:00:00

**Files**:
- `sim_bench/api/services/people_service.py`
- `app/streamlit/components/gallery.py`

**Change**: Improved People feature error handling and debugging

1. **people_service.py `get_person_images()`**:
   - Added debug logging for face_instances count and thumbnail path
   - Added validation to skip faces with empty image_path
   - Added fallback: if no valid face_instances, use thumbnail_image_path

2. **people_service.py `create_from_clusters()`**:
   - Added validation to skip faces with invalid paths (empty, '.', 'None')
   - Added warning logs when skipping invalid faces

3. **gallery.py `render_image_card()`**:
   - Improved error message: now shows filename when path is missing

**Reason**: User reported "No image path" for all images when viewing a person's photos. The root cause is likely stale Person records created before path handling fixes were applied. Added validation and fallbacks to handle edge cases, plus logging to help diagnose issues.

**Action Required**: Re-run the pipeline to regenerate Person records with correct face_instances data.

---

## 2026-02-10 02:00:00

**Files**:
- `sim_bench/pipeline/steps/cluster_people.py`
- `app/streamlit/components/pipeline_runner.py`
- `app/streamlit/components/people_browser.py`
- `configs/pipeline.yaml`

**Change**: Added HDBSCAN cluster merge epsilon to reduce over-segmentation

1. **cluster_people.py**: Added `cluster_selection_epsilon` parameter
   - Merges clusters within this distance of each other
   - Higher value = more merging = fewer clusters
   - Default: 0.3

2. **pipeline_runner.py**: Added "Cluster Merge Distance" slider (0.0-0.8)
   - Only shown when HDBSCAN method selected
   - Also lowered min_cluster_size minimum to 1

3. **people_browser.py**: Fixed deprecation warning
   - Changed `use_column_width=True` to `use_container_width=True`

**Reason**: User reported too many clusters (over-segmentation). The `cluster_selection_epsilon` parameter tells HDBSCAN to merge clusters that are close together, reducing fragmentation of the same person into multiple clusters.

---

## 2026-02-10 01:30:00

**Files**:
- `app/streamlit/components/pipeline_runner.py`
- `app/streamlit/components/metrics.py`

**Change**: Added min face size config and improved metrics table

1. **pipeline_runner.py**: Added "Min Face Size (px)" slider (20-100, default 50)
   - Controls minimum face size in pixels to be considered
   - Applied to insightface_detect_faces, insightface_score_expression/eyes/pose

2. **metrics.py**: Enhanced per-image metrics table with clearer body/face columns
   - "Body" column: ✓ if body detected
   - "Face" column: Face count
   - "BodyPose": Body facing camera score
   - "FacePose": Face frontal score
   - Renamed "Sharpness" to "Sharp" for column width

**Reason**: User requested min face size threshold control and clearer display of body vs face detection with their respective pose scores.

---

## 2026-02-10 01:00:00

**Files**:
- `sim_bench/api/services/pipeline_service.py`
- `sim_bench/pipeline/scoring/person_penalty.py`
- `sim_bench/pipeline/steps/cluster_by_identity.py`

**Change**: Fixed face scores showing as None and improved multi-face handling

1. **pipeline_service.py `_build_image_metrics()`**:
   - Normalized paths for cache key lookups (forward slashes)
   - Fixed InsightFace face score retrieval using correct face_index

2. **person_penalty.py**:
   - Normalized all paths for cache key lookups
   - Changed from only looking at `face_0` to checking ALL faces
   - Now uses WORST score across all faces (as user requested)
   - Added `_get_face_count()` helper for both MediaPipe and InsightFace

3. **cluster_by_identity.py**:
   - Fixed to work with InsightFace faces (was only using MediaPipe `context.faces`)
   - Normalized paths in face-to-person lookup
   - Now checks both `context.faces` and `context.insightface_faces`

**Reason**: User reported face Pose/Eyes/Smile scores all showing as None. Root cause was path format mismatch (backslashes vs forward slashes on Windows). Also fixed penalty computation to use worst score from all faces, not just first face.

---

## 2026-02-10 00:30:00

**Files**:
- `app/streamlit/api_client.py`
- `app/streamlit/components/gallery.py`

**Change**: Fixed People image viewing errors

1. **api_client.py**: `_parse_image()` now handles both `path` and `image_path` keys
   - The `get_person_images` API returns `image_path` but `_parse_image` was looking for `path`
   - Now checks both keys: `data.get("path") or data.get("image_path", "")`

2. **gallery.py**: Added error handling to thumbnail loading
   - `_load_thumbnail_cached()` now returns None on error instead of crashing
   - `_load_thumbnail()` handles None bytes gracefully
   - `render_image_card()` checks for empty path before trying to load

**Reason**: User got error when clicking on a photo in the People tab. Root cause: API endpoint returns `image_path` but parser expected `path`, resulting in empty path and file-not-found errors.

---

## 2026-02-10 00:15:00

**Files**:
- `sim_bench/pipeline/steps/cluster_people.py`
- `app/streamlit/components/pipeline_runner.py`
- `configs/pipeline.yaml`

**Change**: Added HDBSCAN support for people clustering (now the default)

1. **cluster_people.py**: Added HDBSCAN method alongside agglomerative
   - HDBSCAN auto-determines optimal clusters based on density
   - Handles noise (outlier faces not forced into clusters)
   - Uses normalized embeddings (euclidean on normalized ≈ cosine distance)
   - Logs noise point count

2. **pipeline_runner.py**: Updated UI with method selector
   - Dropdown to choose: "hdbscan" (default) or "agglomerative"
   - HDBSCAN shows "Min Faces per Person" slider (2-5)
   - Agglomerative shows "Identity Distance Threshold" slider (0.3-0.9)

3. **pipeline.yaml**: Updated cluster_people config
   - method: hdbscan (default)
   - min_cluster_size: 2
   - min_samples: 2

**Reason**: User asked about intelligent threshold selection. HDBSCAN automatically finds natural clusters without requiring manual threshold tuning - it only needs `min_cluster_size` (minimum faces to form a "person").

---

## 2026-02-09 12:00:00

**Files**:
- `sim_bench/api/services/people_service.py`
- `sim_bench/pipeline/steps/cluster_people.py`
- `sim_bench/pipeline/steps/extract_face_embeddings.py`
- `sim_bench/pipeline/steps/insightface_detect_faces.py`
- `sim_bench/api/services/pipeline_service.py`

**Change**: Fixed People feature data flow with multiple fixes:

1. **BBox format handling in PeopleService**: Both `create_from_clusters()` and `_get_thumbnail_info()` now handle both dict-style and object-style bbox (InsightFace stores as dict, MediaPipe as objects)

2. **Path normalization for cache keys**: Normalized paths to forward slashes across all steps to ensure consistent cache key lookup:
   - `extract_face_embeddings._generate_cache_key()`: uses forward slashes
   - `insightface_detect_faces._get_cache_config()`: normalizes paths
   - `cluster_people._collect_faces_with_embeddings()`: normalizes paths when looking up embeddings

3. **Enhanced logging**: Added detailed debug logging to trace face embedding storage and people cluster creation:
   - `cluster_people`: Logs counts of faces found from each source (MediaPipe vs InsightFace), matched vs unmatched embeddings
   - `extract_face_embeddings`: Logs number of embeddings stored with sample keys
   - `pipeline_service`: Logs people cluster count and Person record creation

**Reason**: User reported People tab is empty, Person column is empty. Investigation revealed:
- Path format mismatch on Windows (backslash vs forward slash) caused face embeddings to not be found when looked up in `cluster_people` step
- BBox stored as dict by InsightFace but `PeopleService` expected object with `.x`, `.y` attributes
- No logging made it difficult to trace where the data flow broke

---

## 2026-02-08 10:30:00

**Files**:
- `sim_bench/pipeline/steps/detect_faces.py` (removed debug code)
- Deleted `yolov8s-pose.pt` files (version mismatch)

**Change**: Removed debug traceback logging; deleted old YOLO model files causing version mismatch

**Reason**: MediaPipe error is FIXED (pipeline correctly uses InsightFace). YOLO error `'Conv' object has no attribute 'bn'` was caused by model files saved with different ultralytics version. Deleting them allows ultralytics to download fresh compatible versions.

---

## 2026-02-08 10:15:00

**Files**: `sim_bench/pipeline/steps/detect_faces.py`

**Change**: Added traceback logging to `_get_crop_service()` to debug why MediaPipe is being loaded

**Reason**: Pipeline steps list does NOT include `detect_faces`, yet MediaPipe is still loading. Added stack trace logging to identify exactly which code path is calling `_get_crop_service()`. (Now removed after confirming MediaPipe is no longer called)

---

## 2026-02-08 10:00:00

**Files**: `sim_bench/pipeline/executor.py`

**Change**: Added logging to show resolved pipeline steps after dependency resolution

**Reason**: Debugging MediaPipe error - need to verify which steps are actually being executed after `PipelineBuilder.build()` resolves dependencies. This will reveal if `detect_faces` step is being incorrectly added by dependency resolution.

---

## 2026-02-03 16:30:00 ✅ COMPLETE

**Files**: 
- `.gitattributes`
- `models/album_app/arcface_resnet50.pt`
- `models/album_app/ava_resnet50.pt`
- `models/album_app/siamese_comparison_model.pt`
- Git history (rewritten)

**Change**: Migrated all PyTorch model files (.pt) to Git LFS and rewrote repository history

**Reason**: User requested moving .pt files to Git LFS to reduce repository size and improve clone/push/pull performance for large binary files.

**Details**:
- Ran `git lfs install` to initialize Git LFS
- Ran `git lfs track "*.pt"` to configure LFS tracking
- Updated `.gitattributes` with `*.pt filter=lfs diff=lfs merge=lfs -text`
- Committed all 3 model files as LFS objects (99% rewrite)
- Rewrote entire Git history using `git lfs migrate import --everything`
- Uploaded 309 MB LFS objects to remote storage
- Force-pushed rewritten history (commit 851bbe4) to GitHub
- Verified: local and remote in sync, all 3 model files tracked by LFS
- Repository size reduced by ~300 MB

**Commits**: 
- `9df805c` - "chore: migrate model files (.pt) to Git LFS"
- `6f4f1ee` - History rewrite commit
- `851bbe4` - Final documentation commit

**Verification**: `git lfs ls-files` shows 3 files, `git status` shows "up to date with origin/main"

---

## 2026-02-03 16:00:00

**Files**: 
- `CLAUDE.md`
- `CHANGES_LOG.md` (created)

**Change**: Updated CLAUDE.md with current architecture and added change tracking requirement

**Reason**: User requested CLAUDE.md be updated to reflect current state (Streamlit + FastAPI, recent bug fixes, etc.) and instructed to always maintain a change log after every modification.

**Details**:
- Corrected architecture section: Now correctly states Streamlit + FastAPI (not NiceGUI)
- Added "Recent Updates (Feb 2026)" section documenting bug fixes and features
- Updated app launch commands to show backend + frontend startup
- Added pipeline steps information (18-step engine)
- Added API endpoint development instructions
- Added debugging tips section with common issues and solutions
- Created CHANGES_LOG.md with template and initial entries
- Added prominent instruction at top: "After EVERY code change, append to CHANGES_LOG.md"

---

## 2026-02-03 15:45:00

**Files**: 
- `sim_bench/api/schemas/result.py`
- `sim_bench/api/services/result_service.py`
- `app/streamlit/components/gallery.py`

**Change**: Added "Final Score" (composite_score) column to cluster debug spreadsheet

**Reason**: User requested visibility of the final selection score used for ranking images. This helps debug why certain images were selected over others by showing the weighted combination of IQA, AVA, sharpness, and face scores.

**Details**:
- Added `composite_score` field to `ImageMetrics` schema
- Updated `_build_image_dict()` to include composite_score from stored metrics
- Added "Final Score" column to cluster debug table (4th column after Selected)
- Score displayed with 3 decimal places (e.g., 0.856)

---

## 2026-02-03 15:30:00

**Files**: `app/streamlit/pages/results.py`

**Change**: Simplified cluster view by removing redundant API call and manual is_selected marking loop

**Reason**: Completed Task 6 of cluster debug view implementation. The enriched `get_clusters()` API now returns images with `is_selected` already set, making the separate `get_selected_images()` call and manual marking loop unnecessary.

**Details**:
- Removed `client.get_selected_images(job_id)` call in "By Cluster" mode
- Removed 7 lines of manual is_selected marking code
- Reduced code from 10 lines to 3 lines
- Improved performance by eliminating redundant API call

---

## 2026-02-03 (Earlier - Claude Code CLI Session)

**Files**: Multiple (11 files total)

**Changes**: Bug fixes and feature enhancements for Streamlit + FastAPI album app

**Bug Fixes**:
1. **Siamese model config loading** (`sim_bench/pipeline/steps/select_best.py`)
   - Fixed config key mismatch: now reads from nested `config["siamese"]` dict
   - Properly extracts checkpoint_path, tiebreaker_range, duplicate_threshold

2. **HDBSCAN clustering parameters** (`sim_bench/pipeline/steps/cluster_scenes.py`)
   - Fixed parameter pass-through: now passes min_samples, metric, cluster_selection_epsilon, cluster_selection_method
   - Updated config: min_cluster_size changed from 3 to 2

3. **Image EXIF rotation** (`app/streamlit/components/gallery.py`)
   - Added `_load_image_for_display()` helper using `ImageOps.exif_transpose()`
   - Applied to all st.image() call sites

**Feature Enhancements**:
1. **Per-image score display** (result_service.py, schemas, api_client, models)
   - Backend now returns: face_pose_scores, face_eyes_scores, face_smile_scores, is_selected, sharpness
   - Extracted `_build_image_dict()` helper to avoid code duplication

2. **Portrait indicators** (gallery.py)
   - Gallery shows sharpness in score line
   - Portrait indicators: Eyes (Open/Closed), Expression (Smiling/Neutral)

3. **Metrics table with CSV export** (metrics.py, results.py)
   - Added "Metrics Table" tab with DataFrame showing all image scores
   - CSV download button for export
   - Fixed PyArrow mixed-type error by converting Cluster column to string

4. **Cluster debug view** (Multiple files - Tasks 1-5)
   - Expanded ClusterInfo schema with selected_count, has_faces, face_count, person_labels
   - Enriched get_clusters() to return full ImageMetrics objects
   - Added persona sub-grouping: groups images by people appearing in them
   - Added thumbnail spreadsheet with base64-encoded 80px image previews
   - Shows all scores in table format for debugging selection decisions

---

## 2026-02-04 10:30:00

**Files**: 
- `SCORE_PERSISTENCE_DEBUG_PLAN.md` (created and updated)

**Change**: Created comprehensive debug and fix plan for missing scores + People Management feature

**Reason**: User reported that Pose, Eyes, Smile, and Final Score columns are showing None values, and People column is empty. User also clarified they want full people management (identify, name, filter).

**Details**:
- Documented 4 main problems: face scores None, composite_score None, people empty, face detection threshold
- Root cause hypothesis: scores computed but not persisted to database image_metrics JSON
- **Split into 3 sprints**:
  - **Sprint 1** (1.5 hrs): Fix score persistence to database
  - **Sprint 2** (2 hrs): Full People Management UI (gallery, naming, filtering)
  - **Sprint 3** (30 min): Tune face detection (DECREASE threshold 0.5→0.3 to catch more faces)
- Clarified face detection issue is FALSE NEGATIVES (missing real faces), not false positives
- Expanded People feature to include:
  - New People Gallery page with thumbnails and naming interface
  - Person filtering in Results page
  - Backend API endpoints for get_people() and update_person_name()
- Test strategy with 10-image test album
- Clear success criteria per sprint
- Identified 11 files (4 new, 7 modified)
- Total estimated fix time: 4-5 hours (can split across sessions)

**Next Steps**: User to choose Option A (Sprint 1 first) or Option B (all sprints together)

---

## Historical Changes (Pre-Log)

For changes before this log was created, see:
- `FEBRUARY_2026_UPDATES.md` - Recent session details
- `CLUSTER_DEBUG_VIEW_COMPLETE.md` - Complete cluster view implementation
- `FINAL_SCORE_COLUMN_ADDED.md` - Final score column addition
- `DOCUMENTATION_UPDATE_SUMMARY.md` - Documentation updates
- `MILESTONES.md` - Major project milestones

---

## Instructions for Claude Code

**After EVERY code modification**:
1. Append a new entry to this file
2. Use ISO 8601 timestamp format (YYYY-MM-DD HH:MM:SS)
3. List ALL files modified
4. Describe WHAT changed (be specific)
5. Explain WHY (user request, bug fix, refactor, etc.)
6. Include relevant details (line numbers, function names, key values)

**Example Template**:
```markdown
## YYYY-MM-DD HH:MM:SS

**Files**: 
- `path/to/file1.py`
- `path/to/file2.py`

**Change**: [One-line summary]

**Reason**: [Why this change was needed]

**Details**:
- [Specific change 1]
- [Specific change 2]
```

This log helps:
- Debug issues by tracking when changes were made
- Understand evolution of codebase
- Coordinate between different AI sessions
- Provide context for future development

---

### 2026-02-04 12:00:00
**Files**: `sim_bench/pipeline/context.py`
**Change**: Added `composite_scores: dict[str, float]` field to PipelineContext dataclass
**Reason**: Composite scores were computed transiently in select_best but never persisted; needed a field to store them for database persistence

### 2026-02-04 12:01:00
**Files**: `sim_bench/api/services/pipeline_service.py`
**Change**: Replaced inline image_metrics dict comprehension with `_build_image_metrics()` helper method that correctly aggregates per-face scores by iterating over detected faces using cache keys (`"path:face_N"`), includes composite_score, and calls `PeopleService.create_from_clusters()` to persist Person records after pipeline completion
**Reason**: Face scores (pose/eyes/smile) were always None because `pipeline_service.py` looked up scores by image path, but face scoring steps store scores keyed by cache key format (`"path:face_0"`). Also, Person records were never created because `create_from_clusters()` was never called from the pipeline execution flow.

### 2026-02-04 12:02:00
**Files**: `sim_bench/pipeline/steps/select_best.py`
**Change**: Added `context.composite_scores[path] = score` loop after scoring images in `_select_from_cluster()` to persist computed composite scores back into the pipeline context
**Reason**: Composite scores were computed for ranking but discarded after selection; they need to be stored in context so `pipeline_service.py` can persist them to the database

### 2026-02-04 12:03:00
**Files**: `sim_bench/api/services/people_service.py`
**Change**: Fixed BoundingBox serialization in `create_from_clusters()` - replaced `list(face.bbox)` with explicit `[face.bbox.x, face.bbox.y, face.bbox.w, face.bbox.h]` for both face_instances and thumbnail_bbox
**Reason**: BoundingBox is a dataclass and not iterable; `list(bbox)` would raise TypeError at runtime

### 2026-02-04 12:04:00
**Files**: `sim_bench/api/services/result_service.py`
**Change**: Changed person display name from `f"Person {person.person_index}"` to `f"Person {person.person_index + 1}"` (1-based indexing)
**Reason**: person_index is 0-based (cluster ID), but user-facing display should be 1-based for readability

### 2026-02-04 12:05:00
**Files**: `sim_bench/pipeline/steps/detect_faces.py`, `sim_bench/face_pipeline/crop_service.py`, `configs/pipeline.yaml`, `configs/global_config.yaml`
**Change**: Lowered face detection confidence threshold from 0.5 to 0.3 across all config defaults and code defaults
**Reason**: Threshold of 0.5 was causing false negatives (missing real faces); lowering to 0.3 catches more real faces while the existing min_face_ratio (2%) filter still rejects tiny artifacts

### 2026-02-05 10:00:00
**Files**: `sim_bench/api/services/pipeline_service.py`
**Change**: Added `cluster_people` step to DEFAULT_PIPELINE (after `extract_face_embeddings`, before `cluster_by_identity`)
**Reason**: People tab was empty because faces were extracted but never globally clustered by identity. Without `cluster_people`, `people_clusters` dict is empty, so no Person records were created.

### 2026-02-05 10:01:00
**Files**: `app/streamlit/pages/results.py`, `app/streamlit/components/gallery.py`
**Change**: Changed `render_cluster_gallery(clusters, show_all_images=True)` and increased column count to 6 when showing all images
**Reason**: Results view was only showing 4 images per cluster due to `show_all_images=False` and `max_preview=4` limit

### 2026-02-05 10:02:00
**Files**: `configs/pipeline.yaml`
**Change**: Lowered `detection_confidence` from 0.3 to 0.2, lowered `min_face_ratio` from 0.02 to 0.01
**Reason**: Still missing faces in many cases; more aggressive detection thresholds to catch smaller and less confident faces

### 2026-02-05 11:00:00
**Files**: `app/streamlit/components/pipeline_runner.py`
**Change**: Added comprehensive UI controls for pipeline configuration:
- Face Detection: detection_confidence slider, min_face_ratio slider
- Selection: max_score_gap slider, duplicate_threshold slider, siamese_enabled checkbox
- Added `cluster_people` to DEFAULT_PIPELINE and STEP_DISPLAY_NAMES
**Reason**: Most pipeline parameters were only editable via YAML; now controllable from UI

### 2026-02-05 11:01:00
**Files**: `sim_bench/pipeline/context.py`, `sim_bench/pipeline/steps/select_best.py`
**Change**: Added `siamese_comparisons` list field to PipelineContext; updated `_apply_siamese_tiebreaker` and `_check_near_duplicate` to log each comparison with type, images, winner, confidence, method
**Reason**: Siamese comparisons were invisible; now stored for debugging and display

### 2026-02-05 11:02:00
**Files**: `sim_bench/api/database/models.py`, `sim_bench/api/services/pipeline_service.py`, `sim_bench/api/services/result_service.py`, `sim_bench/api/routers/results.py`
**Change**: Added `siamese_comparisons` JSON column to PipelineResult model, persist comparisons to DB, added `get_comparisons()` service method and `/comparisons` API endpoint
**Reason**: Comparison log needs to be persisted and accessible via API

### 2026-02-05 11:03:00
**Files**: `app/streamlit/api_client.py`, `app/streamlit/pages/results.py`
**Change**: Added `get_comparisons()` API client method; added "Comparisons" tab showing tiebreaker results (which image won) and duplicate checks (accepted/rejected)
**Reason**: Users can now see exactly which Siamese comparisons were made and their outcomes

### 2026-02-05 12:00:00
**Files**: `sim_bench/pipeline/steps/cluster_people.py`
**Change**: Rewrote `process()` to collect faces from `context.faces` and `context.face_embeddings` instead of requiring `context.all_faces` (which was never populated)
**Reason**: People tab was empty because `cluster_people` required `all_faces` from `filter_best_faces` step which wasn't in the pipeline

### 2026-02-05 12:01:00
**Files**: `app/streamlit/pages/results.py`
**Change**: Added thumbnails to Comparisons tab - both tiebreaker and duplicate check sections now show image thumbnails side by side
**Reason**: User requested visual comparison of images in the comparisons view

### 2026-02-05 12:02:00
**Files**: `app/streamlit/components/metrics.py`
**Change**: Added thumbnails to per-image metrics table, added "Final" (composite_score) column, uses `st.column_config.ImageColumn` for thumbnail display
**Reason**: User requested thumbnails in metrics table for easier identification

### 2026-02-05 12:03:00
**Files**: `configs/pipeline.yaml`
**Change**: Lowered `min_face_ratio` from 0.01 to 0.005 (0.5%), lowered `detection_confidence` from 0.2 to 0.15
**Reason**: Still missing faces; allowing very small faces to be detected

### 2026-02-05 14:00:00
**Files**: `app/streamlit/api_client.py`
**Change**: Fixed `_parse_person()` to map `thumbnail_image_path` to `representative_face` field; added `get_subclusters()` method
**Reason**: API returns `thumbnail_image_path` but client model expected `representative_face`; also need API method for fetching face sub-clusters

### 2026-02-05 14:01:00
**Files**: `sim_bench/api/database/models.py`
**Change**: Added `face_subclusters = Column(JSON)` to PipelineResult model
**Reason**: Need to persist face-based sub-clusters (images grouped by face identity within each scene cluster)

### 2026-02-05 14:02:00
**Files**: `sim_bench/api/services/pipeline_service.py`
**Change**: Added serialization of `context.face_clusters` to `face_subclusters` JSON in PipelineResult when saving completed pipeline
**Reason**: Sub-clusters computed by `cluster_by_identity` step were not being persisted to database

### 2026-02-05 14:03:00
**Files**: `sim_bench/api/services/result_service.py`, `sim_bench/api/routers/results.py`
**Change**: Added `get_subclusters(job_id)` service method and `GET /{job_id}/subclusters` API endpoint
**Reason**: Need to expose face sub-clusters via REST API for frontend display

### 2026-02-06 10:00:00
**Files**: `app/streamlit/pages/results.py`
**Change**: Added "Sub-Clusters" tab to results page showing face-based sub-clusters within each scene cluster
**Reason**: User requested sub-clusters to be displayed - shows images grouped by unique face combinations (e.g., A+B, A-only, B-only, no faces)

**Details**:
- Added `_render_subclusters_tab()` function
- Uses expandable sections for each scene cluster
- Sub-clusters sorted by face count (descending)
- Shows face count, identity signature, and thumbnail grid (up to 6 images)
- Uses emoji indicators: 👥 for faces, 📷 for no-face clusters

### 2026-02-06 11:00:00
**Files**:
- `app/streamlit/components/people_browser.py`
- `app/streamlit/models.py`
- `app/streamlit/api_client.py`

**Change**: Fixed People tab to show cropped face thumbnails and added inline rename

**Reason**: Person thumbnails were showing full image instead of just the face; user requested ability to edit person name from grid view

**Details**:
- Added `thumbnail_bbox` field to Person model
- Updated `_parse_person()` to include thumbnail_bbox from API
- Updated `_render_person_thumbnail()` to crop face from image using bbox with 30% padding
- Added inline rename functionality to `render_person_card()` - click pencil icon to rename
- Pass album_id to render_person_card for rename API calls

### 2026-02-06 12:00:00
**Files**: `sim_bench/pipeline/steps/cluster_by_identity.py`

**Change**: Fixed critical bug - sub-clustering now uses global person IDs from cluster_people instead of independent embedding quantization

**Reason**: Selection logic was inconsistent with People tab. "Person 3" in People tab was computed by global clustering, but sub-clustering used a different quantized embedding hash. This caused images with the same person to be placed in different sub-clusters and not compete properly.

**Details**:
- Added `_build_face_to_person_lookup()` to map (image_path, face_index) → person_id
- Modified `process()` to look up person IDs from `context.people_clusters`
- Removed old `_compute_identity_signature()` that used embedding quantization
- Updated dependency: now depends on `cluster_people` instead of `extract_face_embeddings`
- Sub-cluster identity now shows "Person_0+Person_1" format for clarity
- Added `person_ids` list to sub-cluster metadata for downstream use

### 2026-02-06 12:30:00
**Files**:
- `sim_bench/pipeline/steps/detect_faces.py`
- `sim_bench/face_pipeline/types.py`
- `sim_bench/api/services/people_service.py`

**Change**: Store cropped face images to disk for faster thumbnail loading

**Reason**: Previously cropped faces in memory but discarded them. For People tab thumbnails, had to re-crop from full image every time. Now save to `.faces/` directory.

**Details**:
- Added `_get_faces_dir()`, `_get_face_crop_path()`, `_save_face_crop()` helpers
- Modified `_serialize_faces()` to save crops to `{album}/.faces/{image}_face_{n}.jpg`
- Added `crop_path` field to `CroppedFace` dataclass
- Modified `_deserialize_faces()` to load from saved crop if available
- Updated `people_service.create_from_clusters()` to use `crop_path` for thumbnail if available
- Thumbnail stored as direct path to cropped face (no bbox needed when pre-cropped)

### 2026-02-06 12:31:00
**Files**: `app/streamlit/components/gallery.py`

**Change**: Make gallery images display with consistent square aspect ratio

**Reason**: Portrait and landscape images had different heights in grid, causing inconsistent visual layout

**Details**:
- Updated `_load_image_for_display()` to crop to center square by default
- Added `make_square` parameter (default True) for control
- Gallery now shows uniform thumbnail grid

### 2026-02-06 12:32:00
**Files**: `app/streamlit/components/people_browser.py`

**Change**: Fixed bbox coordinate conversion in person thumbnail cropping

**Reason**: Bbox values are stored in relative coordinates (0-1 range) but code was using them as pixel values, causing incorrect crop regions

**Details**:
- Added conversion: `x = x_rel * img_w`, etc.
- Padding calculation now works correctly with pixel values

### 2026-02-06 13:00:00
**Files**:
- `sim_bench/pipeline/steps/select_best.py`
- `app/streamlit/components/people_browser.py`
- `configs/pipeline.yaml`

**Change**: Fixed duplicate detection logic and added missing People page features

**Reason**: Multiple issues reported:
1. Duplicate detection was incorrectly using Siamese confidence instead of embedding similarity
2. People page missing `enable_selection` parameter and `render_merge_dialog` function
3. Name editing didn't show proper error messages
4. Near-identical images being selected due to too-strict threshold

**Details**:
- Rewrote `_check_near_duplicate()` to use embedding similarity only (Siamese CNN compares quality, not similarity)
- Added `_get_embedding_similarity()` helper method
- Lowered duplicate threshold from 0.95 to 0.85 (more aggressive filtering)
- Added `enable_selection` parameter to `render_people_grid()`
- Added `render_merge_dialog()` function for merging people
- Added `_add_to_merge_selection()` and `_remove_from_merge_selection()` helpers
- Added error handling and messages to inline rename functionality

### 2026-02-06 13:15:00
**Files**:
- `app/streamlit/components/gallery.py`
- `app/streamlit/components/metrics.py`
- `app/streamlit/pages/results.py`

**Change**: Fixed ALL thumbnail functions to produce consistent square images

**Reason**: Images were displaying at different sizes because `thumbnail()` only shrinks and maintains aspect ratio

**Details**:
- Fixed `_load_thumbnail` in gallery.py - crop to center square, resize to exact 300x300
- Fixed `_image_to_base64_thumbnail` in gallery.py - crop to center square, resize to exact size
- Fixed `_image_to_base64_thumbnail` in metrics.py - crop to center square, resize to exact size
- Fixed `_load_thumbnail` in results.py - crop to center square, resize to exact size
- All functions now: 1) crop to center square, 2) resize to exact requested size with LANCZOS

### 2026-02-06 13:45:00
**Files**:
- `sim_bench/api/services/pipeline_service.py`
- `configs/pipeline.yaml`

**Change**: Switched default pipeline from MediaPipe to InsightFace

**Reason**: User requested InsightFace as the default backend

**Details**:
- Updated `DEFAULT_PIPELINE` in pipeline_service.py to use InsightFace steps:
  - `detect_persons` (YOLOv8-Pose)
  - `insightface_detect_faces` (InsightFace SCRFD)
  - `insightface_score_expression/eyes/pose`
- Added `cluster_people` step (missing from original InsightFace config)
- Updated `select_best` config with InsightFace scoring:
  - `scoring_backend: insightface`
  - `scoring_strategy: insightface_penalty`
  - Penalty weights for body/face/eyes/smile/pose

### 2026-02-06 14:00:00
**Files**: `app/streamlit/components/gallery.py`

**Change**: Fixed gallery image sizes with caching and fixed pixel width

**Reason**: Images were still displaying at inconsistent sizes despite previous fixes

**Details**:
- Changed from `use_column_width=True` to `width=THUMBNAIL_SIZE` (200px)
- Added `@st.cache_data` decorator for thumbnail caching (faster loading)
- Thumbnails now stored as JPEG bytes in Streamlit cache
- All images display at exactly 200x200 pixels (fits 4 per row)
- Truncated long filenames to prevent layout issues

### 2026-02-06 14:30:00
**Files**:
- `app/streamlit/components/gallery.py`
- `app/streamlit/components/metrics.py`
- `app/streamlit/models.py`
- `app/streamlit/api_client.py`
- `sim_bench/api/services/pipeline_service.py`

**Change**: Added InsightFace metrics to Results view

**Reason**: User requested new InsightFace metrics (person detection, body facing score) to be displayed

**Details**:
- Added `person_detected`, `body_facing_score`, `person_confidence` to ImageInfo model
- Updated `_build_image_metrics()` to include InsightFace person detection data
- Updated `_parse_image()` in API client to parse new fields
- Updated `_render_face_info()` to show body facing score
- Updated cluster score table to include Person and Body columns
- Updated per-image metrics table to include InsightFace metrics

### 2026-02-06 14:35:00
**Files**: `app/streamlit/components/gallery.py`

**Change**: Changed thumbnail resize to preserve aspect ratio

**Reason**: User requested images not be cropped, just resized to fixed width

**Details**:
- Changed from square crop + resize to width-only resize
- Thumbnails now 200px wide with proportional height
- Full image content preserved (no cropping)

### 2026-02-07 10:00:00
**Files**: `notebooks/eda_yolo_insightface.ipynb` (created)

**Change**: Created EDA notebook for YOLOv8 person detection and InsightFace face analysis

**Reason**: User requested notebook to explore model outputs and experiment with detection results

**Details**:
- **Part 1: YOLOv8 Person Detection**
  - Loads YOLOv8-Pose model (configurable size: n/s/m/l/x)
  - Shows model outputs: bounding boxes, confidence, 17 COCO keypoints, body facing score
  - Top 5 highest/lowest confidence detections
  - Front-facing vs side-facing analysis
  - Cell to run on specific user-selected image
- **Part 2: InsightFace Face Analysis**
  - Loads InsightFace buffalo_l model
  - Shows outputs: bbox, confidence, 5-point landmarks, age, gender, pose angles
  - Heuristic smile score from mouth/eye ratio
  - 5 images with faces / 5 without
  - 5 smiling / 5 not smiling
  - Age/gender distribution charts
  - Cell to run on specific image
- **Part 3: Combined Analysis**
  - Merges YOLOv8 + InsightFace results
  - Categories: person+face, person-only, face-only, neither
  - 5 person+face images, 5 person-no-face (back turned)
  - 5 smiling persons, 5 not smiling
  - Combined visualization on specific image
- Dataset: `D:\Budapest2025_Google`
- Helper functions for visualization with bounding boxes and keypoints

---

### 2026-02-07 09:00:00
**Files**:
- `sim_bench/pipeline/context.py`
- `sim_bench/pipeline/steps/insightface_score_expression.py`
- `sim_bench/pipeline/steps/insightface_score_eyes.py`
- `sim_bench/pipeline/steps/insightface_score_pose.py`
- `sim_bench/pipeline/scoring/strategy.py`
- `sim_bench/pipeline/steps/extract_face_embeddings.py`

**Change**: Fixed 4 bugs in InsightFace pipeline

**Reason**: Loop logic bug caused face lookup to always return first face or None; missing context attributes caused AttributeError; scoring strategy assumed attributes existed without defensive checks; extract_face_embeddings had incomplete dependency list

**Details**:
1. **Bug 1 (CRITICAL) - Loop logic error**: Fixed `_find_face()` in 3 files (insightface_score_expression.py:91-93, insightface_score_eyes.py:91-93, insightface_score_pose.py:91-93). Changed from `return face if face_matches else None` (returns on first iteration!) to `if face.get('face_index') == face_index: return face` followed by `return None` outside loop.

2. **Bug 2 (HIGH) - Missing context attributes**: Added `persons: dict[str, dict]` and `insightface_faces: dict[str, dict]` fields to PipelineContext dataclass in context.py (after line 32).

3. **Bug 3 (HIGH) - Missing defensive checks**: Changed `context.persons.get()` and `context.insightface_faces.get()` to use `getattr(context, 'persons', {})` pattern in strategy.py at lines 77, 94, 100, 122. This prevents AttributeError when context doesn't have these attributes.

4. **Bug 4 (MEDIUM) - Wrong dependency metadata**: Updated `depends_on` in extract_face_embeddings.py from `["detect_faces"]` to `["detect_faces", "insightface_detect_faces"]` so step runs after either MediaPipe or InsightFace face detection.

---

### 2026-02-07 11:30:00
**Files**: `CLAUDE.md`

**Change**: Improved CLAUDE.md for better clarity and reduced verbosity

**Reason**: User ran `/init` command to improve the Claude Code guidance file

**Details**:
- Condensed project overview to bullet points
- Consolidated common commands into single code block
- Added concrete code example for creating new pipeline steps
- Added table format for key entry points
- Documented both pipelines (default + insightface)
- Removed "Recent Updates" section (transient info that becomes stale)
- Removed verbose module structure list (easily discoverable)
- Removed redundant "Adding New Components" section (replaced with code example)
- Added model weights location section
- Streamlined debugging tips
- Reduced overall length by ~40% while preserving essential information

---

### 2026-02-07 12:00:00
**Files**: `docs/architecture/PIPELINE_CALL_CHAIN.md` (created)

**Change**: Created comprehensive documentation explaining why MediaPipe is being called

**Reason**: User encountered protobuf/MediaPipe compatibility error and wanted to understand the full call chain

**Details**:
- Traced complete call chain from user click → frontend → API → executor → step → MediaPipe
- **Root cause identified**: Frontend `pipeline_runner.py:12-27` has outdated `DEFAULT_PIPELINE` using MediaPipe steps (`score_face_eyes`), while backend `pipeline_service.py:22-38` has updated InsightFace pipeline
- Frontend sends its step list to backend, overriding backend's default
- Documented both pipelines (MediaPipe vs InsightFace) with step comparisons
- Explained dependency resolution and why `detect_faces` gets auto-added
- Included ASCII diagrams showing the problem flow
- Provided 3 fix options:
  1. Update frontend DEFAULT_PIPELINE to use InsightFace steps (recommended)
  2. Don't pass steps from frontend, let backend use its default
  3. Downgrade protobuf (not recommended)

---

### 2026-02-08 12:30:00
**Files**: `docs/architecture/CONFIG_SINGLE_SOURCE_OF_TRUTH_PLAN.md` (created)

**Change**: Created comprehensive plan to make YAML config the single source of truth

**Reason**: User identified that there are 3 conflicting sources for pipeline definition (YAML, frontend, backend) and wants a clean architecture

**Details**:
- **Problem**: Frontend hardcodes `DEFAULT_PIPELINE` (MediaPipe), backend has different `DEFAULT_PIPELINE` (InsightFace), YAML has yet another version
- **Solution**: YAML → DB (on startup sync) → Frontend (fetches from API)
- **5 Phases**:
  1. Update YAML to current InsightFace pipeline
  2. Improve config sync (sync YAML to DB on every startup, not just first run)
  3. Remove hardcoded pipelines from frontend and backend
  4. Add user settings persistence (save/load user preferences to DB)
  5. Migration script for existing installations
- **New DB columns**: `is_system`, `user_id`, `parent_profile_id` for ConfigProfile
- **New API endpoints**: `GET/POST /config/user/{user_id}` for user settings
- **Key principle**: User profiles store only OVERRIDES, not full config - they inherit from default and automatically get updates when YAML changes

---

### 2026-02-08 13:00:00
**Files**:
- `configs/pipeline.yaml`
- `sim_bench/api/database/models.py`
- `sim_bench/api/services/config_service.py`
- `sim_bench/api/services/pipeline_service.py`
- `sim_bench/api/routers/config.py`
- `app/streamlit/api_client.py`
- `app/streamlit/components/pipeline_runner.py`

**Change**: Implemented "YAML as Single Source of Truth" for pipeline configuration

**Reason**: Resolve the 3-source-of-truth problem where YAML, frontend, and backend all had different pipeline definitions

**Details**:
- **Phase 1**: Updated `pipeline.yaml` with `minimal_pipeline` option (InsightFace already set as default)
- **Phase 2**:
  - Added `is_system`, `user_id`, `parent_profile_id` columns to ConfigProfile model
  - Updated `config_service.py` with `sync_default_profile()` that syncs YAML→DB on every startup
  - Added `get_available_pipelines()` helper function
  - Added user profile methods: `get_user_profile()`, `save_user_profile()`, `get_user_config()`, `delete_user_profile()`
- **Phase 3**:
  - Removed hardcoded `DEFAULT_PIPELINE` from `pipeline_service.py`
  - Updated `start_pipeline()` to load steps from config service when not provided
  - Removed hardcoded pipelines from `pipeline_runner.py`
  - Frontend now fetches pipelines from API via `get_available_pipelines()`
- **Phase 4**:
  - Added API endpoints: `GET/POST/DELETE /config/user/{user_id}` and `GET /config/pipelines`
  - Added API client methods: `get_available_pipelines()`, `get_user_config()`, `save_user_config()`
  - Frontend loads saved user settings on page load
  - Added "Save Settings" button to persist user preferences
  - Config slider values now restore from saved settings

**To apply**: Delete `sim_bench.db` and restart the API to create fresh database from YAML

---

### 2026-02-08 14:00:00
**Files**:
- `sim_bench/pipeline/steps/score_ava.py`
- `sim_bench/pipeline/context.py`
- `sim_bench/pipeline/scoring/strategy.py`
- `sim_bench/pipeline/steps/insightface_score_pose.py`
- `sim_bench/pipeline/steps/insightface_score_eyes.py`
- `sim_bench/pipeline/steps/insightface_score_expression.py`
- `sim_bench/pipeline/insightface_pipeline/face_cropper.py` (created)
- `sim_bench/pipeline/insightface_pipeline/__init__.py`
- `configs/pipeline.yaml`

**Change**: Fixed face scoring (AVA, Pose, Eyes, Smile) to produce meaningful quality metrics

**Reason**: Scoring steps were returning hardcoded 0.5 values or using inconsistent scales, making quality-based selection ineffective

**Details**:

1. **AVA Score Normalization**:
   - Modified `score_ava.py:_store_results()` to divide scores by 10 before storing
   - AVA model returns 1-10 scale, now normalized to 0-1 at storage time
   - Removed redundant normalization from `context.py:get_image_score()` (line 104)
   - Removed redundant normalization from `strategy.py:InsightFacePenaltyScoring.compute_score()` (line 60)
   - Changed default fallback from `5.0 / 10.0` to `0.5` in strategy.py

2. **Pose Scoring from InsightFace Landmarks**:
   - Replaced stub `FacePoseScorer.compute_score()` in `insightface_score_pose.py`
   - New algorithm computes frontal score from 5-point landmarks (left_eye, right_eye, nose)
   - Calculates eye center, eye vector, and nose deviation from eye line
   - Normalizes yaw by eye distance to get frontal score (1 = frontal, 0 = profile)
   - Added `import numpy as np` for calculations

3. **Face Cropping Utility**:
   - Created `face_cropper.py` with `InsightFaceCropper` class
   - Takes InsightFace bbox, applies configurable margin (default 30%), resizes to 256x256
   - Handles EXIF rotation with `ImageOps.exif_transpose()`
   - Exported from `__init__.py`

4. **Eye Scoring via MediaPipe on Cropped Faces**:
   - Replaced stub `EyeStateScorer.compute_score()` in `insightface_score_eyes.py`
   - Uses `InsightFaceCropper` to get 256x256 face crop
   - Runs MediaPipe Face Mesh on crop
   - Calls existing `detect_eye_state()` from `portrait_analysis/eye_state.py`
   - Normalizes EAR (Eye Aspect Ratio) to 0-1 score
   - Added config parameters: `crop_margin`, `target_size`, `ear_threshold`
   - Updated `_find_face()` to enrich face_data with `original_path`
   - Removed unused `NeutralScorer` class

5. **Smile Scoring via MediaPipe on Cropped Faces**:
   - Replaced stub `ExpressionScorer.compute_score()` in `insightface_score_expression.py`
   - Uses same `InsightFaceCropper` approach as eye scoring
   - Runs MediaPipe Face Mesh on crop
   - Calls existing `detect_smile()` from `portrait_analysis/smile_detection.py`
   - Returns normalized smile score (already 0-1 from utility)
   - Added config parameters: `crop_margin`, `target_size`, `width_threshold`
   - Updated `_find_face()` to enrich face_data with `original_path`
   - Removed unused `NeutralScorer` class

6. **Pipeline Config Updates**:
   - Updated `pipeline.yaml` with new config parameters for InsightFace scoring steps
   - `insightface_score_expression`: crop_margin, target_size, width_threshold
   - `insightface_score_eyes`: crop_margin, target_size, ear_threshold
   - `insightface_score_pose`: simplified config (uses 5-point landmarks, no external model needed)

---

### 2026-02-09 15:00:00
**Files**:
- `requirements.txt`
- `sim_bench/pipeline/utils/__init__.py` (created)
- `sim_bench/pipeline/utils/image_cache.py` (created)
- `sim_bench/pipeline/insightface_pipeline/face_analyzer.py`
- `sim_bench/pipeline/insightface_pipeline/face_cropper.py`
- `sim_bench/pipeline/steps/extract_face_embeddings.py`
- `sim_bench/portrait_analysis/analyzer.py`
- `sim_bench/face_pipeline/crop_service.py`

**Change**: Fixed protobuf compatibility and created global image cache with EXIF normalization

**Reason**: Two issues: (1) protobuf 6.x incompatible with MediaPipe causing `'MessageFactory' object has no attribute 'GetPrototype'` error, (2) EXIF transpose happening inconsistently causing bbox coordinate mismatches ("Coordinate 'right' is less than 'left'" warnings)

**Details**:

1. **Protobuf Version Fix**:
   - Added `protobuf>=3.20,<4` to requirements.txt
   - This version works with both MediaPipe and Streamlit

2. **Global Image Cache** (`sim_bench/pipeline/utils/image_cache.py`):
   - Created `ImageCache` singleton class with persistent disk cache
   - Cache location: `~/.sim_bench/image_cache/`
   - EXIF-first cache key strategy:
     - If image has EXIF DateTimeOriginal: `SHA256(datetime + make + model + size)`
     - Fallback: `SHA256(first_64KB + last_64KB + size)`
   - Images normalized once (EXIF transposed, RGB converted) and cached as JPEG
   - SQLite index for fast lookups
   - API: `get()`, `get_pil()`, `get_dimensions()`, `clear()`, `evict()`, `get_stats()`

3. **Updated Image Consumers**:
   - `face_analyzer.py`: Use `get_image_cache().get()` instead of `Image.open()`
   - `face_cropper.py`: Use `get_image_cache().get_pil()`, added bbox validation
   - `extract_face_embeddings.py`: Use cache, added crop coordinate validation
   - `portrait_analysis/analyzer.py`: Use cache in `_load_image()`
   - `face_pipeline/crop_service.py`: Use cache in `_load_image()`

4. **Benefits**:
   - Consistent EXIF handling across all pipeline steps
   - Bbox coordinates always match image orientation
   - Performance: images normalized once, cached for reuse
   - Shared across albums (same image = one cached copy)

---

### 2026-02-09 16:00:00
**Files**:
- `sim_bench/pipeline/steps/cluster_people.py`
- `sim_bench/pipeline/scoring/quality_strategy.py`
- `app/streamlit/pages/results.py`

**Change**: Fixed People tab, Siamese comparisons logging, and UI clarity

**Reason**: Multiple issues reported: People tab empty, no Siamese comparisons displayed, "All Filtered" confusing

**Details**:

1. **Fix cluster_people for InsightFace pipeline** (`cluster_people.py`):
   - Root cause: Step only looked at `context.faces` (MediaPipe), not `context.insightface_faces` (InsightFace)
   - Added `_collect_faces_with_embeddings()` method that works with both pipelines
   - Created `FaceForClustering` dataclass for lightweight face representation
   - Properly looks up embeddings using cache key format (`"path:face_N"`)
   - Now `context.people_clusters` gets populated, Person records get created

2. **Log Siamese comparisons from quality strategies** (`quality_strategy.py`):
   - Root cause: `_apply_siamese_refinement()` and `_run_tournament()` didn't have access to `context`
   - Updated `SiameseRefinementQuality._apply_siamese_refinement()` to accept `context` and log comparisons
   - Updated `SiameseTournamentQuality._run_tournament()` to accept `context` and log comparisons
   - Comparisons now logged to `context.siamese_comparisons` with type='refinement' or type='tournament'

3. **Rename "All Filtered" to "All Processed"** (`results.py`):
   - Changed view mode option from "All Filtered" to "All Processed"
   - Updated description from "passed quality filter" to "processed by pipeline"
   - Updated metric label to "All Processed"

**Cascading effects**:
- People tab will now show detected people
- Person column in cluster view will be populated
- Sub-clustering by identity will work properly
- Comparisons tab will show Siamese refinement/tournament comparisons
