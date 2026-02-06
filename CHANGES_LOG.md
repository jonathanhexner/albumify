# Change Log

**Purpose**: Track all code modifications with timestamps for debugging and history.

**Format**: Each entry includes date/time (ISO 8601), files modified, change description, and reason.

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
