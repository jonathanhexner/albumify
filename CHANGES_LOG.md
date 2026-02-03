# Change Log

**Purpose**: Track all code modifications with timestamps for debugging and history.

**Format**: Each entry includes date/time (ISO 8601), files modified, change description, and reason.

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
