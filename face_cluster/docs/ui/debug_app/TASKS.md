# Face Clustering Debug App - Task Breakdown

**Date:** 2026-02-17
**Status:** Draft - Pending Approval
**Prerequisites:** [REQUIREMENTS.md](REQUIREMENTS.md) and [ARCHITECTURE.md](ARCHITECTURE.md) approved

---

## Task Overview

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1 | T1-T3 | Setup & Models |
| Phase 2 | T4-T7 | Services Layer |
| Phase 3 | T8-T14 | Components |
| Phase 4 | T15-T21 | Pages |
| Phase 5 | T22-T24 | Integration & Cleanup |

**Total: 24 tasks**

---

## Phase 1: Setup & Models

### T1: Create folder structure
**Size:** S (Small)
**Depends on:** None

**Work:**
- Create `app/face_clustering_debug/` directory
- Create subdirectories: `pages/`, `components/`, `services/`, `models/`
- Create empty `__init__.py` in each directory

**Acceptance Criteria:**
- [ ] All directories exist
- [ ] All `__init__.py` files created
- [ ] `python -c "from app.face_clustering_debug import *"` works

**Files:**
```
app/face_clustering_debug/
├── __init__.py
├── pages/__init__.py
├── components/__init__.py
├── services/__init__.py
└── models/__init__.py
```

---

### T2: Implement data models (schemas.py)
**Size:** S
**Depends on:** T1

**Work:**
- Create dataclasses for: `FaceInfo`, `ClusterInfo`, `MergeDecision`, `AttachDecision`, `ClusteringResult`
- Include all fields from ARCHITECTURE.md
- Add type hints

**Acceptance Criteria:**
- [ ] All 5 dataclasses implemented
- [ ] `FaceInfo` includes landmarks and pose_angles
- [ ] `python -m py_compile models/schemas.py` passes
- [ ] < 120 lines

**Files:**
- `app/face_clustering_debug/models/schemas.py`

---

### T3: Define service protocols (protocols.py)
**Size:** S
**Depends on:** T2

**Work:**
- Create `DataLoaderProtocol` using `typing.Protocol`
- Define method signatures: `load_embeddings()`, `load_faces()`, `load_clustering_result()`, `get_available_methods()`, `get_face_crop()`

**Acceptance Criteria:**
- [ ] Protocol class defined with all 5 methods
- [ ] Type hints for all parameters and returns
- [ ] < 50 lines

**Files:**
- `app/face_clustering_debug/services/protocols.py`

---

## Phase 2: Services Layer

### T4: Implement FileLoader service
**Size:** M (Medium)
**Depends on:** T3

**Work:**
- Implement `FileLoader` class conforming to `DataLoaderProtocol`
- Parse benchmark JSON files for metadata and clustering results
- Load embeddings from NPY files
- Load face crops from `face_crops/` directory
- Extract from existing `face_clustering_comparison.py` logic

**Acceptance Criteria:**
- [ ] All 5 protocol methods implemented
- [ ] Works with existing `results/face_clustering_benchmark/` data
- [ ] Unit test: load embeddings, verify shape
- [ ] Unit test: load faces, verify count matches embeddings
- [ ] < 150 lines

**Files:**
- `app/face_clustering_debug/services/file_loader.py`
- `tests/face_clustering_debug/test_file_loader.py`

---

### T5: Implement DBLoader service
**Size:** M
**Depends on:** T3

**Work:**
- Implement `DBLoader` class conforming to `DataLoaderProtocol`
- Query `universal_cache` table for embeddings and face metadata
- Query `people` table for clustering results
- Handle case where data doesn't exist

**Acceptance Criteria:**
- [ ] All 5 protocol methods implemented
- [ ] Works with existing database at `~/.sim_bench/sim_bench.db`
- [ ] Returns empty/None gracefully when no data
- [ ] < 150 lines

**Files:**
- `app/face_clustering_debug/services/db_loader.py`
- `tests/face_clustering_debug/test_db_loader.py`

---

### T6: Implement ClusteringRunner service
**Size:** M
**Depends on:** T2

**Work:**
- Implement `ClusteringRunner` class with static methods
- `get_available_algorithms()` - return list of algorithm names
- `get_algorithm_params(algorithm)` - return param definitions for UI
- `run(algorithm, params, embeddings)` - execute clustering via `sim_bench.clustering.base.load_clustering_method()`
- Parse stats dict into `ClusteringResult` schema

**Acceptance Criteria:**
- [ ] All 3 methods implemented
- [ ] `get_algorithm_params()` returns complete param definitions for `hybrid_hdbscan_knn` and `hybrid_closest_face`
- [ ] `run()` calls existing clustering code, not reimplementation
- [ ] Unit test: run clustering on small synthetic data
- [ ] < 100 lines

**Files:**
- `app/face_clustering_debug/services/clustering_runner.py`
- `tests/face_clustering_debug/test_clustering_runner.py`

---

### T7: Services __init__.py exports
**Size:** S
**Depends on:** T4, T5, T6

**Work:**
- Export all services from `services/__init__.py`
- Verify imports work correctly

**Acceptance Criteria:**
- [ ] `from app.face_clustering_debug.services import FileLoader, DBLoader, ClusteringRunner` works

**Files:**
- `app/face_clustering_debug/services/__init__.py`

---

## Phase 3: Components

### T8: Implement face_grid component
**Size:** S
**Depends on:** T2

**Work:**
- Implement `render_face_grid()` function
- Display faces in configurable grid
- Highlight exemplars with marker
- Return selected face index when clicked

**Acceptance Criteria:**
- [ ] Renders grid of face thumbnails
- [ ] Exemplars show ⭐ marker
- [ ] Clicking face returns its index
- [ ] < 60 lines

**Files:**
- `app/face_clustering_debug/components/face_grid.py`

---

### T9: Implement face_detail component
**Size:** S
**Depends on:** T2

**Work:**
- Implement `render_face_detail()` function
- Draw 5-point landmarks on face image
- Show pose angles
- Show confidence score

**Acceptance Criteria:**
- [ ] Renders enlarged face image
- [ ] Landmarks drawn with colored dots
- [ ] Pose angles displayed
- [ ] < 50 lines

**Files:**
- `app/face_clustering_debug/components/face_detail.py`

---

### T10: Implement distance_heatmap component
**Size:** S
**Depends on:** None

**Work:**
- Implement `render_distance_heatmap()` function
- Use matplotlib to create heatmap
- Show distance values in cells
- Color scale: green (close) to red (far)

**Acceptance Criteria:**
- [ ] Renders heatmap from distance matrix
- [ ] Axis labels show face/cluster IDs
- [ ] < 60 lines

**Files:**
- `app/face_clustering_debug/components/distance_heatmap.py`

---

### T11: Implement threshold_display component
**Size:** S
**Depends on:** T2

**Work:**
- Implement `render_threshold_info()` function
- Display threshold value with Q1, Q3, IQR, raw threshold
- Visual indicator showing where threshold falls

**Acceptance Criteria:**
- [ ] Shows all threshold statistics
- [ ] Clear visual formatting
- [ ] < 40 lines

**Files:**
- `app/face_clustering_debug/components/threshold_display.py`

---

### T12: Implement decision_card component
**Size:** S
**Depends on:** T2

**Work:**
- Implement `render_merge_decision()` function
- Implement `render_attach_decision()` function
- Show decision details: clusters involved, threshold, result, reason

**Acceptance Criteria:**
- [ ] Merge decision shows: clusters, threshold, pairs, result, reason
- [ ] Attach decision shows: face, target cluster, candidates, result
- [ ] Color-coded: green=merged/attached, red=rejected
- [ ] < 80 lines

**Files:**
- `app/face_clustering_debug/components/decision_card.py`

---

### T13: Implement param_sliders component
**Size:** S
**Depends on:** None

**Work:**
- Implement `render_param_sliders()` function
- Generate sliders from param definitions dict
- Support int and float types
- Return dict of current values

**Acceptance Criteria:**
- [ ] Dynamically creates sliders from definitions
- [ ] Returns current values as dict
- [ ] < 50 lines

**Files:**
- `app/face_clustering_debug/components/param_sliders.py`

---

### T14: Components __init__.py exports
**Size:** S
**Depends on:** T8-T13

**Work:**
- Export all components from `components/__init__.py`

**Acceptance Criteria:**
- [ ] All components importable from `app.face_clustering_debug.components`

**Files:**
- `app/face_clustering_debug/components/__init__.py`

---

## Phase 4: Pages

### T15: Implement overview page
**Size:** M
**Depends on:** T7, T8, T9, T11

**Work:**
- Implement `render_overview_page()` function
- Two-column layout: clusters grid + face detail panel
- Show cluster count, noise count metrics
- Expandable sections per cluster
- Click face to show detail with landmarks

**Acceptance Criteria:**
- [ ] Shows all clusters with face grids
- [ ] Exemplars highlighted
- [ ] Face detail panel works on click
- [ ] Threshold info shown per cluster
- [ ] < 150 lines

**Files:**
- `app/face_clustering_debug/pages/overview.py`

---

### T16: Implement merge_decisions page
**Size:** M
**Depends on:** T7, T10, T12

**Work:**
- Implement `render_merge_decisions_page()` function
- List all merge decisions
- Filter by: merged/rejected, cluster ID
- Show cross-distance matrix for selected pair
- Explain why merge did/didn't happen

**Acceptance Criteria:**
- [ ] Lists all merge decisions
- [ ] Filtering works
- [ ] Distance matrix shown for selected pair
- [ ] Reason clearly explained
- [ ] < 150 lines

**Files:**
- `app/face_clustering_debug/pages/merge_decisions.py`

---

### T17: Implement attach_decisions page
**Size:** M
**Depends on:** T7, T12

**Work:**
- Implement `render_attach_decisions_page()` function
- List all attachment decisions
- Filter by: attached/remained noise
- Show candidate clusters with distances

**Acceptance Criteria:**
- [ ] Lists all attachment decisions
- [ ] Shows candidates per noise point
- [ ] Explains why attached or stayed noise
- [ ] < 120 lines

**Files:**
- `app/face_clustering_debug/pages/attach_decisions.py`

---

### T18: Implement distance_lookup page
**Size:** S
**Depends on:** T7, T9

**Work:**
- Implement `render_distance_lookup_page()` function
- Two face selectors (dropdowns or number inputs)
- Show both faces side-by-side
- Compute and display Euclidean and cosine distance
- Show whether distance is within each cluster's threshold

**Acceptance Criteria:**
- [ ] Select any two faces
- [ ] Shows both distances
- [ ] Compares to thresholds
- [ ] < 100 lines

**Files:**
- `app/face_clustering_debug/pages/distance_lookup.py`

---

### T19: Implement parameter_tuning page
**Size:** M
**Depends on:** T6, T7, T8, T13

**Work:**
- Implement `render_parameter_tuning_page()` function
- Algorithm selector dropdown
- Dynamic parameter sliders
- "Run Clustering" button
- Show results: cluster count, sizes, compare to previous

**Acceptance Criteria:**
- [ ] All algorithm params adjustable via sliders
- [ ] Clustering runs when button clicked
- [ ] Results displayed after run
- [ ] < 150 lines

**Files:**
- `app/face_clustering_debug/pages/parameter_tuning.py`

---

### T20: Implement algorithm_comparison page
**Size:** M
**Depends on:** T7, T8

**Work:**
- Implement `render_algorithm_comparison_page()` function
- Side-by-side comparison of two methods
- Metrics comparison: cluster count, noise count, sizes
- Show which faces changed clusters

**Acceptance Criteria:**
- [ ] Two methods shown side-by-side
- [ ] Metrics comparison table
- [ ] Identifies differences
- [ ] < 150 lines

**Files:**
- `app/face_clustering_debug/pages/algorithm_comparison.py`

---

### T21: Pages __init__.py exports
**Size:** S
**Depends on:** T15-T20

**Work:**
- Export all page render functions from `pages/__init__.py`

**Acceptance Criteria:**
- [ ] All pages importable from `app.face_clustering_debug.pages`

**Files:**
- `app/face_clustering_debug/pages/__init__.py`

---

## Phase 5: Integration & Cleanup

### T22: Implement main.py entry point
**Size:** M
**Depends on:** T21

**Work:**
- Implement main entry point
- Sidebar: data source selection (Files vs DB)
- Sidebar: page navigation
- Route to appropriate page based on selection
- Handle errors gracefully

**Acceptance Criteria:**
- [ ] App launches with `streamlit run app/face_clustering_debug/main.py`
- [ ] Data source switching works
- [ ] All pages accessible via navigation
- [ ] < 80 lines

**Files:**
- `app/face_clustering_debug/main.py`

---

### T23: End-to-end testing
**Size:** M
**Depends on:** T22

**Work:**
- Manual testing of all pages
- Test with benchmark files
- Test with database (if data exists)
- Fix any integration bugs

**Acceptance Criteria:**
- [ ] All pages render without errors
- [ ] Face grid selection works
- [ ] Landmarks display correctly
- [ ] Parameter tuning runs clustering
- [ ] No import errors

**Files:**
- (No new files - bug fixes to existing)

---

### T24: Cleanup old files
**Size:** S
**Depends on:** T23

**Work:**
- Delete deprecated files:
  - `app/face_clustering_comparison.py`
  - `app/debug_hybrid_closest.py`
  - `scripts/debug_specific_faces.py`
  - `scripts/trace_face_83.py`
- Update CLAUDE.md with new app location
- Update CHANGES_LOG.md

**Acceptance Criteria:**
- [ ] Old files deleted
- [ ] Documentation updated
- [ ] Git commit with cleanup

**Files:**
- Delete 4 files
- Update `CLAUDE.md`
- Update `CHANGES_LOG.md`

---

## Task Dependency Graph

```
T1 (folder structure)
 │
 ├── T2 (schemas) ──────────────────────────────────┐
 │    │                                              │
 │    ├── T3 (protocols)                             │
 │    │    │                                         │
 │    │    ├── T4 (FileLoader) ──┐                   │
 │    │    │                     │                   │
 │    │    ├── T5 (DBLoader) ────┼── T7 (services init)
 │    │    │                     │         │
 │    │    └── T6 (ClusteringRunner)──┘    │
 │    │                                     │
 │    ├── T8 (face_grid) ───────────────────┼────────┐
 │    │                                     │        │
 │    ├── T9 (face_detail) ─────────────────┼────────┤
 │    │                                     │        │
 │    ├── T11 (threshold_display) ──────────┼────────┤
 │    │                                     │        │
 │    └── T12 (decision_card) ──────────────┼────────┤
 │                                          │        │
 ├── T10 (distance_heatmap) ────────────────┼────────┤
 │                                          │        │
 └── T13 (param_sliders) ───────────────────┼────────┤
                                            │        │
                              T14 (components init)  │
                                            │        │
                    ┌───────────────────────┴────────┘
                    │
    ┌───────────────┼───────────────┬────────────────┐
    │               │               │                │
T15 (overview)  T16 (merge)   T17 (attach)    T18 (distance)
    │               │               │                │
    │           T19 (param_tuning)  │                │
    │               │               │                │
    │           T20 (comparison)    │                │
    │               │               │                │
    └───────────────┴───────────────┴────────────────┘
                              │
                      T21 (pages init)
                              │
                      T22 (main.py)
                              │
                      T23 (e2e testing)
                              │
                      T24 (cleanup)
```

---

## Execution Order (Recommended)

**Batch 1 (Foundation):** T1, T2, T3
**Batch 2 (Services):** T4, T5, T6, T7
**Batch 3 (Components):** T8, T9, T10, T11, T12, T13, T14
**Batch 4 (Pages):** T15, T16, T17, T18, T19, T20, T21
**Batch 5 (Integration):** T22, T23, T24

---

## Size Estimates

| Size | Count | Estimated Lines per Task |
|------|-------|-------------------------|
| S (Small) | 14 | 30-60 lines |
| M (Medium) | 10 | 80-150 lines |

**Total estimated:** ~1050 lines of new code

---

## Approval

- [ ] Task breakdown approved by user
- [ ] Ready to begin implementation

