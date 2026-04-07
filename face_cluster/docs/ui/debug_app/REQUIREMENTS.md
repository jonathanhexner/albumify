# Face Clustering Debug App - Requirements Document

**Date:** 2026-02-17
**Status:** Draft - Pending Approval
**Author:** Claude Code

---

## 1. Background & Problem Statement

### Current State Issues

1. **No Separation of Concerns**: The current `app/face_clustering_comparison.py` (1261 lines) mixes:
   - Data loading
   - UI rendering
   - Distance calculations
   - Clustering visualization
   - Algorithm re-implementation

2. **Algorithm Duplication**: The app re-implements clustering logic instead of:
   - Reading results from the database (`universal_cache`, `pipeline_results`, `people` tables)
   - Calling existing clustering modules (`sim_bench.clustering.*`)

3. **No Modularity**: Single monolithic files violate SOLID principles:
   - `face_clustering_comparison.py` - 1261 lines
   - `debug_hybrid_closest.py` - 402 lines
   - `benchmark_face_clustering.py` - 612 lines

4. **Scattered Files**: Related modules spread across:
   - `app/face_clustering_comparison.py`
   - `app/debug_hybrid_closest.py`
   - `scripts/benchmark_face_clustering.py`
   - `scripts/debug_specific_faces.py`
   - `scripts/trace_face_83.py`
   - `notebooks/debug_face_clusters.ipynb`
   - `notebooks/debug_face_analysis.ipynb`

5. **Broken Imports**: Import errors due to inconsistent module paths

---

## 2. Objectives

### Primary Goals

1. **Single Responsibility**: Each module does ONE thing well
2. **Consume, Don't Reimplement**: Use existing clustering APIs and database
3. **Clear Folder Structure**: Dedicated `app/face_clustering_debug/` directory
4. **Testable Components**: Business logic separated from UI
5. **Maintainable**: No file exceeds ~200 lines

### Success Criteria

- [ ] All face clustering debug functionality in one folder
- [ ] Zero algorithm reimplementation in UI code
- [ ] Each Python file < 250 lines
- [ ] All imports work without sys.path hacks
- [ ] Unit tests for data access layer

---

## 3. Functional Requirements

### FR-1: Data Sources

The app MUST support two data sources:

| Source | Description | Use Case |
|--------|-------------|----------|
| **Database** | Read from `universal_cache`, `people`, `pipeline_results` | Production: analyze completed pipeline runs |
| **Benchmark Files** | Read from JSON/NPY files in `results/` | Development: compare algorithm variants |

### FR-2: Core Features

| ID | Feature | Description |
|----|---------|-------------|
| FR-2.1 | Cluster Overview | Display all clusters with face thumbnails, exemplar markers, threshold values |
| FR-2.2 | Merge Decisions | Show why clusters did/didn't merge (threshold, pairs, distances) |
| FR-2.3 | Attachment Decisions | Show why noise points attached or remained noise |
| FR-2.4 | Distance Lookup | Query distance between any two faces |
| FR-2.5 | Parameter Tuning | Adjust params and re-run clustering (calls existing algorithm) |
| FR-2.6 | Algorithm Comparison | Side-by-side comparison of HDBSCAN vs Hybrid methods |

### FR-3: Clustering Execution

When user requests re-clustering:
- App MUST call `sim_bench.clustering.base.load_clustering_method(config)`
- App MUST NOT implement clustering logic
- App MAY pass `collect_debug_data=True` to get decision logs

### FR-4: Data Access

| Data | Source |
|------|--------|
| Face embeddings | `UniversalCache` table OR `.npy` file |
| Face metadata (bbox, confidence) | `UniversalCache` table OR `.json` file |
| Cluster labels | `people` table OR clustering result dict |
| Merge/attach decisions | Clustering `stats` dict (when `collect_debug_data=True`) |

---

## 4. Non-Functional Requirements

### NFR-1: Code Organization

```
app/face_clustering_debug/
├── __init__.py
├── main.py                    # Streamlit entry point (<100 lines)
├── pages/
│   ├── __init__.py
│   ├── overview.py            # Cluster overview page
│   ├── merge_decisions.py     # Merge analysis page
│   ├── attach_decisions.py    # Attachment analysis page
│   ├── distance_lookup.py     # Distance query page
│   ├── parameter_tuning.py    # Re-run with new params
│   └── comparison.py          # Side-by-side comparison
├── services/
│   ├── __init__.py
│   ├── data_loader.py         # Load from DB or files (interface)
│   ├── db_loader.py           # Database implementation
│   ├── file_loader.py         # JSON/NPY file implementation
│   └── clustering_runner.py   # Wrapper around sim_bench.clustering
├── components/
│   ├── __init__.py
│   ├── face_grid.py           # Reusable face thumbnail grid
│   ├── distance_matrix.py     # Heatmap component
│   ├── threshold_chart.py     # Threshold visualization
│   └── decision_card.py       # Merge/attach decision display
└── models/
    ├── __init__.py
    └── schemas.py             # Pydantic models for data transfer
```

### NFR-2: File Size Limits

| File Type | Max Lines |
|-----------|-----------|
| Page modules | 200 |
| Service modules | 150 |
| Component modules | 100 |
| Main entry point | 100 |

### NFR-3: Dependencies

- UI code imports ONLY from `services/` and `components/`
- Services import from `sim_bench.*` (existing modules)
- No circular imports
- No `sys.path` manipulation

### NFR-4: Testing

| Layer | Test Type |
|-------|-----------|
| Services | Unit tests with mocked DB |
| Components | Snapshot tests (optional) |
| Integration | Manual testing via Streamlit |

---

## 5. Out of Scope

- Modifying existing clustering algorithms
- Changing database schema
- Modifying the main album app (`app/streamlit/`)
- Training or model changes

---

## 6. Migration Plan

### Phase 1: Setup Structure
- Create `app/face_clustering_debug/` folder
- Create empty module files with docstrings

### Phase 2: Extract Services
- Implement `data_loader.py` interface
- Implement `db_loader.py` (read from database)
- Implement `file_loader.py` (read from benchmark files)
- Implement `clustering_runner.py` (wrapper)

### Phase 3: Extract Components
- Extract reusable UI components from monolith
- Each component: single responsibility, < 100 lines

### Phase 4: Build Pages
- One page per feature
- Pages use services + components only

### Phase 5: Wire Up Main
- Create `main.py` entry point
- Add navigation between pages

### Phase 6: Cleanup
- Delete old monolithic files
- Update documentation
- Add to CLAUDE.md

---

## 7. Files to Deprecate

After migration, these files will be deleted:

| File | Reason |
|------|--------|
| `app/face_clustering_comparison.py` | Replaced by modular app |
| `app/debug_hybrid_closest.py` | Merged into pages |
| `scripts/debug_specific_faces.py` | Functionality in new app |
| `scripts/trace_face_83.py` | One-off debug script |

Keep (but may refactor):
| File | Reason |
|------|--------|
| `scripts/benchmark_face_clustering.py` | Produces benchmark data (separate concern) |

---

## 8. Open Questions

1. Should we keep JSON/NPY file support long-term, or migrate everything to DB?
2. Should parameter tuning write results back to DB?
3. Do we need authentication/multi-user support?

---

## Approval

- [ ] Requirements approved by user
- [ ] Ready to proceed to Architecture document

