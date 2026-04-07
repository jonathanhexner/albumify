# Face Clustering Documentation Organization Proposal

**Date**: 2026-03-28

---

## Current State (Scattered Documentation)

### Root Directory (11 files)
- ARCHITECTURE_DOCUMENTATION_COMPLETE.md
- ARCHITECTURE_EXPERT_REVIEW.md
- CLUSTER_DEBUG_VIEW_COMPLETE.md
- CURRENT_IMPLEMENTATION_STATUS.md
- FACE_CLUSTERING_ARCHITECTURE.md ⭐
- FACE_CLUSTERING_WORKBENCH_GUIDE.md
- IMPLEMENTATION_RECOMMENDATIONS.md ⭐
- IMPLEMENTATION_SUMMARY.md
- PHASE_1A_IMPLEMENTATION_SUMMARY.md ⭐
- README_CLUSTERING_BENCHMARK.md
- TROUBLESHOOTING_CLUSTERING.md

### docs/ Directory (15 files)
- docs/FACE_CLUSTERING_COMPLETE_GUIDE.md
- docs/FACE_CLUSTERING_WORKFLOW.md
- docs/FACIAL_CLUSTERING_DEBUG.md
- docs/HYBRID_CLUSTERING.md
- docs/KNN_CLUSTERING_PIPELINE.md
- docs/ML_CLUSTER_MERGING_GUIDE.md
- docs/ML_CLUSTER_MERGING_WORKFLOW.md
- docs/PLAN_ML_CLUSTER_MERGING.md
- docs/SPRINT_PLANS_CLUSTERING_DEBUG.md
- docs/FACE_FILTERING_PLAN.md
- docs/FACE_MANAGEMENT_MODULES.md
- docs/FACE_MANAGEMENT_UI_PLAN.md
- docs/FACE_RECOGNITION_FIX_PLAN.md
- docs/PROPOSAL_FACE_ALIGNMENT_REFACTOR.md
- docs/face_clustering_debug_app/ (4 files)

### docs/architecture/ (5 files)
- docs/architecture/COMMON_INTERFACE_SOLUTION.md
- docs/architecture/FACE_DATA_FLOW.md
- docs/architecture/FACE_PIPELINE_PLAN.md
- docs/architecture/INSIGHTFACE_PIPELINE.md
- docs/architecture/INSIGHTFACE_SUMMARY.md

### face_cluster/ (1 file)
- face_cluster/README.md

### Recent (3 files just created)
- TEST_PROPOSAL_EMBEDDING_VALIDATION.md ⭐
- EXPERT_REVIEW_EMBEDDING_VALIDATION_TEST.md ⭐

**Total**: 35+ face clustering related docs scattered across 5 locations

---

## Proposed Structure

### Option A: Consolidate under face_cluster/docs/

```
face_cluster/
├── docs/
│   ├── README.md                          # Entry point, links to all docs
│   ├── ARCHITECTURE.md                    # Main architecture (from FACE_CLUSTERING_ARCHITECTURE.md)
│   ├── GETTING_STARTED.md                 # Quick start guide
│   │
│   ├── design/                            # Design decisions and reviews
│   │   ├── EXPERT_REVIEW.md               # Architecture expert review
│   │   ├── IMPLEMENTATION_RECOMMENDATIONS.md
│   │   ├── TEST_DESIGN_REVIEW.md          # Embedding validation test review
│   │   └── PHASE_1A_SUMMARY.md            # Implementation summary
│   │
│   ├── algorithms/                        # Algorithm documentation
│   │   ├── knn_graph.md                   # Mutual kNN graph clustering
│   │   ├── hybrid_clustering.md           # HDBSCAN + kNN hybrid
│   │   ├── quality_gating.md              # Face quality filtering
│   │   ├── exemplar_selection.md          # d10-based exemplars
│   │   └── ml_merging.md                  # ML-based cluster merging
│   │
│   ├── pipeline/                          # Pipeline documentation
│   │   ├── overview.md                    # Pipeline flow diagram
│   │   ├── steps.md                       # Step-by-step guide
│   │   ├── configuration.md               # YAML config reference
│   │   └── troubleshooting.md             # Common issues
│   │
│   ├── ui/                                # UI documentation
│   │   ├── workbench_guide.md             # Streamlit workbench
│   │   ├── labeling_app.md                # Manual labeling interface
│   │   └── debug_view.md                  # Debug visualization
│   │
│   ├── workflows/                         # Complete workflows
│   │   ├── experimentation.md             # Algorithm experimentation
│   │   ├── ml_training.md                 # ML merge training workflow
│   │   └── benchmarking.md                # Benchmarking guide
│   │
│   └── archive/                           # Outdated/superseded docs
│       ├── sprint_plans/
│       ├── old_proposals/
│       └── README.md                      # What's archived and why
│
├── README.md                              # Module overview (existing)
├── types.py
├── embedding.py
└── ...
```

### Option B: Consolidate under docs/face_clustering/

```
docs/
├── face_clustering/                       # All face clustering docs
│   ├── README.md                          # Entry point
│   ├── architecture/                      # Same structure as Option A
│   ├── design/
│   ├── algorithms/
│   ├── pipeline/
│   ├── ui/
│   ├── workflows/
│   └── archive/
│
├── architecture/                          # General architecture docs (non-face)
├── _archive/                              # General archive
└── ...
```

---

## Recommendation: **Option A** (face_cluster/docs/)

**Reasons:**
1. **Colocation** - Documentation lives with the code it documents
2. **Module independence** - face_cluster/ can be extracted as standalone package
3. **Clear ownership** - Anyone working on face_cluster/ knows where docs are
4. **Less clutter** - Keeps root directory clean

**Trade-off:**
- Con: General project docs are in docs/, but face clustering docs are in face_cluster/docs/
- Mitigation: Add `docs/face_clustering.md` as a **redirect file** pointing to face_cluster/docs/README.md

---

## Migration Plan

### Phase 1: Create Structure (5 min)
```bash
mkdir -p face_cluster/docs/{design,algorithms,pipeline,ui,workflows,archive}
touch face_cluster/docs/README.md
```

### Phase 2: Move & Categorize (20 min)

**design/** (5 docs):
- ARCHITECTURE_EXPERT_REVIEW.md → design/EXPERT_REVIEW.md
- IMPLEMENTATION_RECOMMENDATIONS.md → design/IMPLEMENTATION_RECOMMENDATIONS.md
- EXPERT_REVIEW_EMBEDDING_VALIDATION_TEST.md → design/TEST_DESIGN_REVIEW.md
- PHASE_1A_IMPLEMENTATION_SUMMARY.md → design/PHASE_1A_SUMMARY.md
- TEST_PROPOSAL_EMBEDDING_VALIDATION.md → design/TEST_PROPOSAL.md

**Main docs** (3 docs):
- FACE_CLUSTERING_ARCHITECTURE.md → face_cluster/docs/ARCHITECTURE.md
- FACE_CLUSTERING_WORKBENCH_GUIDE.md → ui/workbench_guide.md
- docs/FACE_CLUSTERING_COMPLETE_GUIDE.md → GETTING_STARTED.md

**algorithms/** (3 docs):
- docs/HYBRID_CLUSTERING.md → algorithms/hybrid_clustering.md
- docs/KNN_CLUSTERING_PIPELINE.md → algorithms/knn_graph.md
- docs/ML_CLUSTER_MERGING_GUIDE.md → workflows/ml_training.md

**pipeline/** (2 docs):
- TROUBLESHOOTING_CLUSTERING.md → pipeline/troubleshooting.md
- docs/FACE_CLUSTERING_WORKFLOW.md → pipeline/overview.md

**ui/** (2 docs):
- docs/face_clustering_debug_app/ → ui/debug_app/
- CLUSTER_DEBUG_VIEW_COMPLETE.md → ui/debug_view.md

**workflows/** (2 docs):
- docs/ML_CLUSTER_MERGING_WORKFLOW.md → workflows/ml_training_workflow.md
- README_CLUSTERING_BENCHMARK.md → workflows/benchmarking.md

**archive/** (15+ docs):
- Move outdated/superseded docs:
  - IMPLEMENTATION_SUMMARY.md (superseded by PHASE_1A_SUMMARY.md)
  - CURRENT_IMPLEMENTATION_STATUS.md (outdated)
  - docs/SPRINT_PLANS_CLUSTERING_DEBUG.md
  - docs/FACIAL_CLUSTERING_DEBUG.md
  - docs/PLAN_ML_CLUSTER_MERGING.md
  - docs/FACE_*_PLAN.md (all old plans)
  - docs/PROPOSAL_FACE_ALIGNMENT_REFACTOR.md

### Phase 3: Create Entry Point (10 min)

**face_cluster/docs/README.md**:
```markdown
# Face Clustering Documentation

## Quick Links
- [Architecture](ARCHITECTURE.md) - System overview, components, data flow
- [Getting Started](GETTING_STARTED.md) - Quickstart guide
- [Configuration](pipeline/configuration.md) - YAML config reference

## Documentation Map

### Design & Planning
- [Expert Review](design/EXPERT_REVIEW.md) - Architecture review by experts
- [Implementation Recommendations](design/IMPLEMENTATION_RECOMMENDATIONS.md)
- [Test Design Review](design/TEST_DESIGN_REVIEW.md) - Embedding validation tests

### Algorithms
- [KNN Graph Clustering](algorithms/knn_graph.md)
- [Hybrid HDBSCAN+KNN](algorithms/hybrid_clustering.md)
- [Quality Gating](algorithms/quality_gating.md)
- [ML-Based Merging](algorithms/ml_merging.md)

### Pipeline
- [Pipeline Overview](pipeline/overview.md) - Complete pipeline flow
- [Pipeline Steps](pipeline/steps.md) - Individual step documentation
- [Configuration](pipeline/configuration.md) - YAML config options
- [Troubleshooting](pipeline/troubleshooting.md) - Common issues

### UI
- [Workbench Guide](ui/workbench_guide.md) - Streamlit experimentation app
- [Labeling App](ui/labeling_app.md) - Manual labeling interface
- [Debug View](ui/debug_view.md) - Distance visualization

### Workflows
- [Experimentation Workflow](workflows/experimentation.md)
- [ML Training Workflow](workflows/ml_training.md)
- [Benchmarking Guide](workflows/benchmarking.md)

## Archive
See [archive/README.md](archive/README.md) for outdated documentation.
```

### Phase 4: Create Redirect (2 min)

**docs/face_clustering.md**:
```markdown
# Face Clustering Documentation

**Note:** Face clustering documentation has moved to:

👉 **[face_cluster/docs/README.md](../face_cluster/docs/README.md)**

This consolidates all face clustering documentation with the code module.

## Quick Links
- [Architecture](../face_cluster/docs/ARCHITECTURE.md)
- [Getting Started](../face_cluster/docs/GETTING_STARTED.md)
- [Pipeline Configuration](../face_cluster/docs/pipeline/configuration.md)
```

### Phase 5: Update References (10 min)
- Update CLAUDE.md to reference face_cluster/docs/
- Update main README.md to point to face_cluster/docs/
- Search for broken links in remaining docs

---

## Missing Documentation (To Create)

Based on scan, we're **missing**:
1. ✅ **Test design documentation** - Now created (TEST_DESIGN_REVIEW.md)
2. ❌ **Quality gating algorithm** - Need algorithms/quality_gating.md
3. ❌ **Exemplar selection algorithm** - Need algorithms/exemplar_selection.md
4. ❌ **Pipeline configuration reference** - Need pipeline/configuration.md
5. ❌ **Labeling app guide** - Need ui/labeling_app.md

---

## Summary

**Current**: 35+ docs scattered across 5 locations
**Proposed**: ~25 active docs + 15 archived, organized in face_cluster/docs/

**Benefits**:
- ✅ Single source of truth
- ✅ Logical categorization
- ✅ Clear what's current vs archived
- ✅ Easy to find documentation
- ✅ Module can be extracted as standalone

**Next Steps**:
1. Get user approval
2. Execute migration plan (45 min total)
3. Verify all links work
4. Archive outdated docs with README explaining why
