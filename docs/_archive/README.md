# Archived Documentation

**Status**: These documents are from the earlier sim-bench research/benchmarking phase (Oct-Nov 2025) and are preserved for historical reference. The code they reference may have been deprecated, moved to `sim_bench/legacy/`, or significantly refactored.

---

## Why Archived?

The project evolved from a research benchmarking tool to the **Album Organization App**. The new architecture uses:
- `sim_bench/album/` - Workflow orchestration
- `sim_bench/model_hub/` - Unified model coordination
- `app/album/` - Streamlit UI

## Current Documentation

For up-to-date documentation, see the parent `docs/` folder:
- [GETTING_STARTED.md](../GETTING_STARTED.md) - Quick start guide
- [ARCHITECTURE_INDEX.md](../ARCHITECTURE_INDEX.md) - Documentation navigation
- [ALBUM_APP_ARCHITECTURE.md](../ALBUM_APP_ARCHITECTURE.md) - Complete system design
- [FILE_DEPENDENCY_MAP.md](../FILE_DEPENDENCY_MAP.md) - Code connections
- [MODEL_USAGE_QUICK_REFERENCE.md](../MODEL_USAGE_QUICK_REFERENCE.md) - Model details

---

## Archive Contents

### Standalone Documents

| File | Description |
|------|-------------|
| AGENT_IMPLEMENTATION_STATUS.md | Gemini agent implementation status |
| QUALITY_ASSESSMENT_QUICKSTART.md | Old API examples (NIMA, ViT classes) |
| QUALITY_BENCHMARK_GUIDE.md | Benchmark evaluation guide |
| CLIP_AESTHETIC_SCORING.md | CLIP aesthetic scoring notes |
| WHY_CLIP_AESTHETIC_WORKS.md | Research notes on CLIP aesthetics |
| SYNTHETIC_DEGRADATION_TESTING.md | Synthetic test methodology |
| AGGREGATION_CODE_PATH.md | Aggregation code reference |
| EXPAND_JSON_COLUMN.md | JSON data processing |
| MULTI_MODEL_RANKING_PROPOSAL.md | Multi-model ranking design |
| photo_triage_attribute_contrastive_plan.md | Attribute-based triage plan |
| FACE_LANDMARK_BENCHMARKING.md | Face detection benchmarks |
| DEPENDENCIES.md | Old dependency documentation |

### Subdirectories

| Folder | Contents |
|--------|----------|
| architecture/ | Old agent/orchestration architecture docs |
| clustering/ | Clustering methods (KMeans, DBSCAN, HDBSCAN) |
| datasets/ | Dataset documentation (UKBench, Holidays, PhotoTriage) |
| development/ | Development experiments and plans |
| guides/ | Usage guides |
| quality_assessment/ | Quality assessment methods and benchmarks |

---

## Original Documentation Index

The documentation below is preserved but may reference deprecated code.

### Image Similarity / Retrieval
- Datasets: UKBench (10,200 images), Holidays (1,491 images), PhotoTriage (12,988 images)
- Methods: SIFT BoVW, Chi-Square, EMD, DINOv2, OpenCLIP

### Clustering
- Methods: KMeans, DBSCAN, HDBSCAN
- Visualization: HTML gallery generator

### Quality Assessment
- Methods: Rule-based (sharpness, contrast, exposure), CNN (NIMA), Transformers (MUSIQ)
- Benchmarks: Comprehensive comparison framework

### Architecture
- Feature caching, unified logging, dataset abstraction

---

*Archived: January 2026*
