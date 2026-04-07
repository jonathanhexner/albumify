# Face Clustering Documentation

**Purpose**: Experimentation platform for face clustering algorithms (separate from main app scene clustering)

---

## Quick Links

- 📖 [Architecture](ARCHITECTURE.md) - System overview, components, data flow
- 🚀 [Getting Started](GETTING_STARTED.md) - Quick start guide
- ⚙️ [Configuration](pipeline/configuration.md) - YAML config reference (TODO)
- 🔧 [Troubleshooting](pipeline/troubleshooting.md) - Common issues

---

## Documentation Map

### 📐 Design & Planning
- [Expert Review](design/EXPERT_REVIEW.md) - Architecture review by CV researcher, SW engineer, UI expert
- [Implementation Recommendations](design/IMPLEMENTATION_RECOMMENDATIONS.md) - Prioritized action items from expert review
- [Test Design Review](design/TEST_DESIGN_REVIEW.md) - Embedding validation test design (reviewed by CV/SW/QA experts)
- [Test Proposal](design/TEST_PROPOSAL.md) - Original test proposal
- [Phase 1A Summary](design/PHASE_1A_SUMMARY.md) - Implementation summary for Phase 1A pipeline steps

### 🧮 Algorithms
- [KNN Graph Clustering](algorithms/knn_graph.md) - Mutual k-NN graph + connected components
- [Hybrid HDBSCAN+KNN](algorithms/hybrid_clustering.md) - Two-stage clustering approach
- Quality Gating (TODO) - Face quality filtering (pose, blur, area)
- Exemplar Selection (TODO) - d10-based representative face selection
- ML-Based Merging (TODO) - Machine learning cluster merging

### 🔄 Pipeline
- [Pipeline Overview](pipeline/overview.md) - Complete pipeline flow
- Pipeline Steps (TODO) - Individual step documentation
- Configuration Reference (TODO) - YAML config options
- [Troubleshooting](pipeline/troubleshooting.md) - Common issues and solutions

### 🖥️ UI
- [Workbench Guide](ui/workbench_guide.md) - Streamlit experimentation app
- [Debug View](ui/debug_view.md) - Distance visualization and cluster inspection
- [Debug App](ui/debug_app/) - Standalone debug application
- Labeling App (TODO) - Manual labeling interface

### 📊 Workflows
- [ML Training Guide](workflows/ml_training_guide.md) - Complete ML merge training workflow
- [ML Training Workflow](workflows/ml_training_workflow.md) - Step-by-step ML training process
- [Benchmarking Guide](workflows/benchmarking.md) - Face clustering benchmarking
- Experimentation Workflow (TODO) - Algorithm experimentation guide

### 📦 Archive
See [archive/README.md](archive/README.md) for outdated/superseded documentation.

---

## Module Overview

The `face_cluster/` module provides:
- **Face detection & embedding** - InsightFace-based face analysis
- **Quality gating** - Filter low-quality faces (pose, blur, size)
- **KNN graph clustering** - Mutual k-NN + connected components
- **Exemplar selection** - d10-based representative faces
- **Cluster merging** - Conservative heuristic + ML-based merging
- **Analysis tools** - Distance matrices, statistics, visualization

See [README.md](../README.md) in module root for API documentation.

---

## Distinction from Main App

**Main App** (`sim_bench/pipeline/`):
- Production face clustering for photo albums
- Scene clustering (NOT face clustering)
- Integrated with Streamlit UI and FastAPI backend

**Face Cluster Module** (`face_cluster/`):
- **Experimentation platform** for face clustering algorithms
- Algorithm development and parameter tuning
- ML training data generation
- Separate from main app, can be extracted as standalone package

---

## Getting Help

- **Architecture questions**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Setup issues**: See [GETTING_STARTED.md](GETTING_STARTED.md)
- **Pipeline errors**: See [pipeline/troubleshooting.md](pipeline/troubleshooting.md)
- **Missing docs**: Check [archive/](archive/) or create an issue

---

**Last Updated**: 2026-03-28
