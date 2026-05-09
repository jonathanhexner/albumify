# Face Clustering Documentation

**⚠️ Documentation Moved**

Face clustering documentation has been reorganized and moved to:

👉 **[face_cluster/docs/README.md](../face_cluster/docs/README.md)**

This consolidates all face clustering documentation with the code module.

---

## Quick Links

- [Architecture](../face_cluster/docs/ARCHITECTURE.md) - System overview and design
- [Getting Started](../face_cluster/docs/GETTING_STARTED.md) - Quick start guide
- [Expert Review](../face_cluster/docs/design/EXPERT_REVIEW.md) - Architecture review
- [Test Design](../face_cluster/docs/design/TEST_DESIGN_REVIEW.md) - Embedding validation tests
- [Workbench Guide](../face_cluster/docs/ui/workbench_guide.md) - Streamlit app
- [Troubleshooting](../face_cluster/docs/pipeline/troubleshooting.md) - Common issues

---

## Documentation Map

All face clustering docs are now organized under `face_cluster/docs/`:
- `design/` - Expert reviews, implementation plans, test designs
- `algorithms/` - KNN graph, hybrid clustering, quality gating
- `pipeline/` - Pipeline overview, steps, configuration
- `ui/` - Workbench, debug view, labeling app
- `workflows/` - ML training, benchmarking, experimentation
- `archive/` - Outdated/superseded documentation

---

## Distinction: Face vs Scene Clustering

**Face Clustering** (`face_cluster/`):
- Experimentation platform for face clustering algorithms
- Separate module, can be extracted as standalone package
- Documentation: `face_cluster/docs/`

**Scene Clustering** (main app):
- Production clustering for photo album organization
- Integrated in `sim_bench/pipeline/`
- Documentation: `docs/` and `docs/architecture/`

---

**Moved on**: 2026-03-28
