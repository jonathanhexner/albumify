# Face Clustering System Architecture

**Version:** 2026-03-27
**Purpose:** Experimentation platform for face clustering algorithms

---

## 1. Requirements

### Primary Use Case
**Algorithm experimentation** - Test different clustering approaches, parameters, and merge strategies to find optimal settings before integrating into production.

### Key Requirements
1. **Detect faces** in photo albums and extract embeddings
2. **Cluster faces** by identity using various algorithms
3. **Manual labeling** - Correct clustering mistakes for ground truth
4. **ML training** - Train merge classifier from labeled data
5. **Export results** - Generate training datasets for ML

### Non-Requirements
- ❌ Real-time performance (batch processing is fine)
- ❌ Integration with main album app (separate tool)
- ❌ Web API (local Streamlit app sufficient)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Album Application                        │
│  (app/streamlit/main.py + sim_bench/api/ + sim_bench/pipeline/) │
│                                                                  │
│  Uses: sim_bench/clustering/* for production clustering          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (separate - no shared code)
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│              Face Clustering Experimentation                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  face_cluster/ (Core Algorithms)                         │  │
│  │  ├── embedding.py      - InsightFaceEmbedder            │  │
│  │  ├── quality.py        - QualityGater                   │  │
│  │  ├── knn_graph.py      - KNNGraphBuilder                │  │
│  │  ├── clustering.py     - ConnectedComponentsClusterer   │  │
│  │  ├── exemplars.py      - D10ExemplarSelector            │  │
│  │  ├── merge.py          - ConservativeMerger, MLMerger   │  │
│  │  ├── attach.py         - HoldoutAttacher                │  │
│  │  ├── analysis.py       - ClusterSnapshot, statistics    │  │
│  │  ├── features.py       - FeatureComputer                │  │
│  │  └── types.py          - FaceRecord, ClusterResult      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ▲                                      │
│                           │                                      │
│  ┌────────────────────────┴─────────────────────────────────┐  │
│  │  User Interfaces                                          │  │
│  │  ├── app/face_clustering_workbench.py (Streamlit)        │  │
│  │  ├── notebooks/debug_*.ipynb                             │  │
│  │  └── scripts/export_clustering_data.py                   │  │
│  └──thi────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Output: results/run_*/                                          │
│  └── faces.csv, clusters.csv, face_crops/, labels.csv           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Responsibilities

| Module | Purpose | Input | Output | Used By |
|--------|---------|-------|--------|---------|
| **InsightFaceEmbedder** | Detect faces, extract 512-dim embeddings | Image paths | `List[FaceRecord]` | All pipelines |
| **QualityGater** | Filter low-quality faces (pose, blur) | `List[FaceRecord]` | `(core_indices, holdout_indices)` | Clustering pipeline |
| **KNNGraphBuilder** | Build mutual k-NN graph | Embeddings array | `GraphResult` (edges, neighbors) | Clustering |
| **ConnectedComponentsClusterer** | Cluster via graph components | `GraphResult` | `ClusterResult` (labels, clusters) | Clustering |
| **D10ExemplarSelector** | Select representative faces | `ClusterResult` + embeddings | Updated `ClusterResult` | Merge stage |
| **ConservativeMerger** | Merge similar clusters | `ClusterResult` + embeddings | Merged `ClusterResult` | Post-clustering |
| **HoldoutAttacher** | Attach noise to clusters | `ClusterResult` + holdout faces | Updated `ClusterResult` | Optional |
| **FeatureComputer** | Compute merge features | Cluster pairs + embeddings | `ClusterPairFeatures` | ML training |

---

## 4. Key Data Structures

### FaceRecord
```python
@dataclass
class FaceRecord:
    face_id: int                    # Unique ID
    image_id: str                   # Source image stem
    image_path: Path                # Full image path
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    landmarks: np.ndarray           # 5-point landmarks
    aligned_face: np.ndarray        # 112x112 aligned crop
    embedding: np.ndarray           # 512-dim normalized vector
    is_core: bool = True            # Passed quality gate?
    pose: Optional[Tuple] = None    # (yaw, pitch, roll)
```

### ClusterResult
```python
@dataclass
class ClusterResult:
    labels: np.ndarray              # Cluster ID per face (-1 = noise)
    clusters: Dict[int, List[int]]  # cluster_id -> face indices
    exemplars: Dict[int, List[int]] # cluster_id -> exemplar indices
    n_clusters: int
    n_noise: int
```

### PipelineConfig
```python
@dataclass
class PipelineConfig:
    K: int = 5                      # kNN neighbors
    distance_threshold: float = 0.35  # Max distance for edge
    min_cluster_size: int = 2
    # + 30+ other parameters for quality, merge, attach...
```

---

## 5. Typical Workflow

### Experimentation Pipeline
```python
# 1. Detect & Embed
embedder = InsightFaceEmbedder(model_name='buffalo_l')
faces = embedder.detect_and_embed(image_paths)

# 2. Quality Gate
config = PipelineConfig(K=5, distance_threshold=0.35)
gater = QualityGater(config)
core_idx, holdout_idx = gater.select_core_set(faces)
core_faces = [faces[i] for i in core_idx]

# 3. Extract embeddings
embeddings = np.array([f.embedding for f in core_faces])

# 4. Build kNN graph
graph_builder = KNNGraphBuilder(k=config.K, distance_threshold=config.distance_threshold)
graph = graph_builder.build(embeddings)

# 5. Cluster
clusterer = ConnectedComponentsClusterer()
result = clusterer.cluster(graph, len(embeddings))

# 6. Select exemplars
exemplar_selector = D10ExemplarSelector()
result = exemplar_selector.select(result, embeddings)

# 7. Merge similar clusters
merger = ConservativeMerger()
result = merger.merge(result, embeddings)

# 8. Export
# Save to faces.csv, clusters.csv, face_crops/
```

---

## 6. File Structure & Outputs

### Input
```
album/
├── photo1.jpg
├── photo2.jpg
└── ...
```

### Output Structure
```
results/run_YYYYMMDD_HHMMSS/
├── faces.csv              # One row per face
│   ├── face_id           # Unique ID
│   ├── image_path        # Source image
│   ├── cluster_id        # Assigned cluster (-1 = noise)
│   ├── is_core           # Passed quality gate
│   └── bbox_x, bbox_y    # Face location
│
├── clusters.csv           # One row per cluster
│   ├── cluster_id
│   ├── size              # Number of faces
│   ├── diameter          # Max intra-cluster distance
│   └── exemplar_face_id  # Representative face
│
├── face_crops/            # Aligned 112x112 crops
│   ├── face_0000_aligned.jpg
│   ├── face_0001_aligned.jpg
│   └── ...
│
├── export_summary.json    # Run metadata
│   ├── timestamp
│   ├── source_directory
│   ├── n_faces, n_clusters, n_noise
│   └── config (K, distance_threshold, etc.)
│
└── corrected_labels.csv   # Manual corrections (after labeling)
    ├── cluster_id
    └── corrected_identity  # User-assigned name
```

---

## 7. Interfaces Summary

### InsightFaceEmbedder
- `detect_and_embed(image_paths: List[str]) -> List[FaceRecord]`

### QualityGater
- `__init__(config: PipelineConfig)`
- `select_core_set(faces: List[FaceRecord]) -> Tuple[List[int], List[int]]`

### KNNGraphBuilder
- `__init__(k: int, distance_threshold: float)`
- `build(embeddings: np.ndarray) -> GraphResult`

### ConnectedComponentsClusterer
- `cluster(graph: GraphResult, n_faces: int) -> ClusterResult`

### D10ExemplarSelector
- `select(result: ClusterResult, embeddings: np.ndarray) -> ClusterResult`

### ConservativeMerger
- `merge(result: ClusterResult, embeddings: np.ndarray) -> ClusterResult`

---

## 8. Usage: Streamlit Workbench

```bash
streamlit run app/face_clustering_workbench.py
```

**Features:**
- Tab 1: Process Album - Run full pipeline with live progress
- Tab 2: View Results - Browse clusters, quality metrics
- Tab 3: Label Clusters - Manual corrections for ground truth
- Tab 4: History - Load previous runs

**Self-documenting:**
- Sidebar shows which module is used at each stage
- Module descriptions (purpose, input, output, file location)
- History automatically tracked in `results/.face_clustering_history.json`

---

## 9. Key Design Decisions

### Why Two Systems?
- **Main app** (`sim_bench/`) - Production, stable, integrated
- **Experimentation** (`face_cluster/`) - Fast iteration, parameter tuning, ML training

### Why File-Based?
- Each stage writes output before next stage starts
- If stage N crashes, stages 1..N-1 are recoverable
- Easy debugging (inspect intermediate outputs)
- Clear lineage tracing

### Why Separate Quality Gate?
- Many faces are low-quality (side view, blur, tiny)
- Clustering noise hurts accuracy
- Core set = high-quality faces for clustering
- Holdout set = attach after clustering (optional)

### Why Mutual kNN?
- More robust than single-linkage
- Prevents chain effect
- Interpretable (can visualize graph)
- Good for small batches (10-1000 faces)

---

## 10. Current Status

✅ **Working:**
- `face_cluster/` modules all implemented
- Streamlit workbench app created
- Module documentation in sidebar
- History tracking

⚠️ **In Progress:**
- Testing full pipeline end-to-end
- Fixing API mismatches (return types)

❌ **Not Started:**
- ML merge classifier training (Phase 3)
- Integration into main app

---

## 11. Next Steps

1. **Complete testing** - Verify workbench runs on test data
2. **Fix API issues** - Align method signatures with actual usage
3. **Document API** - Add docstrings with concrete examples
4. **Run experiments** - Test different K, distance_threshold values
5. **Generate training data** - Export labeled clusters
6. **Train ML classifier** - Use labeled data for merge model

---

**Last Updated:** 2026-03-27
**Owner:** Face Clustering Experimentation
**Contact:** See `face_cluster/README.md`
