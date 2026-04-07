# Pipeline Steps Reference

**Purpose**: Detailed documentation of each pipeline step

---

## Pipeline Flow

```
discover_images
    ↓
insightface_detect_faces
    ↓
align_faces
    ↓
extract_face_embeddings
    ↓
filter_quality_gate  ← Face clustering steps start here
    ↓
build_knn_graph
    ↓
cluster_connected_components
    ↓
select_exemplars
    ↓
compute_debug_distances
    ↓
export_for_labeling
```

---

## Step Documentation

### 1. discover_images

**File**: `sim_bench/pipeline/steps/discover_images.py`

**Purpose**: Scan album directory for image files

**Inputs**: None (uses `context.source_directory`)

**Outputs**:
- `context.image_paths`: List of image file paths

**Config**:
```yaml
discover_images:
  extensions: [".jpg", ".jpeg", ".png", ".heic"]
  recursive: true
```

**Typical Output**: 100-1000 image paths

---

### 2. insightface_detect_faces

**File**: `sim_bench/pipeline/steps/insightface_detect_faces.py`

**Purpose**: Detect faces using InsightFace SCRFD model

**Inputs**:
- `context.image_paths`

**Outputs**:
- `context.insightface_faces`: Dict mapping image_path → face data
  ```python
  {
    "image1.jpg": {
      "faces": [
        {
          "face_index": 0,
          "bbox": {"x_px": 100, "y_px": 200, "w_px": 150, "h_px": 150},
          "landmarks": [[x1, y1], [x2, y2], ...],  # 5-point
          "confidence": 0.98
        },
        ...
      ]
    },
    ...
  }
  ```

**Config**:
```yaml
insightface_detect_faces:
  model_name: 'buffalo_l'
  det_size: [640, 640]
  conf_threshold: 0.5
```

**Typical Output**: 0-10 faces per image

---

### 3. align_faces

**File**: `sim_bench/pipeline/steps/align_faces.py`

**Purpose**: Align face crops to canonical pose (112×112)

**Inputs**:
- `context.insightface_faces`
- Raw images

**Outputs**:
- `context.aligned_faces`: Dict mapping face_key → aligned crop (112×112 RGB)
  ```python
  {
    "image1.jpg:face_0": np.ndarray (112, 112, 3),
    "image1.jpg:face_1": np.ndarray (112, 112, 3),
    ...
  }
  ```

**Config**:
```yaml
align_faces:
  target_size: 112
  use_insightface_alignment: true
```

**Note**: Face key format is `"{image_path}:face_{face_index}"`

---

### 4. extract_face_embeddings

**File**: `sim_bench/pipeline/steps/extract_face_embeddings.py`

**Purpose**: Extract 512-dim ArcFace embeddings from aligned crops

**Inputs**:
- `context.aligned_faces`
- `context.insightface_faces` (for metadata)

**Outputs**:
- `context.face_embeddings`: Dict mapping face_key → embedding (512-dim)
  ```python
  {
    "image1.jpg:face_0": np.array([0.023, -0.145, ...]),  # 512-dim
    ...
  }
  ```

**Config**:
```yaml
extract_face_embeddings:
  backend: 'insightface'
  normalize: true
  verify_norm: true
```

**Validation**: Checks all embeddings are L2-normalized (norm ≈ 1.0)

---

### 5. filter_quality_gate ⭐

**File**: `sim_bench/pipeline/steps/filter_quality_gate.py`

**Purpose**: Filter low-quality faces (pose, blur, area)

**Inputs**:
- `context.face_embeddings`
- `context.aligned_faces`
- `context.insightface_faces`

**Outputs**:
- `context.core_indices`: List of high-quality face indices (for clustering)
- `context.holdout_indices`: List of low-quality face indices (excluded)
- `context.face_records`: List of `FaceRecord` objects

**Config**:
```yaml
filter_quality_gate:
  yaw_max: 45.0
  pitch_max: 30.0
  roll_max: 30.0
  blur_min: 100.0
  max_faces_per_image_core: 10
```

**Typical Output**: 60-80% of faces pass to core set

**Validation**: Asserts `len(core_indices) > 0`

---

### 6. build_knn_graph ⭐

**File**: `sim_bench/pipeline/steps/build_knn_graph.py`

**Purpose**: Build mutual k-NN graph for clustering

**Inputs**:
- `context.face_records`
- `context.core_indices`

**Outputs**:
- `context.knn_graph_result`: `GraphResult` object
  ```python
  GraphResult(
    neighbors: List[List[int]],          # k-NN neighbors per node
    neighbor_distances: List[List[float]], # Corresponding distances
    edges: List[Tuple[int, int, float]], # (i, j, distance) for mutual edges
    G: nx.Graph,                          # NetworkX graph
    distance_matrix: np.ndarray           # Full distance matrix (n×n)
  )
  ```

**Config**:
```yaml
build_knn_graph:
  K: 5
  distance_threshold: 0.35
```

**Typical Output**: 3-8 edges per node

**Validation**: Checks embeddings are normalized (0.9 < norm < 1.1)

---

### 7. cluster_connected_components ⭐

**File**: `sim_bench/pipeline/steps/cluster_connected_components.py`

**Purpose**: Find connected components in kNN graph to form clusters

**Inputs**:
- `context.knn_graph_result`
- `context.core_indices`

**Outputs**:
- `context.initial_clusters`: `ClusterResult` object
  ```python
  ClusterResult(
    labels: np.ndarray,                     # Cluster ID per face (-1 = noise)
    clusters: Dict[int, List[int]],         # cluster_id → face indices
    cluster_stats: Dict[int, Dict],         # cluster_id → statistics
    exemplars: Dict[int, List[int]],        # cluster_id → exemplar indices (empty initially)
    n_clusters: int,
    n_noise: int
  )
  ```

**Config**:
```yaml
cluster_connected_components:
  min_cluster_size: 2
```

**Typical Output**: 10-50 clusters depending on album size

---

### 8. select_exemplars ⭐

**File**: `sim_bench/pipeline/steps/select_exemplars.py`

**Purpose**: Select representative faces for each cluster

**Inputs**:
- `context.initial_clusters`
- `context.knn_graph_result`

**Outputs**:
- `context.initial_clusters` (updated with exemplars)

**Config**:
```yaml
select_exemplars:
  d10_k: 10
  exemplars_d10_threshold: 0.25
  exemplar_suppression_radius: 0.15
  N_exemplars_max: 5
```

**Typical Output**: 1-5 exemplars per cluster

---

### 9. compute_debug_distances ⭐

**File**: `sim_bench/pipeline/steps/compute_debug_distances.py`

**Purpose**: Pre-compute neighbor distances for debug UI

**Inputs**:
- `context.initial_clusters`
- `context.knn_graph_result`
- `context.face_records`

**Outputs**:
- `context.debug_neighbors`: Dict with neighbor information
  ```python
  {
    'within_closest': {
      0: [(1, 0.12), (3, 0.15), ...],  # face_id → [(neighbor_id, distance), ...]
      ...
    },
    'within_furthest': {
      0: [(5, 0.42), (7, 0.38), ...],
      ...
    },
    'cross_cluster': {
      0: [(10, 1, 0.55), ...],  # (neighbor_id, cluster_id, distance)
      ...
    },
    'exemplar_distances': {
      (0, 1): 0.45,  # (cluster_i, cluster_j) → min exemplar distance
      ...
    }
  }
  ```

**Config**:
```yaml
compute_debug_distances:
  n_closest_within: 5
  n_furthest_within: 5
  n_closest_cross: 5
```

**Validation**: Asserts all core faces have neighbors (unless single-face cluster)

---

### 10. export_for_labeling ⭐

**File**: `sim_bench/pipeline/steps/export_for_labeling.py`

**Purpose**: Export clustering results for manual labeling

**Inputs**:
- `context.face_records`
- `context.initial_clusters`
- `context.debug_neighbors`
- `context.aligned_faces`

**Outputs** (files written to disk):
- `faces.csv`: Face metadata with cluster assignments
- `clusters.csv`: Cluster statistics
- `face_crops/`: Aligned 112×112 crops (JPEG)
- `debug_neighbors.json`: Pre-computed neighbors
- `export_summary.json`: Metadata and validation results

**Context outputs**:
- `context.export_directory`: Output directory path
- `context.faces_csv_path`: Path to faces.csv
- `context.clusters_csv_path`: Path to clusters.csv

**Config**:
```yaml
export_for_labeling:
  output_dir: null  # Set via CLI
```

**Validation**: Checks 5 criteria in export_summary.json

---

## Step Dependencies

### Automatic Resolution

Pipeline executor automatically resolves dependencies:

```python
# You specify:
steps = ['export_for_labeling']

# Executor resolves to:
[
  'discover_images',
  'insightface_detect_faces',
  'align_faces',
  'extract_face_embeddings',
  'filter_quality_gate',
  'build_knn_graph',
  'cluster_connected_components',
  'select_exemplars',
  'compute_debug_distances',
  'export_for_labeling'
]
```

### Dependency Graph

```
discover_images
    ↓
insightface_detect_faces (depends_on: discover_images)
    ↓
align_faces (depends_on: insightface_detect_faces)
    ↓
extract_face_embeddings (depends_on: align_faces)
    ↓
filter_quality_gate (depends_on: extract_face_embeddings)
    ↓
build_knn_graph (depends_on: filter_quality_gate)
    ↓
cluster_connected_components (depends_on: build_knn_graph)
    ↓
select_exemplars (depends_on: cluster_connected_components)
    ↓
compute_debug_distances (depends_on: select_exemplars)
    ↓
export_for_labeling (depends_on: compute_debug_distances)
```

---

## Context Object

### Key Fields

```python
class PipelineContext:
    # Input
    source_directory: Path

    # Image discovery
    image_paths: List[Path]

    # Face detection
    insightface_faces: Dict[str, Dict]

    # Alignment
    aligned_faces: Dict[str, np.ndarray]

    # Embeddings
    face_embeddings: Dict[str, np.ndarray]

    # Quality gate
    core_indices: List[int]
    holdout_indices: List[int]
    face_records: List[FaceRecord]

    # Clustering
    knn_graph_result: GraphResult
    initial_clusters: ClusterResult

    # Debug
    debug_neighbors: Dict

    # Export
    export_directory: str
    faces_csv_path: str
    clusters_csv_path: str
```

---

## Adding Custom Steps

To add a new step:

1. **Create step file**: `sim_bench/pipeline/steps/my_step.py`

```python
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.registry import register_step

@register_step
class MyStep(BaseStep):
    def __init__(self):
        self._metadata = StepMetadata(
            name="my_step",
            display_name="My Step",
            description="What this step does",
            category="analysis",
            requires={"input_field"},      # Context fields required
            produces={"output_field"},     # Context fields produced
            depends_on=["previous_step"],  # Steps that must run first
        )

    def process(self, context, config):
        # Read from context
        input_data = context.input_field

        # Process
        output_data = do_something(input_data, config)

        # Write to context
        context.output_field = output_data

        # Log
        logger.info(f"Processed {len(input_data)} items")
```

2. **Register**: Import in `sim_bench/pipeline/steps/all_steps.py`

3. **Add to config**: `configs/face_clustering_experiment.yaml`

```yaml
pipeline:
  steps:
    - ...
    - my_step  # Add your step
    - ...

step_configs:
  my_step:
    param1: value1
    param2: value2
```

---

**See Also**:
- [Configuration Reference](configuration.md) - All config parameters
- [Pipeline Overview](overview.md) - High-level pipeline flow
- [Troubleshooting](troubleshooting.md) - Common step errors
