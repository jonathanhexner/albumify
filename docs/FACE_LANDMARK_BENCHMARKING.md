# Face and Landmark Embedding Benchmarking

## Overview

This document describes how to benchmark face recognition and landmark detection embeddings as similarity methods in sim-bench, just like DINOv2, CLIP, and other visual feature extractors.

## Motivation

Face and landmark embeddings can be treated as **specialized similarity methods** for specific use cases:

- **Face Embeddings**: Images are similar if they contain the same people
- **Landmark Embeddings**: Images are similar if they show the same landmarks/locations

This enables:
1. **Person-based photo retrieval**: "Find all photos of John"
2. **Location-based retrieval**: "Find all photos of the Eiffel Tower"
3. **Hierarchical organization**: First cluster by event, then by people within each event

## Architecture

### Feature Extraction Methods

Both methods follow the standard `BaseMethod` interface and integrate seamlessly with the existing benchmark framework.

#### Face Embeddings (`face_embeddings`)

**File**: `sim_bench/feature_extraction/face_embeddings.py`

**How it works**:
1. Detect faces in each image (using RetinaFace, MTCNN, etc.)
2. Extract face embeddings (using VGG-Face, FaceNet, ArcFace, etc.)
3. Aggregate multiple faces per image using one of:
   - `average`: Average all face embeddings (default)
   - `max_confidence`: Use most confident detection
   - `first`: Use first detected face only
   - `concat`: Concatenate up to N face embeddings

**Parameters**:
```yaml
method:
  method: "face_embeddings"
  backend: "retinaface"  # Detection backend
  embedding_model: "VGG-Face"  # Recognition model
  aggregation: "average"  # Multi-face handling
  max_faces: 5  # For concat mode
  min_confidence: 0.9
  device: "cpu"
  distance: "cosine"
```

**Use Cases**:
- Group photos by person
- Find all photos containing a specific person
- Portrait organization

#### Landmark Embeddings (`landmark_embeddings`)

**File**: `sim_bench/feature_extraction/landmark_embeddings.py`

**How it works**:
1. Detect landmarks/places in each image
2. Extract embeddings using one of:
   - `embedding`: Visual embeddings from landmark model
   - `categorical`: One-hot encoding of landmark names
   - `hybrid`: Combination of visual + categorical

**Parameters**:
```yaml
method:
  method: "landmark_embeddings"
  encoding: "embedding"  # embedding, categorical, or hybrid
  embedding_dim: 512
  min_confidence: 0.5
  device: "cpu"
  distance: "cosine"
```

**Use Cases**:
- Group travel photos by landmark
- Find all photos of a specific place
- Location-based organization

### Hierarchical Clustering

**File**: `sim_bench/clustering/hierarchical.py`

Enables **two-level clustering** for sophisticated photo organization:

**Level 1**: Coarse grouping (e.g., events, scenes)
- Uses general visual features (DINOv2, CLIP)
- Groups by overall scene similarity

**Level 2**: Fine grouping within each Level 1 cluster
- Uses specialized features (faces, landmarks)
- Refines organization within each group

**Example Use Cases**:

1. **Event + People**:
   - L1: Cluster by event (birthday, wedding, vacation)
   - L2: Cluster by people within each event

2. **Location + Landmark**:
   - L1: Cluster by country/city
   - L2: Cluster by specific landmarks within each location

3. **Scene + Content**:
   - L1: Cluster by scene type (indoor, outdoor, beach)
   - L2: Cluster by specific content within each scene

## Running Benchmarks

### 1. Face Embedding Benchmark

Evaluate how well face embeddings work for image similarity:

```bash
python -m sim_bench.run_experiment configs/run.face_embeddings.yaml
```

**Configuration**: `configs/run.face_embeddings.yaml`

This will:
- Extract face embeddings from all images
- Compute pairwise similarities
- Evaluate retrieval performance (mAP, Recall@K)
- Compare against ground truth

**Expected Performance**:
- **High** for datasets where people are the primary similarity factor
- **Low** for datasets focused on scenes/objects without people

### 2. Landmark Embedding Benchmark

Evaluate landmark-based similarity:

```bash
python -m sim_bench.run_experiment configs/run.landmark_embeddings.yaml
```

**Configuration**: `configs/run.landmark_embeddings.yaml`

**Expected Performance**:
- **High** for travel/landmark datasets
- **Low** for datasets without landmarks

### 3. Hierarchical Clustering

Run two-level clustering:

```bash
python -m sim_bench.run_clustering configs/run.hierarchical_clustering.yaml
```

**Configuration**: `configs/run.hierarchical_clustering.yaml`

This will:
- Cluster at Level 1 using DINOv2 (scene similarity)
- Cluster at Level 2 using face embeddings (person similarity)
- Generate hierarchical organization

**Output**:
- Hierarchical cluster labels
- Statistics for both levels
- HTML galleries showing the hierarchy

## Aggregation Strategies

### Face Embedding Aggregation

When an image contains multiple faces, we need to aggregate them into a single vector:

#### 1. Average (Default)
```python
aggregation: "average"
```
- Averages all face embeddings
- **Best for**: General person-based similarity
- **Pros**: Captures all people in the image
- **Cons**: May dilute individual person signals

#### 2. Max Confidence
```python
aggregation: "max_confidence"
```
- Uses embedding from most confident face detection
- **Best for**: Portrait photos, single-person focus
- **Pros**: Focuses on primary person
- **Cons**: Ignores other people

#### 3. First
```python
aggregation: "first"
```
- Uses first detected face only
- **Best for**: Quick processing, portraits
- **Pros**: Fast, simple
- **Cons**: May be inconsistent

#### 4. Concat (Concatenate)
```python
aggregation: "concat"
max_faces: 5
```
- Concatenates up to N face embeddings
- Pads with zeros if fewer faces
- **Best for**: Preserving multi-person information
- **Pros**: Retains all face information
- **Cons**: Higher dimensionality, requires more data

## Integration with Agent System

The face and landmark tools in `sim_bench/agent/tools/` use these methods internally:

```python
# Face detection tool uses face embeddings
from sim_bench.photo_analysis.factory import create_photo_analyzer

face_analyzer = create_photo_analyzer(
    analyzer_type='face',
    config={'backend': 'retinaface'}
)

# Landmark detection tool uses landmark embeddings
landmark_analyzer = create_photo_analyzer(
    analyzer_type='landmark',
    config={'device': 'cpu'}
)
```

The **agent workflows** can leverage hierarchical clustering:

```python
# Workflow: Organize by event, then by people within each event
workflow = Workflow(
    name="organize_by_event_and_people",
    steps=[
        # Level 1: Cluster by event
        WorkflowStep("cluster_events", "cluster_images", {
            'method': 'dbscan',
            'feature_type': 'dinov2'
        }),
        # Level 2: Detect faces
        WorkflowStep("detect_faces", "detect_faces", {
            'backend': 'retinaface'
        }),
        # Level 2: Group by person within events
        WorkflowStep("group_by_person", "group_by_person", {
            'similarity_threshold': 0.6
        }, dependencies=["cluster_events", "detect_faces"])
    ]
)
```

## Benchmark Comparison

You can now compare face/landmark embeddings against other methods:

```bash
# Run all methods for comparison
python -m sim_bench.run_experiment configs/run.dinov2.yaml
python -m sim_bench.run_experiment configs/run.openclip.yaml
python -m sim_bench.run_experiment configs/run.face_embeddings.yaml
python -m sim_bench.run_experiment configs/run.landmark_embeddings.yaml

# Compare results
python -m sim_bench.analysis.compare_methods \
    outputs/dinov2_benchmark \
    outputs/openclip_benchmark \
    outputs/face_embeddings_benchmark \
    outputs/landmark_embeddings_benchmark
```

This will show you:
- Which method works best for your specific dataset
- Whether face/landmark features provide complementary information
- Whether hierarchical approaches improve results

## Expected Results

### When Face Embeddings Work Well

- **High performance**:
  - Person-centric datasets (family photos, portraits)
  - Same people across different scenes
  - Ground truth based on people

- **Low performance**:
  - Landscape/architecture datasets
  - No people in images
  - Ground truth based on scenes/objects

### When Landmark Embeddings Work Well

- **High performance**:
  - Travel/tourism datasets
  - Famous landmarks
  - Location-based ground truth

- **Low performance**:
  - Indoor/personal photos
  - No recognizable landmarks
  - Generic scenes

### When to Use Hierarchical Clustering

- **Best for**:
  - Complex organization needs (events + people)
  - Large photo collections with multiple facets
  - When single-level clustering is insufficient

- **Example**: Wedding photo album
  - L1: Pre-wedding, ceremony, reception, after-party
  - L2: Bride/groom, family groups, friends

## Implementation Notes

### Feature Dimension Handling

- **DINOv2**: 768 dims
- **OpenCLIP**: 512 dims
- **Face embeddings**: 512-4096 dims (model-dependent)
- **Landmark embeddings**: Configurable (default 512)

All methods output `[n_images, embedding_dim]` matrices compatible with the benchmark framework.

### Missing Data Handling

- **No faces detected**: Returns zero vector
- **No landmarks detected**: Returns zero vector
- **Failed processing**: Returns zero vector + warning

This ensures robust benchmarking even with incomplete detections.

### Performance Considerations

- **Face detection**: ~0.5-2 sec/image (CPU)
- **Landmark detection**: ~0.3-1 sec/image (CPU)
- **Hierarchical clustering**: 2x the cost of single-level

For large datasets, consider:
- Using GPU (`device: "cuda"`)
- Caching embeddings
- Sampling for initial experiments

## Next Steps

1. **Run benchmarks** on your datasets to see performance
2. **Tune parameters** (confidence thresholds, aggregation strategies)
3. **Compare methods** to understand which works best for your use case
4. **Try hierarchical clustering** for sophisticated organization
5. **Integrate with agent** for natural language photo management

## Example Workflow

```bash
# 1. Benchmark face embeddings
python -m sim_bench.run_experiment configs/run.face_embeddings.yaml

# 2. Benchmark landmark embeddings
python -m sim_bench.run_experiment configs/run.landmark_embeddings.yaml

# 3. Compare with visual methods
python -m sim_bench.analysis.compare_methods \
    outputs/dinov2_benchmark \
    outputs/face_embeddings_benchmark

# 4. Run hierarchical clustering
python -m sim_bench.run_clustering configs/run.hierarchical_clustering.yaml

# 5. Use in agent system
streamlit run app_agent.py
# Then: "Organize my photos by event and person"
```

## References

- Face recognition models: VGG-Face, FaceNet, ArcFace
- Face detection: RetinaFace, MTCNN, DLIB
- Landmark recognition: Vision transformers, place recognition models
- Hierarchical clustering: Two-level DBSCAN/HDBSCAN
