# Specialized Models

Domain-specific models for face detection and landmark recognition.

## Face Model

Extracts face embeddings for person clustering.

### Backends

- **DeepFace** (default): Easy to use, good accuracy
  - Install: `pip install deepface`
  - Models: VGG-Face, Facenet, ArcFace, etc.

- **InsightFace**: High accuracy, fast
  - Install: `pip install insightface`
  - Better for production use

- **MediaPipe**: Fast, lightweight (detection only)
  - Install: `pip install mediapipe`
  - Note: Doesn't provide embeddings, only detection

### Usage

```python
from sim_bench.specialized_models import create_specialized_model

# Create face model
face_model = create_specialized_model('face', backend='deepface')

# Process images
results = face_model.process_batch(
    image_paths=["photo1.jpg", "photo2.jpg"],
    routing_hints={'needs_face_detection': True}
)

# Extract embeddings for clustering
embeddings = face_model.extract_embeddings(image_paths)
```

## Landmark Model

Extracts place/landmark embeddings for location-based clustering.

Uses enhanced CLIP embeddings with landmark-specific prompts.

### Usage

```python
from sim_bench.specialized_models import create_specialized_model

# Create landmark model
landmark_model = create_specialized_model('landmark')

# Process images
results = landmark_model.process_batch(
    image_paths=["photo1.jpg", "photo2.jpg"],
    routing_hints={'needs_landmark_detection': True}
)

# Extract embeddings for clustering
embeddings = landmark_model.extract_embeddings(image_paths)
```

## Integration with Pipeline

The `PhotoAnalysisPipeline` automatically uses these models based on routing decisions:

```python
from sim_bench.photo_analysis import PhotoAnalysisPipeline

pipeline = PhotoAnalysisPipeline(
    face_config={'backend': 'deepface'},
    landmark_config={}
)

# Automatically runs CLIP, then specialized models based on routing
results = pipeline.analyze_with_specialized(image_paths)

# Extract all embeddings for clustering
embeddings = pipeline.extract_embeddings_for_clustering(results)
```




