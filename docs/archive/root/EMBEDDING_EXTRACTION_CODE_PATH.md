# Embedding Extraction Code Path

## How `embeddings_*.npy` files are created

### Call Chain

```
benchmark_face_clustering.py (scripts/)
  ├─> run_pipeline_for_embeddings(album_path, config)  [Line 81-112]
  │     └─> PipelineExecutor.execute(context, step_names, config)
  │           └─> ExtractFaceEmbeddingsStep.process()  [sim_bench/pipeline/steps/extract_face_embeddings.py]
  │                 └─> _get_extractor(config)  [Line 78-84]
  │                       └─> FaceEmbeddingExtractorFactory.create(config)
  │                             └─> InsightFaceNativeExtractor(config)  [Line 24]
  │
  ├─> collect_face_data(context)  [Line 115-160]
  │     └─> Returns (embeddings_array, metadata)
  │           where embeddings come from context.face_embeddings
  │
  └─> save_embeddings(embeddings_array, output_dir)  [Line 567-573]
        └─> np.save(embeddings_{timestamp}.npy, embeddings)
```

### Core Embedding Extraction

**File**: `sim_bench/pipeline/face_embedding/insightface_native.py`
**Class**: `InsightFaceNativeExtractor`
**Method**: `extract_batch(face_images, face_metadata)` [Line 53-128]

```python
# Line 69-97: Direct recognition model path (used for pre-cropped faces)
if self._rec_model is not None:
    for face_img in face_images:
        # 1. Convert RGB to BGR
        face_bgr = face_img[:, :, ::-1].copy()  # Line 81

        # 2. Resize to 112x112
        face_resized = cv2.resize(face_bgr, (112, 112))  # Line 91

        # 3. Extract embedding
        embedding = self._rec_model.get_feat([face_resized])[0]  # Line 94

        # 4. Normalize
        embedding_norm = embedding / np.linalg.norm(embedding)  # Line 96

        embeddings.append(embedding_norm.astype(np.float32))
```

### Key Details

1. **Model Loading**:
   - `FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])`
   - Uses InsightFace's w600k_r50 ArcFace model
   - Recognition model accessed via `app.models['recognition']`

2. **Input Format**:
   - Expects RGB numpy arrays (pre-aligned face crops)
   - Converts RGB → BGR internally
   - Resizes to 112x112 pixels

3. **Output Format**:
   - 512-dim float32 numpy array
   - **L2 normalized** (important!)
   - Stored in `context.face_embeddings` dict with key `"{path}:face_{idx}"`

4. **Normalization**:
   - Line 96: `embedding / np.linalg.norm(embedding)`
   - This is CRITICAL - embeddings are L2 normalized

### To Reproduce Embeddings

Use the exact same class:

```python
from sim_bench.pipeline.face_embedding.insightface_native import InsightFaceNativeExtractor

config = {"backend": "insightface", "device": "cpu", "model_name": "buffalo_l"}
extractor = InsightFaceNativeExtractor(config)

# face_images: List of RGB numpy arrays (112x112 or will be resized)
embeddings = extractor.extract_batch(face_images, metadata)
# Returns: List[np.ndarray] of shape (512,), L2 normalized
```

### vs face_cluster.InsightFaceEmbedder

**Different class!** The `face_cluster.InsightFaceEmbedder` is a separate implementation:
- File: `face_cluster/embedding.py`
- Method: `get_embedding()` [Line 162-200]
- May have different preprocessing/normalization

**This explains the mismatch** - notebook uses different code path than benchmark script.
