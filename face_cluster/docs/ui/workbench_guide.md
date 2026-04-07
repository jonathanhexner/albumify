# Face Clustering Workbench - Quick Guide

## What Is This?

**One-stop experimentation platform** for face clustering algorithm development.

- Uses `face_cluster/` modules (separate from main app)
- No separate scripts needed - everything in one UI
- Self-explanatory with module documentation
- Tracks history automatically

## Launch

```bash
streamlit run app/face_clustering_workbench.py
```

## Features

### 📂 Tab 1: Process Album
**Run full 7-stage pipeline in one click:**

1. 🔍 Detect & Embed (`InsightFaceEmbedder`)
2. 🎯 Quality Gate (`QualityGater`)
3. 💾 Save Crops
4. 📊 Build kNN Graph (`KNNGraphBuilder`)
5. 🧩 Cluster (`ConnectedComponentsClusterer`)
6. 🎯 Merge Clusters (`D10ExemplarSelector` + `ConservativeMerger`)
7. 💾 Export (faces.csv, clusters.csv)

**What you see:**
- Live progress for each stage
- Which module is currently running
- Summary of each stage result
- Clear success/failure feedback

### 📊 Tab 2: View Results
- Summary stats (faces, clusters, noise)
- Browse clusters one by one
- Cluster quality indicators (diameter)
- Face gallery for each cluster
- Source image paths

### 🏷️ Tab 3: Label Clusters
- Assign names to clusters
- Visual inspection while labeling
- Saves to `corrected_labels.csv`
- Used for ML training data

### 📜 Tab 4: History
- Automatic tracking in `.face_clustering_history.json`
- View last 10 runs
- Load any previous run
- Compare different parameter settings

## Sidebar: Module Documentation

**Shows for each module:**
- 💡 Purpose - what it does
- ⬇️ Input - what it needs
- ⬆️ Output - what it produces
- 📍 File location

**Modules used:**
- `InsightFaceEmbedder` - face_cluster/embedding.py
- `QualityGater` - face_cluster/quality.py
- `KNNGraphBuilder` - face_cluster/knn_graph.py
- `ConnectedComponentsClusterer` - face_cluster/clustering.py
- `D10ExemplarSelector` - face_cluster/exemplars.py
- `ConservativeMerger` - face_cluster/merge.py
- `HoldoutAttacher` - face_cluster/attach.py

## Quick Start

1. **Launch app**: `streamlit run app/face_clustering_workbench.py`
2. **Tab 1**: Check "Use test data" → Click "▶️ Run Full Pipeline"
3. **Wait ~30 seconds** for completion
4. **Tab 2**: Browse the 3 clusters
5. **Tab 3**: Label clusters (optional)
6. **Tab 4**: See run history

## Output Structure

```
results/run_YYYYMMDD_HHMMSS/
├── face_records.json      # Raw face detections
├── faces.csv              # Face metadata with cluster assignments
├── clusters.csv           # Cluster statistics
├── face_crops/            # Aligned face images
│   ├── face_0000_aligned.jpg
│   ├── face_0001_aligned.jpg
│   └── ...
└── export_summary.json    # Run metadata
```

## Configuration Parameters

**k (kNN neighbors)**: 3-10
- Higher k = more connections, larger clusters
- Lower k = stricter clustering, more small clusters
- Default: 5

**Distance threshold**: 0.25-0.45
- Lower = stricter (fewer edges, more clusters)
- Higher = looser (more edges, fewer clusters)
- Default: 0.35

## Does NOT Impact Main App

This workbench is **completely separate**:
- Uses `face_cluster/` modules
- Main app uses `sim_bench/pipeline/` and `sim_bench/clustering/`
- No shared state
- Experimentation sandbox

## Next Steps After Experimentation

Once you find good parameters:
1. Note the k and distance_threshold values
2. Update `configs/pipeline.yaml` in main app
3. Main app will use these settings for production

## Troubleshooting

**No faces detected:**
- Check album path exists
- Verify images are .jpg, .png, .heic
- Try different album

**Pipeline fails at Stage X:**
- Check error details in expander
- Verify all face_cluster modules are present
- Check logs for stack trace

**Results look wrong:**
- Try different k or distance_threshold
- Check cluster diameters (should be < 0.5)
- Use labeling tab to inspect clusters

## History File

Location: `results/.face_clustering_history.json`

Tracks:
- Run name and timestamp
- Source album
- Number of faces/clusters
- Configuration parameters

**Automatic cleanup**: Keeps last 50 runs
