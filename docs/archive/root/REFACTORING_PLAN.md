# Face Clustering Refactoring Plan

## Overview

**Goal**: Transform monolithic test script into production-ready, reusable pipeline

**Effort**: 4-6 hours
**Benefit**: Works for ANY album, not just test data

---

## Current State (What We Have Now)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  run_and_label_test_clustering.py (294 lines - monolithic)  │
│                                                               │
│  def run_clustering():                                        │
│      # Stage 1: Detect (30 lines inline)                     │
│      embedder = InsightFaceEmbedder(...)                     │
│      face_records = embedder.detect_and_embed(...)           │
│                                                               │
│      # Stage 2: Quality gate (5 lines inline)                │
│      core_indices = list(range(len(face_records)))           │
│                                                               │
│      # Stage 3: Save crops (15 lines inline)                 │
│      for record in face_records:                             │
│          cv2.imwrite(...)                                    │
│                                                               │
│      # Stage 4: Cluster (10 lines inline)                    │
│      builder = KNNGraphBuilder(config)                       │
│      cluster_result = clusterer.cluster(...)                 │
│                                                               │
│      # Stage 5: Export CSVs (80 lines inline)                │
│      faces_data = []                                         │
│      for record in face_records:                             │
│          faces_data.append({...})                            │
│      faces_df.to_csv(...)                                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │   Results for   │
                    │   TEST DATA     │
                    │   ONLY          │
                    └─────────────────┘
```

### File Structure

```
sim-bench/
├── scripts/
│   └── run_and_label_test_clustering.py    # 294 lines - hardcoded for test data
│
├── app/
│   └── simple_face_labeling.py             # Hardcoded: DATA_DIR = "results/test_face_clustering_e2e"
│
└── face_cluster/                            # Library (existing)
    ├── embedding.py      ✓ (exists)
    ├── quality.py        ✓ (exists)
    ├── knn_graph.py      ✓ (exists)
    ├── clustering.py     ✓ (exists)
    ├── crops.py          ✗ (MISSING - logic in script)
    └── export.py         ✗ (MISSING - logic in script)
```

### Problems

❌ **Hardcoded paths**: Only works for `test_data/face_clustering`
❌ **Not reusable**: Can't run on Google_Germany without rewriting
❌ **Monolithic**: All logic in one 294-line script
❌ **No tests**: Can't test stages independently
❌ **Mixed concerns**: I/O logic + business logic in same file

---

## Proposed State (After Refactoring)

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│         scripts/run_face_clustering.py (100 lines - thin)        │
│                                                                    │
│  def main(album_path, output_dir):                                │
│      # Orchestrate 5 stages (just function calls)                 │
│                                                                    │
│      [1] detect_and_embed(album_path)            ─────┐           │
│           └─> face_records.json                       │           │
│                                                        │           │
│      [2] quality_gate(face_records, config)           │           │
│           └─> core_indices, holdout_indices           │           │
│                                                        │           │
│      [3] save_crops(face_records, output_dir)    <────┤  Reusable │
│           └─> face_crops/, crop_manifest.json         │  Library  │
│                                                        │  Functions│
│      [4] cluster(face_records, core_indices)          │           │
│           └─> cluster_result.json                     │           │
│                                                        │           │
│      [5] export_for_labeling(...)                ─────┘           │
│           └─> faces.csv, clusters.csv, export_summary.json        │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │   Works for ANY album   │
                    │  - test_data            │
                    │  - Google_Germany       │
                    │  - Budapest2025         │
                    └─────────────────────────┘
```

### File Structure

```
sim-bench/
├── scripts/
│   ├── run_face_clustering.py              # NEW: 100 lines - generic orchestrator
│   └── [ARCHIVE] run_and_label_test_clustering.py
│
├── app/
│   └── face_clustering_labeling.py         # UPDATED: Takes --data-dir parameter
│
├── face_cluster/                            # Complete library
│   ├── embedding.py      ✓ (exists)
│   ├── quality.py        ✓ (exists)
│   ├── knn_graph.py      ✓ (exists)
│   ├── clustering.py     ✓ (exists)
│   ├── crops.py          ✓ (NEW - 50 lines)
│   └── export.py         ✓ (NEW - 80 lines)
│
└── tests/face_clustering/                   # NEW: Independent tests
    ├── test_stage1_detection.py
    ├── test_stage3_crops.py
    ├── test_stage5_export.py
    └── test_e2e.py
```

### Benefits

✅ **Generic**: Works for any album
✅ **Testable**: Each stage independently tested
✅ **Reusable**: Library functions used by multiple scripts
✅ **Maintainable**: Separation of concerns (I/O vs business logic)
✅ **Flexible**: Easy to add new stages or modify existing ones

---

## Side-by-Side Comparison

### Running on Different Albums

**CURRENT** (Hardcoded):
```bash
# Test data - works
python scripts/run_and_label_test_clustering.py

# Google_Germany - DOESN'T WORK (need to edit script)
# ❌ Would need to change hardcoded paths in script
# ❌ Would need to change hardcoded paths in app
```

**PROPOSED** (Flexible):
```bash
# Test data
python scripts/run_face_clustering.py \
    --album test_data/face_clustering \
    --output results/test

# Google_Germany - WORKS
python scripts/run_face_clustering.py \
    --album D:/Google_Germany \
    --output results/Germany_v1

# Budapest2025 - WORKS
python scripts/run_face_clustering.py \
    --album D:/Budapest2025 \
    --output results/Budapest_v1

# Then label ANY of them
streamlit run app/face_clustering_labeling.py -- --data-dir results/Germany_v1
```

---

## Data Flow Diagram

### Current Flow (Hardcoded)

```
test_data/face_clustering/
    └─> [Monolithic Script] ─> results/test_face_clustering_e2e/
                                    └─> [Hardcoded App]
```

### Proposed Flow (Flexible)

```
ANY Album Directory
    │
    ├─> test_data/face_clustering/
    ├─> D:/Google_Germany/
    └─> D:/Budapest2025/
        │
        ▼
    [Generic Pipeline Script]
        │
        ├─> Stage 1: face_cluster.embedding.detect_and_embed()
        ├─> Stage 2: face_cluster.quality.QualityGater.select_core_set()
        ├─> Stage 3: face_cluster.crops.save_crops()  ◄── NEW
        ├─> Stage 4: face_cluster.clustering.cluster_faces()
        └─> Stage 5: face_cluster.export.export_for_labeling()  ◄── NEW
        │
        ▼
    results/{album_name}/
        ├─> face_records.json
        ├─> face_crops/
        ├─> crop_manifest.json
        ├─> cluster_result.json
        ├─> faces.csv
        ├─> clusters.csv
        └─> export_summary.json
        │
        ▼
    [Flexible Labeling App]
        └─> Select ANY results directory
```

---

## What Changes (File-by-File)

### 1. CREATE: `face_cluster/crops.py` (~50 lines)

**What it does**: Save aligned face crops and manifest

**Before** (inline in script):
```python
# In run_and_label_test_clustering.py (lines 120-140)
crops_dir = output_dir / "face_crops"
crops_dir.mkdir(exist_ok=True)
for record in face_records:
    if record.aligned_face is not None:
        crop_path = crops_dir / f"face_{record.face_id:04d}_aligned.jpg"
        crop_bgr = cv2.cvtColor(record.aligned_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(crop_path), crop_bgr)
```

**After** (reusable function):
```python
# face_cluster/crops.py
from pathlib import Path
from typing import List
import cv2
import json
from face_cluster.types import FaceRecord

def save_crops(
    face_records: List[FaceRecord],
    output_dir: Path
) -> Path:
    """Save aligned face crops and write manifest.

    Args:
        face_records: List of face records with aligned_face
        output_dir: Directory to save crops

    Returns:
        Path to crop_manifest.json
    """
    crops_dir = output_dir / "face_crops"
    crops_dir.mkdir(exist_ok=True, parents=True)

    manifest = []

    for record in face_records:
        if record.aligned_face is not None:
            crop_path = crops_dir / f"face_{record.face_id:04d}_aligned.jpg"

            # Convert RGB to BGR for OpenCV
            crop_bgr = cv2.cvtColor(record.aligned_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)

            manifest.append({
                'face_id': record.face_id,
                'crop_path': str(crop_path),
                'source_image': str(record.image_path)
            })

    # Write manifest
    manifest_file = output_dir / 'crop_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump({
            'crops': manifest,
            'n_crops': len(manifest),
            'crops_dir': str(crops_dir)
        }, f, indent=2)

    return manifest_file
```

---

### 2. CREATE: `face_cluster/export.py` (~80 lines)

**What it does**: Generate faces.csv, clusters.csv, export_summary.json

**Before** (inline in script):
```python
# In run_and_label_test_clustering.py (lines 180-240)
faces_data = []
for i, record in enumerate(face_records):
    cluster_id = -1
    if i in core_indices:
        core_idx = core_indices.index(i)
        cluster_id = int(cluster_result.labels[core_idx])

    faces_data.append({
        'face_id': record.face_id,
        'image_path': str(record.image_path),
        'cluster_id': cluster_id,
        'is_core': i in core_indices,
        'person_label': Path(record.image_path).parent.name
    })

faces_df = pd.DataFrame(faces_data)
faces_df.to_csv(output_dir / "faces.csv", index=False)

# ... more inline code for clusters.csv, export_summary.json
```

**After** (reusable function):
```python
# face_cluster/export.py
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from face_cluster.types import FaceRecord, ClusterResult

def export_for_labeling(
    face_records: List[FaceRecord],
    cluster_result: ClusterResult,
    core_indices: List[int],
    output_dir: Path,
    config: dict = None
) -> Tuple[Path, Path, Path]:
    """Export clustering results for labeling app.

    Args:
        face_records: All face records
        cluster_result: Clustering result
        core_indices: Indices of core faces
        output_dir: Output directory
        config: Pipeline config dict

    Returns:
        Tuple of (faces_csv, clusters_csv, summary_json) paths
    """
    # Generate faces.csv
    faces_df = _generate_faces_dataframe(
        face_records, cluster_result, core_indices
    )
    faces_csv = output_dir / 'faces.csv'
    faces_df.to_csv(faces_csv, index=False)

    # Generate clusters.csv
    clusters_df = _generate_clusters_dataframe(
        cluster_result, face_records, core_indices
    )
    clusters_csv = output_dir / 'clusters.csv'
    clusters_df.to_csv(clusters_csv, index=False)

    # Generate export_summary.json
    summary = {
        'timestamp': datetime.now().isoformat(),
        'embeddings_dir': str(output_dir / 'face_crops'),
        'n_faces': len(face_records),
        'n_clusters': cluster_result.n_clusters,
        'n_noise': cluster_result.n_noise,
        'config': config or {}
    }
    summary_json = output_dir / 'export_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    return faces_csv, clusters_csv, summary_json


def _generate_faces_dataframe(
    face_records: List[FaceRecord],
    cluster_result: ClusterResult,
    core_indices: List[int]
) -> pd.DataFrame:
    """Generate faces.csv data."""
    # ... implementation
    pass


def _generate_clusters_dataframe(
    cluster_result: ClusterResult,
    face_records: List[FaceRecord],
    core_indices: List[int]
) -> pd.DataFrame:
    """Generate clusters.csv data."""
    # ... implementation
    pass
```

---

### 3. CREATE: `scripts/run_face_clustering.py` (~100 lines)

**What it does**: Orchestrate the 5 stages (thin wrapper)

```python
#!/usr/bin/env python3
"""
Run face clustering pipeline on any album.

Usage:
    python scripts/run_face_clustering.py \
        --album /path/to/photos \
        --output results/my_album
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import (
    InsightFaceEmbedder,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    PipelineConfig
)
from face_cluster.crops import save_crops
from face_cluster.export import export_for_labeling


def main(album_path: Path, output_dir: Path, config: PipelineConfig):
    """Run 5-stage clustering pipeline."""

    print(f"Album: {album_path}")
    print(f"Output: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Detect and extract embeddings
    print("[1/5] Detecting faces and extracting embeddings...")
    embedder = InsightFaceEmbedder(model_name='buffalo_l', ctx_id=-1)

    image_paths = list(album_path.rglob("*.jpg")) + list(album_path.rglob("*.jpeg"))
    face_records = embedder.detect_and_embed([str(p) for p in image_paths])

    print(f"      Detected {len(face_records)} faces")

    # Stage 2: Quality gating
    print("[2/5] Quality gating...")
    gater = QualityGater(config)
    face_records = gater.compute_blur_scores(face_records)
    core_indices, holdout_indices = gater.select_core_set(face_records)

    print(f"      Core: {len(core_indices)}, Holdout: {len(holdout_indices)}")

    # Stage 3: Save crops
    print("[3/5] Saving face crops...")
    crop_manifest = save_crops(face_records, output_dir)
    print(f"      Saved {len(face_records)} crops")

    # Stage 4: Cluster
    print("[4/5] Clustering...")
    builder = KNNGraphBuilder(config)
    graph_result = builder.build_graph(face_records, core_indices)

    clusterer = ConnectedComponentsClusterer(config)
    cluster_result = clusterer.cluster(graph_result, core_indices)

    print(f"      Found {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

    # Stage 5: Export for labeling
    print("[5/5] Exporting for labeling app...")
    faces_csv, clusters_csv, summary_json = export_for_labeling(
        face_records, cluster_result, core_indices, output_dir,
        config={'K': config.K, 'distance_threshold': config.distance_threshold}
    )

    print(f"      Saved: {faces_csv.name}, {clusters_csv.name}")

    print()
    print("=" * 70)
    print(f"✓ Complete: {output_dir}")
    print("=" * 70)
    print()
    print(f"To label clusters:")
    print(f"  streamlit run app/face_clustering_labeling.py -- --data-dir {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run face clustering pipeline')
    parser.add_argument('--album', type=Path, required=True, help='Path to album directory')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--K', type=int, default=5, help='kNN K parameter')
    parser.add_argument('--threshold', type=float, default=0.35, help='Distance threshold')

    args = parser.parse_args()

    config = PipelineConfig(
        K=args.K,
        distance_threshold=args.threshold,
        min_cluster_size=2
    )

    main(args.album, args.output, config)
```

---

### 4. UPDATE: `app/face_clustering_labeling.py` (5 lines changed)

**Before** (hardcoded):
```python
# Line 20
DATA_DIR = Path("results/test_face_clustering_e2e")  # HARDCODED
CROPS_DIR = DATA_DIR / "face_crops"
```

**After** (flexible):
```python
# Parse command-line argument
import sys
if '--data-dir' in sys.argv:
    idx = sys.argv.index('--data-dir')
    DATA_DIR = Path(sys.argv[idx + 1])
else:
    DATA_DIR = Path("results/test_face_clustering_e2e")  # Default

CROPS_DIR = DATA_DIR / "face_crops"

st.markdown(f"**Data**: `{DATA_DIR}`")
```

---

## Implementation Checklist

### Phase 1: Create New Files (4-5 hours)

- [ ] **Create `face_cluster/crops.py`** (1 hour)
  - [ ] `save_crops()` function
  - [ ] Write crop_manifest.json
  - [ ] Test: `python -c "from face_cluster.crops import save_crops; ..."`

- [ ] **Create `face_cluster/export.py`** (2 hours)
  - [ ] `export_for_labeling()` main function
  - [ ] `_generate_faces_dataframe()` helper
  - [ ] `_generate_clusters_dataframe()` helper
  - [ ] Test: `python -c "from face_cluster.export import export_for_labeling; ..."`

- [ ] **Create `scripts/run_face_clustering.py`** (1 hour)
  - [ ] Argparse setup
  - [ ] Call all 5 stages
  - [ ] Test on test_data: `python scripts/run_face_clustering.py --album test_data/face_clustering --output results/test_refactored`

- [ ] **Update `app/face_clustering_labeling.py`** (30 min)
  - [ ] Add --data-dir parameter support
  - [ ] Test: `streamlit run app/face_clustering_labeling.py -- --data-dir results/test_refactored`

### Phase 2: Test (1-2 hours)

- [ ] **Test on test data**
  - [ ] Run pipeline: `python scripts/run_face_clustering.py --album test_data/face_clustering --output results/test_refactored`
  - [ ] Verify all files created
  - [ ] Open labeling app
  - [ ] Verify images display

- [ ] **Test on Google_Germany**
  - [ ] Run pipeline: `python scripts/run_face_clustering.py --album D:/Google_Germany --output results/Germany_v1`
  - [ ] Verify no null image_paths
  - [ ] Open labeling app
  - [ ] Spot-check clusters

### Phase 3: Archive Old Code (30 min)

- [ ] **Move to `archive/scripts_2026Q1/`**
  - [ ] `benchmark_face_clustering.py`
  - [ ] `export_clustering_data.py`
  - [ ] `run_and_label_test_clustering.py`
  - [ ] `test_e2e_face_clustering.py`
  - [ ] All `debug_*.py` scripts

- [ ] **Update documentation**
  - [ ] Update CLAUDE.md with new workflow
  - [ ] Create WORKFLOW.md showing how to use new pipeline

---

## Success Criteria

After refactoring, you should be able to:

✅ **Run on ANY album with one command**:
```bash
python scripts/run_face_clustering.py --album /any/path --output results/any_name
```

✅ **Test each stage independently**:
```python
from face_cluster.crops import save_crops
# Test crop saving without running full pipeline
```

✅ **Switch between albums easily**:
```bash
# Morning: work on Google_Germany
python scripts/run_face_clustering.py --album D:/Google_Germany --output results/Germany
streamlit run app/face_clustering_labeling.py -- --data-dir results/Germany

# Afternoon: work on Budapest
python scripts/run_face_clustering.py --album D:/Budapest2025 --output results/Budapest
streamlit run app/face_clustering_labeling.py -- --data-dir results/Budapest
```

---

## Alternative: Keep Current Code (Skip Refactoring)

**If you want to skip refactoring**, you can still use the current working code:

**Modify `run_and_label_test_clustering.py` to take album path**:
```python
# Line 40: Change from
test_data_dir = Path("test_data/face_clustering")

# To
import sys
if len(sys.argv) > 1:
    test_data_dir = Path(sys.argv[1])
else:
    test_data_dir = Path("test_data/face_clustering")
```

**Then run**:
```bash
python scripts/run_and_label_test_clustering.py D:/Google_Germany
```

**Pros**: 5 minutes of work
**Cons**: Still hardcoded app, no tests, not maintainable long-term

---

## Decision Time

**Option A: Refactor (4-6 hours)**
- Get clean, production-ready pipeline
- Reusable for all albums
- Testable components
- Maintainable long-term

**Option B: Quick fix (5 minutes)**
- Make current script take album path as argument
- Keep everything else as-is
- Get to labeling faster

**Option C: Hybrid**
- Just create `scripts/run_face_clustering.py` (skip tests)
- Skip creating library modules
- Quick (~2 hours)

Which do you want?
