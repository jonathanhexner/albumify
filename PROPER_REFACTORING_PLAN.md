# Face Clustering Refactoring Plan (Using Existing Pipeline)

## Overview

**Discovery**: We already have a complete generic pipeline framework!

**Current Problem**: We're NOT using the existing `sim_bench/pipeline/` framework for face clustering export

**Solution**: Add 1 new step to the existing pipeline

**Effort**: 1-2 hours (not 4-6!)

---

## Existing Architecture (Already Built!)

### Generic Pipeline Framework

```
sim_bench/pipeline/
├── base.py          ✓ BaseStep abstract class
├── context.py       ✓ PipelineContext (shared mutable state)
├── executor.py      ✓ PipelineExecutor (orchestrates steps)
├── builder.py       ✓ Dependency resolution (topological sort)
├── registry.py      ✓ @register_step decorator
└── steps/
    ├── insightface_detect_faces.py   ✓ Stage 1
    ├── filter_faces.py               ✓ Stage 2
    ├── extract_face_embeddings.py    ✓ Stage 3
    ├── cluster_people.py             ✓ Stage 4
    └── export_for_labeling.py        ✗ MISSING (Stage 5)
```

### How It Works

```python
# 1. Define context (shared state)
context = PipelineContext(
    source_directory=Path("D:/Google_Germany")
)

# 2. List steps to run
step_names = [
    "discover_images",
    "insightface_detect_faces",
    "filter_faces",
    "extract_face_embeddings",
    "cluster_people",
    "export_for_labeling"  # NEW
]

# 3. Execute pipeline
executor = PipelineExecutor(registry)
result = executor.execute(context, step_names, config)

# 4. Pipeline automatically:
#    - Resolves dependencies (topological sort)
#    - Validates each step's inputs exist in context
#    - Executes each step: step.process(context, config)
#    - Each step reads from context, processes, writes back to context
#    - Reports progress
```

### Example: Existing Step

```python
# sim_bench/pipeline/steps/cluster_people.py

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.registry import register_step

@register_step
class ClusterPeopleStep(BaseStep):
    """Cluster faces into people identities using HDBSCAN or hybrid methods."""

    _metadata = StepMetadata(
        name="cluster_people",
        display_name="Cluster People",
        description="Cluster face embeddings into distinct people",
        category="clustering",
        requires={
            "all_faces",           # Reads from context
            "all_face_embeddings"
        },
        produces={
            "people_clusters",     # Writes to context
            "people_thumbnails"
        },
        depends_on=["extract_face_embeddings"]  # Auto-resolved
    )

    def process(self, context: PipelineContext, config: dict):
        """Execute clustering."""
        # 1. Read from context
        faces = context.all_faces
        embeddings = context.all_face_embeddings

        # 2. Process
        clustering_method = config.get('method', 'hdbscan')
        clusters = self._run_clustering(embeddings, clustering_method)

        # 3. Write to context
        context.people_clusters = clusters
        context.people_thumbnails = self._generate_thumbnails(faces, clusters)
```

---

## What We Need to Add (1 Step)

### New Step: `export_for_labeling.py`

```python
# sim_bench/pipeline/steps/export_for_labeling.py

from pathlib import Path
import pandas as pd
import json
from datetime import datetime

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.context import PipelineContext


@register_step
class ExportForLabelingStep(BaseStep):
    """Export clustering results for labeling app."""

    _metadata = StepMetadata(
        name="export_for_labeling",
        display_name="Export for Labeling",
        description="Export faces.csv, clusters.csv for manual labeling",
        category="export",
        requires={
            "all_faces",
            "people_clusters",
            "source_directory"
        },
        produces={
            "export_directory",
            "faces_csv_path",
            "clusters_csv_path"
        },
        depends_on=["cluster_people"]
    )

    def process(self, context: PipelineContext, config: dict):
        """Export clustering data."""

        # Get output directory from config or use default
        output_dir = Path(config.get('output_dir', 'results/clustering_export'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate faces.csv
        faces_df = self._generate_faces_dataframe(
            context.all_faces,
            context.people_clusters
        )
        faces_csv = output_dir / 'faces.csv'
        faces_df.to_csv(faces_csv, index=False)

        # 2. Generate clusters.csv
        clusters_df = self._generate_clusters_dataframe(
            context.people_clusters,
            context.all_face_embeddings
        )
        clusters_csv = output_dir / 'clusters.csv'
        clusters_df.to_csv(clusters_csv, index=False)

        # 3. Save face crops (already in context as aligned faces)
        crops_dir = output_dir / 'face_crops'
        crops_dir.mkdir(exist_ok=True)

        import cv2
        for face_info in context.all_faces:
            if hasattr(face_info, 'aligned_face') and face_info.aligned_face is not None:
                crop_path = crops_dir / f"face_{face_info.face_id:04d}_aligned.jpg"
                # Convert RGB to BGR for OpenCV
                crop_bgr = cv2.cvtColor(face_info.aligned_face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr)

        # 4. Generate export_summary.json
        summary = {
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(context.source_directory),
            'embeddings_dir': str(crops_dir),
            'n_faces': len(context.all_faces),
            'n_clusters': len(context.people_clusters),
            'config': config
        }
        summary_json = output_dir / 'export_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)

        # Write to context
        context.export_directory = str(output_dir)
        context.faces_csv_path = str(faces_csv)
        context.clusters_csv_path = str(clusters_csv)

        context.report_progress("export_for_labeling", 1.0,
                               f"Exported to {output_dir}")

    def _generate_faces_dataframe(self, all_faces, people_clusters):
        """Generate faces.csv data."""
        import pandas as pd

        # Create face_id to cluster_id mapping
        face_to_cluster = {}
        for cluster_id, face_list in people_clusters.items():
            for face_info in face_list:
                face_to_cluster[face_info.face_id] = cluster_id

        faces_data = []
        for face_info in all_faces:
            faces_data.append({
                'face_id': face_info.face_id,
                'image_path': str(face_info.image_path),
                'cluster_id': face_to_cluster.get(face_info.face_id, -1),
                'is_core': face_info.get('is_clusterable', True),
                'bbox_x': face_info.bbox.get('x_px', 0),
                'bbox_y': face_info.bbox.get('y_px', 0),
            })

        return pd.DataFrame(faces_data)

    def _generate_clusters_dataframe(self, people_clusters, embeddings_dict):
        """Generate clusters.csv data."""
        import pandas as pd
        import numpy as np

        clusters_data = []
        for cluster_id, face_list in people_clusters.items():
            if len(face_list) > 0:
                # Compute cluster diameter
                if len(face_list) > 1:
                    cluster_embeddings = np.array([
                        embeddings_dict.get(f.face_id, np.zeros(512))
                        for f in face_list
                    ])
                    similarity = cluster_embeddings @ cluster_embeddings.T
                    distance_matrix = 1.0 - similarity
                    np.fill_diagonal(distance_matrix, 0.0)
                    diameter = float(np.max(distance_matrix))
                else:
                    diameter = 0.0

                clusters_data.append({
                    'cluster_id': cluster_id,
                    'size': len(face_list),
                    'diameter': diameter,
                    'exemplar_face_id': face_list[0].face_id
                })

        return pd.DataFrame(clusters_data)
```

---

## Pipeline Configuration

### Create: `configs/face_clustering_export.yaml`

```yaml
# Pipeline configuration for face clustering + export

pipeline:
  name: "face_clustering_export"
  steps:
    - discover_images
    - insightface_detect_faces
    - filter_faces
    - extract_face_embeddings
    - cluster_people
    - export_for_labeling  # NEW

step_configs:
  discover_images:
    extensions: ['.jpg', '.jpeg', '.png', '.heic']

  insightface_detect_faces:
    model_name: 'buffalo_l'
    det_size: [640, 640]

  filter_faces:
    min_confidence: 0.5
    min_bbox_ratio: 0.02

  extract_face_embeddings:
    backend: 'insightface'
    model_name: 'buffalo_l'

  cluster_people:
    method: 'hybrid_hdbscan_knn'
    min_cluster_size: 2
    knn_k: 5
    distance_threshold: 0.35

  export_for_labeling:  # NEW
    output_dir: null  # Will be set via CLI argument
```

---

## Usage: Generic Script

### Create: `scripts/run_face_clustering_pipeline.py`

```python
#!/usr/bin/env python3
"""
Run face clustering pipeline using existing sim_bench architecture.

Usage:
    python scripts/run_face_clustering_pipeline.py \
        --album /path/to/photos \
        --output results/my_album \
        --config configs/face_clustering_export.yaml
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.executor import PipelineExecutor
from sim_bench.pipeline.registry import get_global_registry
from sim_bench.pipeline.config import load_pipeline_config


def main():
    parser = argparse.ArgumentParser(description='Run face clustering pipeline')
    parser.add_argument('--album', type=Path, required=True, help='Album directory')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--config', type=Path, default='configs/face_clustering_export.yaml')
    args = parser.parse_args()

    # Load pipeline configuration
    config = load_pipeline_config(args.config)

    # Override output directory
    config.step_configs['export_for_labeling']['output_dir'] = str(args.output)

    # Create context
    context = PipelineContext(
        source_directory=args.album
    )

    # Execute pipeline
    registry = get_global_registry()
    executor = PipelineExecutor(registry)

    print(f"Album: {args.album}")
    print(f"Output: {args.output}")
    print(f"Pipeline: {config.pipeline['name']}")
    print(f"Steps: {config.pipeline['steps']}")
    print()

    result = executor.execute(
        context,
        step_names=config.pipeline['steps'],
        config=config
    )

    if result.success:
        print()
        print("=" * 70)
        print("✓ Pipeline completed successfully")
        print("=" * 70)
        print(f"Output: {context.export_directory}")
        print()
        print("To label clusters:")
        print(f"  streamlit run app/face_clustering_labeling.py -- --data-dir {context.export_directory}")
    else:
        print()
        print("=" * 70)
        print(f"✗ Pipeline failed at step: {result.failed_step}")
        print("=" * 70)
        if result.error_message:
            print(f"Error: {result.error_message}")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

---

## Benefits of Using Existing Pipeline

✅ **Generic**: Works for ANY step sequence
✅ **Dependency resolution**: Automatic topological sort
✅ **Validation**: Checks required inputs exist before running
✅ **Progress tracking**: Built-in progress reporting
✅ **Caching**: UniversalCache integration already exists
✅ **Testable**: Each step independently testable
✅ **Config-driven**: Change behavior via YAML, not code
✅ **Reusable**: Same framework used by album app, API, CLI

---

## Comparison

### Current Monolithic Script

```python
# run_and_label_test_clustering.py (294 lines)
def run_clustering():
    # All 5 stages inline
    embedder = InsightFaceEmbedder(...)
    face_records = embedder.detect_and_embed(...)
    # ... 250 more lines
```

### Using Existing Pipeline

```python
# Just 1 new step file (100 lines)
# + 1 config file (30 lines)
# + 1 generic script (50 lines)
# Total: 180 lines

# Run ANY album:
python scripts/run_face_clustering_pipeline.py \
    --album D:/Google_Germany \
    --output results/Germany
```

---

## Implementation Checklist

### Phase 1: Create Export Step (1 hour)

- [ ] Create `sim_bench/pipeline/steps/export_for_labeling.py`
- [ ] Implement `_generate_faces_dataframe()`
- [ ] Implement `_generate_clusters_dataframe()`
- [ ] Save face crops
- [ ] Write export_summary.json

### Phase 2: Configuration (15 min)

- [ ] Create `configs/face_clustering_export.yaml`
- [ ] Add export_for_labeling step config

### Phase 3: Generic Script (30 min)

- [ ] Create `scripts/run_face_clustering_pipeline.py`
- [ ] Argparse setup
- [ ] Load config, create context, execute pipeline

### Phase 4: Test (30 min)

- [ ] Test on test_data
- [ ] Test on Google_Germany
- [ ] Verify labeling app works

### Phase 5: Archive (15 min)

- [ ] Move old scripts to archive/

**Total: ~2.5 hours** (not 6!)

---

## Usage Examples

```bash
# Test data
python scripts/run_face_clustering_pipeline.py \
    --album test_data/face_clustering \
    --output results/test_v1

# Google Germany
python scripts/run_face_clustering_pipeline.py \
    --album D:/Google_Germany \
    --output results/Germany_v1

# Budapest
python scripts/run_face_clustering_pipeline.py \
    --album D:/Budapest2025 \
    --output results/Budapest_v1

# Custom config
python scripts/run_face_clustering_pipeline.py \
    --album D:/My_Album \
    --output results/My_Album \
    --config configs/my_custom_pipeline.yaml

# Then label any of them
streamlit run app/face_clustering_labeling.py -- --data-dir results/Germany_v1
```

---

## Next Steps

1. **Implement export_for_labeling step** (1 hour)
2. **Create pipeline config YAML** (15 min)
3. **Create generic script** (30 min)
4. **Test on test_data** (15 min)
5. **Test on Google_Germany** (15 min)
6. **Archive old code** (15 min)

**Total: ~2.5 hours**

Ready to start?
