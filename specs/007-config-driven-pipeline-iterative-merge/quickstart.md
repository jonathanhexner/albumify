# Quickstart: Config-Driven Pipeline + Iterative Manual Merge

## Programmatic usage

```python
from face_cluster.pipeline import FaceClusteringPipeline
from face_cluster.config import PipelineConfig

pipeline = FaceClusteringPipeline()

# Full run
result = pipeline.run(PipelineConfig.full_run(
    source_dir="path/to/images",
    output_dir="results/my_run",
    K=5, distance_threshold=0.35,
))

# Recluster with different params
result2 = pipeline.run(PipelineConfig.recluster(
    source_dir="results/my_run",
    output_dir="results/my_recluster",
    K=3, distance_threshold=0.4,
))

# Remerge (after manual merge snapshot)
result3 = pipeline.run(PipelineConfig.remerge(
    source_dir="results/my_snapshot",
    output_dir="results/my_remerge",
    merge_enabled=True,
))

# Remerge with exemplar re-selection
result4 = pipeline.run(PipelineConfig.remerge(
    source_dir="results/my_snapshot",
    output_dir="results/my_remerge_ex",
    with_exemplars=True,
    merge_enabled=True,
))
```

## App workflow (Merge Analysis tab)

1. Load a completed run in the app
2. Open Merge Analysis tab — see proposed merge candidates
3. Approve/reject candidates
4. Optionally check "Re-run exemplar selection before merge"
5. Click "Apply Approved Merges"
6. App saves a snapshot, runs remerge, shows new candidates
7. Repeat steps 3-6 until no candidates remain

## Test scenarios

```bash
# Run all face clustering tests
.venv/Scripts/python -m pytest tests/face_clustering/ -v

# Run only the new tests
.venv/Scripts/python -m pytest tests/face_clustering/test_manual_merge_snapshot.py -v
.venv/Scripts/python -m pytest tests/face_clustering/test_pipeline_remerge.py -v

# Run recluster tests (updated to use preset config)
.venv/Scripts/python -m pytest tests/face_clustering/test_merge_stage.py -v -k "Recluster"
```
