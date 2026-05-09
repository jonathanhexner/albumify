# Research: Config-Driven Pipeline + Iterative Manual Merge

**Date**: 2026-04-17

## Resolved Questions

### R1: How does the current `recluster()` load source data?

**Finding**: `pipeline.py` L261-272 — loads via `load_pipeline_result(source_dir)`, then manually extracts `faces`, `core_indices`, `holdout_indices`, and `crop_manifest`. Embeddings are attached to each `FaceRecord.embedding_normalized`.

**Decision**: The remerge loader follows the same pattern but additionally loads `cluster_result` (merged if available) and reconstructs a `graph_result` from the embedding distance matrix.

### R2: How is `graph_result` used by merge and exemplar stages?

**Finding**:
- `_select_exemplars` calls `selector.select_exemplars(cluster_result, graph_result)` — uses `graph_result.distance_matrix`.
- `_merge` calls `merger.merge_clusters_with_logging(cluster_result, graph_result)` — uses `graph_result.distance_matrix`.

**Decision**: Remerge source loader must build a `graph_result`-like object with at least a `distance_matrix` attribute. Can construct it from embeddings via pairwise cosine distance. Need to check what type `graph_result` is.

### R3: What type is `graph_result`?

**Finding**: `knn_graph.py` `KNNGraphBuilder.build_graph()` returns a `GraphResult` dataclass with fields: `distance_matrix`, `edges`, `adjacency`. The merge and exemplar stages only use `distance_matrix`.

**Decision**: The remerge loader can construct a minimal `GraphResult` with just `distance_matrix` populated (edges and adjacency set to empty defaults). Exemplar and merge stages only read `distance_matrix`.

### R4: What callers of `recluster()` exist?

**Finding**:
- `app/face_clustering.py` L1186 — `pipeline.recluster(_source, _output)`
- `tests/face_clustering/test_merge_stage.py` — 4 tests in `ut_ReclusterE2E` class
- `tests/face_clustering/test_pipeline_history_hook.py` L79 — `pipeline.recluster(out_full, out_rc)`

**Decision**: All three call sites are updated in the same change. No external consumers.

### R5: Does `PipelineConfig` serialization need special handling for callables?

**Finding**: `_init_context` serializes config via `{k: v for k, v in vars(self.config).items() if not k.startswith("_")}`. A callable `on_progress` would fail JSON serialization.

**Decision**: Exclude non-serializable fields (`on_progress`, `stages` can stay as list[str]). Filter out callables in the serialization dict comprehension.
