# spec-026: Face Clustering Results DB

**Status**: In Progress
**Date**: 2026-05-04
**Replaces**: CSV + JSON + numpy flat file exports
**Design doc**: `docs/design/app/2026-05-04_face_clustering_db_data_model.md`

## Problem

Face clustering results are stored as flat files (CSVs, JSONs, numpy arrays) that get overwritten during merge iterations. This causes:
- Lost base cluster data (clusters.csv overwritten by merged state)
- No per-iteration traceability (can't see how clusters evolved)
- Face-embedding mismatches undetectable (no queryable chain from face → embedding → cluster)
- Fragile filename gymnastics (clusters_stage_base.csv, faces_merged.csv)
- No way to query "which faces changed clusters during merge iteration 3"

## Solution

One SQLite DB per run: `{output_dir}/face_clustering.db`

7 tables, all relational, no JSON blobs:
- **faces** — immutable detection data
- **embeddings** — 512-dim vector per face
- **cluster_assignments** — face_id + cluster_id + iteration (core traceability)
- **clusters** — per-cluster metrics at each iteration
- **merge_decisions** — full evidence per candidate pair per iteration
- **face_scores** — pose/eyes/expression per face
- **run_metadata** — config and summary

See `docs/design/app/2026-05-04_face_clustering_db_data_model.md` for full schema, example queries, and migration plan.

## Acceptance Criteria

1. Pipeline writes `face_clustering.db` during export (alongside CSVs during transition)
2. Standalone FC app loads from DB when available, falls back to CSVs
3. Full traceability: `SELECT ... FROM faces JOIN embeddings JOIN cluster_assignments WHERE face_id = 42` returns the complete chain
4. Per-iteration cluster state queryable: base clusters, each merge iteration, final state
5. Merge decisions stored per-pair per-iteration with all gate evidence
6. Existing FC app functionality unchanged (all tabs still work)

## Design Decisions

- **Separate from UniversalCache**: UniversalCache is cross-run input caching. This DB is per-run output. Different purposes, different lifecycle.
- **Embeddings as binary blob**: 2048 bytes per face. Not human-readable — loaded into numpy for distance computation.
- **Store per-iteration cluster scores**: Diameter, avg_intra_dist stored per cluster per iteration. Cheap (few hundred rows) and avoids recomputing from embeddings.
- **Crops stay on disk**: JPEG files don't belong in SQLite. `faces.crop_path` column points to `crops/face_NNNN_aligned.jpg`.
- **Config as JSON**: Only exception to "no JSON" rule — PipelineConfig is genuinely a nested blob.

## Migration Path

1. Phase 1: Write DB alongside CSVs (both outputs). FC app reads CSVs as before.
2. Phase 2: FC app loader reads DB when `face_clustering.db` exists, falls back to CSVs.
3. Phase 3: Remove CSV writing. Delete old CSV code paths.
