# Face Clustering Workflow - Quick Reference

## Regenerate Embeddings → Cluster → Label

When you need to regenerate embeddings from existing face crops and label clusters:

```bash
# 1. Regenerate embeddings from crops
python scripts/regenerate_embeddings_from_crops.py --face-crops results/Google_Germany/face_crops --output results/Google_Germany --metadata results/Google_Germany/benchmark_2026-03-01_01-10-04.json

# 2. Export clustering data (runs kNN graph clustering)
python scripts/export_clustering_data.py --embeddings results/Google_Germany/embeddings_FRESH_2026-03-17_01-00-27.npy --output results/Google_Germany/clustering_export

# 3. Open labeling app
streamlit run app/face_clustering_labeling.py
```

## What Each Step Does

1. **regenerate_embeddings_from_crops.py**
   - Reads aligned face crops from `face_crops/`
   - Extracts fresh embeddings using InsightFace
   - Saves `embeddings_FRESH_*.npy` and `embeddings_metadata_FRESH_*.json`

2. **export_clustering_data.py**
   - Runs kNN graph clustering on embeddings
   - Exports `faces.csv`, `clusters.csv`, `candidate_pairs.csv`, `export_summary.json`

3. **face_clustering_labeling.py**
   - Streamlit app to view clusters and assign corrected identities
   - Saves corrections to `corrected_identities.json`

## Other Useful Apps

```bash
# Debug clustering decisions
streamlit run app/face_clustering_debug/main.py

# Compare clustering methods side-by-side
streamlit run app/face_clustering_comparison.py
```
