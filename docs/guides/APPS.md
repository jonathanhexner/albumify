# Streamlit Apps

All commands assume Windows with `.venv` in project root.

| App | Command | Description |
|-----|---------|-------------|
| Albumify | `.venv/Scripts/streamlit run app/streamlit/main.py` | Main album UI (requires FastAPI backend: `.venv/Scripts/python -m uvicorn sim_bench.api.main:app --reload --port 8000`) |
| Face Clustering | `.venv/Scripts/streamlit run app/face_clustering/main.py` | Full clustering pipeline: run, recluster, view clusters, merge analysis |
| Face Clustering Debug | `.venv/Scripts/streamlit run app/face_clustering_debug/main.py` | Algorithm comparison, parameter tuning, decision visualization |
| Face Clustering Comparison | `.venv/Scripts/streamlit run app/face_clustering_comparison.py` | Side-by-side HDBSCAN vs Hybrid clustering comparison |
| Face Clustering Workbench | `.venv/Scripts/streamlit run app/face_clustering_workbench.py` | End-to-end experimentation: process albums, label, train ML merge classifier |
| Face Cluster Labeling | `.venv/Scripts/streamlit run app/face_clustering_labeling.py` | Manual identity labeling for ML training data |
| Simple Face Labeling | `.venv/Scripts/streamlit run app/simple_face_labeling.py` | Lightweight auto-loading labeling (no config needed) |
| Photo Analysis | `.venv/Scripts/streamlit run app/photo_analysis/main.py` | Batch photo analysis with CLIP tags, face detection, landmark recognition |
| Photo Organization | `.venv/Scripts/streamlit run app/photo_organization/main.py` | AI-driven photo organization by events, people, landmarks, quality |
| Albumify (Standalone) | `.venv/Scripts/streamlit run app/album/main.py` | Album organization with config panel and workflow form |
