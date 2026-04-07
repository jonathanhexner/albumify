"""
Face Clustering Workbench - Complete experimentation platform.

Self-contained tool for:
1. Processing albums (detect faces, extract embeddings, cluster)
2. Viewing results (cluster gallery, statistics)
3. Labeling clusters (manual corrections)
4. Training ML merge classifier

All modules from face_cluster/ package - separate from main app.

Usage:
    streamlit run app/face_clustering_workbench.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_cluster import (
    PipelineConfig,
    InsightFaceEmbedder,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
    ConservativeMerger,
    HoldoutAttacher,
    ClusterSnapshot,
    FaceRecord,
)

st.set_page_config(
    page_title="Face Clustering Workbench",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# Constants & Config
# ============================================================

HISTORY_FILE = Path("results/.face_clustering_history.json")
RESULTS_DIR = Path("results")

# Module documentation
MODULE_INFO = {
    "InsightFaceEmbedder": {
        "purpose": "Detects faces and extracts 512-dim embeddings",
        "input": "Album images",
        "output": "List[FaceRecord] with embeddings",
        "file": "face_cluster/embedding.py"
    },
    "QualityGater": {
        "purpose": "Filters low-quality faces (pose, blur, size)",
        "input": "List[FaceRecord]",
        "output": "Marks is_core=True/False",
        "file": "face_cluster/quality.py"
    },
    "KNNGraphBuilder": {
        "purpose": "Builds mutual k-NN graph from embeddings",
        "input": "Embeddings + distance threshold",
        "output": "GraphResult (edges, neighbors)",
        "file": "face_cluster/knn_graph.py"
    },
    "ConnectedComponentsClusterer": {
        "purpose": "Clusters faces via connected components on kNN graph",
        "input": "GraphResult",
        "output": "ClusterResult (labels, clusters)",
        "file": "face_cluster/clustering.py"
    },
    "D10ExemplarSelector": {
        "purpose": "Selects representative faces per cluster (d10 method)",
        "input": "ClusterResult + embeddings",
        "output": "Exemplar indices per cluster",
        "file": "face_cluster/exemplars.py"
    },
    "ConservativeMerger": {
        "purpose": "Merges similar clusters using adaptive thresholds",
        "input": "Clusters + exemplars",
        "output": "Merged ClusterResult",
        "file": "face_cluster/merge.py"
    },
    "HoldoutAttacher": {
        "purpose": "Attaches noise faces to nearest cluster",
        "input": "ClusterResult + noise faces",
        "output": "Updated ClusterResult",
        "file": "face_cluster/attach.py"
    }
}

# ============================================================
# Session State
# ============================================================

if 'current_run' not in st.session_state:
    st.session_state.current_run = None
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'pipeline_stage' not in st.session_state:
    st.session_state.pipeline_stage = None

# ============================================================
# History Management
# ============================================================

def load_history():
    """Load run history from JSON file."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_to_history(run_info):
    """Append run to history file."""
    history = load_history()
    history.append(run_info)
    # Keep last 50 runs
    history = history[-50:]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_run_results(output_dir: Path):
    """Load results from completed run."""
    try:
        faces_csv = output_dir / 'faces.csv'
        clusters_csv = output_dir / 'clusters.csv'
        summary_json = output_dir / 'export_summary.json'

        if not all([faces_csv.exists(), clusters_csv.exists()]):
            return {'success': False, 'error': 'Missing output files'}

        faces_df = pd.read_csv(faces_csv)
        clusters_df = pd.read_csv(clusters_csv)

        if summary_json.exists():
            with open(summary_json) as f:
                summary = json.load(f)
        else:
            summary = {}

        return {
            'success': True,
            'faces_df': faces_df,
            'clusters_df': clusters_df,
            'summary': summary,
            'output_dir': output_dir
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================
# Pipeline Processing
# ============================================================

def run_full_pipeline(album_path: Path, output_dir: Path, config: PipelineConfig,
                     progress_placeholder):
    """Run complete face clustering pipeline."""

    results = {
        'success': False,
        'stages': [],
        'output_dir': output_dir
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Detect & Embed
        progress_placeholder.info("🔍 Stage 1/7: Detecting faces and extracting embeddings...")
        st.session_state.pipeline_stage = "InsightFaceEmbedder"

        # Scan directory for images
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC']:
            image_paths.extend(album_path.rglob(f'*{ext}'))
        image_paths = [str(p) for p in image_paths]

        embedder = InsightFaceEmbedder(model_name='buffalo_l')
        face_records = embedder.detect_and_embed(image_paths)
        results['stages'].append(f"✓ Detected {len(face_records)} faces from {len(image_paths)} images")

        if len(face_records) == 0:
            results['error'] = "No faces detected in album"
            return results

        # Save face records
        face_records_json = output_dir / 'face_records.json'
        with open(face_records_json, 'w') as f:
            json.dump([{
                'face_id': i,
                'image_path': str(fr.image_path),
                'bbox': fr.bbox,
                'is_core': getattr(fr, 'is_core', True)
            } for i, fr in enumerate(face_records)], f, indent=2)

        # Stage 2: Quality Gate
        progress_placeholder.info("🎯 Stage 2/7: Filtering by quality...")
        st.session_state.pipeline_stage = "QualityGater"

        gater = QualityGater(config)
        face_records = gater.select_core_set(face_records)
        n_core = sum(1 for fr in face_records if fr.is_core)
        results['stages'].append(f"✓ {n_core}/{len(face_records)} faces passed quality gate")

        # Stage 3: Save crops
        progress_placeholder.info("💾 Stage 3/7: Saving face crops...")
        crops_dir = output_dir / 'face_crops'
        crops_dir.mkdir(exist_ok=True)

        import cv2
        for i, fr in enumerate(face_records):
            if hasattr(fr, 'aligned_face') and fr.aligned_face is not None:
                crop_path = crops_dir / f"face_{i:04d}_aligned.jpg"
                crop_bgr = cv2.cvtColor(fr.aligned_face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr)

        results['stages'].append(f"✓ Saved {len(face_records)} face crops")

        # Stage 4: Build kNN graph
        progress_placeholder.info("📊 Stage 4/7: Building k-NN graph...")
        st.session_state.pipeline_stage = "KNNGraphBuilder"

        core_records = [fr for fr in face_records if fr.is_core]
        embeddings = np.array([fr.embedding for fr in core_records])

        graph_builder = KNNGraphBuilder(k=config.K, distance_threshold=config.distance_threshold)
        graph_result = graph_builder.build(embeddings)
        results['stages'].append(f"✓ Built kNN graph: {len(graph_result.edges)} edges")

        # Stage 5: Cluster
        progress_placeholder.info("🧩 Stage 5/7: Clustering faces...")
        st.session_state.pipeline_stage = "ConnectedComponentsClusterer"

        clusterer = ConnectedComponentsClusterer()
        cluster_result = clusterer.cluster(graph_result, len(embeddings))
        results['stages'].append(f"✓ Created {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")

        # Stage 6: Select exemplars & merge
        progress_placeholder.info("🎯 Stage 6/7: Selecting exemplars and merging...")
        st.session_state.pipeline_stage = "D10ExemplarSelector"

        exemplar_selector = D10ExemplarSelector()
        cluster_result = exemplar_selector.select(cluster_result, embeddings)

        st.session_state.pipeline_stage = "ConservativeMerger"
        merger = ConservativeMerger()
        cluster_result = merger.merge(cluster_result, embeddings)
        results['stages'].append(f"✓ After merge: {cluster_result.n_clusters} clusters")

        # Stage 7: Export
        progress_placeholder.info("💾 Stage 7/7: Exporting results...")
        st.session_state.pipeline_stage = "Exporting"

        # Generate faces.csv
        faces_data = []
        for i, fr in enumerate(face_records):
            cluster_id = -1
            if i < len(core_records):
                core_idx = i
                if core_idx < len(cluster_result.labels):
                    cluster_id = int(cluster_result.labels[core_idx])

            faces_data.append({
                'face_id': i,
                'image_path': str(fr.image_path),
                'cluster_id': cluster_id,
                'is_core': fr.is_core,
                'bbox_x': fr.bbox.get('x', 0),
                'bbox_y': fr.bbox.get('y', 0)
            })

        faces_df = pd.DataFrame(faces_data)
        faces_df.to_csv(output_dir / 'faces.csv', index=False)

        # Generate clusters.csv
        clusters_data = []
        for cluster_id, members in cluster_result.clusters.items():
            if len(members) > 0:
                # Compute diameter
                cluster_embs = embeddings[members]
                if len(cluster_embs) > 1:
                    norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
                    cluster_embs_norm = cluster_embs / (norms + 1e-8)
                    similarity = cluster_embs_norm @ cluster_embs_norm.T
                    distance_matrix = 1.0 - similarity
                    np.fill_diagonal(distance_matrix, 0.0)
                    diameter = float(np.max(distance_matrix))
                else:
                    diameter = 0.0

                clusters_data.append({
                    'cluster_id': cluster_id,
                    'size': len(members),
                    'diameter': diameter,
                    'exemplar_face_id': members[0]
                })

        clusters_df = pd.DataFrame(clusters_data)
        clusters_df.to_csv(output_dir / 'clusters.csv', index=False)

        # Export summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'source_directory': str(album_path),
            'n_faces': len(face_records),
            'n_clusters': cluster_result.n_clusters,
            'n_noise': cluster_result.n_noise,
            'config': {
                'K': config.K,
                'distance_threshold': config.distance_threshold,
            }
        }
        with open(output_dir / 'export_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        results['stages'].append(f"✓ Exported faces.csv, clusters.csv")
        results['success'] = True
        results['summary'] = summary

        return results

    except Exception as e:
        results['error'] = str(e)
        results['exception'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()
        return results

# ============================================================
# Main UI
# ============================================================

st.title("🔬 Face Clustering Workbench")
st.markdown("**Experimentation platform** using `face_cluster/` modules - separate from main app")

# Show module info in sidebar
with st.sidebar:
    st.markdown("### 📦 Modules Used")
    st.markdown("*From `face_cluster/` package*")

    with st.expander("View Module Details", expanded=False):
        for module_name, info in MODULE_INFO.items():
            st.markdown(f"**{module_name}**")
            st.caption(f"📍 {info['file']}")
            st.caption(f"💡 {info['purpose']}")
            st.caption(f"⬇️ Input: {info['input']}")
            st.caption(f"⬆️ Output: {info['output']}")
            st.markdown("---")

    st.markdown("---")

    if st.session_state.current_run and st.session_state.current_run.get('success'):
        run = st.session_state.current_run
        st.markdown("**📊 Current Run**")
        st.caption(f"Output: `{run['output_dir'].name}`")
        st.caption(f"Faces: {run['summary'].get('n_faces', 0)}")
        st.caption(f"Clusters: {run['summary'].get('n_clusters', 0)}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Process Album",
    "📊 View Results",
    "🏷️ Label Clusters",
    "📜 History"
])

# ============================================================
# TAB 1: Process Album
# ============================================================

with tab1:
    st.header("Process Album")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Album selection
        use_test_data = st.checkbox("Use test data", value=True)

        if use_test_data:
            album_path = Path("test_data/face_clustering")
            st.info(f"📁 Using: `{album_path}` (9 images, 3 people)")
        else:
            album_input = st.text_input("Album path", value="D:/Google_Germany")
            album_path = Path(album_input)

        # Config
        st.subheader("Pipeline Configuration")

        col_k, col_thresh = st.columns(2)
        with col_k:
            k = st.number_input("k (kNN neighbors)", min_value=1, max_value=20, value=5,
                               help="Number of nearest neighbors for graph")
        with col_thresh:
            distance_threshold = st.slider("Distance threshold", 0.0, 1.0, 0.35, 0.05,
                                          help="Max distance for kNN edge")

        config = PipelineConfig(K=k, distance_threshold=distance_threshold)

        # Output name
        default_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_name = st.text_input("Run name", value=default_name)
        output_dir = RESULTS_DIR / output_name

        # Run button
        if st.button("▶️ Run Full Pipeline", type="primary", disabled=st.session_state.pipeline_running):
            if not album_path.exists():
                st.error(f"❌ Album not found: {album_path}")
            else:
                st.session_state.pipeline_running = True
                progress_placeholder = st.empty()

                # Run pipeline
                results = run_full_pipeline(album_path, output_dir, config, progress_placeholder)

                st.session_state.pipeline_running = False
                st.session_state.pipeline_stage = None

                if results['success']:
                    st.success("✅ Pipeline completed successfully!")

                    # Show stages
                    st.markdown("**Pipeline Stages:**")
                    for stage in results['stages']:
                        st.markdown(f"- {stage}")

                    # Load results
                    st.session_state.current_run = load_run_results(output_dir)

                    # Save to history
                    save_to_history({
                        'name': output_name,
                        'timestamp': datetime.now().isoformat(),
                        'source': str(album_path),
                        'n_faces': results['summary']['n_faces'],
                        'n_clusters': results['summary']['n_clusters'],
                        'config': {'k': k, 'distance_threshold': distance_threshold}
                    })

                    st.balloons()
                else:
                    st.error(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
                    if 'traceback' in results:
                        with st.expander("Error details"):
                            st.code(results['traceback'])

    with col2:
        st.subheader("Pipeline Flow")
        st.markdown("""
        **7-Stage Pipeline:**

        1. 🔍 **Detect & Embed**
           `InsightFaceEmbedder`

        2. 🎯 **Quality Gate**
           `QualityGater`

        3. 💾 **Save Crops**
           Export aligned faces

        4. 📊 **Build kNN Graph**
           `KNNGraphBuilder`

        5. 🧩 **Cluster**
           `ConnectedComponentsClusterer`

        6. 🎯 **Merge Clusters**
           `D10ExemplarSelector` + `ConservativeMerger`

        7. 💾 **Export**
           faces.csv, clusters.csv
        """)

        if st.session_state.pipeline_stage:
            st.info(f"**Currently running:** {st.session_state.pipeline_stage}")

# ============================================================
# TAB 2: View Results
# ============================================================

with tab2:
    st.header("View Results")

    if st.session_state.current_run and st.session_state.current_run.get('success'):
        run = st.session_state.current_run
        faces_df = run['faces_df']
        clusters_df = run['clusters_df']
        summary = run['summary']
        output_dir = run['output_dir']

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📷 Faces", summary.get('n_faces', len(faces_df)))
        col2.metric("👥 Clusters", summary.get('n_clusters', len(clusters_df)))
        col3.metric("🔇 Noise", summary.get('n_noise', 0))
        col4.metric("📁 Source", Path(summary.get('source_directory', '')).name)

        st.markdown("---")

        # Cluster selector
        cluster_ids = sorted(clusters_df['cluster_id'].tolist())
        selected_cluster = st.selectbox(
            "Select cluster",
            cluster_ids,
            format_func=lambda x: f"Cluster {x} ({len(faces_df[faces_df['cluster_id'] == x])} faces)"
        )

        # Cluster info
        cluster_info = clusters_df[clusters_df['cluster_id'] == selected_cluster].iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Size", int(cluster_info['size']))
        col2.metric("Diameter", f"{cluster_info['diameter']:.3f}")
        quality = "🟢 Good" if cluster_info['diameter'] < 0.5 else "🟡 Review"
        col3.metric("Quality", quality)

        # Show faces
        cluster_faces = faces_df[faces_df['cluster_id'] == selected_cluster]

        st.markdown(f"**Faces in Cluster {selected_cluster}:**")
        cols = st.columns(min(5, len(cluster_faces)))

        for idx, (_, face) in enumerate(cluster_faces.iterrows()):
            col = cols[idx % len(cols)]
            crop_path = output_dir / 'face_crops' / f"face_{int(face['face_id']):04d}_aligned.jpg"
            if crop_path.exists():
                img = Image.open(crop_path)
                col.image(img, caption=f"Face {int(face['face_id'])}", use_container_width=True)
                col.caption(f"📄 {Path(face['image_path']).name}")
    else:
        st.info("👈 Run pipeline first or load from History")

# ============================================================
# TAB 3: Label Clusters
# ============================================================

with tab3:
    st.header("Label Clusters")

    if st.session_state.current_run and st.session_state.current_run.get('success'):
        run = st.session_state.current_run
        faces_df = run['faces_df']
        clusters_df = run['clusters_df']
        output_dir = run['output_dir']

        st.markdown("Assign names to clusters for training data generation.")

        # Load/create labels
        labels_path = output_dir / 'corrected_labels.csv'
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
        else:
            labels_df = pd.DataFrame({
                'cluster_id': clusters_df['cluster_id'],
                'corrected_identity': [''] * len(clusters_df)
            })

        # Labeling interface
        for _, cluster in clusters_df.iterrows():
            cluster_id = int(cluster['cluster_id'])

            with st.expander(f"Cluster {cluster_id} ({int(cluster['size'])} faces)"):
                # Show faces
                cluster_faces = faces_df[faces_df['cluster_id'] == cluster_id]
                cols = st.columns(min(5, len(cluster_faces)))

                for idx, (_, face) in enumerate(cluster_faces.iterrows()):
                    col = cols[idx % len(cols)]
                    crop_path = output_dir / 'face_crops' / f"face_{int(face['face_id']):04d}_aligned.jpg"
                    if crop_path.exists():
                        img = Image.open(crop_path)
                        col.image(img, use_container_width=True)

                # Label input
                current_label = labels_df[labels_df['cluster_id'] == cluster_id]['corrected_identity'].iloc[0]
                new_label = st.text_input(
                    f"Identity for Cluster {cluster_id}",
                    value=current_label,
                    key=f"label_{cluster_id}",
                    placeholder="e.g., John, Sarah, Person1"
                )
                labels_df.loc[labels_df['cluster_id'] == cluster_id, 'corrected_identity'] = new_label

        # Save button
        if st.button("💾 Save Labels"):
            labels_df.to_csv(labels_path, index=False)
            st.success(f"✅ Labels saved to: {labels_path}")
    else:
        st.info("👈 Load results first")

# ============================================================
# TAB 4: History
# ============================================================

with tab4:
    st.header("Run History")

    history = load_history()

    if not history:
        st.info("No previous runs yet. Run a pipeline to create history.")
    else:
        st.markdown(f"**{len(history)} previous runs:**")

        for run_info in reversed(history[-10:]):  # Show last 10
            with st.expander(
                f"📁 {run_info['name']} - {run_info['n_faces']} faces, {run_info['n_clusters']} clusters"
            ):
                st.markdown(f"**Timestamp:** {run_info['timestamp']}")
                st.markdown(f"**Source:** `{run_info['source']}`")
                st.markdown(f"**Config:** k={run_info['config']['k']}, threshold={run_info['config']['distance_threshold']}")

                if st.button("📂 Load", key=f"load_{run_info['name']}"):
                    output_dir = RESULTS_DIR / run_info['name']
                    st.session_state.current_run = load_run_results(output_dir)
                    st.success("✅ Loaded!")
                    st.rerun()
