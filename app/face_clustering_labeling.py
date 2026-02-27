"""
Streamlit app for manually labeling face clusters.

Allows users to assign corrected_identity to each cluster for ML training.

Usage:
    streamlit run app/face_clustering_labeling.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import json
from datetime import datetime

st.set_page_config(
    page_title="Face Cluster Labeling",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Face Cluster Labeling")
st.markdown("Assign corrected identities to clusters for ML training")

# Session state initialization
if 'export_dir' not in st.session_state:
    st.session_state.export_dir = None
if 'faces_df' not in st.session_state:
    st.session_state.faces_df = None
if 'clusters_df' not in st.session_state:
    st.session_state.clusters_df = None
if 'labels_df' not in st.session_state:
    st.session_state.labels_df = None
if 'identity_suggestions' not in st.session_state:
    st.session_state.identity_suggestions = []


@st.cache_data
def load_csvs(export_dir: Path):
    """Load exported CSV files."""
    faces_df = pd.read_csv(export_dir / 'faces.csv')
    clusters_df = pd.read_csv(export_dir / 'clusters.csv')

    # Try to load existing labels
    labels_path = export_dir / 'corrected_labels.csv'
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
    else:
        # Initialize with cluster IDs as default identities
        labels_df = pd.DataFrame({
            'cluster_id': clusters_df['cluster_id'],
            'corrected_identity': [f"person_{i}" for i in clusters_df['cluster_id']]
        })

    return faces_df, clusters_df, labels_df


def get_cluster_face_images(cluster_id: int, faces_df: pd.DataFrame, crops_dir: Path, max_faces: int = 20):
    """Load face crop images for a cluster."""
    cluster_faces = faces_df[faces_df['cluster_id'] == cluster_id].head(max_faces)
    images = []

    for _, face_row in cluster_faces.iterrows():
        face_id = face_row['face_id']

        # Try to load face crop
        for pattern in [f"face_{face_id:04d}_aligned.jpg", f"face_{face_id:04d}.jpg"]:
            crop_path = crops_dir / pattern
            if crop_path.exists():
                try:
                    img = Image.open(crop_path)
                    images.append((face_id, img))
                    break
                except Exception as e:
                    st.warning(f"Failed to load {crop_path}: {e}")

    return images


def save_labels(export_dir: Path, labels_df: pd.DataFrame):
    """Save corrected labels to CSV."""
    labels_path = export_dir / 'corrected_labels.csv'
    labels_df.to_csv(labels_path, index=False)

    # Also save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_clusters': len(labels_df),
        'n_identities': labels_df['corrected_identity'].nunique(),
        'identity_counts': labels_df['corrected_identity'].value_counts().to_dict()
    }

    metadata_path = export_dir / 'labeling_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return labels_path, metadata_path


# Sidebar: Load data
st.sidebar.header("📂 Data Source")

default_dir = "results/face_clustering_training/test_v2"
export_dir_input = st.sidebar.text_input(
    "Export directory",
    value=default_dir,
    help="Directory containing faces.csv, clusters.csv from export script"
)

if st.sidebar.button("Load Data") or st.session_state.export_dir is None:
    export_dir = Path(export_dir_input)

    if not export_dir.exists():
        st.error(f"Directory not found: {export_dir}")
    elif not (export_dir / 'faces.csv').exists():
        st.error(f"faces.csv not found in {export_dir}")
    elif not (export_dir / 'clusters.csv').exists():
        st.error(f"clusters.csv not found in {export_dir}")
    else:
        st.session_state.export_dir = export_dir
        st.session_state.faces_df, st.session_state.clusters_df, st.session_state.labels_df = load_csvs(export_dir)

        # Extract identity suggestions from existing labels
        st.session_state.identity_suggestions = sorted(
            st.session_state.labels_df['corrected_identity'].unique().tolist()
        )

        st.sidebar.success(f"✅ Loaded {len(st.session_state.clusters_df)} clusters")

# Main UI
if st.session_state.export_dir is not None:
    export_dir = st.session_state.export_dir
    faces_df = st.session_state.faces_df
    clusters_df = st.session_state.clusters_df
    labels_df = st.session_state.labels_df

    # Check for face crops directory
    crops_dir = export_dir / 'face_crops'
    if not crops_dir.exists():
        # Try parent directory
        crops_dir = export_dir.parent / 'face_crops'

    if not crops_dir.exists():
        st.warning(f"⚠️ Face crops directory not found. Expected at: {export_dir / 'face_crops'}")
        st.info("Face images won't be displayed, but you can still assign labels based on cluster IDs.")

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Faces", len(faces_df))
    with col2:
        st.metric("Clusters", len(clusters_df))
    with col3:
        n_identities = labels_df['corrected_identity'].nunique()
        st.metric("Unique Identities", n_identities)
    with col4:
        n_noise = len(faces_df[faces_df['cluster_id'] == -1])
        st.metric("Noise Faces", n_noise)

    # Identity distribution
    with st.expander("📊 Identity Distribution", expanded=False):
        identity_counts = labels_df['corrected_identity'].value_counts()
        st.bar_chart(identity_counts)
        st.dataframe(
            identity_counts.reset_index().rename(columns={'index': 'Identity', 'corrected_identity': 'Count'}),
            hide_index=True
        )

    # Labeling interface
    st.header("🏷️ Assign Identities to Clusters")

    # Sort options
    sort_by = st.selectbox(
        "Sort clusters by",
        ["Cluster ID", "Size (largest first)", "Size (smallest first)"],
        index=0
    )

    if sort_by == "Cluster ID":
        sorted_clusters = clusters_df.sort_values('cluster_id')
    elif sort_by == "Size (largest first)":
        sorted_clusters = clusters_df.sort_values('cluster_size', ascending=False)
    else:
        sorted_clusters = clusters_df.sort_values('cluster_size', ascending=True)

    # Pagination
    clusters_per_page = st.slider("Clusters per page", min_value=5, max_value=50, value=10, step=5)
    n_pages = (len(sorted_clusters) + clusters_per_page - 1) // clusters_per_page
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)

    start_idx = (page - 1) * clusters_per_page
    end_idx = min(start_idx + clusters_per_page, len(sorted_clusters))
    page_clusters = sorted_clusters.iloc[start_idx:end_idx]

    st.markdown(f"Showing clusters {start_idx + 1}-{end_idx} of {len(sorted_clusters)}")

    # Track changes
    changes_made = False

    # Display each cluster
    for idx, cluster_row in page_clusters.iterrows():
        cluster_id = cluster_row['cluster_id']
        cluster_size = cluster_row['cluster_size']

        st.markdown("---")

        # Cluster header
        col_header, col_identity = st.columns([2, 3])

        with col_header:
            st.subheader(f"Cluster {cluster_id}")
            st.caption(f"Size: {cluster_size} faces | Diameter: {cluster_row['diameter']:.3f}")

        with col_identity:
            # Get current identity
            current_identity = labels_df.loc[labels_df['cluster_id'] == cluster_id, 'corrected_identity'].iloc[0]

            # Identity input with suggestions
            new_identity = st.selectbox(
                "Corrected Identity",
                options=[''] + st.session_state.identity_suggestions + ['[New Identity]'],
                index=st.session_state.identity_suggestions.index(current_identity) + 1 if current_identity in st.session_state.identity_suggestions else 0,
                key=f"select_{cluster_id}"
            )

            # Handle new identity
            if new_identity == '[New Identity]':
                new_identity = st.text_input(
                    "Enter new identity",
                    value=current_identity,
                    key=f"text_{cluster_id}"
                )
            elif new_identity == '':
                new_identity = current_identity

            # Update if changed
            if new_identity != current_identity:
                labels_df.loc[labels_df['cluster_id'] == cluster_id, 'corrected_identity'] = new_identity

                # Add to suggestions if new
                if new_identity not in st.session_state.identity_suggestions:
                    st.session_state.identity_suggestions.append(new_identity)
                    st.session_state.identity_suggestions.sort()

                changes_made = True

        # Display face crops
        if crops_dir.exists():
            images = get_cluster_face_images(cluster_id, faces_df, crops_dir, max_faces=20)

            if len(images) > 0:
                # Display in grid
                cols_per_row = 10
                n_rows = (len(images) + cols_per_row - 1) // cols_per_row

                for row in range(n_rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        img_idx = row * cols_per_row + col_idx
                        if img_idx < len(images):
                            face_id, img = images[img_idx]
                            with cols[col_idx]:
                                st.image(img, caption=f"{face_id}", use_container_width=True)
            else:
                st.info(f"No face crops found for cluster {cluster_id}")
        else:
            st.info("Face crops directory not available")

    # Save button
    st.markdown("---")
    col_save, col_reset, col_stats = st.columns([1, 1, 2])

    with col_save:
        if st.button("💾 Save Labels", type="primary", use_container_width=True):
            labels_path, metadata_path = save_labels(export_dir, labels_df)
            st.success(f"✅ Saved to:\n- {labels_path.name}\n- {metadata_path.name}")
            st.session_state.labels_df = labels_df  # Update session state
            st.rerun()

    with col_reset:
        if st.button("🔄 Reset All", use_container_width=True):
            # Reset to default (cluster IDs)
            labels_df['corrected_identity'] = [f"person_{i}" for i in labels_df['cluster_id']]
            st.session_state.labels_df = labels_df
            st.rerun()

    with col_stats:
        if changes_made:
            st.info("⚠️ Unsaved changes")

    # Instructions
    with st.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        ### Labeling Workflow

        1. **Load Data**: Enter the export directory path and click "Load Data"
        2. **Review Clusters**: Look at the face images in each cluster
        3. **Assign Identities**:
           - Select existing identity from dropdown
           - Or choose "[New Identity]" and enter a new name
           - Use consistent names (e.g., "person_A", "person_B", "noise")
        4. **Save Progress**: Click "Save Labels" to save your work
        5. **Next Steps**: Use saved `corrected_labels.csv` for training

        ### Tips

        - **Noise clusters**: Label as "noise" if faces don't belong together
        - **Same person**: Give all clusters of the same person the same identity
        - **Pagination**: Use page controls to navigate through many clusters
        - **Sorting**: Sort by size to handle large clusters first

        ### After Labeling

        Run the training script:
        ```bash
        python scripts/train_merge_classifier.py \\
            --candidate-pairs {export_dir}/candidate_pairs.csv \\
            --corrected-labels {export_dir}/corrected_labels.csv \\
            --output {export_dir}/model
        ```
        """.format(export_dir=export_dir))

else:
    st.info("👈 Enter export directory path in sidebar and click 'Load Data' to begin")

    st.markdown("""
    ### Getting Started

    1. First, export clustering data:
       ```bash
       python scripts/export_clustering_data.py \\
           --embeddings path/to/embeddings.npy \\
           --output results/face_clustering_training/dataset1
       ```

    2. Then load the output directory in this app

    3. Assign corrected identities to each cluster

    4. Save labels and proceed to training
    """)
