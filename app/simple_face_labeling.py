"""
Simple face clustering labeling app - NO manual configuration needed.

Automatically loads from results/test_face_clustering_e2e/

Usage:
    streamlit run app/simple_face_labeling.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import json

# Configuration
DATA_DIR = Path("results/test_face_clustering_e2e")
CROPS_DIR = DATA_DIR / "face_crops"

st.set_page_config(
    page_title="Face Cluster Labeling",
    page_icon="🏷️",
    layout="wide"
)

st.title("🏷️ Face Cluster Labeling")
st.markdown(f"**Data**: `{DATA_DIR}`")

# Load data
@st.cache_data
def load_data():
    """Load clustering data."""
    try:
        faces_df = pd.read_csv(DATA_DIR / 'faces.csv')
        clusters_df = pd.read_csv(DATA_DIR / 'clusters.csv')

        # Load or create labels
        labels_path = DATA_DIR / 'corrected_labels.csv'
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
        else:
            # Initialize with person_label from ground truth
            cluster_to_person = {}
            for _, row in faces_df.iterrows():
                cid = row['cluster_id']
                if cid not in cluster_to_person:
                    cluster_to_person[cid] = row.get('person_label', f'person_{cid}')

            labels_df = pd.DataFrame({
                'cluster_id': list(cluster_to_person.keys()),
                'corrected_identity': list(cluster_to_person.values())
            })

        return faces_df, clusters_df, labels_df

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.error(f"Expected files in: {DATA_DIR}")
        st.stop()

faces_df, clusters_df, labels_df = load_data()

# Statistics
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Faces", len(faces_df))
with col2:
    st.metric("Clusters", len(clusters_df))
with col3:
    st.metric("Unique People", len(labels_df['corrected_identity'].unique()))

st.markdown("---")

# Display clusters
st.subheader("Clusters")

for _, cluster_row in clusters_df.iterrows():
    cluster_id = int(cluster_row['cluster_id'])
    size = int(cluster_row['size'])

    # Get current label
    label_row = labels_df[labels_df['cluster_id'] == cluster_id]
    current_label = label_row['corrected_identity'].iloc[0] if len(label_row) > 0 else f"person_{cluster_id}"

    # Cluster header with label editing
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Cluster {cluster_id} ({size} faces)")
    with col2:
        new_label = st.text_input(
            f"Label for cluster {cluster_id}",
            value=current_label,
            key=f"label_{cluster_id}",
            label_visibility="collapsed"
        )

        # Update label if changed
        if new_label != current_label:
            labels_df.loc[labels_df['cluster_id'] == cluster_id, 'corrected_identity'] = new_label
            st.session_state['labels_changed'] = True

    # Display faces in this cluster
    cluster_faces = faces_df[faces_df['cluster_id'] == cluster_id]

    if len(cluster_faces) > 0:
        # Show face images
        cols = st.columns(min(6, len(cluster_faces)))

        for idx, (_, face_row) in enumerate(cluster_faces.iterrows()):
            face_id = int(face_row['face_id'])
            crop_path = CROPS_DIR / f"face_{face_id:04d}_aligned.jpg"

            with cols[idx % 6]:
                if crop_path.exists():
                    try:
                        img = Image.open(crop_path)
                        st.image(img, caption=f"Face {face_id}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading {crop_path}: {e}")
                else:
                    st.warning(f"Missing: face_{face_id:04d}")

                # Show ground truth for comparison
                if 'person_label' in face_row:
                    st.caption(f"GT: {face_row['person_label']}")

    st.markdown("---")

# Save button
st.markdown("### Save Corrected Labels")

if st.button("💾 Save Labels", type="primary"):
    output_path = DATA_DIR / 'corrected_labels.csv'
    labels_df.to_csv(output_path, index=False)
    st.success(f"✅ Saved to {output_path}")

    # Show summary
    st.markdown("**Summary:**")
    for _, row in labels_df.iterrows():
        cluster_id = row['cluster_id']
        identity = row['corrected_identity']
        size = clusters_df[clusters_df['cluster_id'] == cluster_id]['size'].iloc[0]
        st.markdown(f"- Cluster {cluster_id} ({size} faces) → **{identity}**")

# Debugging info
with st.expander("🔍 Debug Info"):
    st.markdown("**Faces DataFrame:**")
    st.dataframe(faces_df)

    st.markdown("**Clusters DataFrame:**")
    st.dataframe(clusters_df)

    st.markdown("**Labels DataFrame:**")
    st.dataframe(labels_df)

    st.markdown("**Files:**")
    st.code(f"""
Data directory: {DATA_DIR.absolute()}
Crops directory: {CROPS_DIR.absolute()}
Crops exist: {CROPS_DIR.exists()}
Crop count: {len(list(CROPS_DIR.glob('*.jpg'))) if CROPS_DIR.exists() else 0}
    """)
